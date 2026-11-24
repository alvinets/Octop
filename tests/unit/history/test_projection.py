from __future__ import annotations

import asyncio
import json

import pytest
from langchain_core.messages import HumanMessage

from octop.infra.gateway.history_backfill import HistoryBackfillQueue
from octop.infra.gateway.process.message_keys import (
    CHECKPOINT_TS_KEY,
    STREAM_ERROR_FLAG,
    parse_checkpoint_ts_ms,
)
from octop.infra.history.projection import (
    TurnHistoryTracker,
    _wire_text,
    live_message_input,
    message_input,
)


def test_message_input_preserves_existing_checkpoint_ts() -> None:
    item = message_input(
        HumanMessage(
            content="hi",
            additional_kwargs={CHECKPOINT_TS_KEY: 1_700_000_000_000},
        )
    )
    assert item is not None
    wire = json.loads(item.message_json)
    assert wire["data"]["additional_kwargs"][CHECKPOINT_TS_KEY] == 1_700_000_000_000
    assert item.created_at == 1_700_000_000


def test_message_input_does_not_invent_stamp_by_default() -> None:
    item = message_input(HumanMessage(content="hi"))
    assert item is not None
    wire = json.loads(item.message_json)
    assert CHECKPOINT_TS_KEY not in (wire["data"].get("additional_kwargs") or {})


def test_live_message_input_stamps_when_missing() -> None:
    item = live_message_input(
        HumanMessage(content="hi"),
        now_ms=1_700_000_000_123,
    )
    assert item is not None
    wire = json.loads(item.message_json)
    assert wire["data"]["additional_kwargs"][CHECKPOINT_TS_KEY] == 1_700_000_000_123
    assert item.created_at == 1_700_000_000


def test_live_message_input_preserves_existing_stamp() -> None:
    item = live_message_input(
        HumanMessage(
            content="hi",
            additional_kwargs={CHECKPOINT_TS_KEY: 1_700_000_000_000},
        ),
        now_ms=9,
    )
    assert item is not None
    wire = json.loads(item.message_json)
    assert wire["data"]["additional_kwargs"][CHECKPOINT_TS_KEY] == 1_700_000_000_000


def test_parse_checkpoint_ts_ms() -> None:
    assert parse_checkpoint_ts_ms(1_700_000_000) == 1_700_000_000_000
    assert parse_checkpoint_ts_ms(1_700_000_000_000) == 1_700_000_000_000
    assert parse_checkpoint_ts_ms(0) is None
    assert parse_checkpoint_ts_ms(True) is None
    assert parse_checkpoint_ts_ms(None) is None


def test_stream_tokens_stamp_checkpoint_ts() -> None:
    tracker = TurnHistoryTracker.from_request(
        {"messages": [{"role": "user", "content": "hi", "id": "u1"}]}
    )
    tracker.observe({"type": "token", "content": "hello"})
    assistant = next(item for item in tracker.inputs if item.role in ("ai", "assistant"))
    wire = json.loads(assistant.message_json)
    assert CHECKPOINT_TS_KEY in (wire["data"].get("additional_kwargs") or {})


def test_turn_tracker_keeps_only_latest_user_turn_and_dedupes_replay() -> None:
    tracker = TurnHistoryTracker.from_request(
        {"messages": [{"role": "user", "content": "latest", "id": "u2"}]}
    )
    state = [
        {"role": "user", "content": "old", "id": "u1"},
        {"role": "assistant", "content": "old answer", "id": "a1"},
        {"role": "user", "content": "latest", "id": "u2"},
        {"role": "assistant", "content": "new answer", "id": "a2"},
    ]
    tracker.observe({"type": "state_snapshot", "data": {"messages": state}})
    tracker.observe({"type": "state_update", "data": {"messages": state}})

    assert [item.message_id for item in tracker.inputs] == ["u2", "a2"]


def test_turn_tracker_keeps_streamed_tokens_and_error() -> None:
    tracker = TurnHistoryTracker.from_request(
        {"messages": [{"role": "user", "content": "continue this", "id": "u1"}]}
    )
    tracker.observe({"type": "token", "content": "partial "})
    tracker.observe({"type": "token", "content": "answer"})
    tracker.observe(
        {
            "type": "error",
            "message": "模型服务返回余额或额度不足。",
            "error_code": "TOKEN_QUOTA_EXCEEDED",
        }
    )

    texts = [_wire_text(item) for item in tracker.inputs]
    assert any("continue this" in text for text in texts)
    assert "partial answer" in texts
    assert any("余额或额度不足" in text for text in texts)
    error = next(item for item in tracker.inputs if "余额或额度不足" in _wire_text(item))
    assert STREAM_ERROR_FLAG in error.message_json


def test_turn_tracker_does_not_duplicate_state_assistant() -> None:
    tracker = TurnHistoryTracker.from_request(
        {"messages": [{"role": "user", "content": "hi", "id": "u1"}]}
    )
    tracker.observe({"type": "token", "content": "hello"})
    tracker.observe(
        {
            "type": "state_snapshot",
            "data": {
                "messages": [
                    {"role": "user", "content": "hi", "id": "u1"},
                    {"role": "assistant", "content": "hello world", "id": "a1"},
                ]
            },
        }
    )

    texts = [_wire_text(item) for item in tracker.inputs]
    assert texts.count("hello") == 0
    assert any(text == "hello world" for text in texts)


@pytest.mark.asyncio
async def test_backfill_queue_runs_one_job_at_a_time_and_dedupes() -> None:
    queue = HistoryBackfillQueue(max_pending=2)
    first_started = asyncio.Event()
    release_first = asyncio.Event()
    order: list[str] = []

    async def first() -> None:
        order.append("first:start")
        first_started.set()
        await release_first.wait()
        order.append("first:end")

    async def second() -> None:
        order.append("second")

    assert queue.enqueue("thr-1", first) is True
    assert queue.enqueue("thr-1", first) is True
    assert queue.enqueue("thr-2", second) is True
    assert queue.available_slots == 0
    assert queue.active_jobs == 2
    assert queue.contains("thr-1") is True
    await first_started.wait()
    assert order == ["first:start"]
    assert queue.available_slots == 1
    release_first.set()
    await asyncio.wait_for(queue._queue.join(), timeout=1)  # noqa: SLF001
    assert order == ["first:start", "first:end", "second"]
    assert queue.active_jobs == 0
    assert queue.available_slots == 2
    assert queue.contains("thr-1") is False
    await queue.close()


def test_multistep_turn_no_extra_concatenated_ai_message() -> None:
    """Multi-step turn where all step texts are present in state must NOT
    append a bogus concatenated assistant message."""
    t = TurnHistoryTracker.from_request({"messages": []})
    t.observe({"type": "token", "content": "Let me create the file.\n\n"})
    t.observe({"type": "token", "content": "Now I'll build the docs.\n\n"})
    t.observe(
        {
            "type": "state_snapshot",
            "data": {
                "messages": [
                    {"type": "human", "content": "do the task", "id": "h1"},
                    {
                        "type": "ai",
                        "content": "Let me create the file.\n\n",
                        "id": "a1",
                        "tool_calls": [{"name": "write_file", "args": {}, "id": "c1"}],
                    },
                    {"type": "tool", "content": "ok", "id": "t1", "tool_call_id": "c1"},
                    {
                        "type": "ai",
                        "content": "Now I'll build the docs.\n\n",
                        "id": "a2",
                        "tool_calls": [{"name": "execute", "args": {}, "id": "c2"}],
                    },
                ]
            },
        }
    )

    ai_items = [item for item in t.inputs if item.role in ("ai", "assistant")]
    assert len(ai_items) == 2
    texts = [_wire_text(item) for item in ai_items]
    assert "Let me create the file.\n\n" in texts
    assert "Now I'll build the docs.\n\n" in texts


def test_cancelled_turn_appends_only_uncovered_final_segment() -> None:
    """Turn cancelled before the final answer: append ONLY the uncovered final
    segment, never the concatenation of earlier preambles."""
    t = TurnHistoryTracker.from_request({"messages": []})
    t.observe({"type": "token", "content": "Step one text.\n\n"})
    t.observe(
        {
            "type": "state_snapshot",
            "data": {
                "messages": [
                    {"type": "human", "content": "do it", "id": "h1"},
                    {
                        "type": "ai",
                        "content": "Step one text.\n\n",
                        "id": "a1",
                        "tool_calls": [{"name": "write_file", "args": {}, "id": "c1"}],
                    },
                    {"type": "tool", "content": "ok", "id": "t1", "tool_call_id": "c1"},
                ]
            },
        }
    )
    t.observe({"type": "token", "content": "Step two partial...\n\n"})
    t.observe({"type": "token", "content": "Still typing"})

    ai_items = [item for item in t.inputs if item.role in ("ai", "assistant")]
    texts = [_wire_text(item) for item in ai_items]
    concat = "Step one text.\n\nStep two partial...\n\nStill typing"
    assert concat not in texts
    assert "Step two partial...\n\nStill typing" in texts


def test_single_step_no_state_appends_streamed_text_once() -> None:
    """Single-step turn with no state messages: append the streamed text
    exactly once (preserve today's behavior)."""
    t = TurnHistoryTracker.from_request({"messages": []})
    t.observe({"type": "token", "content": "hello"})
    t.observe({"type": "token", "content": " world"})

    ai_items = [item for item in t.inputs if item.role in ("ai", "assistant")]
    assert len(ai_items) == 1
    assert _wire_text(ai_items[0]) == "hello world"


def test_empty_stream_no_reasoning_appends_nothing() -> None:
    """Empty stream + no reasoning: append nothing (preserve)."""
    t = TurnHistoryTracker.from_request(
        {"messages": [{"type": "human", "content": "hi", "id": "h1"}]}
    )

    ai_items = [item for item in t.inputs if item.role in ("ai", "assistant")]
    assert len(ai_items) == 0


def test_reasoning_follows_same_segmentation() -> None:
    """Reasoning text follows the same segmentation; each segment carries its
    own reasoning_content in additional_kwargs."""
    t = TurnHistoryTracker.from_request({"messages": []})
    t.observe({"type": "reasoning", "content": "think1"})
    t.observe({"type": "token", "content": "answer1"})
    t.observe(
        {
            "type": "state_snapshot",
            "data": {
                "messages": [
                    {"type": "human", "content": "do it", "id": "h1"},
                    {
                        "type": "ai",
                        "content": "answer1",
                        "id": "a1",
                        "tool_calls": [{"name": "tool1", "args": {}, "id": "c1"}],
                    },
                    {"type": "tool", "content": "ok", "id": "t1", "tool_call_id": "c1"},
                ]
            },
        }
    )
    t.observe({"type": "reasoning", "content": "think2"})
    t.observe({"type": "token", "content": "answer2"})

    ai_items = [item for item in t.inputs if item.role in ("ai", "assistant")]
    assert len(ai_items) == 2
    texts = [_wire_text(item) for item in ai_items]
    assert "answer1" in texts
    assert "answer2" in texts

    extra2 = json.loads(ai_items[1].message_json)["data"].get("additional_kwargs", {})
    assert extra2.get("reasoning_content") == "think2"
