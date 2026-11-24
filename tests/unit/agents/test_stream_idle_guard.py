"""Tests for :mod:`octop.infra.agents.stream_guard`."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

import pytest

from octop.i18n.domains.stream import STREAM_STALL, format_stream_error
from octop.infra.agents.stream_guard import (
    StreamChunkTimeoutError,
    StreamRepetitionError,
    stream_guard,
)

AGENT_ID = "test-agent"
THREAD_ID = "test-thread"


async def _drain(aiter: AsyncIterator[Any], *, max_chunks: int | None = None) -> list[Any]:
    items: list[Any] = []
    async for item in aiter:
        items.append(item)
        if max_chunks is not None and len(items) >= max_chunks:
            break
    return items


@pytest.mark.asyncio
async def test_timeout_blocks_forever_raises_stream_chunk_timeout() -> None:
    """Chunks yield then block forever → StreamChunkTimeoutError within timeout."""
    timeout = 0.05
    yielded: list[Any] = []
    blocked = asyncio.Event()

    async def fake_stream() -> AsyncIterator[Any]:
        for i in range(2):
            yielded.append(i)
            yield {"type": "token", "content": str(i)}
        await blocked.wait()  # never resolves

    guard = stream_guard(
        fake_stream(),
        idle_timeout_s=timeout,
        cancel_stream=lambda *_: None,
        agent_id=AGENT_ID,
        thread_id=THREAD_ID,
    )
    with pytest.raises(StreamChunkTimeoutError) as exc_info:
        async for _ in guard:
            pass

    assert exc_info.value.timeout_s == timeout
    assert yielded == [0, 1]
    assert blocked.is_set() is False  # generator didn't get far enough to block


@pytest.mark.asyncio
async def test_external_cancel_propagates_as_cancelled_error() -> None:
    """Task.cancel() while blocked must raise CancelledError, not StreamChunkTimeoutError."""
    timeout = 0.5
    blocked = asyncio.Event()

    async def fake_stream() -> AsyncIterator[Any]:
        while True:
            await blocked.wait()
            yield {"type": "token", "content": "x"}

    guard = stream_guard(
        fake_stream(),
        idle_timeout_s=timeout,
        cancel_stream=lambda *_: None,
        agent_id=AGENT_ID,
        thread_id=THREAD_ID,
    )

    async def drain() -> list[Any]:
        return [item async for item in guard]

    task = asyncio.create_task(drain())
    # Let the guard install its wait_for
    await asyncio.sleep(0.01)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_tool_phase_suspends_deadline_then_resumes() -> None:
    """Silence during tool phase does NOT raise; silence after tool_result does."""
    timeout = 0.05

    async def fake_stream() -> AsyncIterator[Any]:
        yield {"type": "token", "content": "hello"}
        # Enter tool phase
        yield {
            "type": "state_update",
            "node": "agent",
            "data": {
                "messages": [
                    {"role": "assistant", "tool_calls": [{"id": "t1", "name": "x", "args": {}}]}
                ]
            },
        }
        # Tool phase: silence longer than timeout — should NOT raise
        await asyncio.sleep(timeout * 4)
        # Leave tool phase
        yield {
            "type": "tool_result",
            "node": "agent",
            "messages": [{"role": "tool", "tool_call_id": "t1", "content": ""}],
        }
        # Back to generating: silence NOW should raise
        await asyncio.sleep(timeout * 4)

    guard = stream_guard(
        fake_stream(),
        idle_timeout_s=timeout,
        cancel_stream=lambda *_: None,
        agent_id=AGENT_ID,
        thread_id=THREAD_ID,
    )
    chunks: list[Any] = []
    with pytest.raises(StreamChunkTimeoutError):
        async for chunk in guard:
            chunks.append(chunk)
    types = [c["type"] for c in chunks]
    assert "token" in types
    assert "state_update" in types
    assert "tool_result" in types


@pytest.mark.asyncio
async def test_idle_timeout_zero_disables_guard() -> None:
    """idle_timeout_s=0 means no deadline — silence never raises."""

    async def fake_stream() -> AsyncIterator[Any]:
        yield {"type": "token", "content": "a"}
        await asyncio.sleep(0.3)
        yield {"type": "token", "content": "b"}

    guard = stream_guard(
        fake_stream(),
        idle_timeout_s=0,
        cancel_stream=lambda *_: None,
        agent_id=AGENT_ID,
        thread_id=THREAD_ID,
    )
    chunks = await asyncio.wait_for(_drain(guard), timeout=1.0)
    assert [c["content"] for c in chunks] == ["a", "b"]


@pytest.mark.asyncio
async def test_chunks_faster_than_timeout_completes_normally() -> None:
    """Rapid chunks complete without timeout."""
    timeout = 0.5

    async def fake_stream() -> AsyncIterator[Any]:
        for i in range(5):
            yield {"type": "token", "content": str(i)}

    guard = stream_guard(
        fake_stream(),
        idle_timeout_s=timeout,
        cancel_stream=lambda *_: None,
        agent_id=AGENT_ID,
        thread_id=THREAD_ID,
    )
    chunks = await _drain(guard)
    assert len(chunks) == 5


@pytest.mark.asyncio
async def test_format_stream_error_classifies_stream_chunk_timeout() -> None:
    """format_stream_error returns EN/ZH stream_stall text for StreamChunkTimeoutError."""
    exc = StreamChunkTimeoutError(timeout_s=120.0)
    from octop.i18n.domains.stream import classify_stream_error_message

    assert classify_stream_error_message(str(exc)) == STREAM_STALL
    en_text = format_stream_error(exc, "en")
    zh_text = format_stream_error(exc, "zh")
    assert zh_text != en_text  # different locales produce different text
    assert en_text  # non-empty
    assert zh_text  # non-empty


@pytest.mark.asyncio
async def test_repetition_aborts_the_turn() -> None:
    """A fragment repeated without converging ends the turn instead of streaming forever."""
    fragment = (
        "Response interrupted - let me restart the document creation process properly, again.\n\n"
    )
    cancelled: list[tuple[str, str]] = []

    async def fake_stream() -> AsyncIterator[Any]:
        for _ in range(40):
            yield {"type": "token", "content": fragment}

    guard = stream_guard(
        fake_stream(),
        idle_timeout_s=0,
        cancel_stream=lambda agent, thread: cancelled.append((agent, thread)),
        agent_id=AGENT_ID,
        thread_id=THREAD_ID,
    )
    with pytest.raises(StreamRepetitionError) as exc_info:
        async for _ in guard:
            pass

    assert cancelled == [(AGENT_ID, THREAD_ID)]
    assert exc_info.value.period == len(fragment)
    assert exc_info.value.repeats >= 6


@pytest.mark.asyncio
async def test_normal_prose_is_not_repetition() -> None:
    text = "".join(f"Sentence {i} explains a distinct step of the plan.\n" for i in range(300))

    async def fake_stream() -> AsyncIterator[Any]:
        for start in range(0, len(text), 200):
            yield {"type": "token", "content": text[start : start + 200]}

    guard = stream_guard(
        fake_stream(),
        idle_timeout_s=0,
        cancel_stream=lambda *_: None,
        agent_id=AGENT_ID,
        thread_id=THREAD_ID,
    )
    chunks = await _drain(guard)
    assert "".join(c["content"] for c in chunks) == text


@pytest.mark.asyncio
async def test_repetition_counter_resets_at_step_boundaries() -> None:
    """Each model step starts clean, so a repeated fragment per step is not one long loop."""
    fragment = "x" * 100

    async def fake_stream() -> AsyncIterator[Any]:
        for _ in range(3):
            for _ in range(4):
                yield {"type": "token", "content": fragment}
            yield {"type": "tool_result", "content": "done"}

    guard = stream_guard(
        fake_stream(),
        idle_timeout_s=0,
        cancel_stream=lambda *_: None,
        agent_id=AGENT_ID,
        thread_id=THREAD_ID,
    )
    chunks = await _drain(guard)
    assert len(chunks) == 15


def test_stream_repetition_is_classified_as_stream_stall() -> None:
    from octop.i18n.domains.stream import classify_stream_error_message

    exc = StreamRepetitionError(period=120, repeats=34)
    assert classify_stream_error_message(str(exc)) == STREAM_STALL
    assert format_stream_error(exc, "zh")
