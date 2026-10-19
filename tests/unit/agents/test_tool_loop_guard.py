"""Tests for ToolLoopGuardMiddleware — breaks repeated identical tool calls."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import ToolMessage

from octop.infra.agents.middleware.tool_loop_guard import (
    ToolLoopAbort,
    ToolLoopGuardMiddleware,
)

_ERR = "send_file_to_user: no such file: /"
_OK = "['/boot/', '/etc/', '/home/']"


def _request(name: str, args: dict[str, Any], call_id: str) -> Any:
    return type("Req", (), {"tool_call": {"name": name, "args": args, "id": call_id}})()


def _handler(calls: list[str], *, content: str, status: str) -> Any:
    async def handler(request: Any) -> ToolMessage:
        calls.append(str(request.tool_call["id"]))
        return ToolMessage(
            content=content,
            tool_call_id=str(request.tool_call["id"]),
            status=status,
        )

    return handler


@pytest.mark.asyncio
async def test_suppresses_repeat_after_threshold() -> None:
    guard = ToolLoopGuardMiddleware(threshold=3)
    calls: list[str] = []
    results = []
    for i in range(6):
        results.append(
            await guard.awrap_tool_call(
                _request("send_file_to_user", {"file_path": "/"}, f"c{i}"),
                _handler(calls, content=_ERR, status="error"),
            )
        )

    assert len(calls) == 3
    assert all(r.status == "error" for r in results)
    assert "cannot produce a different answer" in str(results[-1].content)


@pytest.mark.asyncio
async def test_suppresses_identical_successful_calls() -> None:
    """Regression: 113 x `ls /` all returned success, so a failures-only guard never fired."""
    guard = ToolLoopGuardMiddleware(threshold=3)
    calls: list[str] = []
    results = []
    for i in range(8):
        results.append(
            await guard.awrap_tool_call(
                _request("ls", {"path": "/"}, f"c{i}"),
                _handler(calls, content=_OK, status="success"),
            )
        )

    assert len(calls) == 3
    blocked = results[-1]
    assert blocked.status == "error"
    assert "already been called" in str(blocked.content)


@pytest.mark.asyncio
async def test_different_args_do_not_trip_the_guard() -> None:
    guard = ToolLoopGuardMiddleware(threshold=3)
    calls: list[str] = []
    for i in range(6):
        await guard.awrap_tool_call(
            _request("send_file_to_user", {"file_path": f"/p{i}"}, f"c{i}"),
            _handler(calls, content=_ERR, status="error"),
        )
    assert len(calls) == 6


@pytest.mark.asyncio
async def test_interleaved_tools_reset_the_streak() -> None:
    guard = ToolLoopGuardMiddleware(threshold=3)
    calls: list[str] = []
    h = _handler(calls, content=_OK, status="success")
    for i in range(3):
        await guard.awrap_tool_call(_request("ls", {"path": "/"}, f"a{i}"), h)
        await guard.awrap_tool_call(_request("read_file", {"path": f"f{i}.md"}, f"b{i}"), h)
    assert len(calls) == 6


@pytest.mark.asyncio
async def test_streak_latches_but_a_different_call_recovers() -> None:
    guard = ToolLoopGuardMiddleware(threshold=2)
    calls: list[str] = []
    h = _handler(calls, content=_ERR, status="error")
    for i in range(5):
        await guard.awrap_tool_call(_request("send_file_to_user", {"file_path": "/"}, f"c{i}"), h)
    assert len(calls) == 2

    await guard.awrap_tool_call(_request("ls", {"path": "."}, "c9"), h)
    assert len(calls) == 3


@pytest.mark.asyncio
async def test_threshold_one_blocks_immediately_on_second_call() -> None:
    guard = ToolLoopGuardMiddleware(threshold=1)
    calls: list[str] = []
    h = _handler(calls, content=_OK, status="success")
    for i in range(4):
        await guard.awrap_tool_call(_request("ls", {"path": "/"}, f"c{i}"), h)
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_arg_key_order_is_not_a_distinct_signature() -> None:
    guard = ToolLoopGuardMiddleware(threshold=2)
    calls: list[str] = []
    h = _handler(calls, content=_OK, status="success")
    await guard.awrap_tool_call(_request("t", {"a": 1, "b": 2}, "c0"), h)
    await guard.awrap_tool_call(_request("t", {"b": 2, "a": 1}, "c1"), h)
    await guard.awrap_tool_call(_request("t", {"a": 1, "b": 2}, "c2"), h)
    assert len(calls) == 2


@pytest.mark.asyncio
async def test_aborts_turn_after_max_blocks() -> None:
    """Stage 2: refusals alone do not stop a degenerate model — the turn must end."""
    guard = ToolLoopGuardMiddleware(threshold=3, max_blocks=5)
    calls: list[str] = []
    h = _handler(calls, content=_OK, status="success")

    refusals = 0
    with pytest.raises(ToolLoopAbort):
        for i in range(50):
            await guard.awrap_tool_call(_request("ls", {"path": "/"}, f"c{i}"), h)
            refusals += 1

    # threshold(3) real executions, then max_blocks(5) refusals, then abort
    assert len(calls) == 3
    assert refusals == 7  # 3 real + 4 refusals before the 5th raises


@pytest.mark.asyncio
async def test_abort_bubbles_up_through_toolnode() -> None:
    """ToolNode re-raises GraphBubbleUp before its generic handler; keep that true."""
    from langgraph.errors import GraphBubbleUp

    assert issubclass(ToolLoopAbort, GraphBubbleUp)


@pytest.mark.asyncio
async def test_different_call_clears_the_block_counter() -> None:
    guard = ToolLoopGuardMiddleware(threshold=2, max_blocks=3)
    calls: list[str] = []
    h = _handler(calls, content=_OK, status="success")

    for i in range(6):
        await guard.awrap_tool_call(_request("ls", {"path": "/"}, f"a{i}"), h)
        await guard.awrap_tool_call(_request("read_file", {"path": "x"}, f"b{i}"), h)

    assert len(calls) == 12


@pytest.mark.asyncio
async def test_error_result_does_not_trip_the_guard() -> None:
    """A failing-but-alternating call is normal work, not a loop."""
    guard = ToolLoopGuardMiddleware(threshold=2, max_blocks=3)
    calls: list[str] = []
    for i in range(9):
        h = _handler(calls, content=_ERR if i % 2 == 0 else "other", status="error")
        await guard.awrap_tool_call(_request("t", {"n": i}, f"c{i}"), h)
    assert len(calls) == 9


@pytest.mark.asyncio
async def test_execute_refusal_mentions_no_cd_needed(tmp_path: Path) -> None:
    guard = ToolLoopGuardMiddleware(threshold=1, workspace_dir=tmp_path)
    calls: list[str] = []
    h = _handler(calls, content=_OK, status="success")
    await guard.awrap_tool_call(_request("execute", {"command": "ls"}, "c0"), h)
    await guard.awrap_tool_call(_request("execute", {"command": "ls"}, "c1"), h)
    assert len(calls) == 1
    blocked = await guard.awrap_tool_call(_request("execute", {"command": "ls"}, "c2"), h)
    assert "no `cd` is needed" in blocked.content
    assert str(tmp_path) in blocked.content


@pytest.mark.asyncio
async def test_execute_refusal_without_workspace(tmp_path: Path) -> None:
    guard = ToolLoopGuardMiddleware(threshold=1)
    calls: list[str] = []
    h = _handler(calls, content=_OK, status="success")
    await guard.awrap_tool_call(_request("execute", {"command": "ls"}, "c0"), h)
    await guard.awrap_tool_call(_request("execute", {"command": "ls"}, "c1"), h)
    blocked = await guard.awrap_tool_call(_request("execute", {"command": "ls"}, "c2"), h)
    assert "no `cd` is needed" in blocked.content
    assert str(tmp_path) not in blocked.content


@pytest.mark.asyncio
async def test_file_tool_refusal_mentions_needs_file_path(tmp_path: Path) -> None:
    guard = ToolLoopGuardMiddleware(threshold=1, workspace_dir=tmp_path)
    calls: list[str] = []
    h = _handler(calls, content=_ERR, status="error")
    await guard.awrap_tool_call(_request("write_file", {"file_path": "/"}, "c0"), h)
    await guard.awrap_tool_call(_request("write_file", {"file_path": "/"}, "c1"), h)
    blocked = await guard.awrap_tool_call(_request("write_file", {"file_path": "/"}, "c2"), h)
    assert "needs a file path, not a directory" in blocked.content
    assert str(tmp_path) in blocked.content
