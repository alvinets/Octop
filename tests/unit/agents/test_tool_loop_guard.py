"""Tests for ToolLoopGuardMiddleware — breaks repeated identical tool calls."""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.messages import ToolMessage

from octop.infra.agents.middleware.tool_loop_guard import ToolLoopGuardMiddleware

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
    assert "will not change the answer" in str(results[-1].content)


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
    assert "HOST" in str(blocked.content)


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
