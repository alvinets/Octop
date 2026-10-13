"""Tests for WorkspaceRootAliasMiddleware — bare "/" means the workspace root."""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.messages import ToolMessage

from octop.infra.agents.middleware.workspace_root_alias import (
    WorkspaceRootAliasMiddleware,
    rewrite_root_alias,
)


def _call(name: str, args: dict[str, Any]) -> dict[str, Any]:
    return {"name": name, "args": args, "id": "c1"}


def test_bare_slash_is_rewritten_for_path_tools() -> None:
    assert rewrite_root_alias(_call("ls", {"path": "/"})) == _call("ls", {"path": "."})
    assert rewrite_root_alias(_call("send_file_to_user", {"file_path": "/"})) == _call(
        "send_file_to_user", {"file_path": "."}
    )
    # harness namespaced tools resolve to the same base name
    assert rewrite_root_alias(_call("ns/ls", {"path": "/"})) is not None


def test_other_absolute_paths_are_untouched() -> None:
    assert rewrite_root_alias(_call("ls", {"path": "/home/kzhy"})) is None
    assert rewrite_root_alias(_call("read_file", {"path": "/etc/passwd"})) is None
    assert rewrite_root_alias(_call("read_file", {"path": "/home/"})) is None


def test_workspace_relative_paths_untouched() -> None:
    assert rewrite_root_alias(_call("read_file", {"path": "generated/a.docx"})) is None
    assert rewrite_root_alias(_call("read_file", {"path": "./a.md"})) is None
    assert rewrite_root_alias(_call("read_file", {"path": "a/b/c.png"})) is None


def test_non_path_tools_untouched() -> None:
    assert rewrite_root_alias(_call("glob", {"pattern": "**/*.docx"})) is None
    assert rewrite_root_alias(_call("execute", {"command": "ls /"})) is None
    assert rewrite_root_alias(_call("web_fetch", {"url": "http://x/"})) is None


def test_non_dict_args_untouched() -> None:
    assert rewrite_root_alias({"name": "ls", "args": "/", "id": "c1"}) is None


def test_original_call_object_is_not_mutated() -> None:
    original = _call("ls", {"path": "/"})
    rewrite_root_alias(original)
    assert original["args"]["path"] == "/"


@pytest.mark.asyncio
async def test_handler_receives_rewritten_request() -> None:
    guard = WorkspaceRootAliasMiddleware()
    seen: list[Any] = []

    async def handler(request: Any) -> ToolMessage:
        seen.append(request.tool_call["args"])
        return ToolMessage(content="ok", tool_call_id="c1", status="success")

    from dataclasses import dataclass

    @dataclass
    class Req:
        tool_call: dict[str, Any]

    await guard.awrap_tool_call(Req(_call("ls", {"path": "/"})), handler)  # type: ignore[arg-type]
    assert seen == [{"path": "."}]


@pytest.mark.asyncio
async def test_passthrough_when_nothing_to_rewrite() -> None:
    guard = WorkspaceRootAliasMiddleware()
    seen: list[Any] = []

    async def handler(request: Any) -> ToolMessage:
        seen.append(request.tool_call["args"])
        return ToolMessage(content="ok", tool_call_id="c1", status="success")

    from dataclasses import dataclass

    @dataclass
    class Req:
        tool_call: dict[str, Any]

    await guard.awrap_tool_call(Req(_call("ls", {"path": "generated"})), handler)  # type: ignore[arg-type]
    assert seen == [{"path": "generated"}]
