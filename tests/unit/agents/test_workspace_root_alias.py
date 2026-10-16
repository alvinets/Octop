"""Tests for WorkspaceRootAliasMiddleware — workspace-intent paths land in the workspace.

The backend keeps ``root_dir="/"`` so agents can still mount host paths, which means
bare and relative paths would otherwise resolve against the host root.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import ToolMessage

from octop.infra.agents.middleware.workspace_root_alias import (
    WorkspaceRootAliasMiddleware,
    is_host_mounted_backend,
    resolve_workspace_intent,
    rewrite_workspace_paths,
)

WS = Path("/home/kzhy/.octop/agents/haowise-office")
HOST = {"type": "local_shell", "virtual_mode": True, "root_dir": "/"}


def _call(name: str, args: dict[str, Any]) -> dict[str, Any]:
    return {"name": name, "args": args, "id": "c1"}


def _rewrite(name: str, args: dict[str, Any], *, ws: Path | None = WS, host: bool = True) -> Any:
    return rewrite_workspace_paths(_call(name, args), workspace_dir=ws, host_mounted=host)


@dataclass
class Req:
    tool_call: dict[str, Any]


def test_bare_slash_maps_to_workspace() -> None:
    assert _rewrite("ls", {"path": "/"}) == _call("ls", {"path": str(WS)})
    assert _rewrite("send_file_to_user", {"file_path": "/"}) == _call(
        "send_file_to_user", {"file_path": str(WS)}
    )


def test_dot_maps_to_workspace() -> None:
    assert _rewrite("ls", {"path": "."}) == _call("ls", {"path": str(WS)})
    assert _rewrite("read_file", {"file_path": "./"}) == _call("read_file", {"file_path": str(WS)})


def test_relative_paths_map_into_workspace() -> None:
    assert _rewrite("ls", {"path": "generated"}) == _call("ls", {"path": f"{WS}/generated"})
    assert _rewrite("read_file", {"file_path": "generated/a.docx"}) == _call(
        "read_file", {"file_path": f"{WS}/generated/a.docx"}
    )
    assert _rewrite("read_file", {"path": "a/b/c.png"}) == _call(
        "read_file", {"path": f"{WS}/a/b/c.png"}
    )


def test_host_absolute_paths_are_preserved() -> None:
    assert _rewrite("ls", {"path": "/etc"}) is None
    assert _rewrite("read_file", {"file_path": "/etc/passwd"}) is None
    assert _rewrite("read_file", {"file_path": "/home/kzhy/x"}) is None
    assert _rewrite("write_file", {"file_path": "/opt/host/keep.md", "content": "c"}) is None


def test_glob_relative_pattern_is_scoped_to_workspace() -> None:
    assert _rewrite("glob", {"pattern": "**/*.docx"}) == _call(
        "glob", {"pattern": f"{WS}/**/*.docx"}
    )
    assert _rewrite("glob", {"pattern": "generated/*.md"}) == _call(
        "glob", {"pattern": f"{WS}/generated/*.md"}
    )


def test_grep_pattern_is_a_regex_and_must_not_be_rewritten() -> None:
    assert _rewrite("grep", {"pattern": "TODO", "path": "src"}) == _call(
        "grep", {"pattern": "TODO", "path": f"{WS}/src"}
    )
    assert _rewrite("grep", {"pattern": "class .*Agent"}) is None


def test_non_path_tools_are_untouched() -> None:
    assert _rewrite("execute", {"command": "ls /"}) is None
    assert _rewrite("web_fetch", {"url": "http://x/"}) is None


def test_non_dict_args_untouched() -> None:
    assert (
        rewrite_workspace_paths(
            {"name": "ls", "args": "/", "id": "c1"}, workspace_dir=WS, host_mounted=True
        )
        is None
    )


def test_original_call_object_is_not_mutated() -> None:
    original = _call("ls", {"path": "/"})
    rewrite_workspace_paths(original, workspace_dir=WS, host_mounted=True)
    assert original["args"]["path"] == "/"


def test_harness_namespaced_tools_resolve_by_base_name() -> None:
    assert _rewrite("harness/ls", {"path": "/"}) == _call("harness/ls", {"path": str(WS)})


@pytest.mark.parametrize(
    "spec,expected",
    [
        ({"type": "local_shell", "root_dir": "/"}, True),
        ({"type": "filesystem"}, True),
        ({"type": "docker"}, False),
        ({"type": "opensandbox"}, False),
        ({"type": "s3"}, False),
        ({"type": "cos"}, False),
        ({"type": "postgres"}, False),
        (None, False),
        ("local_shell", False),
    ],
)
def test_host_mounted_backend_gating(spec: Any, expected: bool) -> None:
    assert is_host_mounted_backend(spec) is expected


def test_remote_and_sandbox_backends_are_never_rewritten() -> None:
    for spec in ({"type": "s3"}, {"type": "docker"}, {"type": "opensandbox"}, None):
        middleware = WorkspaceRootAliasMiddleware(workspace_dir=WS, backend_spec=spec)
        assert middleware._rewrite(Req(_call("ls", {"path": "/"}))) is None


def test_missing_workspace_is_a_no_op() -> None:
    assert _rewrite("ls", {"path": "/"}, ws=None) is None


def test_no_workspace_yet_leaves_everything_alone() -> None:
    middleware = WorkspaceRootAliasMiddleware(backend_spec=HOST)
    assert middleware._workspace is None
    assert middleware._rewrite(Req(_call("ls", {"path": "/"}))) is None


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("/", str(WS)),
        (".", str(WS)),
        ("generated", f"{WS}/generated"),
        ("/etc/passwd", None),
        ("file:///a", None),
        ("", None),
        (None, None),
        (123, None),
    ],
)
def test_resolve_workspace_intent(raw: Any, expected: str | None) -> None:
    assert resolve_workspace_intent(raw, WS) == expected


@pytest.mark.asyncio
async def test_handler_receives_rewritten_request() -> None:
    middleware = WorkspaceRootAliasMiddleware(workspace_dir=WS, backend_spec=HOST)
    seen: list[Any] = []

    async def handler(request: Any) -> ToolMessage:
        seen.append(request.tool_call["args"])
        return ToolMessage(content="ok", tool_call_id="c1", status="success")

    await middleware.awrap_tool_call(Req(_call("ls", {"path": "/"})), handler)  # type: ignore[arg-type]
    assert seen == [{"path": str(WS)}]


@pytest.mark.asyncio
async def test_passthrough_when_nothing_to_rewrite() -> None:
    middleware = WorkspaceRootAliasMiddleware(workspace_dir=WS, backend_spec=HOST)
    seen: list[Any] = []

    async def handler(request: Any) -> ToolMessage:
        seen.append(request.tool_call["args"])
        return ToolMessage(content="ok", tool_call_id="c1", status="success")

    original = _call("execute", {"command": "ls /"})
    await middleware.awrap_tool_call(Req(original), handler)  # type: ignore[arg-type]
    assert seen == [{"command": "ls /"}]
