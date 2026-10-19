"""Redirect workspace-intent tool paths at the agent's workspace directory.

Why this exists
---------------
Local backends default to ``root_dir="/"`` on POSIX so an agent can mount global
host paths (globally installed skill package roots, system tooling). That default
is correct and deliberately kept. The side effect is that *every* path the file
tools see resolves against the host root, so:

* ``ls "generated"``  -> 0 entries (resolves to ``/generated``)
* ``read SOUL.md``    -> fails  (resolves to ``/SOUL.md``)
* ``ls "/"``          -> the whole host filesystem

The agent could still *write* its workspace through ``execute``, which runs with
cwd set there — it just could not read anything back. That asymmetry is what let
an agent generate a .docx and then be unable to locate it.

Fix
---
Because ``root_dir="/"`` stays, an *absolute* workspace path resolves correctly.
So this middleware rewrites only workspace-intent arguments:

* bare ``"/"`` and ``"."``  -> the workspace directory
* relative ``"generated/x"`` -> ``"<workspace>/generated/x"``
* any other absolute path    -> untouched, so deliberate host access still works

Also rewrites relative ``glob``/``grep`` *patterns*, which have no separate path
argument: under ``root_dir="/"`` a bare ``**/*.docx`` would otherwise walk the
entire host filesystem — slow, and it leaks host paths into model context.

Not applied to remote backends (s3/postgres/cos/oss/obs) or sandbox backends
(docker/opensandbox): there the workspace is not a host path, so rewriting to one
would point the agent at a directory that does not exist.
"""

from __future__ import annotations

import asyncio
import logging
import posixpath
from collections.abc import Awaitable, Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import ToolCall, ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.types import Command

from octop.infra.backend.compat import is_remote_backend_spec
from octop.infra.gateway.media.tool_media import tool_name_base

logger = logging.getLogger(__name__)

# Tools whose arguments name a workspace file or directory.
_PATH_TOOL_BASES = frozenset(
    {
        "ls",
        "read_file",
        "write_file",
        "edit_file",
        "send_file",
        "send_file_to_user",
        "mkdir",
        "move_file",
        "list_dir",
        "list_directory",
        "grep",
    }
)

_PATH_ARG_KEYS = ("path", "file_path", "filepath", "target_path", "dir", "directory")

# Tools that operate on files only; a directory at the resolved path is an error.
FILE_TOOL_BASES = frozenset(
    {"read_file", "write_file", "edit_file", "send_file", "send_file_to_user"}
)

# glob's pattern is a path pattern, so a relative one is workspace intent.
# grep is excluded: its pattern is a regex, never a path.
_GLOB_TOOL_BASES = frozenset({"glob"})

# Backend specs whose workspace is a real host path.
_HOST_BACKEND_TYPES = frozenset({"local_shell", "filesystem"})
_SANDBOX_BACKEND_TYPES = frozenset({"docker", "opensandbox"})


def is_host_mounted_backend(backend_spec: Any) -> bool:
    """True when the agent's workspace is a host path we may rewrite arguments to."""
    if not isinstance(backend_spec, dict):
        return False
    if is_remote_backend_spec(backend_spec):
        return False
    kind = str(backend_spec.get("type") or "").lower()
    if kind in _SANDBOX_BACKEND_TYPES:
        return False
    return kind in _HOST_BACKEND_TYPES


def resolve_workspace_intent(raw: Any, workspace: Path) -> str | None:
    """Return the workspace-absolute form of *raw*, or None to leave it alone."""
    if not isinstance(raw, str):
        return None
    text = raw.strip()
    if not text or text.startswith("file://"):
        return None
    ws = workspace.as_posix().rstrip("/")
    # Bare "/" means "my workspace" to the model, so it maps to the workspace, not the host root.
    if text == "/":
        return ws
    if text.startswith("/") or (len(text) >= 2 and text[1] == ":") or text.startswith("\\\\"):
        return None
    if text in {".", "./", ".."}:
        return ws
    return posixpath.join(ws, text)


def rewrite_workspace_paths(
    tool_call: ToolCall,
    *,
    workspace_dir: Path | None,
    host_mounted: bool,
) -> ToolCall | None:
    """Return a rewritten tool_call, or None when nothing needed changing."""
    if workspace_dir is None or not host_mounted:
        return None

    base = tool_name_base(str(tool_call.get("name") or "")).lower()
    args = tool_call.get("args")
    if not isinstance(args, dict):
        return None

    if base in _PATH_TOOL_BASES:
        keys: tuple[str, ...] = _PATH_ARG_KEYS
    elif base in _GLOB_TOOL_BASES:
        keys = ("pattern",)
    else:
        return None

    new_args: dict[str, Any] | None = None
    for key in keys:
        replacement = resolve_workspace_intent(args.get(key), workspace_dir)
        if replacement is None:
            continue
        if new_args is None:
            new_args = dict(args)
        new_args[key] = replacement

    if new_args is None:
        return None
    return {**tool_call, "args": new_args}


def _file_tool_directory_hint(tool_call: ToolCall, workspace_dir: Path) -> str | None:
    """Return a model-facing error when a file-only tool targets a directory."""
    base = tool_name_base(str(tool_call.get("name") or "")).lower()
    if base not in FILE_TOOL_BASES:
        return None
    args = tool_call.get("args")
    if not isinstance(args, dict):
        return None
    ws = str(workspace_dir)
    for key in _PATH_ARG_KEYS:
        path = args.get(key)
        if not isinstance(path, str):
            continue
        p = Path(path)
        if p.exists() and p.is_dir():
            return (
                f"`{path}` is a directory, but `{base}` needs a file path.\n"
                f"Your workspace is `{ws}`. Pass a file path inside it — for example "
                f"`generated/<name>.md` — not the workspace root."
            )
    return None


class WorkspaceRootAliasMiddleware(AgentMiddleware[Any, Any]):
    """Point workspace-intent tool paths at the agent workspace, keep host paths as-is."""

    def __init__(
        self,
        workspace_dir: Path | None = None,
        backend_spec: Any = None,
    ) -> None:
        self._workspace = workspace_dir.expanduser().resolve() if workspace_dir else None
        self._host_mounted = is_host_mounted_backend(backend_spec)

    def _rewrite(self, request: ToolCallRequest) -> ToolCallRequest | None:
        rewritten = rewrite_workspace_paths(
            request.tool_call,
            workspace_dir=self._workspace,
            host_mounted=self._host_mounted,
        )
        if rewritten is None:
            return None
        # Replace the request rather than mutating it, so the transcript still
        # records the arguments the model actually emitted.
        return replace(request, tool_call=rewritten)

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        if self._workspace is not None and self._host_mounted:
            rewritten = self._rewrite(request)
            check_call = rewritten.tool_call if rewritten is not None else request.tool_call
            dir_hint = await asyncio.to_thread(
                _file_tool_directory_hint, check_call, self._workspace
            )
            if dir_hint is not None:
                return ToolMessage(
                    content=dir_hint,
                    tool_call_id=str(request.tool_call.get("id") or ""),
                    status="error",
                )
            if rewritten is None:
                return await handler(request)
            logger.info(
                "WorkspaceRootAlias redirected %s into the workspace",
                str(request.tool_call.get("name")),
            )
            return await handler(rewritten)
        return await handler(request)

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
    ) -> ToolMessage | Command[Any]:
        if self._workspace is not None and self._host_mounted:
            rewritten = self._rewrite(request)
            check_call = rewritten.tool_call if rewritten is not None else request.tool_call
            dir_hint = _file_tool_directory_hint(check_call, self._workspace)
            if dir_hint is not None:
                return ToolMessage(
                    content=dir_hint,
                    tool_call_id=str(request.tool_call.get("id") or ""),
                    status="error",
                )
            if rewritten is None:
                return handler(request)
            return handler(rewritten)
        return handler(request)


__all__ = [
    "FILE_TOOL_BASES",
    "WorkspaceRootAliasMiddleware",
    "is_host_mounted_backend",
    "resolve_workspace_intent",
    "rewrite_workspace_paths",
]
