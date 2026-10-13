"""Normalise a bare ``/`` path argument to the workspace root.

Harness path convention (see ``BackendWorkspace.resolve_path``) is:

* ``foo`` / ``./foo`` -> ``{workspace_dir}/foo``
* ``/foo``             -> **host** filesystem absolute

Models overwhelmingly mean "the workspace root" when they pass ``/``, and the
mismatch is actively harmful: ``ls /`` returns the whole host filesystem instead
of the workspace, so the model learns nothing and retries. One production thread
spent 113 consecutive calls on ``ls {"path": "/"}`` before it happened to try a
workspace-relative path.

Only an exact bare ``/`` is rewritten, to ``.``. Any other absolute path
(``/home/...``, ``/etc/...``) is left alone so deliberate host access keeps
working, and rewriting would silently redirect a real target.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from dataclasses import replace
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import ToolCall, ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.types import Command

from octop.infra.gateway.media.tool_media import tool_name_base

logger = logging.getLogger(__name__)

# Harness tools whose arguments name a workspace file or directory.
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
    }
)

_PATH_ARG_KEYS = ("path", "file_path", "filepath", "target_path", "dir", "directory")

WORKSPACE_ROOT_ARG = "."


def rewrite_root_alias(tool_call: ToolCall) -> ToolCall | None:
    """Return a rewritten tool_call, or None when nothing needs changing."""
    if tool_name_base(str(tool_call.get("name") or "")).lower() not in _PATH_TOOL_BASES:
        return None
    args = tool_call.get("args")
    if not isinstance(args, dict):
        return None

    new_args: dict[str, Any] | None = None
    for key in _PATH_ARG_KEYS:
        value = args.get(key)
        if isinstance(value, str) and value.strip() == "/":
            if new_args is None:
                new_args = dict(args)
            new_args[key] = WORKSPACE_ROOT_ARG

    if new_args is None:
        return None
    return {**tool_call, "args": new_args}


class WorkspaceRootAliasMiddleware(AgentMiddleware[Any, Any]):
    """Rewrite a bare ``/`` path argument to the workspace root before dispatch."""

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        rewritten = rewrite_root_alias(request.tool_call)
        if rewritten is None:
            return await handler(request)

        logger.info(
            "WorkspaceRootAlias mapped bare '/' to '.' for %s",
            rewritten.get("name"),
        )
        # Replace the request rather than mutating it, so the caller's transcript
        # keeps the arguments the model actually emitted.
        return await handler(replace(request, tool_call=rewritten))

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
    ) -> ToolMessage | Command[Any]:
        rewritten = rewrite_root_alias(request.tool_call)
        if rewritten is None:
            return handler(request)
        return handler(replace(request, tool_call=rewritten))


__all__ = [
    "WORKSPACE_ROOT_ARG",
    "WorkspaceRootAliasMiddleware",
    "rewrite_root_alias",
]
