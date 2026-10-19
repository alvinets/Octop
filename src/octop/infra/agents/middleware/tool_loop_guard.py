"""Loop-breaker for repeated identical tool calls.

A model that keeps re-issuing the same call never converges, and each identical
response is fed straight back, reinforcing the repeat. Two shapes were observed
in production on one thread:

* ``send_file_to_user {"file_path": "/"}`` x36 after the real path had already
  been discovered — identical *failing* calls.
* ``ls {"path": "/"}`` x113 — identical calls that returned the same answer each
  time. A guard that only counts failures resets on every one of these and never
  trips, so the streak counts repeats regardless of outcome.

Two stages, because refusing the call alone does not stop a degenerate model:

1. ``threshold`` repeats return a refusal *without* executing the tool. Cheap,
   and it costs the model nothing but a message.
2. ``max_blocks`` refusals in a row raise :class:`ToolLoopAbort`, which ends the
   turn. A model stuck in a repetition loop will happily keep emitting calls
   forever against stage 1 alone — observed emitting 19+ refusals in one turn —
   and every one of those still costs a generation.

A different call resets both counters, so ordinary multi-step work is unaffected.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import ToolMessage
from langgraph.errors import GraphBubbleUp
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.types import Command

from octop.infra.agents.middleware.workspace_root_alias import FILE_TOOL_BASES
from octop.infra.gateway.media.tool_media import tool_name_base

logger = logging.getLogger(__name__)

DEFAULT_THRESHOLD = 3
DEFAULT_MAX_BLOCKS = 8
_MAX_ARGS_CHARS = 400


class ToolLoopAbort(RuntimeError, GraphBubbleUp):
    """Ends a turn stuck re-issuing one identical call past the block cap.

    Inherits ``GraphBubbleUp`` deliberately: ToolNode's run loop does
    ``except GraphBubbleUp: raise`` *before* its ``except Exception`` handler,
    so this propagates instead of being converted back into an error ToolMessage
    and answered again.
    """


def _args_fingerprint(args: Any) -> str:
    try:
        text = json.dumps(args, sort_keys=True, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        text = repr(args)
    return text[:_MAX_ARGS_CHARS]


class ToolLoopGuardMiddleware(AgentMiddleware[Any, Any]):
    """Stop a call repeated identically ``threshold`` times; abort the turn after ``max_blocks``."""

    def __init__(
        self,
        threshold: int = DEFAULT_THRESHOLD,
        max_blocks: int = DEFAULT_MAX_BLOCKS,
        workspace_dir: Path | None = None,
    ) -> None:
        self._threshold = max(1, int(threshold))
        self._max_blocks = max(1, int(max_blocks))
        self._workspace = workspace_dir.expanduser().resolve() if workspace_dir else None
        self._signature: tuple[str, str] | None = None
        self._streak = 0
        self._blocks = 0

    def _reset(self) -> None:
        self._signature = None
        self._streak = 0
        self._blocks = 0

    def _blocked_message(self, tool_name: str, args_fp: str) -> str:
        base = tool_name_base(tool_name).lower()
        lines = [
            f"{tool_name} has already been called with these exact arguments "
            f"{self._threshold} times and returned the same result each time. "
            f"Repeating it cannot produce a different answer.\n"
            f"args: {args_fp}\n"
            f"You already have this result. Act on it, or change your approach: "
            f"narrow the path (for example `ls generated` instead of `ls .`), use a "
            f"different tool, or tell the user what is blocking you.",
        ]
        ws = str(self._workspace) if self._workspace else None
        if base == "execute":
            extra = f" (`{ws}`)" if ws else ""
            lines.append(
                f"`execute` already runs inside your workspace{extra}, "
                f"so no `cd` is needed — run the command directly, e.g. `ls generated`."
            )
        elif base in FILE_TOOL_BASES:
            extra = f"; your workspace is `{ws}`, e.g. `generated/<name>.md`" if ws else ""
            lines.append(f"`{base}` needs a file path, not a directory{extra}.")
        return "\n".join(lines)

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        tool_call = request.tool_call
        tool_name = str(tool_call.get("name") or "")
        raw_args = tool_call.get("args") or {}
        args_fp = _args_fingerprint(raw_args if isinstance(raw_args, dict) else {})

        signature = (tool_name, args_fp)
        if signature != self._signature:
            self._signature = signature
            self._streak = 0
            self._blocks = 0
        self._streak += 1

        if self._streak > self._threshold:
            self._blocks += 1
            if self._blocks >= self._max_blocks:
                logger.warning(
                    "ToolLoopGuard aborting turn: %s repeated %d times, %d refusals unanswered",
                    tool_name,
                    self._streak - 1,
                    self._blocks,
                )
                self._reset()
                raise ToolLoopAbort(
                    f"{tool_name} was called with the same arguments "
                    f"({args_fp[:200]}) {self._max_blocks} times in a row without "
                    f"changing its approach; ending the turn to stop the loop."
                )
            logger.warning(
                "ToolLoopGuard suppressed repeat of %s (identical call already ran %d times)",
                tool_name,
                self._streak - 1,
            )
            return ToolMessage(
                content=self._blocked_message(tool_name, args_fp),
                tool_call_id=str(tool_call.get("id") or ""),
                status="error",
            )

        # Deliberately no reset based on the result: an identical call that keeps
        # failing is the original degenerate case, and an identical call that keeps
        # succeeding is the other one. Only a different call clears the streak.
        return await handler(request)


__all__ = [
    "DEFAULT_MAX_BLOCKS",
    "DEFAULT_THRESHOLD",
    "ToolLoopAbort",
    "ToolLoopGuardMiddleware",
]
