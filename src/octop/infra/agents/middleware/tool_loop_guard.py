"""Loop-breaker for repeated identical tool calls.

A model that keeps re-issuing the same call never converges, and each identical
response is fed straight back, reinforcing the repeat. Two shapes were observed
in production on one thread:

* ``send_file_to_user {"file_path": "/"}`` x36 after the real path had already
  been discovered — identical *failing* calls.
* ``ls {"path": "/"}`` x113 — identical calls that *succeed*, because ``/``
  resolves to the host root and returns a listing every time. A guard that only
  counts failures resets on every one of these and never trips.

So the signature is ``(tool name, args)`` and the streak counts repeats
regardless of outcome: an identical call that has already returned the same
answer three times cannot become useful on the fourth.

A different call resets the streak, so ordinary multi-step work is unaffected.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Awaitable, Callable
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.types import Command

logger = logging.getLogger(__name__)

DEFAULT_THRESHOLD = 3
_MAX_ARGS_CHARS = 400


def _args_fingerprint(args: Any) -> str:
    try:
        text = json.dumps(args, sort_keys=True, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        text = repr(args)
    return text[:_MAX_ARGS_CHARS]


class ToolLoopGuardMiddleware(AgentMiddleware[Any, Any]):
    """Stop a call that has been repeated identically ``threshold`` times in a row."""

    def __init__(self, threshold: int = DEFAULT_THRESHOLD) -> None:
        self._threshold = max(1, int(threshold))
        self._signature: tuple[str, str] | None = None
        self._streak = 0

    def _reset(self) -> None:
        self._signature = None
        self._streak = 0

    def _blocked_message(self, tool_name: str, args_fp: str) -> str:
        return (
            f"{tool_name} was already called with these exact arguments "
            f"{self._threshold} times in a row and returned the same result each "
            f"time. Calling it again will not change the answer — stop and move on.\n"
            f"args: {args_fp}\n"
            f"If you expected something different, note that a leading '/' is a HOST "
            f"absolute path here, not the workspace root. Use a workspace-relative "
            f"path such as `generated/report.docx`, or list the workspace with `ls` "
            f"and `.`. Otherwise do something else, or tell the user what is stuck."
        )

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
        self._streak += 1

        if self._streak > self._threshold:
            logger.warning(
                "ToolLoopGuard suppressed repeat of %s (identical call already ran %d times)",
                tool_name,
                self._streak - 1,
            )
            # Deliberately NOT reset: the streak stays latched so every further
            # identical call in this turn is suppressed cheaply. A changed
            # argument still clears it, so recovery is possible.
            return ToolMessage(
                content=self._blocked_message(tool_name, args_fp),
                tool_call_id=str(tool_call.get("id") or ""),
                status="error",
            )

        return await handler(request)


__all__ = ["ToolLoopGuardMiddleware", "DEFAULT_THRESHOLD"]
