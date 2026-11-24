"""Guards for agent stream consumption: idle silence and degenerate repetition.

Two failure modes, both observed in production on the same thread:

* **Idle** — the HTTP connection stays open but no chunk arrives, so the turn
  hangs until the client gives up.
* **Repetition** — the model keeps re-emitting the same fragment (a provider
  interruption notice, in the observed case), so the turn never converges even
  though tokens keep flowing.

Both are turned into bounded failures here. The idle deadline is suspended while
a tool runs, because silence is expected while the graph executes a tool.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import AsyncIterator, Callable
from typing import Any

logger = logging.getLogger(__name__)

_TOOL_PHASE_ENTRY_TYPES = frozenset({"state_update", "state_snapshot"})
_STEP_BOUNDARY_TYPES = frozenset({"state_update", "state_snapshot", "tool_result"})

# A tail of at least _REPEAT_MIN_PERIOD chars repeated _REPEAT_MIN_COUNT times is
# a stuck model, not prose. Re-check every _REPEAT_CHECK_STEP chars to keep the
# per-chunk cost bounded.
_REPEAT_WINDOW = 4096
_REPEAT_MIN_PERIOD = 80
_REPEAT_MIN_COUNT = 6
_REPEAT_CHECK_STEP = 1024


class StreamChunkTimeoutError(RuntimeError):
    """Raised when no streaming chunk arrives within the idle deadline.

    The message carries the literal token ``stream_chunk_timeout`` so the
    existing ``stream_errors.stream_stall`` bundle localizes it; no new i18n
    key or error code is needed.
    """

    def __init__(self, timeout_s: float) -> None:
        self.timeout_s = timeout_s
        super().__init__(f"stream_chunk_timeout: no streaming chunk received for {timeout_s}s")


class StreamRepetitionError(RuntimeError):
    """Raised when the assistant stream repeats one fragment without converging.

    Carries ``stream_repetition`` so the same ``stream_stall`` bundle localizes it.
    """

    def __init__(self, period: int, repeats: int) -> None:
        self.period = period
        self.repeats = repeats
        super().__init__(
            f"stream_repetition: model output repeated the same {period}-char fragment "
            f"at least {repeats} times without converging"
        )


def _enters_tool_phase(chunk: Any) -> bool:
    """True when *chunk* is a model checkpoint that queued a tool call."""
    if not isinstance(chunk, dict) or chunk.get("type") not in _TOOL_PHASE_ENTRY_TYPES:
        return False
    data = chunk.get("data")
    messages = data.get("messages") if isinstance(data, dict) else None
    if not isinstance(messages, list):
        messages = chunk.get("messages")
    if not isinstance(messages, list):
        return False
    return any(
        isinstance(message, dict)
        and isinstance(message.get("tool_calls"), list)
        and bool(message["tool_calls"])
        for message in messages
    )


def _repetition(text: str) -> tuple[int, int] | None:
    """Return ``(period, repeats)`` when *text* ends in a stuck repetition loop.

    The smallest period of the tail comes from the KMP prefix function, so a
    fixed fragment repeated back-to-back is found without trying every length.
    """
    tail = text[-_REPEAT_WINDOW:]
    if len(tail) < _REPEAT_MIN_PERIOD * _REPEAT_MIN_COUNT:
        return None
    prefix = [0] * len(tail)
    for index in range(1, len(tail)):
        matched = prefix[index - 1]
        while matched and tail[index] != tail[matched]:
            matched = prefix[matched - 1]
        if tail[index] == tail[matched]:
            matched += 1
        prefix[index] = matched
    period = len(tail) - prefix[-1]
    repeats = len(tail) // period
    if period < _REPEAT_MIN_PERIOD or repeats < _REPEAT_MIN_COUNT:
        return None
    return period, repeats


async def stream_guard(
    source: AsyncIterator[Any],
    *,
    idle_timeout_s: float,
    cancel_stream: Callable[[str, str], None],
    agent_id: str,
    thread_id: str,
) -> AsyncIterator[Any]:
    """Yield chunks from *source* while enforcing the idle and repetition limits.

    ``idle_timeout_s <= 0`` disables the idle deadline; repetition is always
    checked, because a stuck loop never produces a useful result.
    """
    iterator = source.__aiter__()
    in_tool_phase = False
    accumulated = ""
    total_chars = 0
    next_check = _REPEAT_CHECK_STEP
    try:
        while True:
            try:
                if idle_timeout_s > 0 and not in_tool_phase:
                    chunk = await asyncio.wait_for(iterator.__anext__(), timeout=idle_timeout_s)
                else:
                    chunk = await iterator.__anext__()
            except StopAsyncIteration:
                return
            except TimeoutError:
                logger.warning(
                    "agent stream idle for %ss (agent=%s thread=%s); cancelling turn",
                    idle_timeout_s,
                    agent_id,
                    thread_id,
                )
                cancel_stream(agent_id, thread_id)
                raise StreamChunkTimeoutError(idle_timeout_s) from None
            if isinstance(chunk, dict):
                kind = chunk.get("type")
                if kind == "token":
                    piece = str(chunk.get("content") or "")
                    total_chars += len(piece)
                    # Keep only the tail: a long turn must not grow this buffer.
                    accumulated = (accumulated + piece)[-_REPEAT_WINDOW:]
                    if total_chars >= next_check:
                        found = _repetition(accumulated)
                        if found is not None:
                            period, repeats = found
                            logger.warning(
                                "agent stream repeating a %d-char fragment %d+ times "
                                "(agent=%s thread=%s); cancelling turn",
                                period,
                                repeats,
                                agent_id,
                                thread_id,
                            )
                            cancel_stream(agent_id, thread_id)
                            raise StreamRepetitionError(period, repeats)
                        next_check = total_chars + _REPEAT_CHECK_STEP
                elif kind in _STEP_BOUNDARY_TYPES:
                    accumulated = ""
                    total_chars = 0
                    next_check = _REPEAT_CHECK_STEP
                    if kind == "tool_result":
                        in_tool_phase = False
                    elif _enters_tool_phase(chunk):
                        in_tool_phase = True
            yield chunk
    finally:
        close = getattr(iterator, "aclose", None)
        if callable(close):
            with contextlib.suppress(Exception, GeneratorExit):
                await close()
