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

# A periodic tail is a stuck model, not prose. Long fragments keep the original
# bar (80 chars x 6). Short periods are exactly how degeneration looks (a single
# char, ".2"), so they are detected by coverage: they must fill a much larger
# suffix before the turn is cancelled. Re-check every _REPEAT_CHECK_STEP chars
# to keep the per-chunk cost bounded.
_REPEAT_WINDOW = 4096
_REPEAT_MAX_PERIOD = 256
_REPEAT_SHORT_PERIOD = 4
_REPEAT_SHORT_COVER = 2048
_REPEAT_MEDIUM_COVER = 1024
_REPEAT_MIN_PERIOD = 80
_REPEAT_MIN_COUNT = 6
_REPEAT_CHECK_STEP = 1024

# Text kinds the guard accumulates. Reasoning degenerates the same way tokens do.
_TEXT_TYPES = frozenset({"token", "reasoning"})

# Absolute ceiling for one turn's text (tokens + reasoning). A model can stream
# non-periodic noise forever, so periodicity alone is not a sufficient bound.
_MAX_TURN_CHARS = 200_000


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


class StreamOutputLimitError(RuntimeError):
    """Raised when one turn streams more text than any real answer needs.

    Carries ``stream_output_limit`` so the same ``stream_stall`` bundle localizes it.
    """

    def __init__(self, chars: int) -> None:
        self.chars = chars
        super().__init__(
            f"stream_output_limit: model output reached {chars} characters in one turn"
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


def _periodic_suffix_coverage(text: str, period: int) -> int:
    """Return the length of the longest suffix of *text* that repeats with *period*.

    Only the tail has to be periodic: degeneration starts mid-answer, after
    otherwise finite prose.
    """
    if period < 1 or len(text) < period * 2:
        return 0
    matched = 0
    index = len(text) - 1
    while index - period >= 0 and text[index] == text[index - period]:
        matched += 1
        index -= 1
    if matched < period:
        return 0
    return matched + period


def _repetition(text: str) -> tuple[int, int] | None:
    """Return ``(period, repeats)`` when *text* ends in a stuck repetition loop.

    Short periods get a coverage bar instead of a repeat count so a 1-char or
    2-char loop is caught without flagging normal short repeats (separators,
    tables). A long fragment is also detected by the whole-window fallback.
    """
    tail = text[-_REPEAT_WINDOW:]
    best_period = 0
    best_coverage = 0
    max_period = min(_REPEAT_MAX_PERIOD, len(tail) // 2)
    for period in range(1, max_period + 1):
        coverage = _periodic_suffix_coverage(tail, period)
        if coverage > best_coverage:
            best_period = period
            best_coverage = coverage
    if best_period:
        repeats = best_coverage // best_period
        if repeats < _REPEAT_MIN_COUNT:
            return None
        if best_period <= _REPEAT_SHORT_PERIOD:
            required = _REPEAT_SHORT_COVER
        elif best_period < _REPEAT_MIN_PERIOD:
            required = _REPEAT_MEDIUM_COVER
        else:
            required = _REPEAT_MIN_PERIOD * _REPEAT_MIN_COUNT
        if best_coverage >= required:
            return best_period, repeats
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
    repeats = len(tail) // period if period else 0
    if period >= _REPEAT_MIN_PERIOD and repeats >= _REPEAT_MIN_COUNT:
        return period, repeats
    return None


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
    buffers: dict[str, str] = dict.fromkeys(_TEXT_TYPES, "")
    segment_chars: dict[str, int] = dict.fromkeys(_TEXT_TYPES, 0)
    next_check: dict[str, int] = dict.fromkeys(_TEXT_TYPES, _REPEAT_CHECK_STEP)
    turn_chars = 0
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
                if kind in _TEXT_TYPES:
                    piece = str(chunk.get("content") or "")
                    turn_chars += len(piece)
                    if turn_chars >= _MAX_TURN_CHARS:
                        logger.warning(
                            "agent stream exceeded %d chars in one turn "
                            "(agent=%s thread=%s); cancelling turn",
                            _MAX_TURN_CHARS,
                            agent_id,
                            thread_id,
                        )
                        cancel_stream(agent_id, thread_id)
                        raise StreamOutputLimitError(turn_chars)
                    # Keep only the tail: a long turn must not grow this buffer.
                    buffer = (buffers[kind] + piece)[-_REPEAT_WINDOW:]
                    buffers[kind] = buffer
                    segment_chars[kind] += len(piece)
                    if segment_chars[kind] >= next_check[kind]:
                        found = _repetition(buffer)
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
                        next_check[kind] = segment_chars[kind] + _REPEAT_CHECK_STEP
                elif kind in _STEP_BOUNDARY_TYPES:
                    for text_kind in _TEXT_TYPES:
                        buffers[text_kind] = ""
                        segment_chars[text_kind] = 0
                        next_check[text_kind] = _REPEAT_CHECK_STEP
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
