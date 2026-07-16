"""Single source of truth for parsing Claude Code assistant-message token usage.

Two paths read the SAME four usage buckets from the SAME assistant-message shape:
  * the out-of-band statusline (``cli/cmd_statusline``) — its re-read %, and
  * the backfill ingest path (``core/backfill``) — the Cost tab's numbers.

Keeping the four-bucket parse, the dedup KEY precedence, and the last-wins dedup
POLICY here keeps the two from drifting. A divergence (e.g. first-wins vs
last-wins) makes the statusline's re-read % disagree with the Cost tab for the
same session — small, but inconsistent numbers read like a bug and undercut
trust in the figures.

Pure stdlib on purpose: the statusline imports this and must stay
zero-dependency, fail-safe, and DB-free.
"""
from __future__ import annotations

import json
from typing import Iterable, Iterator, NamedTuple


class AssistantUsage(NamedTuple):
    """The four token buckets Claude Code reports per assistant message."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    @property
    def total(self) -> int:
        return (
            self.input_tokens
            + self.output_tokens
            + self.cache_read_tokens
            + self.cache_write_tokens
        )


def parse_usage(usage: object) -> AssistantUsage:
    """Read the four token buckets from a message ``usage`` mapping.

    Non-dict / missing / falsy fields degrade to 0 — the transcript is external
    data and any single record may be partial.
    """
    if not isinstance(usage, dict):
        return AssistantUsage()
    return AssistantUsage(
        input_tokens=int(usage.get("input_tokens") or 0),
        output_tokens=int(usage.get("output_tokens") or 0),
        cache_read_tokens=int(usage.get("cache_read_input_tokens") or 0),
        cache_write_tokens=int(usage.get("cache_creation_input_tokens") or 0),
    )


def assistant_message_key(record: dict, msg: dict, line_no: int) -> str:
    """Stable dedup key for an assistant record.

    Prefers the Anthropic API response id (``message.id``, ``msg_…``), which is
    stable across resume/branch re-ingest where the record ``uuid`` is
    regenerated; falls back to the record ``uuid`` then the 1-based line number.
    NEVER key on the token-usage signature — distinct real calls can share
    identical counts (#294).
    """
    return msg.get("id") or record.get("uuid") or str(line_no)


def _iter_assistant_records(
    lines: Iterable[str],
) -> Iterator[tuple[dict, dict, int, AssistantUsage]]:
    """Yield ``(record, msg, line_no, usage)`` for each billable assistant record.

    Skips non-JSON lines, non-assistant records, and empty/zero-usage records
    (no cost contribution) — mirroring the filters ``core/backfill`` applies so
    all readers see the same set of billable turns. Shared by
    ``iter_assistant_usage`` and ``iter_assistant_turns`` so the two can never
    disagree on which records count as a turn.
    """
    for line_no, line in enumerate(lines, start=1):
        try:
            record = json.loads(line)
        except Exception:
            continue
        if not isinstance(record, dict) or record.get("type") != "assistant":
            continue
        msg = record.get("message")
        if not isinstance(msg, dict):
            continue
        if not msg.get("usage"):
            continue
        usage = parse_usage(msg.get("usage"))
        if usage.total == 0:
            continue
        yield record, msg, line_no, usage


def iter_assistant_usage(lines: Iterable[str]) -> Iterator[tuple[str, AssistantUsage]]:
    """Yield ``(message_key, usage)`` for each assistant record carrying usage."""
    for record, msg, line_no, usage in _iter_assistant_records(lines):
        yield assistant_message_key(record, msg, line_no), usage


def iter_assistant_turns(
    lines: Iterable[str],
) -> Iterator[tuple[str, str | None, AssistantUsage]]:
    """Yield ``(message_key, model, usage)`` for each assistant record carrying usage.

    Same record + key as ``iter_assistant_usage`` but also surfaces the raw
    ``model`` id Claude Code stamped on that record. Needed by point-in-time
    previews (e.g. ``tj quickstart``'s "what you'd see live" section) that show
    the model in effect at a specific turn, not just the session aggregate.
    """
    for record, msg, line_no, usage in _iter_assistant_records(lines):
        yield assistant_message_key(record, msg, line_no), msg.get("model"), usage


def iter_cumulative_usage(
    lines: Iterable[str],
) -> Iterator[tuple[int, str | None, AssistantUsage]]:
    """Yield ``(turn_index, model, cumulative_usage)`` after each assistant turn.

    ``turn_index`` is the 1-based count of distinct assistant turns seen so far
    (after last-wins dedup); ``model`` is the raw model id on the record that
    produced this yield; ``cumulative_usage`` is the running last-wins-deduped
    total through that turn. The final yielded ``cumulative_usage`` always
    equals ``session_usage`` over the same lines — same dedup key + last-wins
    policy, just surfaced incrementally.

    Powers point-in-time previews (``tj quickstart``'s "what you'd have seen
    live" section) that need the statusline's numbers as they stood mid-session,
    not only the final total.
    """
    latest: dict[str, AssistantUsage] = {}
    running = AssistantUsage()
    turn_index = 0
    for key, model, usage in iter_assistant_turns(lines):
        prev = latest.get(key)
        if prev is None:
            turn_index += 1
            running = AssistantUsage(
                running.input_tokens + usage.input_tokens,
                running.output_tokens + usage.output_tokens,
                running.cache_read_tokens + usage.cache_read_tokens,
                running.cache_write_tokens + usage.cache_write_tokens,
            )
        else:
            running = AssistantUsage(
                running.input_tokens - prev.input_tokens + usage.input_tokens,
                running.output_tokens - prev.output_tokens + usage.output_tokens,
                running.cache_read_tokens - prev.cache_read_tokens + usage.cache_read_tokens,
                running.cache_write_tokens - prev.cache_write_tokens + usage.cache_write_tokens,
            )
        latest[key] = usage
        yield turn_index, model, running


def last_turn_context_tokens(lines: Iterable[str]) -> int:
    """Context-window occupancy at the most recent assistant turn.

    The prompt sent on a turn is all the prior context re-materialized: uncached
    new input + the cached prefix re-read + anything newly written to cache this
    turn (``input + cache_read + cache_write``). Output is NOT part of the window
    that was sent, so it's excluded. This is the size of the LIVE context window
    right now — a distinct signal from the session-total re-read *share*: it's how
    close the window is to a forced auto-compact, which is exactly when a
    user-chosen ``/compact`` is the right call regardless of what's driving the
    re-reads.

    Deduped last-wins by message key like :func:`session_usage` (streaming/resume
    appends several growing records per message; the final one is authoritative),
    then the final distinct turn's occupancy is returned. ``0`` when there are no
    billable turns — the caller treats that as "unknown / can't tell".
    """
    latest: dict[str, AssistantUsage] = {}
    order: list[str] = []
    for key, usage in iter_assistant_usage(lines):
        if key not in latest:
            order.append(key)
        latest[key] = usage
    if not order:
        return 0
    final = latest[order[-1]]
    return final.input_tokens + final.cache_read_tokens + final.cache_write_tokens


def session_usage(lines: Iterable[str]) -> AssistantUsage:
    """Total assistant usage over a session, deduped LAST-WINS by message key.

    Claude Code appends multiple transcript records for one API response during
    streaming/resume/branch — same ``message.id``, growing usage. The FINAL
    record carries the finalized cumulative usage, so the last write wins. This
    matches ``core/backfill``'s span dedup (#294); counting every record instead
    inflates the totals, and first-wins undercounts a still-streaming turn.
    """
    by_key: dict[str, AssistantUsage] = {}
    for key, usage in iter_assistant_usage(lines):
        by_key[key] = usage  # last-wins
    total = AssistantUsage()
    for usage in by_key.values():
        total = AssistantUsage(
            total.input_tokens + usage.input_tokens,
            total.output_tokens + usage.output_tokens,
            total.cache_read_tokens + usage.cache_read_tokens,
            total.cache_write_tokens + usage.cache_write_tokens,
        )
    return total
