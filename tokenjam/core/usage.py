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


def iter_assistant_usage(lines: Iterable[str]) -> Iterator[tuple[str, AssistantUsage]]:
    """Yield ``(message_key, usage)`` for each assistant record carrying usage.

    Skips non-JSON lines, non-assistant records, and empty-usage records (no
    cost contribution) — mirroring the filters ``core/backfill`` applies so both
    paths see the same set of billable turns.
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
        yield assistant_message_key(record, msg, line_no), usage


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
