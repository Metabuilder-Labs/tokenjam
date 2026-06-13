"""Context-cost diagnostic over Claude Code sessions (`tj context`).

The validated Claude-Code wedge (issue #4). Claude Code's built-ins leave a real
gap: ``/compact`` is reactive, lossy and single-session; ``/context`` shows
current-session totals only. Neither attributes *what* is burning quota across
sessions nor suggests a structural fix. A whole DIY ecosystem (ccusage,
codeburn, context-analyzer, session-recall, ...) has sprung up to fill it — the
strongest possible revealed-demand signal. Proof point: anthropics/claude-code
#24147, where a dev parsed 30 days of JSONL to find CLAUDE.md re-reads consumed
99.93% of their quota.

This module is the pure-logic core; :mod:`tokenjam.cli.cmd_context` renders it.
It reports three things over a window of CC sessions:

1. **Per-turn context composition** — for every assistant turn, how many tokens
   were spent *re-reading* prior context (cache-read tokens: conversation
   history, CLAUDE.md, tool-output accumulation) versus doing *net-new work*
   (uncached input + output). The headline is the re-read share.

2. **Recurring inclusions** — the same file read (Read tool, identical
   ``file_path``) repeated across many sessions, frequency-counted, each with a
   concrete structural fix (``@file`` reference / a CLAUDE.md entry). Requires
   ``[capture] tool_inputs = true``.

3. **Compact candidates** — sessions whose re-read share and accumulated
   context are high enough that a mid-session ``/compact`` (or a fresh session)
   would reclaim the most quota.

**Framing is quota-native** (subscription majority — see the
subscription-vs-cost-framing research): headline numbers render as
token-share / "% of cycle" via :mod:`tokenjam.core.framing`. Dollars are a
secondary calibration signal for API users, never the headline.

Honesty discipline (CLAUDE.md Rule 14): every figure is a *measured* token
share or a *structural* candidate flag, never a guaranteed saving. Cache-read
tokens are billed (at a reduced rate), not free — they are real quota.
"""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from tokenjam.otel.semconv import GenAIAttributes

# Honesty caveat surfaced verbatim next to the headline (CLAUDE.md Rule 14).
CONTEXT_HONESTY_CAVEAT = (
    "Re-read tokens are cache reads (billed at a reduced rate, not free) — real "
    "quota. Figures are measured shares and structural candidates, not "
    "guaranteed savings. Review before restructuring."
)

# A re-read share at or above this fraction is "context-heavy" and worth a
# compact / restructure look. Calibrated against the community signal that
# steady-state CC turns can run well above this once history + CLAUDE.md grow.
HIGH_REREAD_SHARE = 0.80

# A session needs at least this much accumulated cache-read to be a compact
# candidate — below it, the absolute quota reclaimed isn't worth the disruption.
COMPACT_MIN_CACHE_TOKENS = 200_000

# A recurring inclusion must appear in at least this many distinct sessions
# before we flag it — a file read in one session isn't "recurring".
RECURRING_MIN_SESSIONS = 3

# Cap on rows carried in the finding payload; aggregates are over ALL rows.
TOP_N = 10

# Tools whose identical input across sessions is a structural-fix candidate.
# Read re-pastes a file every session → `@file` / CLAUDE.md is the fix.
_FILE_READ_TOOLS = {"read", "view", "cat"}


@dataclass
class TurnComposition:
    """One assistant turn's token composition (re-read vs net-new work)."""

    session_id: str
    sub_agent_id: str | None
    model: str
    reread_tokens: int  # cache-read: history + CLAUDE.md + tool-output accrual
    new_input_tokens: int  # uncached input — genuinely new context this turn
    output_tokens: int  # work produced
    cache_write_tokens: int  # cache-creation (first-time caching of a prefix)
    cost_usd: float

    @property
    def total_tokens(self) -> int:
        return (
            self.reread_tokens
            + self.new_input_tokens
            + self.output_tokens
            + self.cache_write_tokens
        )

    @property
    def work_tokens(self) -> int:
        """Net-new work: uncached input + output (excludes re-reading)."""
        return self.new_input_tokens + self.output_tokens

    @property
    def reread_share(self) -> float:
        total = self.total_tokens
        return (self.reread_tokens / total) if total else 0.0


@dataclass
class RecurringInclusion:
    """A file/blob re-included across many sessions, with a structural fix."""

    label: str  # human label, e.g. "Read /path/to/db/schema.prisma"
    tool_name: str
    target: str  # the file_path / identifier re-included
    sessions: int  # distinct sessions it appears in
    occurrences: int  # total times across all sessions
    fix: str  # the structural-fix suggestion


@dataclass
class CompactCandidate:
    """A session whose accumulated re-reading makes it a compact candidate."""

    session_id: str
    reread_tokens: int
    total_tokens: int
    reread_share: float
    turns: int


@dataclass
class ContextDiagnostic:
    """Result of the context-cost diagnostic over a window."""

    since: datetime
    until: datetime
    sessions: int
    turns: int
    # Aggregate composition across all turns in the window.
    total_reread_tokens: int = 0
    total_new_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cache_write_tokens: int = 0
    total_cost_usd: float = 0.0
    # Per-turn breakdown lists (capped, ranked by re-read tokens).
    heaviest_turns: list[TurnComposition] = field(default_factory=list)
    recurring: list[RecurringInclusion] = field(default_factory=list)
    compact_candidates: list[CompactCandidate] = field(default_factory=list)
    # Capture state — drives the "enable [capture] and re-run backfill" nudge.
    tool_inputs_captured: bool = False
    notes: list[str] = field(default_factory=list)
    caveat: str = CONTEXT_HONESTY_CAVEAT

    @property
    def total_tokens(self) -> int:
        return (
            self.total_reread_tokens
            + self.total_new_input_tokens
            + self.total_output_tokens
            + self.total_cache_write_tokens
        )

    @property
    def total_work_tokens(self) -> int:
        return self.total_new_input_tokens + self.total_output_tokens

    @property
    def reread_share(self) -> float:
        total = self.total_tokens
        return (self.total_reread_tokens / total) if total else 0.0

    @property
    def has_data(self) -> bool:
        return self.turns > 0


def _parse_attrs(raw: Any) -> dict:
    """Coerce a span's ``attributes`` column into a dict (it may be JSON text)."""
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:  # noqa: BLE001 — malformed JSON is non-fatal
            return {}
    return {}


def _file_target(tool_name: str, tool_input: Any) -> str | None:
    """Extract a recurring-inclusion target from a tool's input.

    ``tool_input`` may itself be a dict or a JSON string (the test factory
    double-encodes it; real backfill stores the raw dict). We flag file reads
    by their ``file_path`` — the classic "re-paste the same file every session"
    pattern from #24147 — which structurally maps to an ``@file`` / CLAUDE.md
    fix. Other tools are skipped in v1 (named as a follow-up, not faked).
    """
    if tool_name.lower() not in _FILE_READ_TOOLS:
        return None
    if isinstance(tool_input, str):
        try:
            tool_input = json.loads(tool_input)
        except Exception:  # noqa: BLE001
            return None
    if not isinstance(tool_input, dict):
        return None
    path = tool_input.get("file_path") or tool_input.get("path")
    if not path or not isinstance(path, str):
        return None
    return path


def _fix_for(target: str) -> str:
    """Structural-fix suggestion for a recurring file inclusion."""
    return (
        f"Reference {target} as `@{target}` in your prompt or add it to "
        f"CLAUDE.md instead of re-reading it every session."
    )


def compute_context_diagnostic(
    conn: Any,
    since: datetime,
    until: datetime,
    *,
    agent_id: str | None = None,
    tool_inputs_captured: bool = False,
) -> ContextDiagnostic:
    """Run the context-cost diagnostic over a window of spans.

    ``conn`` is a direct DuckDB connection (the diagnostic needs the raw
    ``attributes`` column, not exposed over the API shim). Reads aggregate
    token counts for composition (no content capture required) and the tool
    ``attributes`` for recurring-inclusion detection (needs
    ``[capture] tool_inputs``).
    """
    result = ContextDiagnostic(
        since=since,
        until=until,
        sessions=0,
        turns=0,
        tool_inputs_captured=tool_inputs_captured,
    )

    turns = _load_turns(conn, since, until, agent_id)
    if not turns:
        return result

    result.turns = len(turns)
    result.sessions = len({t.session_id for t in turns})
    result.total_reread_tokens = sum(t.reread_tokens for t in turns)
    result.total_new_input_tokens = sum(t.new_input_tokens for t in turns)
    result.total_output_tokens = sum(t.output_tokens for t in turns)
    result.total_cache_write_tokens = sum(t.cache_write_tokens for t in turns)
    result.total_cost_usd = round(sum(t.cost_usd for t in turns), 8)

    result.heaviest_turns = sorted(
        turns, key=lambda t: t.reread_tokens, reverse=True
    )[:TOP_N]

    result.compact_candidates = _compact_candidates(turns)
    result.recurring = _recurring_inclusions(conn, since, until, agent_id)

    if not tool_inputs_captured:
        result.notes.append(
            "Recurring-inclusion detection needs `[capture] tool_inputs = true` "
            "in tj.toml, then `tj backfill claude-code` re-run on a fresh DB "
            "(re-ingest skips already-stored spans — see `tj context --help`)."
        )
    return result


def _load_turns(
    conn: Any,
    since: datetime,
    until: datetime,
    agent_id: str | None,
) -> list[TurnComposition]:
    """One row per assistant LLM turn, with its token composition."""
    clauses = [
        "name = $1",
        "start_time >= $2",
        "start_time < $3",
        "model IS NOT NULL",
    ]
    params: list[Any] = [GenAIAttributes.SPAN_LLM_CALL, since, until]
    if agent_id:
        clauses.append("agent_id = $" + str(len(params) + 1))
        params.append(agent_id)
    where = " AND ".join(clauses)
    rows = conn.execute(
        "SELECT session_id, agent_id, model, "
        "COALESCE(input_tokens, 0), COALESCE(output_tokens, 0), "
        "COALESCE(cache_tokens, 0), COALESCE(cache_write_tokens, 0), "
        "COALESCE(cost_usd, 0.0) "
        "FROM spans WHERE " + where,
        params,
    ).fetchall()

    turns: list[TurnComposition] = []
    for (sid, said, model, in_tok, out_tok, cache_tok, cache_w_tok, cost) in rows:
        turns.append(
            TurnComposition(
                session_id=str(sid) if sid is not None else "unknown",
                sub_agent_id=str(said) if said is not None else None,
                model=str(model or "unknown"),
                reread_tokens=int(cache_tok or 0),
                new_input_tokens=int(in_tok or 0),
                output_tokens=int(out_tok or 0),
                cache_write_tokens=int(cache_w_tok or 0),
                cost_usd=float(cost or 0.0),
            )
        )
    return turns


def _compact_candidates(turns: list[TurnComposition]) -> list[CompactCandidate]:
    """Sessions whose accumulated re-reading clears the compact thresholds."""
    by_session: dict[str, list[TurnComposition]] = defaultdict(list)
    for t in turns:
        by_session[t.session_id].append(t)

    candidates: list[CompactCandidate] = []
    for sid, sess_turns in by_session.items():
        reread = sum(t.reread_tokens for t in sess_turns)
        total = sum(t.total_tokens for t in sess_turns)
        share = (reread / total) if total else 0.0
        if reread >= COMPACT_MIN_CACHE_TOKENS and share >= HIGH_REREAD_SHARE:
            candidates.append(
                CompactCandidate(
                    session_id=sid,
                    reread_tokens=reread,
                    total_tokens=total,
                    reread_share=round(share, 4),
                    turns=len(sess_turns),
                )
            )
    candidates.sort(key=lambda c: c.reread_tokens, reverse=True)
    return candidates[:TOP_N]


def _recurring_inclusions(
    conn: Any,
    since: datetime,
    until: datetime,
    agent_id: str | None,
) -> list[RecurringInclusion]:
    """Files re-read across many sessions (capture-gated, structural fix each)."""
    clauses = [
        "name = $1",
        "start_time >= $2",
        "start_time < $3",
        "tool_name IS NOT NULL",
    ]
    params: list[Any] = [GenAIAttributes.SPAN_TOOL_CALL, since, until]
    if agent_id:
        clauses.append("agent_id = $" + str(len(params) + 1))
        params.append(agent_id)
    where = " AND ".join(clauses)
    rows = conn.execute(
        "SELECT session_id, tool_name, attributes FROM spans WHERE " + where,
        params,
    ).fetchall()

    # (tool_name, target) -> {sessions: set, occurrences: int}
    agg: dict[tuple[str, str], dict[str, Any]] = defaultdict(
        lambda: {"sessions": set(), "occurrences": 0}
    )
    for sid, tool_name, attrs_raw in rows:
        attrs = _parse_attrs(attrs_raw)
        tool_input = attrs.get(GenAIAttributes.TOOL_INPUT)
        if tool_input is None:
            continue
        target = _file_target(str(tool_name or ""), tool_input)
        if target is None:
            continue
        key = (str(tool_name), target)
        agg[key]["sessions"].add(str(sid) if sid is not None else "unknown")
        agg[key]["occurrences"] += 1

    inclusions: list[RecurringInclusion] = []
    for (tool_name, target), data in agg.items():
        session_count = len(data["sessions"])
        if session_count < RECURRING_MIN_SESSIONS:
            continue
        inclusions.append(
            RecurringInclusion(
                label=f"{tool_name} {target}",
                tool_name=tool_name,
                target=target,
                sessions=session_count,
                occurrences=int(data["occurrences"]),
                fix=_fix_for(target),
            )
        )
    inclusions.sort(key=lambda r: (r.sessions, r.occurrences), reverse=True)
    return inclusions[:TOP_N]


def diagnostic_to_dict(diag: ContextDiagnostic) -> dict[str, Any]:
    """JSON-serialisable view of the diagnostic for ``--json`` output."""
    return {
        "since": diag.since.isoformat(),
        "until": diag.until.isoformat(),
        "sessions": diag.sessions,
        "turns": diag.turns,
        "total_reread_tokens": diag.total_reread_tokens,
        "total_new_input_tokens": diag.total_new_input_tokens,
        "total_output_tokens": diag.total_output_tokens,
        "total_cache_write_tokens": diag.total_cache_write_tokens,
        "total_work_tokens": diag.total_work_tokens,
        "total_tokens": diag.total_tokens,
        "reread_share": round(diag.reread_share, 4),
        "total_cost_usd": round(diag.total_cost_usd, 6),
        "heaviest_turns": [
            {
                "session_id": t.session_id,
                "sub_agent_id": t.sub_agent_id,
                "model": t.model,
                "reread_tokens": t.reread_tokens,
                "new_input_tokens": t.new_input_tokens,
                "output_tokens": t.output_tokens,
                "reread_share": round(t.reread_share, 4),
            }
            for t in diag.heaviest_turns
        ],
        "recurring": [
            {
                "label": r.label,
                "tool_name": r.tool_name,
                "target": r.target,
                "sessions": r.sessions,
                "occurrences": r.occurrences,
                "fix": r.fix,
            }
            for r in diag.recurring
        ],
        "compact_candidates": [
            {
                "session_id": c.session_id,
                "reread_tokens": c.reread_tokens,
                "total_tokens": c.total_tokens,
                "reread_share": c.reread_share,
                "turns": c.turns,
            }
            for c in diag.compact_candidates
        ],
        "tool_inputs_captured": diag.tool_inputs_captured,
        "notes": diag.notes,
        "caveat": diag.caveat,
    }
