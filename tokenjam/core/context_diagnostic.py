"""Context-cost diagnostic over Claude Code sessions (`tj context`).

The validated Claude-Code wedge (issue #4). Claude Code's built-ins leave a real
gap: ``/compact`` is reactive, lossy and single-session; ``/context`` shows
current-session totals only. Neither attributes *what* is burning quota across
sessions nor suggests a structural fix. A whole DIY ecosystem of quota-tracking
tools has sprung up to fill it — the
strongest possible revealed-demand signal. Proof point: anthropics/claude-code
#24147, where a dev parsed 30 days of JSONL to find CLAUDE.md re-reads consumed
99.93% of their quota.

This module is the pure-logic core; :mod:`tokenjam.cli.cmd_context` renders it.
It reports three things over a window of CC sessions:

1. **Per-turn context composition** — for every assistant turn, how many tokens
   were spent *re-reading* prior context (cache-read tokens: conversation
   history, CLAUDE.md, tool-output accumulation) versus *cache-miss* overhead
   (cache-creation tokens — uncached input that missed the cache and had to be
   written to it, billed at a premium) versus doing *net-new work* (uncached
   input + output). The headline is the re-read share; cache-miss is broken out
   as its own named overhead source (#11).

   **Named overhead sources (#11).** #4 named re-read overhead (cache reads).
   #11 asked to additionally attribute two large, specific overhead sources as
   their own named categories:
     * **prompt-cache MISS** (cache-creation tokens) — *derivable*: the
       on-disk usage block carries ``cache_creation_input_tokens`` separately,
       mapped to ``NormalizedSpan.cache_write_tokens``. These are input that
       missed the cache and had to be written to it (Anthropic bills cache
       writes at a premium over base input), so they are genuine, nameable
       overhead distinct from both re-read and net-new work. Surfaced as the
       ``cache_miss`` category below.
     * **MCP schema-injection** (~25K tok/call) — *NOT derivable from current
       data and deliberately parked.* See ``MCP_INJECTION_PARK_NOTE``: neither
       the Claude Code on-disk transcript nor live spans carry per-tool /
       per-schema token attribution. Tool-definition / system-prompt tokens are
       folded indistinguishably into ``input_tokens`` / ``cache_creation``;
       ``mcp__``-prefixed names appear only on tool-*invocation* spans (which
       carry no schema-injection token count). Attributing it would require a
       new capture path (a per-request tool-schema token delta), so this half
       is parked rather than faked.

2. **Recurring inclusions** — content re-pasted across many sessions/turns,
   frequency-counted, each with a concrete structural fix. Four kinds, all
   capture-gated:
     * **file reads** — the same file (Read tool, identical ``file_path``)
       re-read across sessions → ``@file`` / a CLAUDE.md entry
       (needs ``[capture] tool_inputs = true``);
     * **searches** — the same Grep / Glob / Search query re-run across sessions
       → pin the result / capture it once (needs ``[capture] tool_inputs``);
     * **prompts** — the same user prompt re-sent across turns/sessions
       → save it as a slash-command / CLAUDE.md note
       (needs ``[capture] prompts``);
     * **large tool outputs** — the same large tool output re-pasted across
       turns/sessions → reference the artifact instead of re-running
       (needs ``[capture] tool_outputs``; only the live ingest path captures
       tool output — the on-disk transcript carries none).

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
    "quota. Cache-miss tokens are cache writes (billed at a premium). Figures "
    "are measured shares and structural candidates, not guaranteed savings. "
    "Review before restructuring."
)

# Parked half of #11: MCP schema-injection attribution is not derivable from
# the data we currently capture. Surfaced as a note (not a fabricated number)
# so the gap is honest and the follow-up is precise (CLAUDE.md Rule 14).
MCP_INJECTION_PARK_NOTE = (
    "MCP tool-schema injection (~25K tokens/call when MCP servers are attached) "
    "is not yet broken out: the on-disk transcript and live spans carry no "
    "per-tool / per-schema token attribution — tool-definition tokens are folded "
    "into input/cache-creation, and `mcp__`-prefixed names appear only on "
    "tool-invocation spans (no schema-injection token count). Capturing a "
    "per-request tool-schema token delta would let `tj context` attribute it."
)

# Subagent accounting (#60). A Claude Code session that delegates to a Task
# subagent records the handoff as a `Task` tool call in the PARENT transcript;
# each subagent's own LLM turns live in a separate `subagents/agent-*.jsonl`
# file that backfill folds in under the same session with a `sub_agent_id`.
# When that subagent transcript is present its turns are already in the weighted
# quota below (this metric never filters `sub_agent_id`). But if the subagent
# file is missing — pruned, or the delegation ran outside the on-disk transcript
# — the parent shows the Task delegation with NO matching subagent turns, so the
# weighted quota is silently LOW for that session (the original #60 A/B measured
# ~half of Claude Code's subagent-inclusive total_cost_usd). Rather than report
# a confidently-wrong total we FLAG those sessions as a lower bound (Rule 14):
# the number is honest about what it can't see.
SUBAGENT_UNACCOUNTED_NOTE = (
    "{n} delegating session(s) recorded a Task subagent handoff whose subagent "
    "turns aren't in the data — their weighted quota is a LOWER BOUND, not a "
    "complete total. Re-run `tj backfill claude-code` while the subagent "
    "transcripts still exist (~/.claude/projects/**/subagents/), or use the live "
    "ingest path, to account for the delegated work."
)

# Tool name Claude Code stamps on the parent-transcript span that delegates to a
# subagent. Backfill copies it verbatim from the tool_use block's `name`, so a
# case-insensitive match is the delegation signal. Kept as a constant so the
# detection query and any future renderer agree on it.
DELEGATION_TOOL_NAME = "task"

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

# Repeated PROMPTS and repeated tool OUTPUTS re-paste WITHIN a session (across
# turns) too, not only across sessions — so they're flagged on an occurrence
# count rather than a distinct-session count. A prompt/output seen at least this
# many times (anywhere in the window) is "recurring".
RECURRING_MIN_OCCURRENCES = 3

# Only LARGE tool outputs are worth flagging — a tiny repeated output (a status
# string, a short JSON) costs almost no quota to re-paste. Gate on the captured
# output's character length as a cheap proxy for token weight.
LARGE_OUTPUT_MIN_CHARS = 2_000

# Cap on rows carried in the finding payload; aggregates are over ALL rows.
TOP_N = 10

# Inclusion-type tags carried on each RecurringInclusion. Stable strings — the
# JSON payload and the renderer key off these.
INCLUSION_FILE_READ = "file_read"
INCLUSION_SEARCH = "search"
INCLUSION_PROMPT = "prompt"
INCLUSION_TOOL_OUTPUT = "tool_output"

# Tools whose identical input across sessions is a structural-fix candidate.
# Read re-pastes a file every session → `@file` / CLAUDE.md is the fix.
_FILE_READ_TOOLS = {"read", "view", "cat"}

# Search/query tools: re-running the SAME query across sessions re-pastes its
# result every time → pin / capture the result instead.
_SEARCH_TOOLS = {"grep", "glob", "search", "ripgrep", "rg"}

# Tool-input keys that hold a search query/pattern, in priority order.
_SEARCH_QUERY_KEYS = ("pattern", "query", "regex", "q", "search")


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
    def cache_miss_tokens(self) -> int:
        """Cache-MISS overhead: cache-creation tokens (input that missed the
        cache and had to be written to it, billed at a premium). A named
        overhead source distinct from re-read and net-new work (#11)."""
        return self.cache_write_tokens

    @property
    def reread_share(self) -> float:
        total = self.total_tokens
        return (self.reread_tokens / total) if total else 0.0

    @property
    def cache_miss_share(self) -> float:
        total = self.total_tokens
        return (self.cache_miss_tokens / total) if total else 0.0


@dataclass
class RecurringInclusion:
    """A blob re-included across many sessions/turns, with a structural fix.

    ``inclusion_type`` distinguishes the four detected kinds (file read, search,
    prompt, large tool output) so the renderer and JSON payload can group and
    label them. ``tool_name`` is empty for prompt inclusions (no tool span).
    """

    label: str  # human label, e.g. "Read /path/to/db/schema.prisma"
    tool_name: str
    target: str  # the file_path / query / prompt-excerpt / output-excerpt
    sessions: int  # distinct sessions it appears in
    occurrences: int  # total times across all sessions/turns
    fix: str  # the structural-fix suggestion
    inclusion_type: str = INCLUSION_FILE_READ


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
    prompts_captured: bool = False
    tool_outputs_captured: bool = False
    # Subagent accounting (#60). `subagent_turns` counts LLM turns attributed to
    # a Task subagent (`sub_agent_id` set) — already folded into the weighted
    # quota above; surfaced for transparency. `delegating_sessions` is how many
    # sessions recorded a Task delegation in the window.
    # `unaccounted_subagent_sessions` are the delegating sessions whose subagent
    # turns are MISSING from the data, so their weighted quota is a lower bound.
    subagent_turns: int = 0
    delegating_sessions: int = 0
    unaccounted_subagent_sessions: list[str] = field(default_factory=list)
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
    def total_cache_miss_tokens(self) -> int:
        """Cache-MISS overhead across the window: cache-creation tokens (#11).

        Aliases ``total_cache_write_tokens`` under the named-overhead framing —
        cache writes are input that missed the cache and were written to it,
        billed at a premium, so they're a nameable overhead source distinct
        from re-read (cache reads) and net-new work."""
        return self.total_cache_write_tokens

    @property
    def reread_share(self) -> float:
        total = self.total_tokens
        return (self.total_reread_tokens / total) if total else 0.0

    @property
    def cache_miss_share(self) -> float:
        total = self.total_tokens
        return (self.total_cache_miss_tokens / total) if total else 0.0

    @property
    def subagent_accounting_partial(self) -> bool:
        """True when at least one delegating session's subagent turns are
        missing, so the window's weighted quota under-counts delegated work and
        is a lower bound rather than a complete total (#60)."""
        return bool(self.unaccounted_subagent_sessions)

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


def _coerce_input(tool_input: Any) -> dict | None:
    """Coerce a tool input (dict or JSON string) into a dict, else ``None``.

    The test factory double-encodes the input as a JSON string; real backfill
    stores the raw dict. Both must work.
    """
    if isinstance(tool_input, str):
        try:
            tool_input = json.loads(tool_input)
        except Exception:  # noqa: BLE001
            return None
    return tool_input if isinstance(tool_input, dict) else None


def _file_target(tool_name: str, tool_input: Any) -> str | None:
    """Extract a file-read recurring-inclusion target from a tool's input.

    Flags file reads by their ``file_path`` — the classic "re-paste the same
    file every session" pattern from #24147 — which structurally maps to an
    ``@file`` / CLAUDE.md fix.
    """
    if tool_name.lower() not in _FILE_READ_TOOLS:
        return None
    data = _coerce_input(tool_input)
    if data is None:
        return None
    path = data.get("file_path") or data.get("path")
    if not path or not isinstance(path, str):
        return None
    return path


def _search_target(tool_name: str, tool_input: Any) -> str | None:
    """Extract a search-query recurring-inclusion target from a tool's input.

    Re-running the SAME Grep/Glob/Search query across sessions re-pastes its
    result every time — keying on the query/pattern (the tool *input*, present
    on the backfill path too) makes this detectable without the output.
    """
    if tool_name.lower() not in _SEARCH_TOOLS:
        return None
    data = _coerce_input(tool_input)
    if data is None:
        return None
    for key in _SEARCH_QUERY_KEYS:
        val = data.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return None


def _normalize_blob(text: str) -> str:
    """Collapse whitespace so trivially-different copies hash to one signature.

    Repeated prompts/outputs that differ only in trailing whitespace or line
    endings are the same inclusion for our purposes.
    """
    return " ".join(text.split())


def _excerpt(text: str, limit: int = 80) -> str:
    """A short single-line label for a long blob."""
    flat = _normalize_blob(text)
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"


def _fix_for(inclusion_type: str, target: str) -> str:
    """Structural-fix suggestion for a recurring inclusion, per kind."""
    if inclusion_type == INCLUSION_FILE_READ:
        return (
            f"Reference {target} as `@{target}` in your prompt or add it to "
            f"CLAUDE.md instead of re-reading it every session."
        )
    if inclusion_type == INCLUSION_SEARCH:
        return (
            f"Pin the result of this search (`{target}`) — capture it once into "
            f"CLAUDE.md or an `@file` note instead of re-running it every session."
        )
    if inclusion_type == INCLUSION_PROMPT:
        return (
            "Save this repeated prompt as a slash-command (or a CLAUDE.md note) "
            "and reference it, instead of re-typing the full text each turn."
        )
    if inclusion_type == INCLUSION_TOOL_OUTPUT:
        return (
            "This large output re-pastes across turns — write it to a file once "
            "and reference the artifact (`@file`) instead of re-running the tool "
            "and re-feeding its output."
        )
    return "Capture this once and reference it instead of re-including it."


def compute_context_diagnostic(
    conn: Any,
    since: datetime,
    until: datetime,
    *,
    agent_id: str | None = None,
    tool_inputs_captured: bool = False,
    prompts_captured: bool = False,
    tool_outputs_captured: bool = False,
) -> ContextDiagnostic:
    """Run the context-cost diagnostic over a window of spans.

    ``conn`` is a direct DuckDB connection (the diagnostic needs the raw
    ``attributes`` column, not exposed over the API shim). Reads aggregate
    token counts for composition (no content capture required) and the span
    ``attributes`` for recurring-inclusion detection.

    Recurring-inclusion detection is per-kind capture-gated, mirroring the four
    ``[capture]`` toggles:
      * file reads + searches need ``tool_inputs_captured`` (read the tool input);
      * repeated prompts need ``prompts_captured`` (``gen_ai.prompt.content``);
      * large repeated outputs need ``tool_outputs_captured``
        (``gen_ai.tool.output`` — only the live ingest path captures it).
    A kind whose capture flag is off is simply not detected; default-off
    behavior (every flag False) is byte-for-byte unchanged.
    """
    result = ContextDiagnostic(
        since=since,
        until=until,
        sessions=0,
        turns=0,
        tool_inputs_captured=tool_inputs_captured,
        prompts_captured=prompts_captured,
        tool_outputs_captured=tool_outputs_captured,
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
    result.recurring = _recurring_inclusions(
        conn,
        since,
        until,
        agent_id,
        tool_inputs_captured=tool_inputs_captured,
        prompts_captured=prompts_captured,
        tool_outputs_captured=tool_outputs_captured,
    )

    _apply_subagent_accounting(result, conn, since, until, agent_id, turns)

    if not (tool_inputs_captured or prompts_captured or tool_outputs_captured):
        result.notes.append(
            "Recurring-inclusion detection needs content capture: "
            "`[capture] tool_inputs = true` (repeated file reads / searches), "
            "`prompts = true` (repeated prompts), `tool_outputs = true` (large "
            "repeated outputs) in tj.toml, then `tj backfill claude-code "
            "--reingest` (re-ingest backfills content onto stored spans — see "
            "`tj context --help`). Tool outputs are only captured on the live "
            "ingest path, not the on-disk transcript."
        )

    # Parked half of #11 — only worth surfacing once there are turns to attribute
    # against. Honest gap, not a fabricated figure (CLAUDE.md Rule 14).
    result.notes.append(MCP_INJECTION_PARK_NOTE)
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
    # Select the real `sub_agent_id` column — NOT `agent_id`. This row's
    # `TurnComposition.sub_agent_id` is the Task-subagent attribution the #60
    # accounting reads (and the `heaviest_turns[].sub_agent_id` JSON emits);
    # before this fix the query selected `agent_id` and bound it to that field,
    # so the metric never actually saw which turns were subagent turns.
    rows = conn.execute(
        "SELECT session_id, sub_agent_id, model, "
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


def _delegating_session_ids(
    conn: Any,
    since: datetime,
    until: datetime,
    agent_id: str | None,
) -> set[str]:
    """Session ids that recorded a Task delegation (subagent handoff) in window.

    Claude Code stamps ``tool_name = "Task"`` on the parent-transcript tool span
    that spawns a subagent, so a distinct-session scan over those spans tells us
    which sessions delegated — independent of whether the subagent's own
    transcript was captured. Case-insensitive on the tool name (#60).
    """
    clauses = [
        "name = $1",
        "start_time >= $2",
        "start_time < $3",
        "LOWER(tool_name) = $4",
    ]
    params: list[Any] = [
        GenAIAttributes.SPAN_TOOL_CALL, since, until, DELEGATION_TOOL_NAME,
    ]
    if agent_id:
        clauses.append("agent_id = $" + str(len(params) + 1))
        params.append(agent_id)
    where = " AND ".join(clauses)
    rows = conn.execute(
        "SELECT DISTINCT session_id FROM spans WHERE " + where, params
    ).fetchall()
    return {str(r[0]) for r in rows if r[0] is not None}


def _apply_subagent_accounting(
    result: ContextDiagnostic,
    conn: Any,
    since: datetime,
    until: datetime,
    agent_id: str | None,
    turns: list[TurnComposition],
) -> None:
    """Populate the subagent-accounting fields + partial-quota note (#60).

    The weighted quota already includes captured subagent turns (``_load_turns``
    never filters ``sub_agent_id``). This adds the honesty half the ticket asks
    for: when a session *delegated* (a ``Task`` tool span in the parent) but no
    subagent turns were captured for it, its weighted quota is a lower bound —
    so we flag it instead of letting the number read as complete (Rule 14).
    """
    result.subagent_turns = sum(1 for t in turns if t.sub_agent_id)
    sessions_with_subagent_turns = {
        t.session_id for t in turns if t.sub_agent_id
    }
    delegating = _delegating_session_ids(conn, since, until, agent_id)
    result.delegating_sessions = len(delegating)
    result.unaccounted_subagent_sessions = sorted(
        delegating - sessions_with_subagent_turns
    )
    if result.unaccounted_subagent_sessions:
        result.notes.append(
            SUBAGENT_UNACCOUNTED_NOTE.format(
                n=len(result.unaccounted_subagent_sessions)
            )
        )


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
    *,
    tool_inputs_captured: bool,
    prompts_captured: bool,
    tool_outputs_captured: bool,
) -> list[RecurringInclusion]:
    """All recurring inclusions across the window, generalized over kind.

    Every kind aggregates into one ``(inclusion_type, tool_name, signature)``
    bucket carrying a distinct-session set + an occurrence count, then is
    frequency-filtered and turned into a :class:`RecurringInclusion` with a
    type-appropriate structural fix. Each kind is gated on its capture flag so a
    flag-off kind contributes nothing.
    """
    # key -> {sessions: set, occurrences: int, target: str, label: str}
    agg: dict[tuple[str, str, str], dict[str, Any]] = defaultdict(
        lambda: {"sessions": set(), "occurrences": 0, "target": "", "label": ""}
    )

    def _add(
        itype: str,
        tool_name: str,
        signature: str,
        sid: Any,
        *,
        target: str | None = None,
        label: str | None = None,
    ) -> None:
        key = (itype, tool_name, signature)
        bucket = agg[key]
        bucket["sessions"].add(str(sid) if sid is not None else "unknown")
        bucket["occurrences"] += 1
        if not bucket["target"]:
            bucket["target"] = target if target is not None else signature
        if not bucket["label"]:
            bucket["label"] = label or bucket["target"]

    if tool_inputs_captured:
        _collect_tool_inputs(conn, since, until, agent_id, _add)
    if prompts_captured:
        _collect_prompts(conn, since, until, agent_id, _add)
    if tool_outputs_captured:
        _collect_tool_outputs(conn, since, until, agent_id, _add)

    inclusions: list[RecurringInclusion] = []
    for (itype, tool_name, _sig), data in agg.items():
        session_count = len(data["sessions"])
        occurrences = int(data["occurrences"])
        # File reads / searches must recur across DISTINCT sessions to count
        # (a single session re-reading a file isn't structural). Prompts /
        # large outputs re-paste across TURNS within a session too, so they gate
        # on the raw occurrence count instead.
        if itype in (INCLUSION_FILE_READ, INCLUSION_SEARCH):
            if session_count < RECURRING_MIN_SESSIONS:
                continue
        elif occurrences < RECURRING_MIN_OCCURRENCES:
            continue
        inclusions.append(
            RecurringInclusion(
                label=str(data["label"]),
                tool_name=tool_name,
                target=str(data["target"]),
                sessions=session_count,
                occurrences=occurrences,
                fix=_fix_for(itype, str(data["target"])),
                inclusion_type=itype,
            )
        )
    # Rank by occurrence then session spread, so the heaviest re-paste leads.
    inclusions.sort(key=lambda r: (r.occurrences, r.sessions), reverse=True)
    return inclusions[:TOP_N]


def _collect_tool_inputs(conn, since, until, agent_id, add) -> None:
    """File-read + search inclusions, both keyed off the tool *input*."""
    rows = _tool_span_rows(
        conn, since, until, agent_id, GenAIAttributes.TOOL_INPUT
    )
    for sid, tool_name, attrs in rows:
        tool_input = attrs.get(GenAIAttributes.TOOL_INPUT)
        if tool_input is None:
            continue
        name = str(tool_name or "")
        path = _file_target(name, tool_input)
        if path is not None:
            add(INCLUSION_FILE_READ, str(tool_name), path, sid,
                target=path, label=f"{tool_name} {path}")
            continue
        query = _search_target(name, tool_input)
        if query is not None:
            add(INCLUSION_SEARCH, str(tool_name), query, sid,
                target=query, label=f"{tool_name} {query}")


def _collect_prompts(conn, since, until, agent_id, add) -> None:
    """Repeated identical user prompts across turns/sessions."""
    rows = _llm_span_rows(
        conn, since, until, agent_id, GenAIAttributes.PROMPT_CONTENT
    )
    for sid, _tool_name, attrs in rows:
        content = attrs.get(GenAIAttributes.PROMPT_CONTENT)
        if not isinstance(content, str) or not content.strip():
            continue
        sig = _normalize_blob(content)
        if not sig:
            continue
        add(INCLUSION_PROMPT, "", sig, sid,
            target=_excerpt(content), label=_excerpt(content))


def _collect_tool_outputs(conn, since, until, agent_id, add) -> None:
    """Large identical tool outputs re-pasted across turns/sessions."""
    rows = _tool_span_rows(
        conn, since, until, agent_id, GenAIAttributes.TOOL_OUTPUT
    )
    for sid, tool_name, attrs in rows:
        output = attrs.get(GenAIAttributes.TOOL_OUTPUT)
        if not isinstance(output, str):
            continue
        if len(output) < LARGE_OUTPUT_MIN_CHARS:
            continue
        sig = _normalize_blob(output)
        if not sig:
            continue
        add(INCLUSION_TOOL_OUTPUT, str(tool_name or ""), sig, sid,
            target=_excerpt(output), label=f"{tool_name} → {_excerpt(output)}")


def _span_rows(
    conn: Any,
    span_name: str,
    since: datetime,
    until: datetime,
    agent_id: str | None,
    extra_clause: str,
) -> list[tuple[Any, Any, dict]]:
    """Shared SELECT of (session_id, tool_name, parsed-attributes) rows."""
    clauses = [
        "name = $1",
        "start_time >= $2",
        "start_time < $3",
        extra_clause,
    ]
    params: list[Any] = [span_name, since, until]
    if agent_id:
        clauses.append("agent_id = $" + str(len(params) + 1))
        params.append(agent_id)
    where = " AND ".join(clauses)
    rows = conn.execute(
        "SELECT session_id, tool_name, attributes FROM spans WHERE " + where,
        params,
    ).fetchall()
    return [(sid, tool_name, _parse_attrs(attrs_raw))
            for sid, tool_name, attrs_raw in rows]


def _tool_span_rows(conn, since, until, agent_id, _attr):
    return _span_rows(
        conn, GenAIAttributes.SPAN_TOOL_CALL, since, until, agent_id,
        "tool_name IS NOT NULL",
    )


def _llm_span_rows(conn, since, until, agent_id, _attr):
    return _span_rows(
        conn, GenAIAttributes.SPAN_LLM_CALL, since, until, agent_id,
        "model IS NOT NULL",
    )


def diagnostic_from_dict(payload: dict[str, Any]) -> ContextDiagnostic:
    """Reconstruct a :class:`ContextDiagnostic` from :func:`diagnostic_to_dict`.

    The inverse of :func:`diagnostic_to_dict`, used by the API-shim path: when
    ``tj serve`` holds the DuckDB write lock, ``tj context`` fetches the
    server-computed diagnostic over HTTP (the daemon owns the direct connection
    that can read the raw ``attributes`` column) and rebuilds the dataclass here
    to render it exactly as the direct-connection path would (#63).

    Only the fields the renderer reads are reconstructed with full fidelity;
    ``heaviest_turns`` (carried for ``--json`` consumers, never rendered) is
    rebuilt best-effort, with the fields the payload doesn't serialize
    (``cost_usd``) defaulted. Missing keys fall back to the dataclass defaults so
    a schema drift degrades gracefully rather than raising.
    """
    diag = ContextDiagnostic(
        since=_parse_dt(payload.get("since")),
        until=_parse_dt(payload.get("until")),
        sessions=int(payload.get("sessions", 0) or 0),
        turns=int(payload.get("turns", 0) or 0),
        total_reread_tokens=int(payload.get("total_reread_tokens", 0) or 0),
        total_new_input_tokens=int(payload.get("total_new_input_tokens", 0) or 0),
        total_output_tokens=int(payload.get("total_output_tokens", 0) or 0),
        total_cache_write_tokens=int(payload.get("total_cache_write_tokens", 0) or 0),
        total_cost_usd=float(payload.get("total_cost_usd", 0.0) or 0.0),
        tool_inputs_captured=bool(payload.get("tool_inputs_captured", False)),
        prompts_captured=bool(payload.get("prompts_captured", False)),
        tool_outputs_captured=bool(payload.get("tool_outputs_captured", False)),
        notes=list(payload.get("notes", []) or []),
        caveat=str(payload.get("caveat", CONTEXT_HONESTY_CAVEAT)),
    )
    diag.heaviest_turns = [
        TurnComposition(
            session_id=str(t.get("session_id", "unknown")),
            sub_agent_id=t.get("sub_agent_id"),
            model=str(t.get("model", "unknown")),
            reread_tokens=int(t.get("reread_tokens", 0) or 0),
            new_input_tokens=int(t.get("new_input_tokens", 0) or 0),
            output_tokens=int(t.get("output_tokens", 0) or 0),
            cache_write_tokens=int(t.get("cache_miss_tokens", 0) or 0),
            cost_usd=0.0,
        )
        for t in payload.get("heaviest_turns", []) or []
    ]
    diag.recurring = [
        RecurringInclusion(
            label=str(r.get("label", "")),
            tool_name=str(r.get("tool_name", "")),
            target=str(r.get("target", "")),
            sessions=int(r.get("sessions", 0) or 0),
            occurrences=int(r.get("occurrences", 0) or 0),
            fix=str(r.get("fix", "")),
            inclusion_type=str(r.get("inclusion_type", INCLUSION_FILE_READ)),
        )
        for r in payload.get("recurring", []) or []
    ]
    diag.compact_candidates = [
        CompactCandidate(
            session_id=str(c.get("session_id", "unknown")),
            reread_tokens=int(c.get("reread_tokens", 0) or 0),
            total_tokens=int(c.get("total_tokens", 0) or 0),
            reread_share=float(c.get("reread_share", 0.0) or 0.0),
            turns=int(c.get("turns", 0) or 0),
        )
        for c in payload.get("compact_candidates", []) or []
    ]
    return diag


def _parse_dt(raw: Any) -> datetime:
    """Parse an ISO datetime string back to a ``datetime`` (defaults to now)."""
    if isinstance(raw, datetime):
        return raw
    if isinstance(raw, str) and raw:
        try:
            return datetime.fromisoformat(raw)
        except ValueError:
            pass
    from tokenjam.utils.time_parse import utcnow
    return utcnow()


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
        # Named overhead source: prompt-cache MISS (cache-creation), #11.
        "total_cache_miss_tokens": diag.total_cache_miss_tokens,
        "total_work_tokens": diag.total_work_tokens,
        "total_tokens": diag.total_tokens,
        "reread_share": round(diag.reread_share, 4),
        "cache_miss_share": round(diag.cache_miss_share, 4),
        "total_cost_usd": round(diag.total_cost_usd, 6),
        "heaviest_turns": [
            {
                "session_id": t.session_id,
                "sub_agent_id": t.sub_agent_id,
                "model": t.model,
                "reread_tokens": t.reread_tokens,
                "new_input_tokens": t.new_input_tokens,
                "output_tokens": t.output_tokens,
                "cache_miss_tokens": t.cache_miss_tokens,
                "reread_share": round(t.reread_share, 4),
                "cache_miss_share": round(t.cache_miss_share, 4),
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
                "inclusion_type": r.inclusion_type,
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
        "prompts_captured": diag.prompts_captured,
        "tool_outputs_captured": diag.tool_outputs_captured,
        # Subagent accounting (#60). `subagent_turns` are already in the weighted
        # quota; `subagent_accounting_partial` marks the total a LOWER BOUND when
        # a delegating session's subagent turns are missing.
        "subagent_turns": diag.subagent_turns,
        "delegating_sessions": diag.delegating_sessions,
        "unaccounted_subagent_sessions": diag.unaccounted_subagent_sessions,
        "subagent_accounting_partial": diag.subagent_accounting_partial,
        # Parked #11 half — surfaced as an honest gap, not a fabricated number.
        "mcp_injection_parked": MCP_INJECTION_PARK_NOTE,
        "notes": diag.notes,
        "caveat": diag.caveat,
    }
