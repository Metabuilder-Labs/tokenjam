"""
Reuse analyzer.

Detects clusters of sessions that share a **planning skeleton**: the structural
shape of the model's first planning output. Agents re-plan the same work
constantly — a "patch release" run thirty times pays for thirty plans whose
tool sequences and structure are identical, only the version string and date
varying. Reuse names that waste and quantifies what reusing the skeleton would
recover.

Planning portion of a session (locked decision, issue #115):
  The first LLM call in the session that precedes any tool call. Mechanically:
    - Order spans within a session by start_time.
    - The most recent LLM span before the first tool span is the planning call.
    - If the session has no tool calls, the first LLM span is the planning call.
    - If the session has no LLM spans, the session is skipped.

  Note: this codebase has no `operation_name` column. An LLM span is one with
  `model IS NOT NULL` (and no tool_name); a tool span is one with
  `tool_name IS NOT NULL`. This mirrors how summarize_window and the script
  analyzer already distinguish the two.

Tiered detection:
  Mode 1 (always) — tool-sequence signature: the ordered tuple of tool names
    following the planning call. Runs against any telemetry, including raw
    Claude Code JSONL backfills with no capture toggles.
  Mode 2 (capture.prompts = true) — also hash a variable-stripped prefix of the
    planning prompt. The cluster key becomes (tool_signature, prompt_prefix_hash),
    so clusters are narrower and more accurate.

Honesty discipline (CLAUDE.md Rule 14): structural detection only. Every
user-visible string says "skeleton match" / "candidate" / "recoverable" /
"review before reusing" — never "identical plan" or "saves you".
"""
from __future__ import annotations

import hashlib
import json
import re
from typing import Any, NamedTuple

from tokenjam.core.optimize.clustering import group_by_key, mask_variables, recurring
from tokenjam.core.optimize.registry import register
from tokenjam.core.optimize.types import AnalyzerContext, ReuseCluster, ReuseFinding
from tokenjam.otel.semconv import GenAIAttributes
from tokenjam.core.persona_scope import add_persona_clause

# Cluster thresholds (issue #115 AC6 / savings contract). Module-level for
# explicit visibility and easy tuning. A cluster is surfaced only when it
# clears all three.
MIN_REPETITIONS = 3            # at least this many sessions share the skeleton
MIN_PLANNING_TOKENS = 200      # tiny "ok, let me think" outputs aren't a "plan"
MIN_RECOVERABLE_USD = 0.01     # already-cheap planning isn't worth surfacing

# How much of the planning prompt to hash for the prefix signature.
PREFIX_LEN = 200

# Hint surfaced in Mode 1 (capture.prompts off) to nudge the richer mode.
_MODE1_HINT = (
    "Clustering ran on tool-sequence signatures only. Set [capture] prompts = "
    "true in tj.toml to also match planning-prompt prefixes for narrower, more "
    "accurate clusters."
)

# Hint for the case the config toggle alone can never describe: capture is ON
# and the analyzed window still carries no prompt text, so clustering silently
# ran in the weaker mode. The usual cause is that the sessions were ingested
# live before the live path carried prompt content, or that Claude Code was run
# without OTEL_LOG_USER_PROMPTS=1 (see api/routes/logs.py).
_CAPTURE_ON_NO_CONTENT_HINT = (
    "[capture] prompts is on, but no planning call in this window carried "
    "prompt text. So clustering ran on tool-sequence signatures only. "
    "Sessions ingested live carry prompt text only when Claude Code ran with "
    "OTEL_LOG_USER_PROMPTS=1. `tj backfill claude-code` recovers it from the "
    "transcripts on disk."
)

# The partial case: prompt text was there for some planning calls and not
# others, so the window's clusters were not all built the same way. Naming the
# split is the point — a reader comparing two runs otherwise cannot tell a
# fully content-matched result from one that was half tool-signature guesswork.
_MIXED_CAPTURE_HINT = (
    "Only some planning calls in this window carried prompt text; the rest "
    "were clustered on tool-sequence signatures alone. So these clusters do "
    "not all rest on the same evidence. Sessions ingested live carry prompt "
    "text only when Claude Code ran with OTEL_LOG_USER_PROMPTS=1. "
    "`tj backfill claude-code` recovers it from the transcripts on disk."
)


class _SpanRow(NamedTuple):
    session_id: str
    start_time: Any
    model: str | None
    tool_name: str | None
    input_tokens: int | None
    output_tokens: int | None
    cache_tokens: int | None
    cache_write_tokens: int | None
    cost_usd: float | None
    attributes: Any


class _SessionPlan(NamedTuple):
    session_id: str
    planning_tokens: int
    planning_cost_usd: float
    # DuckDB returns the TIMESTAMPTZ column as a tz-aware datetime, so this is
    # directly comparable — used to order cluster examples by recency.
    plan_start_time: Any
    tool_signature: tuple[str, ...]
    prompt_prefix_hash: str | None


# Variable-stripping for the prompt-prefix hash. Order matters: posix paths and
# ISO dates are matched before bare digit runs so their internal digits don't
# get partially replaced first. The goal: "release v0.3.4 on 2026-06-15" and
# "release v0.3.5 on 2026-06-17" normalize identically.
#
# _PATH_RE is intentionally greedy and will also swallow the tail of a URL
# (e.g. "https://host/path" → "https:<PATH>"). That's acceptable here: both
# sides of a comparison normalize the same way, and this only feeds a
# 200-char prefix hash used for coarse clustering — not anything reversible.
_PATH_RE = re.compile(r"(?:~|\.{0,2})?/[\w.\-/]+")
_DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")
_NUM_RE = re.compile(r"\d+")

#: Ordered substitutions for the prompt-prefix normalizer (path before date
#: before bare digits, so internal digits aren't partially replaced first).
_STRIP_SUBS = [(_PATH_RE, "<PATH>"), (_DATE_RE, "<DATE>"), (_NUM_RE, "<NUM>")]


def _strip_variables(text: str) -> str:
    """Replace path-, date-, and number-looking spans with fixed placeholders."""
    return mask_variables(text, _STRIP_SUBS)


def _basis_with_coverage(basis: str, with_prompt: int, total: int) -> str:
    """Append the MEASURED prompt-capture coverage to the basis string.

    A finding that degraded to a weaker signal has to say so where its figure
    is explained, not only in a hint a renderer may drop: clustering on tool
    sequences alone is coarser than clustering on tool sequences plus prompt
    prefixes, and a reader comparing two runs cannot otherwise tell which one
    they are looking at. Recomputed per window, never a stored constant.
    """
    if not total:
        return basis
    if with_prompt == total:
        detail = "every planning call in the window carried prompt text"
    elif with_prompt:
        detail = (
            f"{with_prompt} of {total} planning calls carried prompt text; the "
            f"rest clustered on tool sequence alone"
        )
    else:
        detail = (
            "no planning call in the window carried prompt text, so clustering "
            "ran on tool sequence alone"
        )
    return f"{basis} Capture coverage: {detail}."


def _is_llm(row: _SpanRow) -> bool:
    return row.model is not None and row.tool_name is None


def _identify_planning_call(rows: list[_SpanRow]) -> _SpanRow | None:
    """Return the planning LLM span for a session, or None if there's no plan."""
    first_tool_idx = next(
        (i for i, r in enumerate(rows) if r.tool_name is not None),
        None,
    )
    llm_indices = [i for i, r in enumerate(rows) if _is_llm(r)]
    if not llm_indices:
        return None
    if first_tool_idx is None:
        # No tool calls in the session: the first LLM span is the plan.
        return rows[llm_indices[0]]
    # Otherwise: the most recent LLM span before the first tool call.
    before = [i for i in llm_indices if i < first_tool_idx]
    return rows[before[-1]] if before else None


def _tool_signature(rows: list[_SpanRow], plan_idx: int) -> tuple[str, ...]:
    """Ordered tuple of tool names following the planning call."""
    return tuple(
        r.tool_name for r in rows[plan_idx + 1:] if r.tool_name is not None
    )


def _planning_prompt(plan_row: _SpanRow) -> str | None:
    """Extract the planning span's prompt text from its attributes JSON."""
    attrs = plan_row.attributes
    if isinstance(attrs, str):
        try:
            attrs = json.loads(attrs)
        except Exception:
            return None
    if not isinstance(attrs, dict):
        return None
    prompt = attrs.get(GenAIAttributes.PROMPT_CONTENT)
    if prompt is None or prompt == "":
        return None
    if not isinstance(prompt, str):
        prompt = json.dumps(prompt, sort_keys=True)
    return prompt


def _prompt_prefix_hash(plan_row: _SpanRow) -> str | None:
    """SHA-256 (first 16 hex) of the variable-stripped planning-prompt prefix."""
    prompt = _planning_prompt(plan_row)
    if not prompt:
        return None
    normalized = _strip_variables(prompt[:PREFIX_LEN])
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


def _cluster_key(sig: tuple[str, ...], prefix_hash: str | None) -> str:
    """Deterministic cluster id from the tool signature and optional prefix."""
    sig_hash = hashlib.sha256(repr(sig).encode("utf-8")).hexdigest()[:12]
    return f"{sig_hash}-{prefix_hash}" if prefix_hash else sig_hash


def _planning_tokens(row: _SpanRow) -> int:
    return (
        (row.input_tokens or 0)
        + (row.output_tokens or 0)
        + (row.cache_tokens or 0)
        + (row.cache_write_tokens or 0)
    )


def _cluster_sessions(plans: list[_SessionPlan]) -> dict[str, list[_SessionPlan]]:
    """
    Single clustering entry point (architecture §forward-compat): group session
    plans by their deterministic cluster key. Swapping in semantic clustering
    later is a one-function change.
    """
    return group_by_key(
        plans, lambda plan: _cluster_key(plan.tool_signature, plan.prompt_prefix_hash),
    )


@register("reuse")
def run(ctx: AnalyzerContext) -> None:
    """Registry entry point. Attaches a ReuseFinding to ctx.report.findings."""
    capture = getattr(ctx.config, "capture", None)
    prompts_captured = bool(capture and getattr(capture, "prompts", False))
    optimize_cfg = getattr(ctx.config, "optimize", None)
    min_repetitions = getattr(
        optimize_cfg, "min_reuse_repetitions", MIN_REPETITIONS,
    )
    # NOTE: `capture_mode` is NOT decided here. The config toggle says what the
    # user asked for; the finding has to say what clustering actually ran on,
    # and those diverged silently for as long as this field echoed the config —
    # a report declared "with_prompt_prefix" while every cluster member's
    # `prompt_prefix_hash` was None. It is derived below, after the planning
    # calls have been read, from how many of them carried prompt text.

    # Single windowed query; all per-session walking happens in Python. No N+1.
    # Values are bound via DuckDB positional placeholders ($1, $2, ...). The
    # f-string only interpolates the placeholder *index* (len(params)+1), never
    # user data, so this stays parameterized (CLAUDE.md Rule 7) — same pattern
    # as runner.summarize_window.
    clauses = ["start_time >= $1", "start_time < $2", "session_id IS NOT NULL"]
    params: list[Any] = [ctx.since, ctx.until]
    if ctx.agent_id:
        clauses.append(f"agent_id = ${len(params) + 1}")
        params.append(ctx.agent_id)
    # The persona POPULATION scope. Without it this analyzer's dollar figure is
    # computed over the whole mixed corpus and then published under whichever
    # persona the reader picked. See `core/persona_scope.py`.
    add_persona_clause(clauses, ctx.persona_scope)
    where = " AND ".join(clauses)
    rows = ctx.conn.execute(
        f"SELECT session_id, start_time, model, tool_name, "
        f"input_tokens, output_tokens, cache_tokens, cache_write_tokens, "
        f"cost_usd, attributes "
        f"FROM spans WHERE {where} "
        f"ORDER BY session_id, start_time",
        params,
    ).fetchall()

    finding = ReuseFinding(
        capture_mode="tool_sequence_only",
        hint="" if prompts_captured else _MODE1_HINT,
        min_repetitions=min_repetitions,
    )

    if not rows:
        ctx.report.findings["reuse"] = finding
        return

    # Group rows per session, preserving start_time order from the query.
    per_session: dict[str, list[_SpanRow]] = {}
    for r in rows:
        row = _SpanRow(*r)
        per_session.setdefault(str(row.session_id), []).append(row)

    # Resolve each session's planning call and build its skeleton signature.
    plans: list[_SessionPlan] = []
    for session_id, session_rows in per_session.items():
        plan_row = _identify_planning_call(session_rows)
        if plan_row is None:
            continue
        plan_idx = session_rows.index(plan_row)
        plans.append(_SessionPlan(
            session_id=session_id,
            planning_tokens=_planning_tokens(plan_row),
            planning_cost_usd=float(plan_row.cost_usd or 0.0),
            plan_start_time=plan_row.start_time,
            tool_signature=_tool_signature(session_rows, plan_idx),
            prompt_prefix_hash=(
                _prompt_prefix_hash(plan_row) if prompts_captured else None
            ),
        ))

    # Measured, not configured: the share of planning calls that actually
    # carried prompt text, and the mode clustering therefore really ran in.
    # A window can be entirely live-ingested, or predate the live path
    # carrying prompt content, and produce zero coverage with the toggle on.
    with_prompt = sum(1 for p in plans if p.prompt_prefix_hash is not None)
    finding.prompt_capture_coverage = (
        round(with_prompt / len(plans), 4) if plans else None
    )
    if with_prompt and with_prompt == len(plans):
        finding.capture_mode = "with_prompt_prefix"
    elif with_prompt:
        # Partly degraded. The prompt-free calls in this window were clustered
        # on tool signature alone, so the window's clusters do not share one
        # basis and no surface may present them as content-matched. Claiming
        # the full mode here contradicted the basis string built two lines
        # down, which has always said "the rest clustered on tool sequence
        # alone".
        finding.capture_mode = "mixed_prompt_prefix"
        finding.hint = _MIXED_CAPTURE_HINT
    elif prompts_captured and plans:
        # Asked for, not delivered — the case the toggle alone cannot describe.
        finding.hint = _CAPTURE_ON_NO_CONTENT_HINT
    finding.estimate_basis = _basis_with_coverage(
        finding.estimate_basis, with_prompt, len(plans),
    )

    clusters_raw = _cluster_sessions(plans)

    surfaced: list[ReuseCluster] = []
    # Recurrence gate first (the shared threshold filter); the remaining
    # per-cluster gates (avg tokens, recoverable floor) stay analyzer-specific.
    for cluster_id, members in recurring(clusters_raw, min_members=min_repetitions).items():
        reps = len(members)

        avg_tokens = round(sum(m.planning_tokens for m in members) / reps)
        if avg_tokens < MIN_PLANNING_TOKENS:
            continue

        avg_cost = sum(m.planning_cost_usd for m in members) / reps
        cache_reuse_usd = round(avg_cost * (reps - 1), 6)
        if cache_reuse_usd < MIN_RECOVERABLE_USD:
            continue
        script_usd = round(avg_cost * reps, 6)

        # Recency-ordered examples (most recent planning call first).
        by_recency = sorted(members, key=lambda m: m.plan_start_time, reverse=True)
        example_ids = [m.session_id for m in by_recency[:3]]

        surfaced.append(ReuseCluster(
            cluster_id=cluster_id,
            tool_signature=members[0].tool_signature,
            prompt_prefix_hash=members[0].prompt_prefix_hash,
            repetitions=reps,
            avg_planning_tokens=avg_tokens,
            avg_planning_cost_usd=round(avg_cost, 6),
            cache_reuse_recoverable_usd=cache_reuse_usd,
            script_replacement_recoverable_usd=script_usd,
            cache_reuse_recoverable_tokens=avg_tokens * (reps - 1),
            script_replacement_recoverable_tokens=avg_tokens * reps,
            example_session_ids=example_ids,
            skeleton_session_id=by_recency[0].session_id,
            member_session_ids=tuple(m.session_id for m in members),
        ))

    # Cluster listing order (independent of the finding-level headline basis
    # below): rank by the conservative cache-reuse amount, descending.
    surfaced.sort(key=lambda c: c.cache_reuse_recoverable_usd, reverse=True)

    finding.clusters = surfaced
    if surfaced:
        # Headline uses the CONSERVATIVE cache-reuse premise (a template only
        # removes the re-plan delta: avg cost x (reps - 1)), not the
        # script-replacement upper bound (avg cost x reps). The first planning
        # call in a cluster had to happen — there was nothing to reuse yet —
        # so the headline may not charge for it. This matters more now that
        # every surface states these figures in the PAST tense ("this is what
        # you already overspent"): "you wasted $X" where X includes necessary
        # work is a claim a user can disprove on inspection, and on a
        # 2-repetition cluster it is a 2x overclaim. The upper bound stays
        # available per-cluster as script_replacement_recoverable_usd/_tokens
        # for the template-authoring case. See REUSE_ESTIMATE_BASIS.
        finding.past_overspend_usd = round(
            sum(c.cache_reuse_recoverable_usd for c in surfaced), 6
        )
        finding.past_overspend_tokens = sum(
            c.cache_reuse_recoverable_tokens for c in surfaced
        )

    ctx.report.findings["reuse"] = finding
