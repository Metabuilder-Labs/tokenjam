"""
Model-downgrade analyzer.

Flags sessions whose structural shape (short input, short output, few tool
calls) matches a class of work where a cheaper model in the same provider
family is worth reviewing. Never claims quality equivalence — only that the
*shape* matches a class worth a closer look.
"""
from __future__ import annotations

import re as _re
from datetime import datetime
from typing import Any

from tokenjam.core.context_diagnostic import (
    TurnComposition,
    load_turn_compositions,
)
from tokenjam.core.model_tiers import is_premium_tier
from tokenjam.core.optimize.registry import register
from tokenjam.core.optimize.stats import bootstrap_ci
from tokenjam.core.optimize.types import (
    AnalyzerContext,
    DowngradeExample,
    DowngradeFinding,
    OpusAuditExample,
    OpusQuotaAudit,
)
from tokenjam.core.pricing import get_rates

# Structural heuristic thresholds. Sessions are flagged only when ALL three
# hold; the analyzer never claims the cheaper model would have produced the
# same answer — it claims the *shape* matches a class of work worth reviewing.
SMALL_INPUT_TOKENS = 5_000
SMALL_OUTPUT_TOKENS = 500
SMALL_TOOL_CALLS = 5

# Per-TURN cheap-shape thresholds for the segment-level premium quota audit. A
# turn is a fraction of a session, so these sit well below the session-level
# constants above. Measured on NEW work (uncached input + output + this turn's
# tool fan-out), deliberately NOT on re-read/cache tokens: a mechanical turn can
# still re-read a huge context — that cache-read quota is exactly what a cheaper
# model would have burned more cheaply, so it belongs in the reclaimable metric,
# not the shape test. Starting calibration (non-blocking follow-up); left as
# named constants so a real-DB sweep can retune them in one place.
TURN_SMALL_INPUT = 2_000
TURN_SMALL_OUTPUT = 300
TURN_SMALL_TOOL_CALLS = 2

# A cheap segment is a maximal contiguous run of cheap-shaped turns. A run of at
# least this many is a flagged stretch; a shorter run is only flagged when it
# spans the session's ENTIRE set of turns (the whole-session floor — a session
# that was Sonnet-shaped end to end, which the old audit already flagged). This
# stops a lone mechanical turn wedged between two hard turns from counting.
MIN_STRETCH_TURNS = 2

# Cap on the number of spot-check example sessions carried in the audit.
OPUS_AUDIT_MAX_EXAMPLES = 5

# Premium → cheaper alternative in the same provider family. Pricing for both
# sides is resolved at runtime from pricing/models.toml; if either is missing
# the candidate is silently skipped (we won't invent a savings number).
# Premium-tier ladder: Fable → Sonnet and Opus → Haiku each drop two tiers, the
# aggressive step justified because the quota audit only proposes these for
# structurally tiny (Sonnet-shaped) sessions. Keep new premium families in sync
# with tokenjam.core.model_tiers.PREMIUM_TIERS so every flagged session has a
# real routing target.
DOWNGRADE_CANDIDATES: dict[str, dict[str, str]] = {
    "anthropic": {
        "claude-fable-5":    "claude-sonnet-4-6",
        "claude-opus-4-8":   "claude-haiku-4-5",
        "claude-opus-4-7":   "claude-haiku-4-5",
        "claude-opus-4-6":   "claude-haiku-4-5",
        "claude-sonnet-4-6": "claude-haiku-4-5",
        "claude-sonnet-4-5": "claude-haiku-4-5",
    },
    "openai": {
        "gpt-4o":      "gpt-4o-mini",
        "o3":          "o4-mini",
    },
    "google": {
        "gemini-2-5-pro": "gemini-2-5-flash",
    },
}


def lookup_downgrade(provider: str, model: str) -> str | None:
    """The cheaper same-family alternative for ``(provider, model)``, or ``None``.

    Tolerates trailing ``-YYYYMMDD`` date suffixes on the model id. Public (the
    premium quota audit and its routing export both need it); the underscore
    alias is kept so existing internal call sites don't churn.
    """
    mapping = DOWNGRADE_CANDIDATES.get(provider, {})
    if model in mapping:
        return mapping[model]
    m = _re.match(r"^(.*)-(\d{8})$", model)
    if m and m.group(1) in mapping:
        return mapping[m.group(1)]
    return None


# Backward-compatible private alias — internal call sites predate the promotion.
_lookup_downgrade = lookup_downgrade


def _alt_unit_cost(provider: str, original_model: str, alt_model: str,
                   input_tokens: int, output_tokens: int, cache_tokens: int) -> float | None:
    rates = get_rates(provider, alt_model)
    if rates is None:
        return None
    return (
        (input_tokens / 1_000_000) * rates.input_per_mtok
        + (output_tokens / 1_000_000) * rates.output_per_mtok
        + (cache_tokens / 1_000_000) * rates.cache_read_per_mtok
    )


def analyze_model_downgrade(
    conn,
    since: datetime,
    until: datetime,
    agent_id: str | None,
    window_days: float,
) -> DowngradeFinding | None:
    """
    Walk sessions in the window. For each:
      - aggregate input/output/cache tokens, tool count, cost, dominant model
      - if model is in DOWNGRADE_CANDIDATES and shape matches the heuristic,
        treat as a candidate and compute alternative cost
    """
    clauses = ["start_time >= $1", "start_time < $2", "session_id IS NOT NULL"]
    params: list[Any] = [since, until]
    if agent_id:
        clauses.append(f"agent_id = ${len(params) + 1}")
        params.append(agent_id)
    where = " AND ".join(clauses)

    # First pass: LLM spans grouped by session.
    llm_rows = conn.execute(
        f"SELECT session_id, "
        f"FIRST(trace_id) AS trace_id, "
        f"FIRST(agent_id) AS agent_id, "
        f"MIN(start_time) AS start_time, MAX(end_time) AS end_time, "
        f"MIN(provider) AS provider, "
        f"MODE(model) AS model, "
        f"COALESCE(SUM(input_tokens),0)  AS input_tokens, "
        f"COALESCE(SUM(output_tokens),0) AS output_tokens, "
        f"COALESCE(SUM(cache_tokens),0)  AS cache_tokens, "
        f"COALESCE(SUM(cost_usd),0.0)    AS cost_usd "
        f"FROM spans WHERE {where} AND model IS NOT NULL "
        f"GROUP BY session_id",
        params,
    ).fetchall()

    if not llm_rows:
        return None

    # Tool span counts per session (separate query — tool spans have model=NULL).
    tool_rows = conn.execute(
        f"SELECT session_id, COUNT(*) FROM spans "
        f"WHERE {where} AND tool_name IS NOT NULL "
        f"GROUP BY session_id",
        params,
    ).fetchall()
    tool_counts: dict[str, int] = {r[0]: int(r[1] or 0) for r in tool_rows if r[0]}

    total_sessions = len(llm_rows)
    candidate_sessions = 0
    actual_cost = 0.0
    alt_cost = 0.0
    candidate_tokens = 0
    window_total_tokens = 0
    examples: list[DowngradeExample] = []
    suggestions: dict[str, str] = {}
    swaps: list[tuple[str, str, str]] = []
    # Per-candidate-session window savings, for the sampling-confidence interval
    # (#308). One value per candidate session = (actual cost − cheaper-model cost).
    per_session_savings: list[float] = []

    for row in llm_rows:
        session_id, trace_id, _agent, start_time, end_time, provider, model, \
            in_tok, out_tok, cache_tok, cost = row
        # Accumulate window-wide token totals (used for subscription-mode
        # token-share rendering even when the row isn't a candidate).
        window_total_tokens += int(in_tok or 0) + int(out_tok or 0) + int(cache_tok or 0)
        if not provider or not model:
            continue
        alt = _lookup_downgrade(provider, model)
        if not alt:
            continue
        tool_calls = tool_counts.get(session_id, 0)
        if not (
            in_tok < SMALL_INPUT_TOKENS
            and out_tok < SMALL_OUTPUT_TOKENS
            and tool_calls <= SMALL_TOOL_CALLS
        ):
            continue

        alt_unit = _alt_unit_cost(provider, model, alt, int(in_tok), int(out_tok), int(cache_tok))
        if alt_unit is None:
            # No pricing data for the alternative — refuse to invent a savings number.
            continue

        candidate_sessions += 1
        actual_cost += float(cost or 0.0)
        alt_cost += alt_unit
        candidate_tokens += int(in_tok or 0) + int(out_tok or 0) + int(cache_tok or 0)
        # This session's recoverable saving (clamped at 0 — a cheaper model
        # never costs more in our candidate set, but guard against pricing noise).
        per_session_savings.append(max(float(cost or 0.0) - alt_unit, 0.0))
        suggestions[model] = alt
        if (provider, model, alt) not in swaps:
            swaps.append((provider, model, alt))

        if len(examples) < 3:
            duration = None
            try:
                if start_time and end_time:
                    duration = (end_time - start_time).total_seconds()
            except Exception:
                duration = None
            examples.append(DowngradeExample(
                trace_id=str(trace_id) if trace_id else "",
                session_id=str(session_id) if session_id else None,
                model=str(model),
                tool_calls=tool_calls,
                duration_seconds=duration,
                cost_usd=float(cost or 0.0),
            ))

    if candidate_sessions == 0:
        return None

    savings_window = max(actual_cost - alt_cost, 0.0)
    monthly_savings = (savings_window / window_days * 30.0) if window_days > 0 else 0.0
    percent = (candidate_sessions / total_sessions * 100.0) if total_sessions else 0.0

    # Sampling confidence on the monthly projection (#308). Resample the
    # candidate sessions with replacement and recompute the projected monthly
    # savings, so the interval widens when the estimate rests on few sessions.
    # Same `scale` as monthly_savings so the CI brackets that exact figure.
    monthly_scale = (30.0 / window_days) if window_days > 0 else 1.0
    ci = bootstrap_ci(per_session_savings, scale=monthly_scale)
    ci_low = round(ci[0], 2) if ci else None
    ci_high = round(ci[1], 2) if ci else None
    percent_tokens = (
        candidate_tokens / window_total_tokens * 100.0
        if window_total_tokens > 0 else 0.0
    )
    monthly_tokens_in_candidates = (
        int(candidate_tokens / window_days * 30.0) if window_days > 0 else 0
    )

    commands = [f"tjb run --original {p}:{orig} --candidate {p}:{alt}" for p, orig, alt in swaps]
    bench_command = "\n".join(commands) if commands else None

    return DowngradeFinding(
        candidate_sessions=candidate_sessions,
        total_sessions=total_sessions,
        actual_cost_usd=round(actual_cost, 6),
        alternative_cost_usd=round(alt_cost, 6),
        monthly_savings_usd=round(monthly_savings, 2),
        percent_of_sessions=round(percent, 1),
        examples=examples,
        suggestions=suggestions,
        bench_command=bench_command,
        candidate_tokens=candidate_tokens,
        window_total_tokens=window_total_tokens,
        percent_of_tokens=round(percent_tokens, 1),
        monthly_tokens_in_candidates=monthly_tokens_in_candidates,
        # Recoverable-savings contract (#111). Use the WINDOW savings (not the
        # 30-day projection) so every analyzer's estimated_recoverable_usd shares
        # one time basis — "recoverable over the analyzed window" — and the
        # Overview tiles are directly comparable (#122). monthly_savings_usd
        # remains for the CLI's own projected-savings line.
        estimated_recoverable_usd=round(savings_window, 6),
        estimated_recoverable_tokens=candidate_tokens,
        estimate_basis=(
            "candidate sessions routed to a cheaper model over the window — "
            "structural fit only, no quality validation"
        ),
        n_sessions=candidate_sessions,
        ci_low=ci_low,
        ci_high=ci_high,
    )


@register("downsize")
def run(ctx: AnalyzerContext) -> None:
    """Registry entry point. Mutates ctx.report.downgrade."""
    ctx.report.downgrade = analyze_model_downgrade(
        ctx.conn, ctx.since, ctx.until, ctx.agent_id, ctx.window_days,
    )


# ---------------------------------------------------------------------------
# Premium-tier quota audit (retroactive Downsize, reframed as accountability)
# ---------------------------------------------------------------------------
# This reframes the structural downsize heuristic as an accountability "quota
# audit" scoped to premium-tier work (Fable + Opus, via the shared model_tiers
# predicate; issue #5, research/evidence/feature-downsize.md). The headline is
# "% of your premium (Opus/Fable) quota that went to Sonnet-shaped work" — a
# retrospective behaviour mirror in premium token share, not dollars and not
# "reclaimable" (the tokens are already spent) — because the subscription
# majority is on flat-rate plans where dollar framing mis-targets them
# (subscription-vs-cost-framing.md). It complements opusplan / `/model`
# (forward-looking) by answering the backward-looking question those tools
# can't: "which stretches of my PAST premium work were Sonnet-shaped?" Honest
# framing throughout — candidates to spot-check, never "safe to downgrade".
#
# Grain: a PER-TURN walk over the `TurnComposition` rows `tj context` already
# produces, NOT the old whole-session `MODE(model)` + SUM aggregate. Two
# structural fixes fall out:
#   (a) a mechanical STRETCH inside an otherwise-hard session is now detectable
#       (the whole-session heuristic structurally missed it — the hard part blew
#       the sums past the thresholds); and
#   (b) each turn's tokens count under ITS OWN model, so a Sonnet turn inside an
#       Opus session no longer inflates the premium numerator or denominator.
#
# The `audit_opus_quota` name and the `opus_*` fields on OpusQuotaAudit are kept
# for API/serialization stability; they now denote the whole premium tier, not
# Opus alone (the audit counts and flags Fable turns too), and are quota-weighted
# (the #119 weighting) rather than raw token sums.


def _is_cheap_shaped(turn: TurnComposition) -> bool:
    """True when a turn's *reasoning* footprint is Sonnet-shaped (design §2.1).

    Measured on NEW work — uncached input, output, this turn's tool fan-out — and
    on the absence of a Task delegation. Deliberately NOT gated on re-read /
    cache-write tokens: a mechanical turn can still re-read a huge context, and
    that cache-read quota is exactly what a cheaper model would have burned more
    cheaply — it belongs in the reclaimable metric, not the shape test.
    """
    return (
        turn.new_input_tokens < TURN_SMALL_INPUT
        and turn.output_tokens < TURN_SMALL_OUTPUT
        and turn.tool_fanout <= TURN_SMALL_TOOL_CALLS
        and not turn.delegates
    )


def _cheap_segments(
    session_turns: list[TurnComposition],
) -> list[list[TurnComposition]]:
    """Maximal contiguous runs of cheap-shaped turns (design §2.2).

    A non-cheap turn (or a delegation, which makes a turn non-cheap) breaks the
    run. Turns must already be ordered by ``start_time``.
    """
    segments: list[list[TurnComposition]] = []
    run: list[TurnComposition] = []
    for turn in session_turns:
        if _is_cheap_shaped(turn):
            run.append(turn)
        elif run:
            segments.append(run)
            run = []
    if run:
        segments.append(run)
    return segments


def _segment_counts(segment: list[TurnComposition], session_turn_count: int) -> bool:
    """Whether a cheap segment is flagged (design §2.2 + the whole-session floor).

    A stretch of at least ``MIN_STRETCH_TURNS`` counts; a shorter run counts only
    when it spans the session's ENTIRE turn set (a session that was Sonnet-shaped
    end to end — the whole-session floor the old audit already flagged, and the
    reason a single-turn cheap session is still counted). This is what keeps a
    lone mechanical turn wedged between two hard turns from being flagged.
    """
    return (
        len(segment) >= MIN_STRETCH_TURNS
        or len(segment) == session_turn_count
    )


class _SessionAgg:
    """Per-session aggregate of the flagged cheap-segment PREMIUM turns, for the
    spot-check example list (largest-quota first)."""

    __slots__ = ("session_id", "quota_by_model", "quota", "new_input",
                 "output", "reread", "tool_calls", "cost")

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        # (provider, model) -> flagged premium quota, so the example reports the
        # model that actually carried the most misallocated quota in this session
        # rather than freezing on whichever premium turn happened to appear first
        # (a mixed-model session could otherwise mislabel the routing suggestion).
        self.quota_by_model: dict[tuple[str | None, str], float] = {}
        self.quota = 0.0
        self.new_input = 0
        self.output = 0
        self.reread = 0
        self.tool_calls = 0
        self.cost = 0.0

    def dominant_model(self) -> tuple[str | None, str]:
        """The (provider, model) carrying the most flagged premium quota in this
        session — the honest label for the spot-check example + routing hint."""
        provider, model = max(
            self.quota_by_model.items(), key=lambda kv: kv[1],
        )[0]
        return provider, model


def audit_opus_quota(
    conn,
    since: datetime,
    until: datetime,
    agent_id: str | None,
    window_days: float,
) -> OpusQuotaAudit:
    """Retroactive segment-level premium-tier quota audit (issue #5).

    Walks the window's assistant turns per session in ``start_time`` order and
    reports ONE honest figure (founder decision D1): the share of premium
    (Opus/Fable) quota that went to Sonnet-shaped *work* — whole Sonnet-shaped
    sessions PLUS mechanical stretches inside otherwise-hard sessions — on
    exact per-turn model attribution (D2). Quota-weighted (the #119 weighting),
    never a dollar headline; the `opus_*` field names are kept for serialization
    stability but denote the whole premium tier.

    The figure is a labelled ESTIMATE (D3): `segment_estimate_confidence` +
    `estimate_basis` mark it heuristic, and a WIDE bootstrap CI over resampled
    SEGMENTS brackets it (`segment_ci_low/high`) so the band widens honestly when
    few segments carry it.

    Computes purely from already-backfilled token/model metadata — no captured
    content (#3). Always returns an :class:`OpusQuotaAudit` (never ``None``) so
    the renderer can show an honest empty state.
    """
    turns = load_turn_compositions(
        conn, since, until, agent_id,
        ordered=True, with_tool_activity=True,
    )
    audit = OpusQuotaAudit(window_days=window_days)
    if not turns:
        return audit

    # Group into sessions, preserving the (already-applied) start_time ordering.
    by_session: dict[str, list[TurnComposition]] = {}
    for turn in turns:
        by_session.setdefault(turn.session_id, []).append(turn)

    premium_quota = 0.0            # denominator: quota-weighted premium turns
    misallocated_quota = 0.0       # numerator: premium turns in flagged segments
    segment_values: list[float] = []   # one premium-quota value per flagged segment
    suggestions: dict[str, str] = {}
    aggs: dict[str, _SessionAgg] = {}
    actual_cost = 0.0
    alt_cost = 0.0

    for session_id, session_turns in by_session.items():
        premium_turns = [t for t in session_turns if is_premium_tier(t.model)]
        if not premium_turns:
            continue
        audit.opus_sessions += 1
        premium_quota += sum(t.quota_weighted_tokens for t in premium_turns)

        session_flagged = False
        for segment in _cheap_segments(session_turns):
            if not _segment_counts(segment, len(session_turns)):
                continue
            seg_premium = [t for t in segment if is_premium_tier(t.model)]
            if not seg_premium:
                continue
            seg_quota = sum(t.quota_weighted_tokens for t in seg_premium)
            misallocated_quota += seg_quota
            segment_values.append(seg_quota)
            audit.segment_count += 1
            session_flagged = True

            agg = aggs.setdefault(session_id, _SessionAgg(session_id))
            for t in seg_premium:
                agg.quota += t.quota_weighted_tokens
                agg.new_input += t.new_input_tokens
                agg.output += t.output_tokens
                agg.reread += t.reread_tokens
                agg.tool_calls += t.tool_fanout
                agg.cost += t.cost_usd
                key = (t.provider, t.model)
                agg.quota_by_model[key] = (
                    agg.quota_by_model.get(key, 0.0) + t.quota_weighted_tokens
                )
                alt = (
                    lookup_downgrade(t.provider, t.model)
                    if t.provider else None
                )
                if alt:
                    suggestions[t.model] = alt

        if session_flagged:
            audit.candidate_sessions += 1

    # Secondary API-only implied-dollar counterfactual, aggregated over the
    # flagged premium turns and priced at each session's dominant premium model's
    # cheaper alternative (never the headline).
    for agg in aggs.values():
        provider, model = agg.dominant_model()
        alt = lookup_downgrade(provider, model) if provider else None
        if not alt:
            continue
        alt_unit = _alt_unit_cost(
            provider, model, alt, agg.new_input, agg.output, agg.reread,
        )
        if alt_unit is not None:
            actual_cost += agg.cost
            alt_cost += alt_unit

    audit.opus_tokens = int(round(premium_quota))
    audit.candidate_tokens = int(round(misallocated_quota))
    audit.suggestions = suggestions
    audit.percent_quota_misallocated = (
        round(100.0 * misallocated_quota / premium_quota, 1)
        if premium_quota > 0 else 0.0
    )
    audit.percent_sessions = (
        round(100.0 * audit.candidate_sessions / audit.opus_sessions, 1)
        if audit.opus_sessions > 0 else 0.0
    )
    audit.actual_cost_usd = round(actual_cost, 6)
    audit.alternative_cost_usd = round(alt_cost, 6)

    # Confidence interval on the HEADLINE PERCENT (design §6, founder D3). Each
    # flagged segment contributes its premium-quota value; scaling the resampled
    # SUM by 100/premium_quota turns the bootstrap into an interval on the share.
    # Resampling segments (not sessions) widens the band honestly when the
    # estimate rests on few stretches; below 2 segments there is no spread to
    # bracket, so the bounds stay None (the estimate is inherently wide).
    if premium_quota > 0 and len(segment_values) >= 2:
        ci = bootstrap_ci(segment_values, scale=100.0 / premium_quota)
        if ci is not None:
            audit.segment_ci_low = round(ci[0], 1)
            audit.segment_ci_high = round(ci[1], 1)

    # Largest-quota candidate sessions first — the most worthwhile spot-checks.
    ordered_aggs = sorted(aggs.values(), key=lambda a: a.quota, reverse=True)
    audit.examples = [
        _example_for(agg) for agg in ordered_aggs[:OPUS_AUDIT_MAX_EXAMPLES]
    ]
    return audit


def _example_for(agg: _SessionAgg) -> OpusAuditExample:
    """Build a spot-check example labelled with the session's DOMINANT premium
    model (and its cheaper alternative), not whichever premium turn came first."""
    provider, model = agg.dominant_model()
    alt = (lookup_downgrade(provider, model) if provider else None) or ""
    return OpusAuditExample(
        trace_id="",
        session_id=agg.session_id,
        model=model,
        alt_model=alt,
        input_tokens=agg.new_input,
        output_tokens=agg.output,
        cache_tokens=agg.reread,
        tool_calls=agg.tool_calls,
        duration_seconds=None,
        cost_usd=round(agg.cost, 6),
    )
