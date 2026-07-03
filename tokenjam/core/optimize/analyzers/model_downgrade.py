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

# Substring that identifies an Opus-family model name (after provider/date
# normalisation). The quota audit scopes exclusively to Opus sessions — the
# acute quota-burn class (research/evidence/feature-downsize.md). Matching on
# the family substring tolerates version + date suffixes
# (claude-opus-4-7, claude-opus-4-6-20260115, ...).
OPUS_MODEL_SUBSTR = "opus"

# Cap on the number of spot-check example sessions carried in the audit.
OPUS_AUDIT_MAX_EXAMPLES = 5

# Premium → cheaper alternative in the same provider family. Pricing for both
# sides is resolved at runtime from pricing/models.toml; if either is missing
# the candidate is silently skipped (we won't invent a savings number).
DOWNGRADE_CANDIDATES: dict[str, dict[str, str]] = {
    "anthropic": {
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


def _lookup_downgrade(provider: str, model: str) -> str | None:
    """DOWNGRADE_CANDIDATES lookup that tolerates trailing YYYYMMDD suffixes."""
    mapping = DOWNGRADE_CANDIDATES.get(provider, {})
    if model in mapping:
        return mapping[model]
    m = _re.match(r"^(.*)-(\d{8})$", model)
    if m and m.group(1) in mapping:
        return mapping[m.group(1)]
    return None


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
# Opus quota audit (retroactive Downsize, reframed as accountability)
# ---------------------------------------------------------------------------
# This runs the same structural heuristic as `downsize`, but scoped to Opus
# sessions only and reframed as a "quota audit" rather than "cost optimization"
# (issue #5; research/evidence/feature-downsize.md). The headline is
# "% of your Opus quota reclaimable from Sonnet-shaped sessions" — Opus token
# share, not dollars — because the subscription majority is on flat-rate plans
# where dollar framing mis-targets them (subscription-vs-cost-framing.md). It
# complements opusplan / `/model` (forward-looking) by answering the
# backward-looking question those tools can't: "which of my PAST Opus sessions
# were Sonnet-shaped?" Honest framing throughout — candidates to spot-check,
# never "safe to downgrade".


def _is_opus(provider: str, model: str) -> bool:
    """True when the model is an Opus-family model worth auditing for downsize.

    We only audit Opus sessions that ALSO have a known cheaper alternative in
    `DOWNGRADE_CANDIDATES` (so the routing suggestion is real, not invented).
    """
    if OPUS_MODEL_SUBSTR not in (model or "").lower():
        return False
    return _lookup_downgrade(provider, model) is not None


def audit_opus_quota(
    conn,
    since: datetime,
    until: datetime,
    agent_id: str | None,
    window_days: float,
) -> OpusQuotaAudit:
    """Retroactive Opus quota audit over Claude Code (and other) sessions.

    Walks Opus sessions in the window and flags those whose structural shape
    (small input, small output, few tool calls) matches a class of work where a
    cheaper same-family model is worth a spot-check. Quantifies the result as a
    **share of Opus quota** (candidate Opus tokens / total Opus tokens) — never
    a dollar figure as the headline.

    Computes purely from already-backfilled token/model metadata; does NOT
    depend on captured content (#3). Always returns an :class:`OpusQuotaAudit`
    (never ``None``) so the renderer can show an honest empty state.
    """
    clauses = ["start_time >= $1", "start_time < $2", "session_id IS NOT NULL"]
    params: list[Any] = [since, until]
    if agent_id:
        clauses.append(f"agent_id = ${len(params) + 1}")
        params.append(agent_id)
    where = " AND ".join(clauses)

    # LLM spans grouped by session (one row per session, dominant model).
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

    audit = OpusQuotaAudit(window_days=window_days)
    if not llm_rows:
        return audit

    # Tool span counts per session (tool spans have model=NULL).
    tool_rows = conn.execute(
        f"SELECT session_id, COUNT(*) FROM spans "
        f"WHERE {where} AND tool_name IS NOT NULL "
        f"GROUP BY session_id",
        params,
    ).fetchall()
    tool_counts: dict[str, int] = {r[0]: int(r[1] or 0) for r in tool_rows if r[0]}

    candidate_examples: list[tuple[int, OpusAuditExample]] = []
    suggestions: dict[str, str] = {}

    for row in llm_rows:
        session_id, trace_id, _agent, start_time, end_time, provider, model, \
            in_tok, out_tok, cache_tok, cost = row
        if not provider or not model:
            continue
        if not _is_opus(provider, model):
            continue

        session_tokens = int(in_tok or 0) + int(out_tok or 0) + int(cache_tok or 0)
        audit.opus_sessions += 1
        audit.opus_tokens += session_tokens

        alt = _lookup_downgrade(provider, model)
        tool_calls = tool_counts.get(session_id, 0)
        is_candidate = (
            alt is not None
            and in_tok < SMALL_INPUT_TOKENS
            and out_tok < SMALL_OUTPUT_TOKENS
            and tool_calls <= SMALL_TOOL_CALLS
        )
        if not is_candidate:
            continue

        audit.candidate_sessions += 1
        audit.candidate_tokens += session_tokens
        if alt:
            suggestions[str(model)] = alt
            # Best-effort implied dollar value (secondary signal for API users).
            alt_unit = _alt_unit_cost(
                provider, str(model), alt,
                int(in_tok or 0), int(out_tok or 0), int(cache_tok or 0),
            )
            if alt_unit is not None:
                audit.actual_cost_usd += float(cost or 0.0)
                audit.alternative_cost_usd += alt_unit

        duration = None
        try:
            if start_time and end_time:
                duration = (end_time - start_time).total_seconds()
        except Exception:  # noqa: BLE001
            duration = None
        candidate_examples.append((
            session_tokens,
            OpusAuditExample(
                trace_id=str(trace_id) if trace_id else "",
                session_id=str(session_id) if session_id else None,
                model=str(model),
                alt_model=str(alt) if alt else "",
                input_tokens=int(in_tok or 0),
                output_tokens=int(out_tok or 0),
                cache_tokens=int(cache_tok or 0),
                tool_calls=tool_calls,
                duration_seconds=duration,
                cost_usd=float(cost or 0.0),
            ),
        ))

    audit.suggestions = suggestions
    # Largest-quota candidates first — those are the most worthwhile spot-checks.
    candidate_examples.sort(key=lambda pair: pair[0], reverse=True)
    audit.examples = [ex for _tokens, ex in candidate_examples[:OPUS_AUDIT_MAX_EXAMPLES]]

    audit.percent_quota_reclaimable = (
        round(100.0 * audit.candidate_tokens / audit.opus_tokens, 1)
        if audit.opus_tokens > 0 else 0.0
    )
    audit.percent_sessions = (
        round(100.0 * audit.candidate_sessions / audit.opus_sessions, 1)
        if audit.opus_sessions > 0 else 0.0
    )
    audit.actual_cost_usd = round(audit.actual_cost_usd, 6)
    audit.alternative_cost_usd = round(audit.alternative_cost_usd, 6)
    return audit
