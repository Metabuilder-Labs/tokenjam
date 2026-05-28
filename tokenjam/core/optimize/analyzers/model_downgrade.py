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
from tokenjam.core.optimize.types import (
    AnalyzerContext,
    DowngradeExample,
    DowngradeFinding,
)
from tokenjam.core.pricing import get_rates

# Structural heuristic thresholds. Sessions are flagged only when ALL three
# hold; the analyzer never claims the cheaper model would have produced the
# same answer — it claims the *shape* matches a class of work worth reviewing.
SMALL_INPUT_TOKENS = 5_000
SMALL_OUTPUT_TOKENS = 500
SMALL_TOOL_CALLS = 5

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
        suggestions[model] = alt

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
    percent_tokens = (
        candidate_tokens / window_total_tokens * 100.0
        if window_total_tokens > 0 else 0.0
    )
    monthly_tokens_in_candidates = (
        int(candidate_tokens / window_days * 30.0) if window_days > 0 else 0
    )

    return DowngradeFinding(
        candidate_sessions=candidate_sessions,
        total_sessions=total_sessions,
        actual_cost_usd=round(actual_cost, 6),
        alternative_cost_usd=round(alt_cost, 6),
        monthly_savings_usd=round(monthly_savings, 2),
        percent_of_sessions=round(percent, 1),
        examples=examples,
        suggestions=suggestions,
        candidate_tokens=candidate_tokens,
        window_total_tokens=window_total_tokens,
        percent_of_tokens=round(percent_tokens, 1),
        monthly_tokens_in_candidates=monthly_tokens_in_candidates,
    )


@register("model-downgrade")
def run(ctx: AnalyzerContext) -> None:
    """Registry entry point. Mutates ctx.report.downgrade."""
    ctx.report.downgrade = analyze_model_downgrade(
        ctx.conn, ctx.since, ctx.until, ctx.agent_id, ctx.window_days,
    )
