"""
tj optimize analyzers.

Two analyzers, both reading from the existing spans/sessions tables:

  - ModelDowngradeAnalyzer: flag sessions whose structural shape (short input,
    short output, few tool calls) matches a class of work where a cheaper model
    in the same provider family is worth reviewing. Does NOT claim equivalence.

  - BudgetProjectionAnalyzer: project current monthly run rate against any
    configured [budget.<provider>] ceiling, per-provider. No claim is made for
    providers without a configured budget.

Both analyzers are pure functions over a DB connection (DuckDBBackend.conn) so
they're cheap to call from CLI, API, and MCP paths.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

from tokenjam.core.config import ProviderBudget, TjConfig
from tokenjam.core.pricing import get_rates

# Mandatory caveat string. Every channel that surfaces the model-downgrade
# finding must include this verbatim; spec rule #2 is non-negotiable.
MODEL_DOWNGRADE_CAVEAT = (
    "Candidate-flagging heuristic, not a quality judgment. "
    "Review the example sessions before changing models."
)

# Structural heuristic thresholds. Sessions are flagged only when ALL three
# hold; the analyzer never claims the cheaper model would have produced the
# same answer — it claims the *shape* matches a class of work worth reviewing.
SMALL_INPUT_TOKENS = 5_000
SMALL_OUTPUT_TOKENS = 500
SMALL_TOOL_CALLS = 5

THIN_DATA_DAYS = 7

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


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class WindowSummary:
    since:       datetime
    until:       datetime
    days:        float
    sessions:    int
    spans:       int
    total_tokens: int
    total_cost_usd: float
    thin_data:   bool


@dataclass
class DowngradeExample:
    trace_id:   str
    session_id: str | None
    model:      str
    tool_calls: int
    duration_seconds: float | None
    cost_usd:   float


@dataclass
class DowngradeFinding:
    candidate_sessions: int
    total_sessions:     int
    actual_cost_usd:    float
    alternative_cost_usd: float
    monthly_savings_usd: float
    percent_of_sessions: float
    examples:           list[DowngradeExample]
    suggestions:        dict[str, str]
    caveat:             str = MODEL_DOWNGRADE_CAVEAT


@dataclass
class BudgetProjection:
    provider:               str
    budget_usd:             float
    cycle_start_day:        int
    cycle_start:            datetime
    cycle_end:              datetime
    days_into_cycle:        float
    days_remaining:         float
    window_spend_usd:       float
    daily_run_rate_usd:     float
    monthly_run_rate_usd:   float
    projected_cycle_total:  float
    projected_overage_usd:  float
    exhaustion_date:        datetime | None
    days_until_exhaustion:  float | None
    over_budget:            bool
    applies_to_services:    list[str]
    downgrade_run_rate_usd: float | None = None


@dataclass
class OptimizeReport:
    window:    WindowSummary
    downgrade: DowngradeFinding | None
    budgets:   list[BudgetProjection]
    notes:     list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Window summary
# ---------------------------------------------------------------------------

def _utcnow() -> datetime:
    return datetime.now(tz=timezone.utc)


def summarize_window(
    conn,
    since: datetime,
    until: datetime,
    agent_id: str | None = None,
) -> WindowSummary:
    clauses = ["start_time >= $1", "start_time < $2", "model IS NOT NULL"]
    params: list[Any] = [since, until]
    if agent_id:
        clauses.append(f"agent_id = ${len(params) + 1}")
        params.append(agent_id)
    where = " AND ".join(clauses)
    row = conn.execute(
        f"SELECT COUNT(*) AS spans, "
        f"COUNT(DISTINCT session_id) AS sessions, "
        f"COALESCE(SUM(COALESCE(input_tokens,0) + COALESCE(output_tokens,0)), 0) AS tokens, "
        f"COALESCE(SUM(cost_usd), 0.0) AS cost "
        f"FROM spans WHERE {where}",
        params,
    ).fetchone()
    spans = int(row[0] or 0)
    sessions = int(row[1] or 0)
    tokens = int(row[2] or 0)
    cost = float(row[3] or 0.0)
    days = max((until - since).total_seconds() / 86400.0, 0.0)
    return WindowSummary(
        since=since,
        until=until,
        days=days,
        sessions=sessions,
        spans=spans,
        total_tokens=tokens,
        total_cost_usd=cost,
        thin_data=days < THIN_DATA_DAYS or sessions < 3,
    )


# ---------------------------------------------------------------------------
# Model-downgrade analyzer
# ---------------------------------------------------------------------------

def _lookup_downgrade(provider: str, model: str) -> str | None:
    """DOWNGRADE_CANDIDATES lookup that tolerates trailing YYYYMMDD suffixes."""
    mapping = DOWNGRADE_CANDIDATES.get(provider, {})
    if model in mapping:
        return mapping[model]
    import re as _re
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
    # Per-session aggregation from spans. We join on session_id; the session
    # table's totals aren't trustworthy for backfilled rows since they're
    # written after-the-fact, so we re-aggregate from spans directly.
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
    examples: list[DowngradeExample] = []
    suggestions: dict[str, str] = {}

    for row in llm_rows:
        session_id, trace_id, _agent, start_time, end_time, provider, model, \
            in_tok, out_tok, cache_tok, cost = row
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

    return DowngradeFinding(
        candidate_sessions=candidate_sessions,
        total_sessions=total_sessions,
        actual_cost_usd=round(actual_cost, 6),
        alternative_cost_usd=round(alt_cost, 6),
        monthly_savings_usd=round(monthly_savings, 2),
        percent_of_sessions=round(percent, 1),
        examples=examples,
        suggestions=suggestions,
    )


# ---------------------------------------------------------------------------
# Budget projection analyzer
# ---------------------------------------------------------------------------

def _cycle_bounds(now: datetime, start_day: int) -> tuple[datetime, datetime]:
    """
    Return (cycle_start, cycle_end) for the cycle that contains `now`,
    given a monthly cycle that begins on `start_day` of each month.
    """
    start_day = max(1, min(start_day, 28))  # clamp; avoids Feb edge cases
    if now.day >= start_day:
        cs = now.replace(day=start_day, hour=0, minute=0, second=0, microsecond=0)
    else:
        prev_month_year = now.year if now.month > 1 else now.year - 1
        prev_month = now.month - 1 if now.month > 1 else 12
        cs = datetime(prev_month_year, prev_month, start_day, tzinfo=now.tzinfo)
    # cycle_end = next cycle_start
    next_month_year = cs.year + (1 if cs.month == 12 else 0)
    next_month = 1 if cs.month == 12 else cs.month + 1
    ce = datetime(next_month_year, next_month, start_day, tzinfo=cs.tzinfo)
    return cs, ce


def _spend_in_window(
    conn,
    provider: str,
    since: datetime,
    until: datetime,
    services: list[str] | None,
) -> float:
    clauses = ["start_time >= $1", "start_time < $2", "provider = $3"]
    params: list[Any] = [since, until, provider]
    if services:
        # agent_id holds the service.name value in tj's data model.
        placeholders = ",".join(f"${len(params) + i + 1}" for i in range(len(services)))
        clauses.append(f"agent_id IN ({placeholders})")
        params.extend(services)
    where = " AND ".join(clauses)
    row = conn.execute(
        f"SELECT COALESCE(SUM(cost_usd), 0.0) FROM spans WHERE {where}",
        params,
    ).fetchone()
    return float(row[0] or 0.0)


def project_budget(
    conn,
    provider: str,
    budget: ProviderBudget,
    window_since: datetime,
    window_until: datetime,
    downgrade_run_rate_usd: float | None = None,
) -> BudgetProjection | None:
    if not budget.usd or budget.usd <= 0:
        return None

    window_days = max((window_until - window_since).total_seconds() / 86400.0, 1.0 / 86400.0)
    window_spend = _spend_in_window(
        conn, provider, window_since, window_until, budget.applies_to_services or None
    )
    daily_rate = window_spend / window_days
    monthly_rate = daily_rate * 30.0

    cs, ce = _cycle_bounds(window_until, budget.cycle_start_day)
    cycle_spend = _spend_in_window(
        conn, provider, cs, window_until, budget.applies_to_services or None
    )
    cycle_days_total = (ce - cs).total_seconds() / 86400.0
    days_into = max((window_until - cs).total_seconds() / 86400.0, 0.0)
    days_remaining = max(cycle_days_total - days_into, 0.0)

    projected_cycle_total = cycle_spend + daily_rate * days_remaining
    projected_overage = max(projected_cycle_total - budget.usd, 0.0)

    exhaustion_date: datetime | None = None
    days_until_exhaustion: float | None = None
    if daily_rate > 0 and budget.usd > cycle_spend:
        days_to_burn = (budget.usd - cycle_spend) / daily_rate
        exhaustion_date = window_until + timedelta(days=days_to_burn)
        days_until_exhaustion = days_to_burn
        if exhaustion_date > ce:
            exhaustion_date = None
            days_until_exhaustion = None
    elif daily_rate > 0 and cycle_spend >= budget.usd:
        exhaustion_date = window_until
        days_until_exhaustion = 0.0

    return BudgetProjection(
        provider=provider,
        budget_usd=budget.usd,
        cycle_start_day=budget.cycle_start_day,
        cycle_start=cs,
        cycle_end=ce,
        days_into_cycle=round(days_into, 2),
        days_remaining=round(days_remaining, 2),
        window_spend_usd=round(window_spend, 4),
        daily_run_rate_usd=round(daily_rate, 4),
        monthly_run_rate_usd=round(monthly_rate, 2),
        projected_cycle_total=round(projected_cycle_total, 2),
        projected_overage_usd=round(projected_overage, 2),
        exhaustion_date=exhaustion_date,
        days_until_exhaustion=round(days_until_exhaustion, 2) if days_until_exhaustion is not None else None,
        over_budget=projected_cycle_total > budget.usd,
        applies_to_services=list(budget.applies_to_services),
        downgrade_run_rate_usd=downgrade_run_rate_usd,
    )


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------

def build_report(
    db,
    config: TjConfig,
    since: datetime,
    until: datetime | None = None,
    agent_id: str | None = None,
    only: str | None = None,
    budget_provider_filter: str | None = None,
    budget_usd_override: float | None = None,
) -> OptimizeReport:
    """
    Build a complete OptimizeReport.

    `only`:
      - None        -> run both analyzers
      - "model"     -> only the model-downgrade analyzer
      - "budget"    -> only the budget-projection analyzer

    `budget_provider_filter`: when set, only project against this provider's
    budget. `budget_usd_override`: replace the configured budget amount with
    this value (requires budget_provider_filter to be set, else applied to all).
    """
    until = until or _utcnow()
    if until <= since:
        raise ValueError("until must be after since")

    conn = getattr(db, "conn", None)
    if conn is None:
        raise RuntimeError("optimize requires a direct DuckDB connection")

    summary = summarize_window(conn, since, until, agent_id=agent_id)
    window_days = max(summary.days, 1.0 / 86400.0)

    downgrade: DowngradeFinding | None = None
    if only in (None, "model"):
        downgrade = analyze_model_downgrade(
            conn, since, until, agent_id, window_days=window_days,
        )

    notes: list[str] = []
    if summary.thin_data:
        notes.append(
            "Window contains less than ~1 week of activity — projections shown "
            "below should be treated as preliminary."
        )

    # Build per-provider downgrade-adjusted run-rate (used to project what
    # the run rate would be if the user acted on the downgrade pattern).
    downgrade_rate_by_provider: dict[str, float] = {}
    if downgrade is not None and downgrade.actual_cost_usd > 0:
        # Distribute the savings proportionally across providers in the window.
        # Cheap and consistent — savings number is a window-level aggregate.
        prov_spend = conn.execute(
            "SELECT provider, COALESCE(SUM(cost_usd),0.0) FROM spans "
            "WHERE start_time >= $1 AND start_time < $2 AND provider IS NOT NULL "
            "GROUP BY provider",
            [since, until],
        ).fetchall()
        total = sum(float(r[1] or 0.0) for r in prov_spend) or 1.0
        for prov, spend in prov_spend:
            share = float(spend or 0.0) / total
            saved_share = (downgrade.actual_cost_usd - downgrade.alternative_cost_usd) * share
            current_daily = float(spend or 0.0) / window_days
            adjusted_daily = max(current_daily - saved_share / window_days, 0.0)
            downgrade_rate_by_provider[prov] = adjusted_daily * 30.0

    budgets: list[BudgetProjection] = []
    if only in (None, "budget"):
        for provider, bcfg in config.budgets.items():
            if budget_provider_filter and provider != budget_provider_filter:
                continue
            effective = bcfg
            if budget_usd_override is not None:
                effective = ProviderBudget(
                    usd=budget_usd_override,
                    cycle_start_day=bcfg.cycle_start_day,
                    applies_to_services=list(bcfg.applies_to_services),
                )
            proj = project_budget(
                conn, provider, effective, since, until,
                downgrade_run_rate_usd=downgrade_rate_by_provider.get(provider),
            )
            if proj is not None:
                budgets.append(proj)

        # Allow inline budget for a provider not in config: --budget X --budget-usd N
        if (
            budget_provider_filter
            and budget_usd_override is not None
            and budget_provider_filter not in config.budgets
        ):
            inline = ProviderBudget(usd=budget_usd_override, cycle_start_day=1)
            proj = project_budget(
                conn, budget_provider_filter, inline, since, until,
                downgrade_run_rate_usd=downgrade_rate_by_provider.get(budget_provider_filter),
            )
            if proj is not None:
                budgets.append(proj)

    return OptimizeReport(
        window=summary,
        downgrade=downgrade,
        budgets=budgets,
        notes=notes,
    )


def report_to_dict(report: OptimizeReport) -> dict:
    """Convert OptimizeReport to a JSON-serialisable dict."""
    def _serialise(o: Any) -> Any:
        if isinstance(o, datetime):
            return o.isoformat()
        if hasattr(o, "__dataclass_fields__"):
            return {k: _serialise(v) for k, v in asdict(o).items()}
        if isinstance(o, list):
            return [_serialise(x) for x in o]
        if isinstance(o, dict):
            return {k: _serialise(v) for k, v in o.items()}
        return o
    return _serialise(report)


__all__ = [
    "MODEL_DOWNGRADE_CAVEAT",
    "DOWNGRADE_CANDIDATES",
    "WindowSummary",
    "DowngradeExample",
    "DowngradeFinding",
    "BudgetProjection",
    "OptimizeReport",
    "summarize_window",
    "analyze_model_downgrade",
    "project_budget",
    "build_report",
    "report_to_dict",
]
