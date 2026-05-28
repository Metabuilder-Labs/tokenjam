"""
Budget-projection analyzer.

Projects current monthly run rate against any configured [budget.<provider>]
ceiling, per-provider. No claim is made for providers without a configured
budget.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

from tokenjam.core.config import ProviderBudget
from tokenjam.core.optimize.registry import register
from tokenjam.core.optimize.types import AnalyzerContext, BudgetProjection


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


@register("budget-projection")
def run(ctx: AnalyzerContext) -> None:
    """
    Registry entry point. Appends one BudgetProjection per configured
    provider budget to ctx.report.budgets.

    Reads ctx.report.downgrade (set by model-downgrade analyzer, if it ran)
    to provide a downgrade-adjusted run-rate for each provider projection.
    """
    config = ctx.config
    downgrade = ctx.report.downgrade

    # Distribute downgrade savings proportionally across providers in the window.
    downgrade_rate_by_provider: dict[str, float] = {}
    if downgrade is not None and downgrade.actual_cost_usd > 0:
        prov_spend = ctx.conn.execute(
            "SELECT provider, COALESCE(SUM(cost_usd),0.0) FROM spans "
            "WHERE start_time >= $1 AND start_time < $2 AND provider IS NOT NULL "
            "GROUP BY provider",
            [ctx.since, ctx.until],
        ).fetchall()
        total = sum(float(r[1] or 0.0) for r in prov_spend) or 1.0
        for prov, spend in prov_spend:
            share = float(spend or 0.0) / total
            saved_share = (downgrade.actual_cost_usd - downgrade.alternative_cost_usd) * share
            current_daily = float(spend or 0.0) / ctx.window_days
            adjusted_daily = max(current_daily - saved_share / ctx.window_days, 0.0)
            downgrade_rate_by_provider[prov] = adjusted_daily * 30.0

    for provider, bcfg in config.budgets.items():
        if ctx.budget_provider_filter and provider != ctx.budget_provider_filter:
            continue
        effective = bcfg
        if ctx.budget_usd_override is not None:
            effective = ProviderBudget(
                usd=ctx.budget_usd_override,
                cycle_start_day=bcfg.cycle_start_day,
                applies_to_services=list(bcfg.applies_to_services),
                plan=bcfg.plan,
            )
        proj = project_budget(
            ctx.conn, provider, effective, ctx.since, ctx.until,
            downgrade_run_rate_usd=downgrade_rate_by_provider.get(provider),
        )
        if proj is not None:
            ctx.report.budgets.append(proj)

    # Allow inline budget for a provider not in config: --budget X --budget-usd N
    if (
        ctx.budget_provider_filter
        and ctx.budget_usd_override is not None
        and ctx.budget_provider_filter not in config.budgets
    ):
        inline = ProviderBudget(usd=ctx.budget_usd_override, cycle_start_day=1)
        proj = project_budget(
            ctx.conn, ctx.budget_provider_filter, inline, ctx.since, ctx.until,
            downgrade_run_rate_usd=downgrade_rate_by_provider.get(ctx.budget_provider_filter),
        )
        if proj is not None:
            ctx.report.budgets.append(proj)
