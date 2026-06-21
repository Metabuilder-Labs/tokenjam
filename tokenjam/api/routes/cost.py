"""GET /api/v1/cost — aggregated cost data."""
from __future__ import annotations

import math

from fastapi import APIRouter, Depends, Request

from tokenjam.api.deps import require_api_key
from tokenjam.core.cycle import cycle_bounds, effective_cycle_start_day
from tokenjam.core.framing import (
    WindowSummary,
    compute_framing,
    plan_determination_mix,
)
from tokenjam.core.models import CostFilters
from tokenjam.utils.time_parse import parse_since, utcnow

router = APIRouter(dependencies=[Depends(require_api_key)])


def _cycle_block(config) -> dict:
    """Current billing-cycle bounds for the run-rate caption (#138).

    Honors `[budget.<provider>] cycle_start_day` when configured, falling back
    to the calendar month (start_day=1). The UI projects the run-rate to
    `cycle.end` instead of assuming a calendar-month boundary, so a user on a
    non-calendar cycle gets an honest "by <cycle end>" caption.
    """
    now = utcnow()
    start_day = effective_cycle_start_day(config)
    cs, ce = cycle_bounds(now, start_day)
    days_remaining = max(1, math.ceil((ce - now).total_seconds() / 86400.0))
    return {
        "start": int(cs.timestamp()),
        "end": int(ce.timestamp()),
        "days_remaining": days_remaining,
        "start_day": start_day,
    }


def _window_series(conn, agent_id, since_dt, until_dt) -> dict:
    """Window-bucketed cost/tokens per (bucket, agent, model) for charting.

    Buckets hourly for short windows (≤ 2 days) and daily otherwise, returning
    epoch-second bucket keys plus the window bounds so the UI can render the
    FULL selected window with zero-fill and window-matched tick density (#133) —
    the grouped `rows` collapse the agent/model dimension and only cover days
    with data. Buckets are UTC (AT TIME ZONE 'UTC', Rule 1).
    """
    start = since_dt
    end = until_dt or utcnow()
    span_days = ((end - start).total_seconds() / 86400.0) if start is not None else None
    bucket = "hour" if (span_days is not None and span_days <= 2) else "day"
    out: dict = {
        "series": [],
        "series_bucket": bucket,
        "window_start": int(start.timestamp()) if start is not None else None,
        "window_end": int(end.timestamp()),
    }
    if conn is None:
        return out
    # $1 is the date_trunc unit (a controlled 'hour'/'day' literal, bound as a
    # parameter — no f-string SQL, Rule 7).
    clauses = ["model IS NOT NULL"]
    params: list = [bucket]
    if agent_id:
        params.append(agent_id)
        clauses.append("agent_id = $" + str(len(params)))
    if since_dt is not None:
        params.append(since_dt)
        clauses.append("start_time >= $" + str(len(params)))
    if until_dt is not None:
        params.append(until_dt)
        clauses.append("start_time <= $" + str(len(params)))
    where = " AND ".join(clauses)
    sql = (
        "SELECT CAST(epoch(date_trunc($1, start_time AT TIME ZONE 'UTC')) AS BIGINT) AS b, "
        "agent_id, model, COALESCE(SUM(cost_usd), 0.0), "
        "COALESCE(SUM(input_tokens), 0), COALESCE(SUM(output_tokens), 0) "
        "FROM spans WHERE " + where + " GROUP BY b, agent_id, model ORDER BY b"
    )
    rows = conn.execute(sql, params).fetchall()
    out["series"] = [
        {
            "bucket": int(r[0]), "agent_id": r[1], "model": r[2],
            "cost_usd": float(r[3] or 0.0),
            "input_tokens": int(r[4] or 0), "output_tokens": int(r[5] or 0),
        }
        for r in rows
    ]
    return out


@router.get("/cost")
async def get_cost(
    request: Request,
    agent_id: str | None = None,
    since: str | None = None,
    until: str | None = None,
    group_by: str = "day",
) -> dict:
    db = request.app.state.db
    config = request.app.state.config
    since_dt = parse_since(since) if since else None
    until_dt = parse_since(until) if until else None
    filters = CostFilters(
        agent_id=agent_id,
        since=since_dt,
        until=until_dt,
        group_by=group_by,
    )
    rows = db.get_cost_summary(filters)
    total = sum(r.cost_usd for r in rows)
    total_tokens = sum(r.input_tokens + r.output_tokens for r in rows)

    # Plan-tier framing block — single source shared with the CLI (#110). Lets
    # the local web UI render the same suppressed/qualified dollar figures.
    # The mix is window-INDEPENDENT (#177): the pricing mode + qualifier banner
    # are a property of the user's plan, so the chart's tokens-vs-dollars unit
    # stays consistent as the user switches windows. Only the window totals
    # (above) and the `series` (below) are window-scoped.
    conn = getattr(db, "conn", None)
    mix = (
        plan_determination_mix(conn, agent_id)
        if conn is not None else {}
    )
    framing = compute_framing(
        config,
        WindowSummary(
            total_cost_usd=total,
            total_tokens=total_tokens,
            sessions=sum(mix.values()),
            plan_tier_mix=mix,
        ),
    )
    return {
        "rows": [
            {
                "group": r.group,
                "agent_id": r.agent_id,
                "model": r.model,
                "input_tokens": r.input_tokens,
                "output_tokens": r.output_tokens,
                "cache_tokens": r.cache_tokens,
                "cache_write_tokens": r.cache_write_tokens,
                "cost_usd": r.cost_usd,
            }
            for r in rows
        ],
        "total_cost_usd": total,
        "total_tokens": total_tokens,
        "total_cache_tokens": sum(r.cache_tokens for r in rows),
        "total_cache_write_tokens": sum(r.cache_write_tokens for r in rows),
        **_window_series(conn, agent_id, since_dt, until_dt),
        "cycle": _cycle_block(config),
        "framing": framing.to_dict(),
    }
