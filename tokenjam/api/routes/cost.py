"""GET /api/v1/cost — aggregated cost data."""
from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from tokenjam.api.deps import require_api_key
from tokenjam.core.framing import WindowSummary, compute_framing, plan_tier_mix
from tokenjam.core.models import CostFilters
from tokenjam.utils.time_parse import parse_since

router = APIRouter(dependencies=[Depends(require_api_key)])


def _daily_series(conn, agent_id, since_dt, until_dt) -> list[dict]:
    """Daily-bucketed cost/tokens per (date, agent, model) for charting (#113).

    The grouped `rows` collapse the agent/model dimension for day grouping, so
    the time-series chart can't split by model/agent from them. This returns the
    finer breakdown the chart needs without a new endpoint. Dates use
    AT TIME ZONE 'UTC' (CLAUDE.md Rule 1) so they match Python's UTC dates.
    """
    if conn is None:
        return []
    clauses = ["model IS NOT NULL"]
    params: list = []
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
        "SELECT CAST(start_time AT TIME ZONE 'UTC' AS DATE) AS d, agent_id, model, "
        "COALESCE(SUM(cost_usd), 0.0), "
        "COALESCE(SUM(input_tokens), 0), "
        "COALESCE(SUM(output_tokens), 0) "
        "FROM spans WHERE " + where + " "
        "GROUP BY d, agent_id, model ORDER BY d"
    )
    rows = conn.execute(sql, params).fetchall()
    return [
        {
            "date": str(r[0]), "agent_id": r[1], "model": r[2],
            "cost_usd": float(r[3] or 0.0),
            "input_tokens": int(r[4] or 0), "output_tokens": int(r[5] or 0),
        }
        for r in rows
    ]


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
    conn = getattr(db, "conn", None)
    mix = (
        plan_tier_mix(conn, since_dt, until_dt, agent_id)
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
                "cost_usd": r.cost_usd,
            }
            for r in rows
        ],
        "total_cost_usd": total,
        "total_tokens": total_tokens,
        "series": _daily_series(conn, agent_id, since_dt, until_dt),
        "framing": framing.to_dict(),
    }
