"""GET /api/v1/cost — aggregated cost data."""
from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from tokenjam.api.deps import require_api_key
from tokenjam.core.framing import WindowSummary, compute_framing, plan_tier_mix
from tokenjam.core.models import CostFilters
from tokenjam.utils.time_parse import parse_since

router = APIRouter(dependencies=[Depends(require_api_key)])


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
        "framing": framing.to_dict(),
    }
