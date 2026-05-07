"""GET /api/v1/cost — aggregated cost data."""
from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from tj.api.deps import require_api_key
from tj.core.models import CostFilters
from tj.utils.time_parse import parse_since

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
    filters = CostFilters(
        agent_id=agent_id,
        since=parse_since(since) if since else None,
        until=parse_since(until) if until else None,
        group_by=group_by,
    )
    rows = db.get_cost_summary(filters)
    total = sum(r.cost_usd for r in rows)
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
    }
