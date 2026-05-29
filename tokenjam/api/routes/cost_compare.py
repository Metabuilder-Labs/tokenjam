"""
GET /api/v1/cost/compare — server-side period comparison.

Restores `tj cost --compare` and `tj optimize --compare` under
daemon-up mode. Like /api/v1/optimize (#68 §12), this exists because
compute_cost_diff queries db.conn directly and DuckDB blocks
non-daemon attaches.
"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from tokenjam.api.deps import require_api_key
from tokenjam.core.cost import compute_cost_diff
from tokenjam.utils.time_parse import parse_since, utcnow

router = APIRouter()


@router.get("/cost/compare", dependencies=[Depends(require_api_key)])
def get_cost_compare(
    request: Request,
    since: str = Query("7d", description="Current window lookback."),
    compare: str = Query(..., description="previous / last-week / last-month / last-7d / last-30d / YYYY-MM-DD:YYYY-MM-DD"),
    agent_id: str | None = Query(None, alias="agent_id"),
    top_n: int = Query(5, description="Top per-agent / per-model shifts."),
) -> dict[str, Any]:
    """
    Return the structured CostDiff payload that cmd_cost / cmd_optimize
    render. Schema mirrors `_diff_to_dict` in cmd_cost.
    """
    db = request.app.state.db
    if db is None:
        raise HTTPException(status_code=503, detail="Server db not initialised.")

    try:
        since_dt = parse_since(since)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid --since: {exc}") from exc

    until_dt = utcnow()

    try:
        diff = compute_cost_diff(
            db, since_dt, until_dt, compare,
            agent_id=agent_id, top_n=top_n,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    def _wt(w):
        return {
            "since": w.since.isoformat(),
            "until": w.until.isoformat(),
            "sessions": w.sessions,
            "input_tokens": w.input_tokens,
            "output_tokens": w.output_tokens,
            "cache_tokens": w.cache_tokens,
            "total_tokens": w.total_tokens,
            "total_cost_usd": w.total_cost_usd,
        }
    return {
        "current": _wt(diff.current),
        "previous": _wt(diff.previous),
        "cost_delta_usd": diff.cost_delta_usd,
        "cost_delta_pct": diff.cost_delta_pct,
        "tokens_delta": diff.tokens_delta,
        "tokens_delta_pct": diff.tokens_delta_pct,
        "by_agent": diff.by_agent,
        "by_model": diff.by_model,
    }
