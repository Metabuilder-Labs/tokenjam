"""GET /api/v1/agents — agent registry."""
from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from tokenjam.api.deps import require_api_key
from tokenjam.core.framing import WindowSummary, compute_framing, plan_determination_mix

router = APIRouter(dependencies=[Depends(require_api_key)])


@router.get("/agents")
async def list_agents(request: Request) -> dict:
    db = request.app.state.db
    config = request.app.state.config
    if not hasattr(db, "conn") or db.conn is None:
        return {"agents": [], "framing": compute_framing(config, WindowSummary()).to_dict()}
    rows = db.conn.execute(
        "SELECT a.agent_id, a.first_seen, a.last_seen, "
        "COALESCE(SUM(s.cost_usd), 0.0) AS lifetime_cost, "
        "COALESCE(SUM(s.input_tokens + s.output_tokens + s.cache_tokens + s.cache_write_tokens), 0) "
        "AS lifetime_tokens "
        "FROM agents a LEFT JOIN spans s ON a.agent_id = s.agent_id "
        "GROUP BY a.agent_id, a.first_seen, a.last_seen "
        "ORDER BY a.last_seen DESC NULLS LAST"
    ).fetchall()
    total_cost = sum(float(r[3]) for r in rows)
    total_tokens = sum(int(r[4] or 0) for r in rows)
    mix = plan_determination_mix(db.conn)
    framing = compute_framing(
        config,
        WindowSummary(
            total_cost_usd=total_cost,
            total_tokens=total_tokens,
            sessions=sum(mix.values()),
            plan_tier_mix=mix,
        ),
    ).to_dict()
    return {
        "agents": [
            {
                "agent_id": r[0],
                "first_seen": r[1].isoformat() if r[1] else None,
                "last_seen": r[2].isoformat() if r[2] else None,
                "lifetime_cost_usd": float(r[3]),
                "lifetime_tokens": int(r[4] or 0),
            }
            for r in rows
        ],
        "framing": framing,
    }
