"""GET /api/v1/agents — agent registry."""
from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from tj.api.deps import require_api_key

router = APIRouter(dependencies=[Depends(require_api_key)])


@router.get("/agents")
async def list_agents(request: Request) -> dict:
    db = request.app.state.db
    if not hasattr(db, "conn") or db.conn is None:
        return {"agents": []}
    rows = db.conn.execute(
        "SELECT a.agent_id, a.first_seen, a.last_seen, "
        "COALESCE(SUM(s.cost_usd), 0.0) AS lifetime_cost "
        "FROM agents a LEFT JOIN spans s ON a.agent_id = s.agent_id "
        "GROUP BY a.agent_id, a.first_seen, a.last_seen "
        "ORDER BY a.last_seen DESC NULLS LAST"
    ).fetchall()
    return {
        "agents": [
            {
                "agent_id": r[0],
                "first_seen": r[1].isoformat() if r[1] else None,
                "last_seen": r[2].isoformat() if r[2] else None,
                "lifetime_cost_usd": float(r[3]),
            }
            for r in rows
        ]
    }
