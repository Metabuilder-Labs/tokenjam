"""GET /api/v1/status — agent status overview."""
from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from tj.api.deps import require_api_key
from tj.core.db import _row_to_session
from tj.core.models import AlertFilters
from tj.utils.time_parse import utcnow

router = APIRouter(dependencies=[Depends(require_api_key)])


@router.get("/status")
async def get_status(
    request: Request,
    agent_id: str | None = None,
) -> dict:
    db = request.app.state.db

    # Discover agent IDs
    if agent_id:
        agent_ids = [agent_id]
    elif hasattr(db, "conn"):
        rows = db.conn.execute(
            "SELECT DISTINCT agent_id FROM sessions WHERE agent_id IS NOT NULL "
            "UNION "
            "SELECT DISTINCT agent_id FROM spans WHERE agent_id IS NOT NULL "
            "ORDER BY agent_id"
        ).fetchall()
        agent_ids = [r[0] for r in rows]
    else:
        agent_ids = []

    has_active_alerts = False
    agents_data = []

    for aid in agent_ids:
        session = None

        # Check for active session first, then fall back to latest completed
        if hasattr(db, "conn"):
            active_rows = db.conn.execute(
                "SELECT * FROM sessions WHERE agent_id = $1 AND status = 'active' "
                "ORDER BY started_at DESC LIMIT 1",
                [aid],
            ).fetchall()
            if active_rows:
                cols = [d[0] for d in db.conn.description]
                session = _row_to_session(active_rows[0], cols)

        if session is None:
            sessions = db.get_completed_sessions(aid, limit=1)
            if sessions:
                session = sessions[0]

        today_cost = db.get_daily_cost(aid, utcnow().date())

        # Active (unacknowledged, unsuppressed) alerts
        alerts = db.get_alerts(AlertFilters(agent_id=aid, unread=True, limit=50))
        active_alerts = [a for a in alerts if not a.acknowledged and not a.suppressed]
        if active_alerts:
            has_active_alerts = True

        agent_data = {
            "agent_id": aid,
            "status": session.status if session else "idle",
            "session_id": session.session_id if session else None,
            "cost_today": today_cost,
            "input_tokens": session.input_tokens if session else 0,
            "output_tokens": session.output_tokens if session else 0,
            "tool_call_count": session.tool_call_count if session else 0,
            "error_count": session.error_count if session else 0,
            "active_alerts": len(active_alerts),
            "duration_seconds": session.duration_seconds if session else None,
            "started_at": session.started_at.isoformat() if session and session.started_at else None,
            "total_cost_usd": float(session.total_cost_usd) if session and session.total_cost_usd is not None else 0.0,
        }
        agents_data.append(agent_data)

    return {
        "agents": agents_data,
        "has_active_alerts": has_active_alerts,
    }
