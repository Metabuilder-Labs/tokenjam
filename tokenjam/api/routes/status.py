"""GET /api/v1/status — agent status overview."""
from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from tokenjam.api.deps import require_api_key
from tokenjam.core.db import _row_to_session
from tokenjam.core.models import AlertFilters, SESSION_STALE_THRESHOLD
from tokenjam.utils.time_parse import utcnow

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
    config = getattr(request.app.state, "config", None)
    # Recency window for "currently active" — a session whose last span is
    # within this window is a live terminal. Several at once = concurrent
    # terminals sharing one agent_id (e.g. 3 Claude Code tabs in one repo).
    active_cutoff = utcnow() - SESSION_STALE_THRESHOLD

    for aid in agent_ids:
        # One tile per concurrently-active session (terminal). Fall back to the
        # latest completed session so an idle agent still shows a single tile.
        sessions: list = []
        if hasattr(db, "conn"):
            rows = db.conn.execute(
                "SELECT * FROM sessions WHERE agent_id = $1 AND status = 'active' "
                "AND COALESCE(ended_at, started_at) > $2 "
                "ORDER BY COALESCE(ended_at, started_at) DESC",
                [aid, active_cutoff],
            ).fetchall()
            if rows:
                cols = [d[0] for d in db.conn.description]
                sessions = [_row_to_session(r, cols) for r in rows]
        if not sessions:
            completed = db.get_completed_sessions(aid, limit=1)
            if completed:
                sessions = [completed[0]]

        today_cost = db.get_daily_cost(aid, utcnow().date())

        # Active (unacknowledged, unsuppressed) alerts for this agent.
        alerts = db.get_alerts(AlertFilters(agent_id=aid, unread=True, limit=50))
        active_alerts = [a for a in alerts if not a.acknowledged and not a.suppressed]
        if active_alerts:
            has_active_alerts = True

        # Project grouping: prefer the namespace captured on the session;
        # fall back to the agent's configured project ([agents.<id>].project)
        # so already-running agents that never sent service.namespace still
        # group correctly without a restart.
        agent_cfg = config.agents.get(aid) if config else None
        configured_project = agent_cfg.project if agent_cfg else None

        if not sessions:
            # Agent known (has spans) but no session row — show an idle tile.
            agents_data.append({
                "agent_id": aid, "namespace": configured_project, "status": "idle",
                "session_id": None, "cost_today": today_cost, "total_cost_usd": 0.0,
                "input_tokens": 0, "output_tokens": 0, "tool_call_count": 0,
                "error_count": 0, "active_alerts": len(active_alerts),
                "duration_seconds": None, "started_at": None, "last_span_time": None,
            })
            continue

        multi = len(sessions) > 1
        for session in sessions:
            namespace = session.service_namespace or configured_project
            # When several sessions share one agent, attribute alerts per
            # session; otherwise use the agent-level count (covers alerts that
            # carry no session_id).
            if multi:
                sess_alerts = sum(
                    1 for a in active_alerts if a.session_id == session.session_id
                )
            else:
                sess_alerts = len(active_alerts)
            agents_data.append({
                "agent_id": aid,
                "namespace": namespace,
                "status": session.effective_status,
                "session_id": session.session_id,
                "cost_today": today_cost,
                "total_cost_usd": (
                    float(session.total_cost_usd)
                    if session.total_cost_usd is not None else 0.0
                ),
                "input_tokens": session.input_tokens,
                "output_tokens": session.output_tokens,
                "tool_call_count": session.tool_call_count,
                "error_count": session.error_count,
                "active_alerts": sess_alerts,
                "duration_seconds": session.duration_seconds,
                "started_at": (
                    session.started_at.isoformat() if session.started_at else None
                ),
                "last_span_time": (
                    session.ended_at.isoformat() if session.ended_at else None
                ),
            })

    return {
        "agents": agents_data,
        "has_active_alerts": has_active_alerts,
    }
