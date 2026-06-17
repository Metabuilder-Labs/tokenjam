"""GET /api/v1/status — agent status overview."""
from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from tokenjam.api.deps import require_api_key
from tokenjam.core.db import (
    _row_to_session,
    session_active_seconds,
    session_token_cost_rollup,
)
from tokenjam.core.framing import (
    WindowSummary,
    compute_framing,
    plan_determination_mix,
)
from tokenjam.core.models import AlertFilters
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

        # Active (compute) time = sum of span durations for the session.
        # Distinct from the wall-clock duration_seconds below — see issue #147.
        active_seconds = None
        if session is not None and hasattr(db, "conn"):
            active_seconds = session_active_seconds(db.conn, session.session_id)

        # Roll up tokens/cost from the session's spans joined via shared trace
        # (#18): a fan-out harness posting raw OTLP keeps the cost on agent/
        # trace-keyed spans while the session row holds only a zero-cost marker,
        # so the denormalized aggregate reads 0. Fall back to the stored row when
        # the session has no spans at all.
        roll = None
        if session is not None and hasattr(db, "conn"):
            roll = session_token_cost_rollup(db.conn, session.session_id)

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
            "input_tokens": roll["input_tokens"] if roll else (session.input_tokens if session else 0),
            "output_tokens": roll["output_tokens"] if roll else (session.output_tokens if session else 0),
            "tool_call_count": roll["tool_call_count"] if roll else (session.tool_call_count if session else 0),
            "error_count": session.error_count if session else 0,
            "active_alerts": len(active_alerts),
            "duration_seconds": session.duration_seconds if session else None,
            "active_seconds": active_seconds,
            "started_at": session.started_at.isoformat() if session and session.started_at else None,
            "total_cost_usd": (
                roll["total_cost_usd"] if roll
                else (float(session.total_cost_usd) if session and session.total_cost_usd is not None else 0.0)
            ),
        }
        agents_data.append(agent_data)

    # Plan-tier framing block so the agent cards' "Cost today" figure suppresses
    # / reframes raw dollars for subscription / local users (#191) — the web UI
    # consumes this rather than re-deriving the rules in JS (single compute
    # path). Window-INDEPENDENT mix (`plan_determination_mix`), as on /traces.
    config = request.app.state.config
    conn = getattr(db, "conn", None)
    mix = plan_determination_mix(conn, agent_id) if conn is not None else {}
    framing = compute_framing(
        config,
        WindowSummary(plan_tier_mix=mix, sessions=sum(mix.values())),
    )

    return {
        "agents": agents_data,
        "has_active_alerts": has_active_alerts,
        "framing": framing.to_dict(),
    }
