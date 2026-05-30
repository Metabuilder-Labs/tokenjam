"""GET /api/v1/status — agent status overview + session archive."""
from __future__ import annotations

from datetime import timedelta

from fastapi import APIRouter, Depends, Request

from tokenjam.api.deps import require_api_key
from tokenjam.core.db import _row_to_session
from tokenjam.core.models import (
    SESSION_IDLE_THRESHOLD,
    AlertFilters,
    SessionRecord,
)
from tokenjam.utils.time_parse import utcnow

router = APIRouter(dependencies=[Depends(require_api_key)])

# Max current (active/idle) tiles to surface per agent. Extra concurrent
# terminals beyond this are reported via the per-tile `overflow` count rather
# than silently dropped.
MAX_SESSION_TILES = 6
# How many archived (closed/stale) sessions to return, most-recent first.
ARCHIVE_LIMIT = 50


def _session_label(
    session_id: str | None,
    instance_id: str | None,
    session_labels: dict[str, str],
) -> str | None:
    """Human display name for a session's terminal.

    Priority: manual [session_labels] override (full id or prefix match, for
    naming already-running terminals) -> OTel service.instance.id (durable,
    set at launch) -> None (UI falls back to the short session id).
    """
    if session_id and session_labels:
        if session_id in session_labels:
            return session_labels[session_id]
        for key, label in session_labels.items():
            if session_id.startswith(key):
                return label
    return instance_id


def _idle_threshold(config) -> timedelta:
    """Configured idle window ([sessions] idle_minutes), else the default."""
    if config is not None:
        return timedelta(minutes=config.session_idle_minutes)
    return SESSION_IDLE_THRESHOLD


def _project_for(config, agent_id: str) -> str | None:
    """Server-side project fallback ([agents.<id>].project) for an agent."""
    if config is None:
        return None
    agent_cfg = config.agents.get(agent_id)
    return agent_cfg.project if agent_cfg else None


def _build_archive(
    db,
    config,
    session_labels: dict[str, str],
    idle_threshold: timedelta,
    cutoff,
    agent_id: str | None,
) -> list[dict]:
    """Closed + stale sessions, most-recent first, capped at ARCHIVE_LIMIT.

    Stale = an 'active' session whose last activity is older than the idle
    window (a zombie that was never explicitly closed). Closed = a session
    explicitly ended via /api/v1/sessions/close.
    """
    if not hasattr(db, "conn"):
        return []

    clause = (
        "status = 'closed' "
        "OR (status = 'active' AND COALESCE(ended_at, started_at) <= $1)"
    )
    params: list = [cutoff]
    sql = f"SELECT * FROM sessions WHERE ({clause})"
    if agent_id:
        params.append(agent_id)
        sql += f" AND agent_id = ${len(params)}"
    params.append(ARCHIVE_LIMIT)
    sql += f" ORDER BY COALESCE(ended_at, started_at) DESC LIMIT ${len(params)}"

    rows = db.conn.execute(sql, params).fetchall()
    cols = [d[0] for d in db.conn.description]
    archived: list[dict] = []
    for r in rows:
        s = _row_to_session(r, cols)
        namespace = s.service_namespace or _project_for(config, s.agent_id)
        archived.append({
            "agent_id": s.agent_id,
            "namespace": namespace,
            "session_id": s.session_id,
            "label": _session_label(
                s.session_id, s.service_instance_id, session_labels
            ),
            "status": s.status_at(idle_threshold),
            "input_tokens": s.input_tokens,
            "output_tokens": s.output_tokens,
            "tool_call_count": s.tool_call_count,
            "total_cost_usd": (
                float(s.total_cost_usd) if s.total_cost_usd is not None else 0.0
            ),
            "started_at": s.started_at.isoformat() if s.started_at else None,
            "last_span_time": s.ended_at.isoformat() if s.ended_at else None,
        })
    return archived


@router.get("/status")
async def get_status(
    request: Request,
    agent_id: str | None = None,
) -> dict:
    db = request.app.state.db
    config = getattr(request.app.state, "config", None)
    session_labels = dict(config.session_labels) if config else {}
    idle_threshold = _idle_threshold(config)
    now = utcnow()
    # Sessions whose last activity is newer than this are "current" (active or
    # idle) and get a tile; older active sessions are stale -> archive only.
    current_cutoff = now - idle_threshold

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
    agents_data: list[dict] = []

    for aid in agent_ids:
        # Current tiles: active sessions (one per live terminal) whose last
        # activity is within the idle window. Closed/completed/stale sessions
        # never become a current tile — they live only in the archive.
        sessions: list[SessionRecord] = []
        if hasattr(db, "conn"):
            rows = db.conn.execute(
                "SELECT * FROM sessions WHERE agent_id = $1 AND status = 'active' "
                "AND COALESCE(ended_at, started_at) > $2 "
                "ORDER BY COALESCE(ended_at, started_at) DESC",
                [aid, current_cutoff],
            ).fetchall()
            if rows:
                cols = [d[0] for d in db.conn.description]
                sessions = [_row_to_session(r, cols) for r in rows]

        today_cost = db.get_daily_cost(aid, now.date())

        # Active (unacknowledged, unsuppressed) alerts for this agent.
        alerts = db.get_alerts(AlertFilters(agent_id=aid, unread=True, limit=50))
        active_alerts = [a for a in alerts if not a.acknowledged and not a.suppressed]
        if active_alerts:
            has_active_alerts = True

        if not sessions:
            # No active/idle session — contribute no current tile.
            continue

        configured_project = _project_for(config, aid)
        # Cap tiles by recency; surface (don't silently drop) the overflow.
        overflow = max(0, len(sessions) - MAX_SESSION_TILES)
        shown = sessions[:MAX_SESSION_TILES]
        multi = len(shown) > 1
        for session in shown:
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
                "status": session.status_at(idle_threshold),
                "session_id": session.session_id,
                "label": _session_label(
                    session.session_id, session.service_instance_id, session_labels
                ),
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
                # Per-agent count of current sessions hidden by the tile cap.
                "overflow": overflow,
            })

    archived = _build_archive(
        db, config, session_labels, idle_threshold, current_cutoff, agent_id
    )

    return {
        "agents": agents_data,
        "archived": archived,
        "has_active_alerts": has_active_alerts,
    }
