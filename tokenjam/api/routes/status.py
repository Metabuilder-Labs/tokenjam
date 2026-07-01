"""GET /api/v1/status — agent status overview + session archive."""
from __future__ import annotations

from datetime import timedelta, timezone

from fastapi import APIRouter, Depends, Request

from tokenjam.api.deps import require_api_key
from tokenjam.core.alerts import is_interactive_coding_agent
from tokenjam.core.db import (
    _row_to_session,
    sdk_service_series,
    session_active_seconds,
    session_token_cost_rollup,
)
from tokenjam.core.framing import (
    WindowSummary,
    compute_framing,
    plan_determination_mix,
)
from tokenjam.core.models import (
    SESSION_IDLE_THRESHOLD,
    SESSION_STALE_THRESHOLD,
    AlertFilters,
    SessionRecord,
)
from tokenjam.core.transcript import resolve_projects_root, session_transcript_mtime
from tokenjam.utils.time_parse import utcnow

router = APIRouter(dependencies=[Depends(require_api_key)])

# Max current (active/idle) tiles to surface per agent. Extra concurrent
# terminals beyond this are reported via the per-tile `overflow` count rather
# than silently dropped.
MAX_SESSION_TILES = 6
# How many archived (closed/stale) sessions to return, most-recent first.
ARCHIVE_LIMIT = 50
# Scan this many times ARCHIVE_LIMIT candidates so the post-rollup zombie filter
# can still surface up to ARCHIVE_LIMIT sessions that did real work.
ARCHIVE_CANDIDATE_FACTOR = 6

# --- SDK-services zone (non-interactive agents) -----------------------------
# An SDK service never "closes" — it just stops emitting telemetry. So its
# lifecycle is keyed on last-seen, not an explicit end:
#   live         : seen within LIVE_WINDOW           -> Prometheus panel
#   went_quiet   : silent for LIVE..QUIET_WINDOW     -> inactive list, amber
#                  (was steady, just stopped — possible outage)
#   long_dormant : silent beyond QUIET_WINDOW        -> inactive list, muted
# A silent service is ambiguous (decommissioned / idle / crashed), so the UI
# surfaces last-seen and flags the recently-quiet case for a human to check.
SDK_LIVE_WINDOW = SESSION_STALE_THRESHOLD          # 5 min
SDK_QUIET_WINDOW = timedelta(minutes=30)
# Only surface SDK services seen within this window at all (keeps the inactive
# list bounded; older services age out entirely).
SDK_DISCOVERY_WINDOW = timedelta(days=7)
SDK_SERVICES_LIMIT = 50
# Per-minute sparkline resolution for the live services panel.
SDK_SPARKLINE_SLOTS = 24
_SDK_STATE_RANK = {"live": 0, "went_quiet": 1, "long_dormant": 2}


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


def _live_status(
    session: SessionRecord, idle_threshold: timedelta, projects_root
) -> str:
    """Tile status, rescued from a stale span signal by transcript activity.

    A live Claude Code session can read idle/stale once its periodically
    backfilled spans age out, even while its transcript is still being written.
    Only stats the transcript when the base status is idle/stale, so active and
    non-CC sessions cost nothing extra.
    """
    base = session.status_at(idle_threshold)
    if base not in ("idle", "stale"):
        return base
    mtime = session_transcript_mtime(session.session_id, projects_root)
    return session.status_with_transcript_mtime(mtime, idle_threshold)


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
    # Fetch more candidates than we return: 0-signal zombie terminals are
    # dropped below (post-rollup, so a fan-out session's trace-keyed cost isn't
    # mistaken for "empty" — see session_token_cost_rollup #18), so scan a wider
    # window to still surface up to ARCHIVE_LIMIT sessions that did real work.
    params.append(ARCHIVE_LIMIT * ARCHIVE_CANDIDATE_FACTOR)
    sql += f" ORDER BY COALESCE(ended_at, started_at) DESC LIMIT ${len(params)}"

    rows = db.conn.execute(sql, params).fetchall()
    cols = [d[0] for d in db.conn.description]
    archived: list[dict] = []
    for r in rows:
        s = _row_to_session(r, cols)
        namespace = s.service_namespace or _project_for(config, s.agent_id)
        # Roll up tokens/cost from the session's spans joined via shared trace
        # (#18): a fan-out harness posting raw OTLP keeps the cost on agent/
        # trace-keyed spans while the session row holds only a zero-cost marker,
        # so the denormalized aggregate reads 0. Fall back to the stored row when
        # the session has no spans at all.
        roll = session_token_cost_rollup(db.conn, s.session_id)
        input_tokens = roll["input_tokens"] if roll else s.input_tokens
        output_tokens = roll["output_tokens"] if roll else s.output_tokens
        tool_call_count = roll["tool_call_count"] if roll else s.tool_call_count
        total_cost_usd = (
            roll["total_cost_usd"] if roll
            else (float(s.total_cost_usd) if s.total_cost_usd is not None else 0.0)
        )
        # Drop 0-signal zombies: a terminal that opened and did nothing (no
        # tokens, no tool calls, no cost) carries no method worth reviewing.
        # Checked post-rollup so a fan-out session's trace-keyed spend counts as
        # signal even when its stored aggregate reads 0 (#18).
        if (input_tokens == 0 and output_tokens == 0
                and tool_call_count == 0 and total_cost_usd == 0):
            continue
        archived.append({
            "agent_id": s.agent_id,
            "kind": "coding" if is_interactive_coding_agent(s.agent_id) else "sdk",
            "namespace": namespace,
            "session_id": s.session_id,
            "label": _session_label(
                s.session_id, s.service_instance_id, session_labels
            ),
            "status": s.status_at(idle_threshold),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "tool_call_count": tool_call_count,
            "total_cost_usd": total_cost_usd,
            "started_at": s.started_at.isoformat() if s.started_at else None,
            "last_span_time": s.ended_at.isoformat() if s.ended_at else None,
        })
        if len(archived) >= ARCHIVE_LIMIT:
            break
    return archived


def _build_sdk_services(db, config, agent_ids: list[str], now) -> list[dict]:
    """SDK (non-interactive) agents seen recently, each with per-minute cost /
    error sparkline series + a live/went_quiet/long_dormant lifecycle keyed on
    last-seen. Best-effort: any failure degrades to [] rather than 500-ing the
    whole /status route.
    """
    conn = getattr(db, "conn", None)
    if conn is None:
        return []
    try:
        sdk_ids = [a for a in agent_ids if not is_interactive_coding_agent(a)]
        if not sdk_ids:
            return []
        window_start = now - timedelta(minutes=SDK_SPARKLINE_SLOTS)
        series = sdk_service_series(
            conn, sdk_ids, window_start, now, slots=SDK_SPARKLINE_SLOTS
        )
        discovery_cutoff = now - SDK_DISCOVERY_WINDOW

        services: list[dict] = []
        for aid in sdk_ids:
            s = series.get(aid) or {}
            last_seen = s.get("last_seen")
            if last_seen is None:
                continue
            if last_seen.tzinfo is None:
                last_seen = last_seen.replace(tzinfo=timezone.utc)
            if last_seen < discovery_cutoff:
                continue

            gap = now - last_seen
            if gap <= SDK_LIVE_WINDOW:
                state = "live"
            elif gap <= SDK_QUIET_WINDOW:
                state = "went_quiet"
            else:
                state = "long_dormant"

            window_calls = s.get("window_calls", 0)
            window_errors = s.get("window_errors", 0)
            err_rate = (window_errors / window_calls * 100.0) if window_calls else 0.0
            # Average request rate over the sparkline window (req/min).
            req_per_min = window_calls / SDK_SPARKLINE_SLOTS

            services.append({
                "agent_id": aid,
                "kind": "sdk",
                "namespace": _project_for(config, aid),
                "state": state,
                "last_seen": last_seen.isoformat(),
                "today_cost": db.get_daily_cost(aid, now.date()),
                "cost_per_min": s.get("cost_per_min", []),
                "calls_per_min": s.get("calls_per_min", []),
                "err_pct_per_min": s.get("err_pct_per_min", []),
                "req_per_min": req_per_min,
                "err_rate": err_rate,
                "window_cost": s.get("window_cost", 0.0),
            })

        # Live first, then went_quiet, then long_dormant; each newest-seen first.
        # Stable sort: order by last_seen desc, then by state rank (ISO strings
        # sort chronologically, so reverse=True gives newest-first).
        services.sort(key=lambda x: x["last_seen"], reverse=True)
        services.sort(key=lambda x: _SDK_STATE_RANK[x["state"]])
        return services[:SDK_SERVICES_LIMIT]
    except Exception:
        return []


@router.get("/status")
async def get_status(
    request: Request,
    agent_id: str | None = None,
) -> dict:
    db = request.app.state.db
    config = getattr(request.app.state, "config", None)
    session_labels = dict(config.session_labels) if config else {}
    idle_threshold = _idle_threshold(config)
    projects_root = resolve_projects_root(
        getattr(request.app.state, "claude_projects_root", None)
    )
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
            # Active (compute) time = sum of span durations for the session.
            # Distinct from the wall-clock duration_seconds below — see #147.
            active_seconds = (
                session_active_seconds(db.conn, session.session_id)
                if hasattr(db, "conn") else None
            )
            agents_data.append({
                "agent_id": aid,
                "kind": "coding" if is_interactive_coding_agent(aid) else "sdk",
                "namespace": namespace,
                "status": _live_status(session, idle_threshold, projects_root),
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
                "active_seconds": active_seconds,
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

    # SDK-services zone: non-interactive agents with per-minute sparkline series
    # + a last-seen-keyed lifecycle. Separate from the coding tiles above, which
    # are session-backed interactive terminals.
    sdk_services = _build_sdk_services(db, config, agent_ids, now)

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
        "archived": archived,
        "sdk_services": sdk_services,
        "has_active_alerts": has_active_alerts,
        "framing": framing.to_dict(),
    }
