"""GET /api/v1/drift — drift baseline and latest session comparison.

``since`` bounds which OBSERVATION the baseline is compared against. It is the
same parameter, the same helper (``utils.time_parse.parse_since``) and the same
400-on-malformed contract ``/cost``, ``/alerts``, ``/traces`` and ``/optimize``
already use, because this route sits on the same Dashboard behind the same
window selector and used to ignore it: the selector visibly moved three tiles
and silently did nothing to "Agents drifting".

A drift signal is a baseline plus a recent session to compare it against, so a
window works on the session side. An agent whose most recent completed session
ended before the window has no in-window observation, and its ``latest_session``
is therefore ``None`` for that window. That null is accompanied by
``latest_session_outside_window`` and the timestamp that fell outside, because a
bare null here reads as "this agent never ran" when the truth is "it ran, just
not inside the window you asked about". The baseline itself is never hidden.

One exception, and it is a persona exception rather than a window one:
interactive coding agents are never baselined. ``DriftDetector.on_session_end``
already refuses to build a baseline for them, because a single mean/stddev over
a heterogeneous, human-driven workload measures nothing. That is a WRITE-path
skip, and baselines are only ever built once and never recomputed, so a row
written before that skip existed (or by an id that has since been reclassified)
outlives it and still renders. This route therefore applies the same gate on the
READ path, through the same ``is_interactive_coding_agent`` helper, so the two
paths cannot disagree and a legacy row cannot surface as a drift verdict.
"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from tokenjam.api.deps import require_api_key
from tokenjam.core.alerts import is_interactive_coding_agent
from tokenjam.core.data_span import available_data_span
from tokenjam.utils.time_parse import parse_since

router = APIRouter(dependencies=[Depends(require_api_key)])


def _last_activity(session: Any) -> Any:
    """When a session was last active: ``ended_at``, else ``started_at``.

    The same COALESCE ``get_completed_sessions`` orders by, so "the latest
    session" and "is the latest session in the window" cannot disagree about
    which timestamp they mean.
    """
    return getattr(session, "ended_at", None) or getattr(session, "started_at", None)


def _iso(value: Any) -> str | None:
    try:
        return value.isoformat()
    except AttributeError:
        return None


def _build_agent_drift(db: Any, agent_id: str, since: Any = None) -> dict:
    """Build drift info dict for a single agent, bounded to ``since``."""
    if is_interactive_coding_agent(agent_id):
        # Read-path mirror of the write-path skip in DriftDetector.on_session_end.
        # A stale row from before that skip must not render as a drift verdict.
        return {
            "agent_id": agent_id,
            "baseline": None,
            "latest_session": None,
            "baseline_skipped_reason": "interactive_coding_agent",
        }

    baseline = db.get_baseline(agent_id)
    if baseline is None:
        return {"agent_id": agent_id, "baseline": None, "latest_session": None}

    sessions = db.get_completed_sessions(agent_id, limit=1)
    latest = None
    outside_window = False
    latest_at: str | None = None
    if sessions:
        s = sessions[0]
        latest_at = _iso(_last_activity(s))
        activity = _last_activity(s)
        # Outside the window: report the baseline and say WHY there is no
        # comparison, rather than emitting a bare null that reads as "never ran".
        if since is not None and activity is not None and activity < since:
            outside_window = True
        else:
            latest = {
                "session_id": s.session_id,
                "input_tokens": s.input_tokens,
                "output_tokens": s.output_tokens,
                "tool_call_count": s.tool_call_count,
                "duration_seconds": s.duration_seconds,
            }

    return {
        "agent_id": agent_id,
        "latest_session_outside_window": outside_window,
        "latest_session_at": latest_at,
        "baseline": {
            "sessions_sampled": baseline.sessions_sampled,
            "computed_at": baseline.computed_at.isoformat() if baseline.computed_at else None,
            "avg_input_tokens": baseline.avg_input_tokens,
            "stddev_input_tokens": baseline.stddev_input_tokens,
            "avg_output_tokens": baseline.avg_output_tokens,
            "stddev_output_tokens": baseline.stddev_output_tokens,
            "avg_session_duration_s": baseline.avg_session_duration_s,
            "stddev_session_duration": baseline.stddev_session_duration,
            "avg_tool_call_count": baseline.avg_tool_call_count,
            "stddev_tool_call_count": baseline.stddev_tool_call_count,
        },
        "latest_session": latest,
    }


@router.get("/drift", response_model=None)
async def get_drift(
    request: Request,
    agent_id: str | None = None,
    since: str | None = Query(
        None,
        description="Lookback window (e.g. 30d, 7d, 24h). Bounds which session "
                    "the baseline is compared against. Omit for the latest "
                    "session whenever it ran.",
    ),
):
    db = request.app.state.db

    since_dt = None
    if since is not None:
        try:
            since_dt = parse_since(since)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=f"Invalid since: {exc}") from exc

    conn = getattr(db, "conn", None)
    # The window this response actually applied, and how much history there is
    # to apply one over. `available_days` is deliberately NOT newest minus
    # oldest — see `core/data_span.py`; one sentinel-dated row makes that
    # measure wrong by orders of magnitude.
    window = {"since": since, "start": _iso(since_dt)}
    data_span = available_data_span(conn).to_dict()

    if agent_id:
        return {
            **_build_agent_drift(db, agent_id, since_dt),
            "window": window,
            "data_span": data_span,
        }

    # No agent_id: return drift info for all agents with baselines.
    if conn is None:
        return {"agents": [], "window": window, "data_span": data_span}
    rows = conn.execute(
        "SELECT DISTINCT agent_id FROM drift_baselines ORDER BY agent_id"
    ).fetchall()
    # Filtered in Python, not in SQL, so the prefix list stays in exactly one
    # place (core/alerts.py) instead of being restated as a LIKE clause here.
    agents = [
        _build_agent_drift(db, row[0], since_dt)
        for row in rows
        if not is_interactive_coding_agent(row[0])
    ]
    return {"agents": agents, "window": window, "data_span": data_span}
