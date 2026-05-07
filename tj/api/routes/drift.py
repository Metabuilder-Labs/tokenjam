"""GET /api/v1/drift — drift baseline and latest session comparison."""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Request

from tj.api.deps import require_api_key

router = APIRouter(dependencies=[Depends(require_api_key)])


def _build_agent_drift(db: Any, agent_id: str) -> dict:
    """Build drift info dict for a single agent."""
    baseline = db.get_baseline(agent_id)
    if baseline is None:
        return {"agent_id": agent_id, "baseline": None, "latest_session": None}

    sessions = db.get_completed_sessions(agent_id, limit=1)
    latest = None
    if sessions:
        s = sessions[0]
        latest = {
            "session_id": s.session_id,
            "input_tokens": s.input_tokens,
            "output_tokens": s.output_tokens,
            "tool_call_count": s.tool_call_count,
            "duration_seconds": s.duration_seconds,
        }

    return {
        "agent_id": agent_id,
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
async def get_drift(request: Request, agent_id: str | None = None):
    db = request.app.state.db

    if agent_id:
        return _build_agent_drift(db, agent_id)

    # No agent_id: return drift info for all agents with baselines.
    if not hasattr(db, "conn"):
        return {"agents": []}
    rows = db.conn.execute(
        "SELECT DISTINCT agent_id FROM drift_baselines ORDER BY agent_id"
    ).fetchall()
    agents = [_build_agent_drift(db, row[0]) for row in rows]
    return {"agents": agents}
