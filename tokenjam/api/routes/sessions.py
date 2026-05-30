"""POST /api/v1/sessions/close — mark a terminal's sessions closed.

Claude Code emits no "session closed" telemetry, so tj can't passively know a
terminal exited. The `claude` shell wrapper installed by `tj onboard
--claude-code` reports the exit explicitly by POSTing here (via `tj
session-end`) when `claude` returns or is interrupted.

This is a write endpoint, so it is gated by the same ingest Bearer auth as
POST /api/v1/spans (see IngestAuthMiddleware.PROTECTED_PATHS).
"""
from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

router = APIRouter()


@router.post("/sessions/close")
async def close_sessions(request: Request) -> JSONResponse:
    """Close all active sessions matching an instance_id and/or session_id.

    Body: ``{"instance_id": "<id>"}`` and/or ``{"session_id": "<id>"}`` —
    at least one is required. Idempotent: closing already-closed sessions is a
    no-op. Returns ``{"closed": <count>}``.
    """
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content={"error": "Invalid JSON body"})

    if not isinstance(body, dict):
        return JSONResponse(
            status_code=400, content={"error": "Expected a JSON object"}
        )

    instance_id = body.get("instance_id")
    session_id = body.get("session_id")
    if not instance_id and not session_id:
        return JSONResponse(
            status_code=400,
            content={"error": "Provide instance_id and/or session_id"},
        )

    db = request.app.state.db
    closed = 0
    if instance_id:
        closed += db.close_sessions_by_instance(instance_id)
    if session_id:
        closed += db.close_session_by_id(session_id)

    return JSONResponse(status_code=200, content={"closed": closed})
