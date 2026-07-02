"""Session routes.

GET /api/v1/sessions — session enumeration (one row per session). Unlike
/status (which collapses to one record per agent via
`ORDER BY started_at DESC LIMIT 1`), this route returns every matching session,
so a single agent running multiple concurrent sessions is fully represented.
The MCP `list_active_sessions` tool proxies here in serve mode so its output
matches the direct-DB path (#35).

POST /api/v1/sessions/close — mark a terminal's sessions closed. Claude Code
emits no "session closed" telemetry, so tj can't passively know a terminal
exited. The `claude` shell wrapper installed by `tj onboard --claude-code`
reports the exit explicitly by POSTing here (via `tj session-end`) when
`claude` returns or is interrupted. This is a write endpoint gated by the same
ingest Bearer auth as POST /api/v1/spans (see IngestAuthMiddleware.PROTECTED_PATHS),
so it is not additionally gated by the read-side API key.

POST /api/v1/sessions/{id}/label — set (or clear) a user-supplied display name
for a session (the dashboard's right-click rename). API-key gated.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse

from tokenjam.api.deps import require_api_key
from tokenjam.core.db import delete_session_label, set_session_label

router = APIRouter()


@router.get("/sessions", dependencies=[Depends(require_api_key)])
async def list_sessions(
    request: Request,
    status: str | None = None,
    agent_id: str | None = None,
) -> dict:
    """Enumerate sessions one row per session, newest first.

    `status` filters by session status (e.g. 'active'); `agent_id` scopes to a
    single agent. Both are optional.
    """
    db = request.app.state.db
    conn = getattr(db, "conn", None)
    if conn is None:
        return {"sessions": [], "count": 0}

    clauses: list[str] = []
    params: list = []
    if status:
        params.append(status)
        clauses.append(f"status = ${len(params)}")
    if agent_id:
        params.append(agent_id)
        clauses.append(f"agent_id = ${len(params)}")
    where = (" WHERE " + " AND ".join(clauses)) if clauses else ""

    rows = conn.execute(
        "SELECT session_id, agent_id, started_at, total_cost_usd, "
        "input_tokens, output_tokens, tool_call_count, error_count "
        f"FROM sessions{where} ORDER BY started_at DESC",
        params,
    ).fetchall()
    sessions = [
        {
            "session_id": r[0],
            "agent_id": r[1],
            "started_at": r[2].isoformat() if r[2] else None,
            "total_cost_usd": float(r[3]) if r[3] else 0.0,
            "input_tokens": r[4] or 0,
            "output_tokens": r[5] or 0,
            "tool_call_count": r[6] or 0,
            "error_count": r[7] or 0,
        }
        for r in rows
    ]
    return {"sessions": sessions, "count": len(sessions)}


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


# Max length of a user-supplied session label; longer input is truncated.
MAX_SESSION_LABEL_LEN = 120


@router.post(
    "/sessions/{session_id}/label",
    dependencies=[Depends(require_api_key)],
)
async def set_session_label_endpoint(
    request: Request, session_id: str
) -> JSONResponse:
    """Set (or clear) a user-supplied display name for a session.

    Body: ``{"label": "<str>"}``. The label is stripped and truncated to
    ``MAX_SESSION_LABEL_LEN``; an empty (or whitespace-only) label CLEARS any
    existing rename, reverting the card to its default (service.instance.id ->
    short session id). The /status route overlays this onto the tile/archive
    label, taking precedence over the OTel instance id but NOT over a config
    ``[session_labels]`` entry (see ``status._session_label``).

    Dashboard action gated by ``require_api_key`` (the UI's ``apiPost`` sends the
    Bearer api key) — deliberately NOT added to the ingest-secret PROTECTED_PATHS.
    Returns ``{"session_id": ..., "label": <str|None>}`` (None when cleared).
    """
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content={"error": "Invalid JSON body"})

    if not isinstance(body, dict):
        return JSONResponse(
            status_code=400, content={"error": "Expected a JSON object"}
        )

    label = (body.get("label") or "").strip()[:MAX_SESSION_LABEL_LEN]
    db = request.app.state.db
    if not label:
        delete_session_label(db, session_id)
        return JSONResponse(
            status_code=200, content={"session_id": session_id, "label": None}
        )
    set_session_label(db, session_id, label)
    return JSONResponse(
        status_code=200, content={"session_id": session_id, "label": label}
    )
