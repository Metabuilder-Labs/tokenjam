"""GET /api/v1/sessions — session enumeration (one row per session).

Unlike /status (which collapses to one record per agent via
`ORDER BY started_at DESC LIMIT 1`), this route returns every matching
session, so a single agent running multiple concurrent sessions is fully
represented. The MCP `list_active_sessions` tool proxies here in serve mode
so its output matches the direct-DB path (#35).
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from tokenjam.api.deps import require_api_key

router = APIRouter(dependencies=[Depends(require_api_key)])


@router.get("/sessions")
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
