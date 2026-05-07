"""GET /api/v1/tools — tool call records with aggregated stats."""
from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from tj.api.deps import require_api_key
from tj.utils.time_parse import parse_since

router = APIRouter(dependencies=[Depends(require_api_key)])


@router.get("/tools")
async def get_tools(
    request: Request,
    agent_id: str | None = None,
    since: str | None = None,
    tool_name: str | None = None,
) -> dict:
    db = request.app.state.db
    since_dt = parse_since(since) if since else None
    rows = db.get_tool_calls(agent_id, since_dt, tool_name)
    return {"tools": rows, "count": len(rows)}
