"""GET /api/v1/policy/* — enforcement-plane read surfaces (#223).

Powers the MCP policy tools (and any UI) when ``tj serve`` holds the DB write
lock. All three are read-only and SUGGEST-MODE: they report what the gated
engine recorded, never enforce, and carry the ``unvalidated`` label. The savings
figure is estimated-recoverable / would-have-saved — never "saved".
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from tokenjam.api.deps import require_api_key
from tokenjam.proxy.audit import reconcile_savings
from tokenjam.proxy.recommend import policy_status, suggest_policies
from tokenjam.utils.time_parse import parse_since

router = APIRouter(dependencies=[Depends(require_api_key)])


@router.get("/policy/status")
def get_policy_status(request: Request, limit: int = 20) -> dict:
    db = request.app.state.db
    config = request.app.state.config
    return policy_status(db, config, limit=limit)


@router.get("/policy/savings")
def get_policy_savings(request: Request, since: str | None = None,
                       provider: str | None = None) -> dict:
    db = request.app.state.db
    since_dt = parse_since(since) if since else None
    return reconcile_savings(db, since=since_dt, provider=provider).to_dict()


@router.get("/policy/suggestions")
def get_policy_suggestions(request: Request) -> dict:
    db = request.app.state.db
    config = request.app.state.config
    return suggest_policies(db, config)
