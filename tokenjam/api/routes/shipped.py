"""GET /api/v1/shipped: what the window's sessions shipped, and what the
rest cost (shipped-value ledger, contracts §4).

Pure SQL over the ledger tables `core/shipped.match_sessions_to_commits`
keeps current on the daemon pass; no analyzer and no git runs here. The
figures are the same `shipped_summary` the `shipped` analyzer stores, so the
Dashboard card (which asks for its own window), `tj status` in serve mode
and the Optimize card cannot disagree on a definition. Every dollar is
MEASURED and travels with the `framing` block (contracts §1).
"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from tokenjam.api.deps import require_api_key
from tokenjam.core.framing import PERSONAS, WindowSummary, compute_framing, plan_determination_mix
from tokenjam.core.shipped import last_match, shipped_summary, start_match
from tokenjam.utils.time_parse import parse_since, utcnow

router = APIRouter()

#: Longest a caller may hold this request open waiting for the pass.
MATCH_MAX_WAIT_S = 60.0


# Gated by the always-on ingest secret (`IngestAuthMiddleware.PROTECTED_PATHS`),
# not the optional read-side API key: this is a write.
@router.post("/shipped/match")
def match_commits(
    request: Request,
    wait_s: float = Query(0.0, ge=0.0, le=MATCH_MAX_WAIT_S,
                          description="Seconds to wait for the pass before answering."),
) -> dict[str, Any]:
    """Run the session -> commit matcher on the daemon, for a CLI that found
    it holding the DuckDB lock (`tj optimize shipped`; issue #770, fix 4).

    The pass runs on its own thread with its own backend (`start_match`),
    never on this request's connection, and git never runs on the request
    thread; `wait_s` only decides how long the caller waits for it. One
    pass at a time: a request arriving mid-pass joins the running one.
    Answers ``{"started", "completed", "running", ...last result}``.
    """
    config = request.app.state.config
    db = getattr(request.app.state, "db", None)
    if config is None or db is None or getattr(db, "conn", None) is None:
        raise HTTPException(status_code=503, detail="Server has no direct database connection.")
    from tokenjam.core.db import DuckDBBackend

    thread, started = start_match(lambda: DuckDBBackend(config.storage), config)
    if wait_s > 0:
        thread.join(wait_s)
    running = thread.is_alive()
    payload: dict[str, Any] = {"started": started, "running": running, "completed": not running}
    if not running:
        payload.update(last_match())
    return payload


@router.get("/shipped", dependencies=[Depends(require_api_key)])
def get_shipped(
    request: Request,
    since: str = Query("30d", description="Lookback window (e.g. 30d, 7d, 24h)."),
    agent_id: str | None = Query(None, alias="agent_id"),
    persona: str | None = Query(None),
) -> dict[str, Any]:
    db = request.app.state.db
    config = request.app.state.config
    conn = getattr(db, "conn", None)
    if db is None or config is None or conn is None:
        raise HTTPException(status_code=503, detail="Server has no direct database connection.")
    if persona is not None and persona not in PERSONAS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown persona {persona!r}. Expected one of {sorted(PERSONAS)}.",
        )
    try:
        since_dt = parse_since(since)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid since: {exc}") from exc
    until_dt = utcnow()
    summary = shipped_summary(
        conn, since_dt, until_dt, agent_id=agent_id, persona_scope=persona,
    )
    payload = summary.to_dict()
    payload["since"] = since
    payload["window_start"] = since_dt.isoformat()
    payload["window_end"] = until_dt.isoformat()
    mix = plan_determination_mix(conn, agent_id)
    framing = compute_framing(
        config,
        WindowSummary(
            total_cost_usd=summary.cost_shipped_usd + summary.cost_unshipped_usd
            + summary.cost_committed_usd,
            sessions=summary.sessions_total,
            plan_tier_mix=mix,
        ),
    )
    payload["framing"] = framing.to_dict()
    return payload
