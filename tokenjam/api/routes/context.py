"""
GET /api/v1/context — server-side context-cost diagnostic (`tj context`).

`tj context` is the launch-hero command (issue #4 wedge), but the diagnostic
reads the raw `attributes` column for recurring-inclusion detection — data the
API shim never exposed. So whenever `tj serve` held the DuckDB write lock the
CLI fell back to the ApiBackend (which has no `.conn`) and refused to run,
telling the user to stop the daemon on the exact command the launch drives them
to (#63). DuckDB permits only one writer OR many readers across processes (a
concurrent read-only connection alongside the serve write-lock raises an
IOException), so routing the compute through the daemon — which already owns the
direct connection — is the parity fix, mirroring `/api/v1/reuse/clusters` (#154)
and `/api/v1/optimize` (#68).

Returns `diagnostic_to_dict(diag)` (the CLI reconstructs it via
`diagnostic_from_dict`) plus the plan-tier `framing` block (reconstructed via
`Framing(**framing)`), so `tj context` renders identically through serve.
"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from tokenjam.api.deps import require_api_key
from tokenjam.core.context_diagnostic import (
    compute_context_diagnostic,
    diagnostic_to_dict,
)
from tokenjam.core.framing import (
    WindowSummary,
    compute_framing,
    plan_determination_mix,
)
from tokenjam.utils.time_parse import parse_since, utcnow

router = APIRouter()


@router.get("/context", dependencies=[Depends(require_api_key)])
def get_context(
    request: Request,
    since: str = Query("30d", description="Lookback window (e.g. 30d, 7d, 24h)."),
    agent_id: str | None = Query(None, alias="agent_id"),
) -> dict[str, Any]:
    """Run the context-cost diagnostic server-side and return it + framing."""
    db = request.app.state.db
    config = request.app.state.config
    if db is None or config is None:
        raise HTTPException(
            status_code=503,
            detail="Server not fully initialised (db or config missing).",
        )

    conn = getattr(db, "conn", None)
    if conn is None:
        # The daemon itself is in API-shim mode (no direct DB) — nothing to
        # compute the diagnostic from. Shouldn't happen for a real `tj serve`.
        raise HTTPException(
            status_code=503,
            detail="Server has no direct database connection.",
        )

    try:
        since_dt = parse_since(since)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid since: {exc}") from exc

    until_dt = utcnow()

    # Capture flags come from the daemon's own config — the same source the CLI
    # reads on the direct-connection path (cmd_context), so recurring-inclusion
    # detection is gated identically whether the daemon is up or not.
    capture = getattr(config, "capture", None)
    diag = compute_context_diagnostic(
        conn,
        since_dt,
        until_dt,
        agent_id=agent_id,
        tool_inputs_captured=bool(getattr(capture, "tool_inputs", False)),
        prompts_captured=bool(getattr(capture, "prompts", False)),
        tool_outputs_captured=bool(getattr(capture, "tool_outputs", False)),
    )

    payload = diagnostic_to_dict(diag)

    # Plan-tier framing, computed exactly as cmd_context._framing_for does —
    # window-INDEPENDENT session mix for the pricing-mode decision (#177),
    # window-scoped totals — so the CLI renders the same units + qualifier
    # through serve as it does daemon-off.
    mix = plan_determination_mix(conn, agent_id)
    framing = compute_framing(
        config,
        WindowSummary(
            total_cost_usd=diag.total_cost_usd,
            total_tokens=diag.total_tokens,
            sessions=diag.sessions,
            plan_tier_mix=mix,
        ),
    )
    payload["framing"] = framing.to_dict()
    return payload
