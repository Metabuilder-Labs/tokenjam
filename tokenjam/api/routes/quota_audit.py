"""GET /api/v1/quota-audit — server-side Opus quota audit (`tj quota-audit`).

`tj quota-audit` aggregates per-session token/model metadata (Opus sessions
whose structural shape is Sonnet-shaped) at a grain the read-only API shim
never exposed. So whenever `tj serve` holds the DuckDB write lock the CLI fell
back to the ApiBackend (which has no `.conn`) and refused to run — one of the
three most Claude-Code-relevant commands broken from the moment onboarding
installs the daemon. DuckDB permits only one writer OR many readers across
processes (a concurrent read-only connection alongside the serve write-lock
raises an IOException), so routing the compute through the daemon — which
already owns the direct connection — is the parity fix, mirroring
`/api/v1/context` (#63), `/api/v1/reuse/clusters` (#154) and `/api/v1/optimize`.

Returns `audit_to_dict(audit)` (the CLI reconstructs it via `audit_from_dict`)
plus the plan-tier `framing` block (reconstructed via `Framing(**framing)`), so
`tj quota-audit` renders identically through serve.
"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from tokenjam.api.deps import require_api_key
from tokenjam.core.framing import (
    WindowSummary,
    compute_framing,
    plan_determination_mix,
)
from tokenjam.core.optimize.analyzers.model_downgrade import audit_opus_quota
from tokenjam.core.optimize.types import audit_to_dict
from tokenjam.utils.time_parse import parse_since, utcnow

router = APIRouter()


@router.get("/quota-audit", dependencies=[Depends(require_api_key)])
def get_quota_audit(
    request: Request,
    since: str = Query("30d", description="Lookback window (e.g. 30d, 7d, 24h)."),
    agent_id: str | None = Query(None, alias="agent_id"),
) -> dict[str, Any]:
    """Run the Opus quota audit server-side and return it + framing."""
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
        # compute the audit from. Shouldn't happen for a real `tj serve`.
        raise HTTPException(
            status_code=503,
            detail="Server has no direct database connection.",
        )

    try:
        since_dt = parse_since(since)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid since: {exc}") from exc

    until_dt = utcnow()
    window_days = max(
        (until_dt - since_dt).total_seconds() / 86400.0, 1.0 / 86400.0
    )

    audit = audit_opus_quota(
        conn, since_dt, until_dt, agent_id, window_days, config=config,
    )
    payload = audit_to_dict(audit)

    # Plan-tier framing, computed exactly as cmd_quota_audit._framing_for does —
    # window-INDEPENDENT session mix for the pricing-mode decision (#177),
    # window-scoped totals — so the CLI renders the same units + qualifier
    # through serve as it does daemon-off.
    mix = plan_determination_mix(conn, agent_id)
    framing = compute_framing(
        config,
        WindowSummary(
            total_cost_usd=audit.actual_cost_usd,
            total_tokens=audit.opus_tokens,
            sessions=audit.opus_sessions,
            plan_tier_mix=mix,
        ),
    )
    payload["framing"] = framing.to_dict()
    return payload
