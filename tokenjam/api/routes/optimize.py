"""
GET /api/v1/optimize — server-side `tj optimize` execution.

Exists because `cmd_optimize` runs analyzers via direct `db.conn.execute(SQL)`
queries that DuckDB blocks when another process (tj serve) holds the write
lock. Routing optimize through the API lets the CLI work in the recommended
operating mode (daemon auto-started by tj onboard) — see issue #68 §12.

The endpoint runs `build_report` server-side using `app.state.db` (the same
connection that handles ingest) and returns the JSON-serialized report. The
CLI's `cmd_optimize` deserializes the response back into an `OptimizeReport`
via `report_from_dict` and feeds the rendering path as if it had built the
report locally — no second code path for rendering.
"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from tokenjam.api.deps import require_api_key
from tokenjam.core.framing import WindowSummary, compute_framing, plan_tier_mix
from tokenjam.core.optimize import build_report, report_to_dict
from tokenjam.utils.time_parse import parse_since, utcnow

router = APIRouter()


@router.get("/optimize", dependencies=[Depends(require_api_key)])
def get_optimize(
    request: Request,
    since: str = Query("30d", description="Lookback window (e.g. 30d, 7d, 24h)."),
    agent_id: str | None = Query(None, alias="agent_id"),
    finding: list[str] | None = Query(
        None,
        description="Optional list of analyzer names to run. Omit to run all.",
    ),
    budget_provider: str | None = Query(None),
    budget_usd: float | None = Query(None),
) -> dict[str, Any]:
    """
    Run the optimize analyzers server-side and return the serialized report.

    Mirrors the CLI `tj optimize` flags: --since, --agent, positional NAME args,
    --budget, --budget-usd. Returns the same dict shape `report_to_dict` produces
    locally, so the CLI can reconstruct an `OptimizeReport` and render it.
    """
    db = request.app.state.db
    config = request.app.state.config
    if db is None or config is None:
        raise HTTPException(
            status_code=503,
            detail="Server not fully initialised (db or config missing).",
        )

    try:
        since_dt = parse_since(since)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid --since: {exc}") from exc

    until_dt = utcnow()
    try:
        report = build_report(
            db=db,
            config=config,
            since=since_dt,
            until=until_dt,
            agent_id=agent_id,
            findings=list(finding) if finding else None,
            budget_provider_filter=budget_provider,
            budget_usd_override=budget_usd,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    payload = report_to_dict(report)

    # Plan-tier mix lets the CLI render subscription / local / unknown
    # framings correctly under daemon mode. Without this the CLI defaults
    # to "api" pricing_mode regardless of the user's actual plan (#68 §12
    # follow-up). Best-effort: depends on db.conn (DuckDBBackend); skip
    # silently if the daemon's storage layer doesn't expose a connection.
    conn = getattr(db, "conn", None)
    if conn is not None:
        try:
            payload["plan_tier_mix"] = plan_tier_mix(conn, since_dt, until_dt, agent_id)
        except Exception:
            payload["plan_tier_mix"] = {}
    else:
        payload["plan_tier_mix"] = {}

    # Plan-tier framing block (#110) — built from the report window + the
    # plan-tier mix above, so the local web UI frames recoverable-savings and
    # spend figures identically to the CLI.
    w = report.window
    payload["framing"] = compute_framing(
        config,
        WindowSummary(
            total_cost_usd=float(getattr(w, "total_cost_usd", 0.0) or 0.0),
            total_tokens=int(getattr(w, "total_tokens", 0) or 0),
            sessions=int(getattr(w, "sessions", 0) or 0),
            plan_tier_mix=payload["plan_tier_mix"],
        ),
    ).to_dict()

    return payload
