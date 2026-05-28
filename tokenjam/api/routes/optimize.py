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

    Mirrors the CLI `tj optimize` flags: --since, --agent, --finding (repeatable),
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

    try:
        report = build_report(
            db=db,
            config=config,
            since=since_dt,
            until=utcnow(),
            agent_id=agent_id,
            findings=list(finding) if finding else None,
            budget_provider_filter=budget_provider,
            budget_usd_override=budget_usd,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return report_to_dict(report)
