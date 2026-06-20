"""
GET /api/v1/reuse/clusters — server-side Reuse analyzer + skeleton-ready data.

`tj report --reuse` renders a per-cluster planning skeleton, which needs both
the Reuse finding AND each cluster's planning-call completion text. Both come
from direct `spans` queries that DuckDB blocks while `tj serve` holds the write
lock, so the report errored out whenever the daemon was up (#154).

This is a *dedicated* endpoint (issue #154 Option B) rather than bolting the
skeleton text onto `/api/v1/optimize`: the per-cluster planning text can be many
KB, and the Overview polls `/optimize` every 30s — we don't make every poll pay
for report-only data. This endpoint is hit only when a report is generated.

Returns `report_to_dict(report)` (so the CLI reconstructs the finding via the
existing `report_from_dict`) plus two report-only extras: `planning_texts`
({session_id: completion text or null}) and `pricing_mode`.
"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from tokenjam.api.deps import require_api_key
from tokenjam.core.export.reuse_report import gather_planning_texts
from tokenjam.core.framing import dominant_plan, plan_tier_mix, pricing_mode_for
from tokenjam.core.optimize import build_report, report_to_dict
from tokenjam.utils.time_parse import parse_since, utcnow

router = APIRouter()


@router.get("/reuse/clusters", dependencies=[Depends(require_api_key)])
def get_reuse_clusters(
    request: Request,
    since: str = Query("30d", description="Lookback window (e.g. 30d, 7d, 24h)."),
    agent_id: str | None = Query(None, alias="agent_id"),
) -> dict[str, Any]:
    """Run the Reuse analyzer server-side and return the finding + skeleton text."""
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
            db=db, config=config, since=since_dt, until=until_dt,
            agent_id=agent_id, findings=["reuse"],
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    payload = report_to_dict(report)

    finding = report.findings.get("reuse")
    conn = getattr(db, "conn", None)
    # Skeleton text + pricing mode both need the DB; the daemon owns it here.
    if finding is not None and finding.clusters and conn is not None:
        payload["planning_texts"] = gather_planning_texts(conn, finding)
        payload["pricing_mode"] = pricing_mode_for(
            dominant_plan(plan_tier_mix(conn, since_dt, until_dt, agent_id))
        )
    else:
        payload["planning_texts"] = {}
        payload["pricing_mode"] = "unknown"

    return payload
