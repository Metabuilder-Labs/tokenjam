"""
GET /api/v1/reuse/clusters — the STORED Reuse finding + skeleton-ready data.

`tj report --reuse` renders a per-cluster planning skeleton, which needs both
the Reuse finding AND each cluster's planning-call completion text. Both come
from direct `spans` queries that DuckDB blocks while `tj serve` holds the write
lock, so the report errored out whenever the daemon was up (#154).

This is a *dedicated* endpoint (issue #154 Option B) rather than bolting the
skeleton text onto `/api/v1/optimize`: the per-cluster planning text can be many
KB, and we don't make every Overview poll pay for report-only data.

Like every other analyzer-consuming route, it does NOT run the analyzer: the
Reuse finding comes out of the stored report `core.optimize.report_store` keeps
warm (daemon boot / scheduled interval / user-pressed rescan). The only live
work here is `gather_planning_texts`, a plain span lookup for clusters the
stored finding already named — no analyzer, no full-corpus scan.

The payload is the STORED DICT served verbatim, plus the freshness envelope and
two report-only extras: `planning_texts` ({session_id: completion text or
null}) and `pricing_mode`. It is never re-serialized from a rehydrated object —
see `report_store.stored_report_dict` for why that distinction is load-bearing.
"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from tokenjam.api.deps import require_api_key
from tokenjam.core.export.reuse_report import gather_planning_texts
from tokenjam.core.optimize import report_store
from tokenjam.utils.time_parse import parse_since

router = APIRouter()


@router.get("/reuse/clusters", dependencies=[Depends(require_api_key)])
def get_reuse_clusters(
    request: Request,
    since: str = Query(
        "30d",
        description="Echoed back as requested_since. The finding comes from the "
                    "stored report; see window_days / scan_since / scan_until.",
    ),
    agent_id: str | None = Query(None, alias="agent_id"),
) -> dict[str, Any]:
    """Serve the stored Reuse finding + its skeleton text."""
    db = request.app.state.db
    config = request.app.state.config
    if db is None or config is None:
        raise HTTPException(
            status_code=503,
            detail="Server not fully initialised (db or config missing).",
        )

    try:
        parse_since(since)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid --since: {exc}") from exc

    envelope = report_store.stored_report_block(config)
    envelope["requested_since"] = since

    body = report_store.stored_report_dict(config)
    if body is None:
        # Cold store: no finding, and deliberately no empty-looking one. The
        # caller must say "not computed yet", not "no reuse clusters found".
        return {**envelope, "report_available": False,
                "planning_texts": {}, "pricing_mode": "api"}

    payload: dict[str, Any] = dict(body)
    payload.update(envelope)
    payload["report_available"] = True

    # Rehydrated ONLY to hand `gather_planning_texts` the typed finding it
    # expects. The payload above is the stored dict verbatim, so nothing a
    # rehydration might drop can reach the wire through this route.
    finding = report_store.stored_finding(config, "reuse")
    conn = getattr(db, "conn", None)
    # Skeleton text needs the DB; the daemon owns it here. `pricing_mode` is
    # always "api": the reuse report no longer differentiates its recoverable
    # figures by billing mode (product decision — dollars are always
    # legitimate regardless of subscription vs API billing). The field is
    # kept in the payload for shape compatibility with existing consumers.
    if finding is not None and finding.clusters and conn is not None:
        payload["planning_texts"] = gather_planning_texts(conn, finding)
    else:
        payload["planning_texts"] = {}
    payload["pricing_mode"] = "api"

    return payload
