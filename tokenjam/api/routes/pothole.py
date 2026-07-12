"""GET/POST /api/v1/pothole/* — the self-improve loop's pothole review inbox.

Serves the (expensive, full-corpus) pothole-detector result from the on-disk
cache the serve-time background job keeps warm (``core.optimize.
pothole_store``) — this route NEVER computes the finding inline on a request,
which would block the UI for the tens of seconds a full local corpus scan
takes. ``POST /refresh`` kicks a background recompute on a fresh DuckDB
connection (the retention-job pattern from ``cli/cmd_serve.py``) so it never
contends with the live request connection's write lock.

Phase 1 (detect + surface): read-only. There is deliberately no
apply/write route here yet — Approve lands in a later phase.
"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request

from tokenjam.api.deps import require_api_key
from tokenjam.core.optimize import pothole_store

router = APIRouter()


@router.get("/pothole/proposals", dependencies=[Depends(require_api_key)])
def get_pothole_proposals(request: Request) -> dict[str, Any]:
    """Cached pothole-detector proposals for the Review inbox.

    Returns ``{"status": "ready"|"computing"|"never_run", "computed_at":
    iso|null, "finding": <PotholeFinding dict>|null}``. A fresh install (no
    background pass has completed yet, and none is running) reports
    ``"never_run"`` — the inbox renders its empty state for that, not an
    error. ``"computing"`` means a recompute is in flight right now; the
    ``finding``/``computed_at`` fields (when present) are still the last
    GOOD result, so the UI can keep showing it while a refresh runs.
    """
    cached = pothole_store.read_cache()
    computing = pothole_store.is_computing()
    if cached is None:
        return {
            "status": "computing" if computing else "never_run",
            "computed_at": None,
            "finding": None,
        }
    return {
        "status": "computing" if computing else "ready",
        "computed_at": cached.get("computed_at"),
        "finding": cached.get("finding"),
    }


@router.post("/pothole/refresh", dependencies=[Depends(require_api_key)])
def refresh_pothole_proposals(request: Request) -> dict[str, Any]:
    """Kick a background recompute. A recompute already in flight is a no-op
    (returns ``already_running``) — never queued twice."""
    config = request.app.state.config
    if config is None:
        raise HTTPException(status_code=503, detail="Server not fully initialised.")

    from tokenjam.core.db import DuckDBBackend

    started = pothole_store.trigger_background_recompute(
        lambda: DuckDBBackend(config.storage)
    )
    return {"status": "started" if started else "already_running"}
