"""GET /api/v1/recommendations — the recommendation-outcome ledger.

Surfaces which recommendations tokenjam observed being acted on (a summarize
apply, a routing-config export) and, for the downsize analyzer, the post-hoc
adoption verdict with a MEASURED delta — kept strictly separate from the
estimated-recoverable figure (honesty discipline, Rule 14).

The daemon owns the DuckDB connection, so this route is the natural home for
adoption detection: it runs :func:`detect_downsize_adoption` server-side (which
appends any newly-ripe verdicts to the shared on-disk sink) before returning the
aggregated ledger, so Lens sees fresh verdicts without a separate trigger.
"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Request

from tokenjam.api.deps import require_api_key
from tokenjam.core.recommendations import (
    detect_downsize_adoption,
    read_outcomes,
    summarize_outcomes,
)

router = APIRouter(dependencies=[Depends(require_api_key)])


@router.get("/recommendations")
def get_recommendations(request: Request) -> dict[str, Any]:
    """Return the aggregated recommendation-outcome ledger.

    Best-effort runs adoption detection first (the daemon holds the connection);
    a detection failure never blocks the read — the already-recorded ledger is
    still returned.
    """
    db = request.app.state.db
    config = request.app.state.config

    conn = getattr(db, "conn", None)
    if conn is not None:
        try:
            detect_downsize_adoption(conn, config)
        except Exception:  # noqa: BLE001 — detection is advisory; never 500 the read
            pass

    return summarize_outcomes(read_outcomes(config))
