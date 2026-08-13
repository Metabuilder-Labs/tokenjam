"""GET /api/v1/version — package version, used by the UI footer.
GET /health — liveness AND storage readiness probe (alias for uptime tooling,
no prefix)."""
from __future__ import annotations

import logging

from fastapi import APIRouter, Request, Response

from tokenjam import __version__

router = APIRouter()
health_router = APIRouter()

logger = logging.getLogger("tokenjam.api.health")


@router.get("/version")
async def get_version() -> dict:
    return {"version": __version__}


@health_router.get("/health")
async def get_health(request: Request, response: Response) -> dict:
    """Report whether this process can still serve, not merely whether it runs.

    This used to return a flat ``{"status": "ok"}`` that never touched storage.
    That is the failure this endpoint exists to catch: a DuckDB
    ``FatalException`` invalidates the whole database instance, so every route
    starts returning 500 and the dashboard renders empty while the process is
    perfectly alive — and a health probe that only proves the process is alive
    reports "ok" throughout. A status that cannot distinguish "serving" from
    "up but unable to read anything" is worse than no status at all.

    So the probe asks the database, and when it finds an invalidated one it
    tries to re-establish the connections in place (``core/db``'s
    ``recover_invalidated_database`` — closing every handle then reopening,
    which is the only in-process recovery DuckDB allows) and rebuilds the table
    whose index fault triggers it. Recovery is attempted here rather than only
    on a timer so that the very polling that notices the outage also ends it.
    If it cannot be recovered the response is **503** with the reason, so
    uptime tooling and the UI both learn the truth instead of a green tick.
    """
    from tokenjam.core.db import (
        fatal_db_error,
        recover_invalidated_database,
    )

    db = getattr(request.app.state, "db", None)
    probe = getattr(db, "check_health", None)
    if probe is None:
        # A backend without the probe (a test double, an HTTP-fallback shim)
        # cannot be asked, so report liveness only and say which it is rather
        # than asserting a storage state nothing checked.
        return {"status": "ok", "version": __version__, "storage": "unknown"}

    if probe():
        return {"status": "ok", "version": __version__, "storage": "ok"}

    reason = fatal_db_error() or "the database did not answer a health query"
    logger.error("health probe found the database unusable (%s); recovering", reason)
    if recover_invalidated_database() and probe():
        logger.warning("database recovered by the health probe")
        return {
            "status": "ok",
            "version": __version__,
            "storage": "recovered",
            "recovered_from": reason,
        }

    response.status_code = 503
    return {
        "status": "unhealthy",
        "version": __version__,
        "storage": "invalidated",
        "detail": reason,
        "remedy": "restart `tj serve`, then run `tj doctor --repair`",
    }
