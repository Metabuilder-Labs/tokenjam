"""GET /api/v1/version — package version, used by the UI footer.
GET /health — process liveness probe (alias for uptime tooling, no prefix)."""
from __future__ import annotations

from fastapi import APIRouter

from tokenjam import __version__

router = APIRouter()
health_router = APIRouter()


@router.get("/version")
async def get_version() -> dict:
    return {"version": __version__}


@health_router.get("/health")
async def get_health() -> dict:
    return {"status": "ok", "version": __version__}
