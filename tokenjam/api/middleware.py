"""Ingest auth middleware — validates Bearer token on POST /api/v1/spans."""
from __future__ import annotations

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware


class IngestAuthMiddleware(BaseHTTPMiddleware):
    """
    Validates the ingest secret on POST /api/v1/spans.
    If security.ingest_secret is empty string, auth is disabled.
    Returns 401 with JSON error if secret is wrong or missing.
    """

    PROTECTED_PATHS = {"/api/v1/spans", "/v1/logs", "/v1/traces"}

    async def dispatch(self, request: Request, call_next):  # type: ignore[override]
        if request.method == "POST" and request.url.path in self.PROTECTED_PATHS:
            secret = request.app.state.config.security.ingest_secret
            if secret:
                auth = request.headers.get("Authorization", "")
                if not auth.startswith("Bearer ") or auth[7:] != secret:
                    return JSONResponse(
                        status_code=401,
                        content={"detail": "Invalid ingest secret"},
                    )
        return await call_next(request)
