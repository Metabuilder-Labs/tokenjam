"""Ingest auth middleware: validates the ingest Bearer token on the write routes."""
from __future__ import annotations

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware


class IngestAuthMiddleware(BaseHTTPMiddleware):
    """
    Validates the ingest secret on every POST in `PROTECTED_PATHS`.
    If security.ingest_secret is empty string, auth is disabled.
    Returns 401 with JSON error if secret is wrong or missing.
    """

    PROTECTED_PATHS = {
        "/api/v1/spans", "/api/v1/sessions/close", "/v1/logs", "/v1/traces",
        # The three write-shaped routes a CLI without the DuckDB lock calls
        # (issue #770): a session write, a daemon-run backfill, a daemon-run
        # matcher pass. `require_api_key` is off by default, so these take
        # the always-on ingest secret, which the CLI holds in its config and
        # `ApiBackend` sends on exactly these posts.
        "/api/v1/sessions/upsert", "/api/v1/backfill/claude-code", "/api/v1/shipped/match",
    }

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
