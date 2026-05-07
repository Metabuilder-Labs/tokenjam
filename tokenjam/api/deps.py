"""Shared FastAPI dependencies."""
from __future__ import annotations

from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

security = HTTPBearer(auto_error=False)


def require_api_key(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(security),
) -> None:
    config = request.app.state.config
    if not config.api.auth.enabled:
        return
    if credentials is None or credentials.credentials != config.api.auth.api_key:
        raise HTTPException(status_code=401, detail="Invalid API key")
