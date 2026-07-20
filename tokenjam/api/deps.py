"""Shared FastAPI dependencies."""
from __future__ import annotations

import secrets
from urllib.parse import urlsplit

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


# --- Unconditional local-write auth (independent of `api.auth.enabled`) -------
#
# `require_api_key` above is a NO-OP by default (`api.auth.enabled` defaults to
# False) -- fine for read-only telemetry routes, but the self-improve loop's
# mutating relearn endpoints (apply/enable/disable/revert/refresh) write files
# and `git commit` on the caller's behalf. On a default local install that
# would otherwise leave them wide open to any local process, or a browser CSRF
# POST to `http://127.0.0.1:<port>/...` from an unrelated page the user has
# open. `require_relearn_write_auth` is a SEPARATE dependency those routes add
# in addition to `require_api_key` -- it never consults `api.auth.enabled` and
# is always enforced.
#
# Mechanism: a random per-process token (`app.state.relearn_write_token`,
# minted once in `create_app`) is embedded ONLY in the same-origin-served UI
# HTML (`<meta name="tj-write-token">`, see `api/app.py::_serve_ui`) and sent
# back by the UI's JS as the `X-TJ-Local-Token` header on every write call. A
# cross-origin page can't read that meta tag (the CORS policy blocks reading
# another origin's response body) and can't set a custom header via a plain
# `<form>` POST (the classic no-preflight CSRF vector), so it can neither
# steal nor forge the token. A same-origin `Origin` check is layered on top as
# defense-in-depth for browsers that do send it.

_LOCAL_HOSTNAMES = {"localhost", "127.0.0.1", "::1"}


def _same_origin(origin_header: str, request: Request) -> bool:
    """True if ``origin_header`` (the request's ``Origin``) names the same
    host:port this server itself is listening on. ``localhost``/``127.0.0.1``/
    ``::1`` are treated as interchangeable (all mean "this machine")."""
    try:
        parsed = urlsplit(origin_header)
    except ValueError:
        return False
    if not parsed.hostname:
        return False
    origin_host = parsed.hostname.lower()
    req_host = (request.url.hostname or "").lower()
    host_ok = origin_host == req_host or (
        origin_host in _LOCAL_HOSTNAMES and req_host in _LOCAL_HOSTNAMES
    )
    return host_ok and parsed.port == request.url.port


def require_relearn_write_auth(request: Request) -> None:
    """Always-on guard for the relearn write endpoints — independent of
    ``config.api.auth.enabled``. Raises 403 on a cross-origin ``Origin``, then
    401 unless ``X-TJ-Local-Token`` matches this server process's
    ``app.state.relearn_write_token``. A request with neither header is
    refused too — there is no bypass-by-omission."""
    origin = request.headers.get("origin")
    if origin and not _same_origin(origin, request):
        raise HTTPException(status_code=403, detail="cross-origin request refused")

    token = getattr(request.app.state, "relearn_write_token", None)
    supplied = request.headers.get("x-tj-local-token")
    if not token or not supplied or not secrets.compare_digest(supplied, token):
        raise HTTPException(status_code=401, detail="missing or invalid local write token")
