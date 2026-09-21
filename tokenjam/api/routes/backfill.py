"""POST /api/v1/backfill/claude-code: run the Claude Code transcript backfill
ON THE DAEMON, for a CLI that found `tj serve` holding the DuckDB write lock.

`tj backfill claude-code` needs bulk span writes, which the serve-mode HTTP
shim (`core/api_backend.py`) cannot carry; before this route existed the
command silently wrote nothing and then failed on the session write (issue
#770). The daemon already owns exactly this job as its scheduled catch-up
(`transcript_sync.start_catch_up`), so the route dispatches that on its own
thread with its own backend and returns at once; the CLI reports progress
through `tj backfill status`, which reads the same tables.

One on-demand pass at a time: a second request while one runs answers
`started: false, running: true` rather than parsing the tree twice.
Gated by the always-on ingest secret, like `POST /sessions/close`: the CLI holds it in its config and the read-side API key is off by default.
"""
from __future__ import annotations

import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse


router = APIRouter()

_RUNNING: threading.Thread | None = None
_LOCK = threading.Lock()


def _running() -> bool:
    return _RUNNING is not None and _RUNNING.is_alive()


# Gated by the always-on ingest secret (`IngestAuthMiddleware.PROTECTED_PATHS`),
# not the optional read-side API key: this is a write.
@router.post("/backfill/claude-code")
async def backfill_claude_code(request: Request) -> JSONResponse:
    """Body: ``{"since": <ISO 8601> | null, "root": <path> | null,
    "reingest": bool}``. Returns ``{"started": bool, "running": bool}``."""
    global _RUNNING

    try:
        body = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content={"error": "Invalid JSON body"})
    if not isinstance(body, dict):
        return JSONResponse(status_code=400, content={"error": "Expected a JSON object"})

    config = request.app.state.config
    db = getattr(request.app.state, "db", None)
    if config is None or db is None or getattr(db, "conn", None) is None:
        return JSONResponse(
            status_code=503, content={"error": "Server has no direct database connection."},
        )

    lookback: timedelta | None = None
    raw_since = body.get("since")
    if raw_since:
        try:
            since = datetime.fromisoformat(str(raw_since))
        except ValueError:
            return JSONResponse(status_code=400, content={"error": "since must be ISO 8601"})
        if since.tzinfo is None:
            since = since.replace(tzinfo=timezone.utc)
        lookback = max(datetime.now(tz=timezone.utc) - since, timedelta(0))
    root_raw = body.get("root")
    root = Path(str(root_raw)).expanduser() if root_raw else None
    reingest = bool(body.get("reingest", False))

    with _LOCK:
        if _running():
            return JSONResponse(status_code=200, content={"started": False, "running": True})
        from tokenjam.core import transcript_sync
        from tokenjam.core.db import DuckDBBackend

        _RUNNING = transcript_sync.start_catch_up(
            lambda: DuckDBBackend(config.storage), config=config, root=root,
            lookback=lookback, reingest=reingest,
        )
    payload: dict[str, Any] = {"started": True, "running": True}
    return JSONResponse(status_code=200, content=payload)
