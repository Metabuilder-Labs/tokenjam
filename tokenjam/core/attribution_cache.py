"""Cheap on-disk hand-off of the top recurring-inclusion driver.

``tj context`` (``core/context_diagnostic``) already knows WHICH file, search,
prompt, or tool output is re-included most often across sessions — but
answering that needs a live DuckDB connection plus the capture-gated attribute
data. The statusline (``cli/cmd_statusline``) is the opposite: zero-token,
pure-stdlib, invoked after every turn, and must never open the DB or do
anything slower than a linear transcript scan.

``tj backfill claude-code`` (and `tj onboard`, which calls the same
``ingest_claude_code`` function directly) already holds that connection and
the ``[capture]`` flags, so it is the one process that computes the window's
top driver and hands it off here as a tiny JSON file. The statusline does a
plain stat+read of that file — no query, no live computation.
"""
from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path
from typing import Any

# Window the cached driver is computed over on each refresh.
ATTRIBUTION_WINDOW_DAYS = 30

# A cached driver older than this is presumed stale enough that the statusline
# should fall back to the plain badge rather than show it as current.
MAX_CACHE_AGE_SECONDS = 7 * 24 * 60 * 60


def _cache_path() -> Path:
    """Resolved lazily (not at import) so tests can patch ``Path.home``."""
    return Path.home() / ".local" / "share" / "tj" / "attribution_cache.json"


def write_attribution_cache(
    label: str, occurrences: int, sessions: int, *, path: Path | None = None
) -> None:
    """Persist the top recurring-inclusion driver. Best-effort; never raises."""
    from tokenjam.utils.time_parse import utcnow

    target = path or _cache_path()
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps({
            "top_label": label,
            "occurrences": occurrences,
            "sessions": sessions,
            "computed_at": utcnow().isoformat(),
        }))
    except Exception:  # noqa: BLE001 - a cache write must never break ingest
        pass


def read_attribution_cache(
    *, path: Path | None = None, max_age_seconds: int = MAX_CACHE_AGE_SECONDS
) -> dict[str, Any] | None:
    """Read the cached top driver, or ``None`` if missing/stale/corrupt.

    Fail-safe for the statusline hook: any error (missing file, malformed
    JSON, an aged-out entry) degrades to ``None`` rather than raising.
    """
    target = path or _cache_path()
    try:
        if not target.is_file():
            return None
        data = json.loads(target.read_text())
        if not isinstance(data, dict):
            return None
        label = data.get("top_label")
        occurrences = data.get("occurrences")
        if not label or not occurrences:
            return None
        computed_at = data.get("computed_at")
        if isinstance(computed_at, str):
            age = _age_seconds(computed_at)
            if age is not None and age > max_age_seconds:
                return None
        return data
    except Exception:  # noqa: BLE001 - fail-safe read for the statusline hook
        return None


def _age_seconds(computed_at: str) -> float | None:
    try:
        from datetime import datetime

        from tokenjam.utils.time_parse import utcnow

        ts = datetime.fromisoformat(computed_at)
        now = utcnow()
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=now.tzinfo)
        return (now - ts).total_seconds()
    except Exception:  # noqa: BLE001
        return None


def refresh_attribution_cache(
    conn: Any, capture: Any, *, path: Path | None = None
) -> None:
    """Compute the window's top recurring-inclusion driver and cache it.

    Called by ``ingest_claude_code`` after a backfill (the same path
    ``tj onboard`` uses), the one process that already holds a live DuckDB
    connection and the ``[capture]`` flags. Best-effort: any failure (empty
    window, capture off, query error) leaves any existing cache file
    untouched rather than raising or clobbering it with an empty result.
    """
    try:
        from tokenjam.core.context_diagnostic import compute_context_diagnostic
        from tokenjam.utils.time_parse import utcnow

        tool_inputs = bool(getattr(capture, "tool_inputs", False))
        prompts = bool(getattr(capture, "prompts", False))
        tool_outputs = bool(getattr(capture, "tool_outputs", False))
        if not (tool_inputs or prompts or tool_outputs):
            return

        until = utcnow()
        since = until - timedelta(days=ATTRIBUTION_WINDOW_DAYS)
        diag = compute_context_diagnostic(
            conn, since, until,
            tool_inputs_captured=tool_inputs,
            prompts_captured=prompts,
            tool_outputs_captured=tool_outputs,
        )
        if not diag.recurring:
            return
        top = diag.recurring[0]
        write_attribution_cache(
            _short_label(top), top.occurrences, top.sessions, path=path
        )
    except Exception:  # noqa: BLE001 - must never break the ingest it follows
        pass


def _short_label(inclusion: Any) -> str:
    """A terse display label for a recurring inclusion (basename for files)."""
    from tokenjam.core.context_diagnostic import INCLUSION_FILE_READ

    if inclusion.inclusion_type == INCLUSION_FILE_READ:
        return Path(inclusion.target).name or inclusion.target
    return inclusion.label[:40]
