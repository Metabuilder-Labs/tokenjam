"""Persist a snapshot of a session's reconstructed Story (its *method*) to the DB.

`/story` and `/workmap` rebuild a session's Story (the recursive narration +
subagent subtree, `core/transcript.py`) from the on-disk Claude Code JSONL
transcript on *every* request, and nothing writes it down. Claude Code PRUNES
those transcripts, so once the file is gone an ephemeral (killed) agent's method
is lost — the cost spans survive in the DB but the *how* does not.

This module captures the Story into the `session_story` table (migration 14) at
session close so it outlives the prune, and reads it back as a read-through
fallback when the live transcript is gone.

Pure-ish core module: it reads files (via `core/transcript.py`) and writes ONE
row; no analysis, no interpretation. It imports only `core`/`utils` — never
`tokenjam.api` or `tokenjam.cli`. Capture is **best-effort and never raises**
into its caller (a missing/pruned/malformed transcript logs a warning and
no-ops), so wiring it into the close path can never break a close.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from tokenjam.core.transcript import build_session_asks, build_session_story
from tokenjam.utils.time_parse import utcnow

logger = logging.getLogger(__name__)

#: Bumped only when the snapshot payload shape (the value persisted in
#: ``story_json``) changes, so a reader can tell an old snapshot from a new one.
SNAPSHOT_SCHEMA_VERSION = 1


def capture_session_method(
    db: Any,
    session_id: str,
    *,
    projects_dir: Path | str | None = None,
    source: str = "live-transcript",
) -> bool:
    """Snapshot a session's reconstructed Story into ``session_story``.

    Builds both the flat/recursive Story (``build_session_story``) and the
    ask-segmented story (``build_session_asks``) from the on-disk transcript and
    upserts ONE row keyed by ``session_id`` (DuckDB has no portable UPSERT here,
    so this deletes any prior snapshot then inserts the fresh one — idempotent,
    so a re-capture overwrites with the latest/fuller read). ``captured_at`` is
    stamped via ``utcnow()`` (Critical Rule 9).

    Best-effort: returns ``True`` only when a snapshot was actually written;
    returns ``False`` (logging a warning/debug) when there is no transcript/story
    to capture or anything fails. NEVER raises — close/ingest must not break on a
    capture failure.
    """
    conn = getattr(db, "conn", None)
    if conn is None or not session_id:
        return False

    try:
        story = build_session_story(
            session_id, projects_root=projects_dir, include_subagents=True
        )
        asks = build_session_asks(
            session_id, projects_root=projects_dir, include_subagents=True
        )
    except Exception as exc:  # noqa: BLE001 - best-effort, must not raise into caller
        logger.warning(
            "method capture: failed to build story for %s: %s", session_id, exc
        )
        return False

    # Both None == no on-disk transcript (SDK session, or already pruned).
    if story is None and asks is None:
        logger.debug(
            "method capture: no transcript for session %s; nothing to snapshot",
            session_id,
        )
        return False

    snapshot = {"story": story, "asks": asks}
    try:
        payload = json.dumps(snapshot)
        captured_at = utcnow()
        # Parameterised DELETE + INSERT (Critical Rule 7: no f-string SQL).
        conn.execute(
            "DELETE FROM session_story WHERE session_id = $1", [session_id]
        )
        conn.execute(
            "INSERT INTO session_story "
            "(session_id, story_json, captured_at, source, schema_version) "
            "VALUES ($1, $2, $3, $4, $5)",
            [session_id, payload, captured_at, source, SNAPSHOT_SCHEMA_VERSION],
        )
    except Exception as exc:  # noqa: BLE001 - best-effort, must not raise into caller
        logger.warning(
            "method capture: failed to persist snapshot for %s: %s", session_id, exc
        )
        return False
    return True


def load_session_method(db: Any, session_id: str) -> dict | None:
    """Return the persisted Story snapshot for ``session_id``, or ``None``.

    Reads the ``session_story`` row and parses ``story_json`` back into the
    snapshot dict (``{"story": <build_session_story>, "asks": <build_session_asks>}``,
    either value possibly ``None``). Returns ``None`` when no snapshot exists.
    """
    conn = getattr(db, "conn", None)
    if conn is None or not session_id:
        return None

    row = conn.execute(
        "SELECT story_json FROM session_story WHERE session_id = $1",
        [session_id],
    ).fetchone()
    if not row or row[0] is None:
        return None

    raw = row[0]
    # DuckDB returns a JSON column as a text string; parse it back to a dict.
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            logger.warning(
                "method capture: corrupt snapshot json for %s; ignoring", session_id
            )
            return None
    return raw if isinstance(raw, dict) else None
