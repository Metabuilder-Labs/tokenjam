"""`tj resume-brief` — hand a resuming session its prior method.

Reconstructs a compact brief (task · progress · what-was-tried/dead-ends ·
where-it-left-off · working files) from a session tj ALREADY persists, so a
continuing / post-compaction session resumes instead of re-investigating.

Source resolution (all fail-soft — a brief must never break a session):

  * ``--transcript PATH`` — build the brief live from an explicit JSONL.
  * ``--session ID`` — prefer the durable ``session_story`` snapshot
    (``core/method_capture``, survives Claude Code's transcript prune); fall
    back to the live transcript when no snapshot exists.
  * ``--last`` — the most recently active session: the newest transcript on
    disk (the session you are resuming / just compacted), else the newest
    persisted snapshot.

Deterministic, no LLM, out-of-band (zero in-loop token cost). Prints the brief
to stdout; prints NOTHING (exit 0) when there is nothing to brief, so a
SessionStart/PostCompact hook can pipe it straight into ``additionalContext``.
"""
from __future__ import annotations

import glob
from pathlib import Path
from typing import Any

import click

from tokenjam.core.method_capture import load_session_method
from tokenjam.core.resume_brief import build_resume_brief
from tokenjam.core.transcript import (
    _read_records,
    build_session_asks,
    build_session_story,
    resolve_projects_root,
)


def _live_transcript_path(session_id: str, projects_root: Path) -> Path | None:
    """Locate ``<projects_root>/*/<session_id>.jsonl`` on disk, or None."""
    if not session_id or not projects_root.exists():
        return None
    matches = sorted(glob.glob(str(projects_root / "*" / f"{glob.escape(session_id)}.jsonl")))
    return Path(matches[0]) if matches else None


def _from_transcript_path(path: Path) -> tuple[str, Any, Any, Any]:
    """Build (session_id, story, asks, records) from an explicit JSONL path.

    Derives the projects root from the file layout
    (``<root>/<project>/<session>.jsonl``) so the same Story machinery resolves it.
    """
    session_id = path.stem
    projects_root = path.parent.parent
    story = _safe(build_session_story, session_id, projects_root)
    asks = _safe(build_session_asks, session_id, projects_root)
    records = _read_records(path) if path.exists() else None
    return session_id, story, asks, records


def _safe(fn, session_id: str, projects_root: Path):
    """Call a Story builder, degrading any failure to None (fail-soft)."""
    try:
        return fn(session_id, projects_root=projects_root)
    except Exception:  # noqa: BLE001 - a brief must never break a session
        return None


def _load_for_session(
    db: Any, session_id: str, projects_root: Path
) -> tuple[Any, Any, Any]:
    """(story, asks, records) for a session id, snapshot-preferred.

    The durable snapshot survives the transcript prune, so it wins for
    story/asks. Interruption markers live only in the raw transcript, so
    ``records`` is read from the live file whenever it still exists.
    """
    path = _live_transcript_path(session_id, projects_root)
    records = _read_records(path) if path else None

    snapshot = None
    try:
        snapshot = load_session_method(db, session_id) if db is not None else None
    except Exception:  # noqa: BLE001 - fail-soft
        snapshot = None
    if isinstance(snapshot, dict) and (snapshot.get("story") or snapshot.get("asks")):
        return snapshot.get("story"), snapshot.get("asks"), records

    story = _safe(build_session_story, session_id, projects_root)
    asks = _safe(build_session_asks, session_id, projects_root)
    return story, asks, records


def _most_recent_transcript(projects_root: Path) -> str | None:
    """Session id of the most-recently-modified transcript under the root."""
    if not projects_root.exists():
        return None
    try:
        files = glob.glob(str(projects_root / "*" / "*.jsonl"))
    except OSError:
        return None
    if not files:
        return None
    newest = max(files, key=lambda f: _mtime(f))
    return Path(newest).stem


def _mtime(path: str) -> float:
    try:
        return Path(path).stat().st_mtime
    except OSError:
        return 0.0


def _latest_snapshot_session(db: Any) -> str | None:
    """Session id of the newest ``session_story`` snapshot, or None."""
    conn = getattr(db, "conn", None)
    if conn is None:
        return None
    try:
        row = conn.execute(
            "SELECT session_id FROM session_story "
            "ORDER BY captured_at DESC LIMIT 1"
        ).fetchone()
    except Exception:  # noqa: BLE001 - fail-soft (table absent / api mode)
        return None
    return row[0] if row and row[0] else None


@click.command("resume-brief")
@click.option("--session", "session_id", default=None,
              help="Session id to brief (snapshot-preferred, live fallback).")
@click.option("--last", is_flag=True, default=False,
              help="Brief the most recently active session.")
@click.option("--transcript", "transcript_path", default=None,
              help="Build the brief live from an explicit transcript JSONL path.")
@click.pass_context
def cmd_resume_brief(
    ctx: click.Context,
    session_id: str | None,
    last: bool,
    transcript_path: str | None,
) -> None:
    """Emit a compact resume brief for a session (out-of-band, fail-soft)."""
    if not (session_id or last or transcript_path):
        raise click.UsageError("Provide --session <id>, --last, or --transcript <path>.")

    db = ctx.obj.get("db")
    verbose = ctx.obj.get("verbose", False)
    projects_root = resolve_projects_root()

    sid = ""
    story = asks = records = None
    try:
        if transcript_path:
            sid, story, asks, records = _from_transcript_path(Path(transcript_path))
        elif session_id:
            sid = session_id
            story, asks, records = _load_for_session(db, session_id, projects_root)
        else:  # --last
            sid = _most_recent_transcript(projects_root) or _latest_snapshot_session(db) or ""
            if sid:
                story, asks, records = _load_for_session(db, sid, projects_root)

        brief = build_resume_brief(story, asks, session_id=sid, records=records)
    except Exception as exc:  # noqa: BLE001 - never break the caller / a session
        if verbose:
            click.echo(f"resume-brief: skipped ({exc})", err=True)
        return

    if brief:
        click.echo(brief)
    elif verbose:
        click.echo("resume-brief: nothing to brief (no method captured)", err=True)
