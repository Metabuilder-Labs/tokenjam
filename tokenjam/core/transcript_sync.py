"""
Continuous transcript ingestion: reconcile what Claude Code has on disk against
what tj has actually ingested, and close the gap on a schedule.

Before this module there were exactly two ingest paths, and neither ran on its
own:

  1. **Live OTLP** into ``tj serve`` — only captures a session when the shell
     that launched ``claude`` had ``CLAUDE_CODE_ENABLE_TELEMETRY`` +
     ``OTEL_EXPORTER_OTLP_ENDPOINT`` loaded AND the daemon was reachable at that
     exact endpoint at the moment each event fired. Claude Code's exporter has
     no retry and no buffer, so a miss is permanent.
  2. **``core.backfill.ingest_claude_code``** — on-demand CLI only (``tj
     onboard``, ``tj backfill claude-code``). Nothing scheduled it.

So completeness depended on a human remembering to run a CLI command, and every
session missed at ingest is lost for good once Claude Code rotates the
transcript (~30 days). That inverts the product's whole premise: tj is supposed
to retain what Claude discards.

This module supplies the missing third thing — a reconciliation the daemon can
run on startup and on an interval:

  * :func:`scan_disk_sessions` derives session ids from file **content**, never
    from the filename. This distinction is load-bearing: roughly half of the
    ``.jsonl`` files under ``~/.claude/projects`` live in nested ``subagents/``
    folders and carry their PARENT session's ``sessionId`` internally, so a
    filename-derived (or file-count-derived) comparison massively over-reports
    the gap. Two earlier measurements of this same gap disagreed by an order of
    magnitude for exactly that reason.
  * :func:`reconcile_claude_code` anti-joins those ids against the ``sessions``
    table and reports what is on disk but not ingested, with the age of the
    oldest missing session so the user can see it while the source still
    exists.
  * :func:`run_catch_up` re-runs the ordinary backfill over a bounded recent
    window. ``ingest_claude_code`` is already idempotent (span ids are
    deterministic hashes of ``(session_id, message id)`` and every run does a
    batched anti-join), so re-running it costs a re-parse and inserts only what
    is genuinely missing.

Sessions with no assistant turn parse to zero spans and are intentionally NOT
ingested — they carry no token usage. :func:`reconcile_claude_code` reports
those separately from the genuinely-missing ones so a clean install doesn't read
as permanently broken.
"""
from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


#: How many leading lines of a transcript to read while looking for the internal
#: ``sessionId``. Claude Code writes it on every record, so the first line is
#: normally enough; the allowance covers a leading summary/meta record or two.
_HEADER_SCAN_LINES = 40

#: Claude Code prunes its own transcripts at roughly this age. Once a file is
#: gone, an un-ingested session is unrecoverable — this is what makes an ingest
#: gap urgent rather than merely untidy.
CLAUDE_CODE_ROTATION_DAYS = 30


@dataclass(frozen=True)
class DiskSession:
    """One distinct on-disk session, keyed by its INTERNAL ``sessionId``."""

    session_id: str
    started_at: datetime | None
    project: str
    #: Every ``.jsonl`` file carrying this session id — the main thread plus any
    #: nested ``subagents/`` files, which share the parent's id. Kept so a
    #: caller can re-parse the session without a second walk of the tree.
    files: tuple[Path, ...] = ()

    @property
    def file_count(self) -> int:
        return len(self.files)

    def age_days(self, now: datetime | None = None) -> float | None:
        if self.started_at is None:
            return None
        ref = now or datetime.now(tz=timezone.utc)
        return (ref - self.started_at).total_seconds() / 86400.0


@dataclass
class SyncReport:
    """Result of comparing on-disk transcripts against ingested sessions."""

    root: Path
    since: datetime | None = None
    files_scanned: int = 0
    #: Files we could not derive a ``sessionId`` from (truncated/corrupt).
    files_unreadable: int = 0
    disk_sessions: int = 0
    ingested_sessions: int = 0
    #: On disk, not in the DB, and carries at least one billable assistant turn.
    missing: list[DiskSession] = field(default_factory=list)
    #: On disk, not in the DB, but parses to zero spans (no assistant turn) —
    #: intentionally skipped, not a defect.
    skipped_empty: list[DiskSession] = field(default_factory=list)

    @property
    def missing_count(self) -> int:
        return len(self.missing)

    @property
    def skipped_empty_count(self) -> int:
        return len(self.skipped_empty)

    @property
    def oldest_missing(self) -> DiskSession | None:
        dated = [s for s in self.missing if s.started_at is not None]
        if not dated:
            return None
        return min(dated, key=lambda s: s.started_at)  # type: ignore[arg-type,return-value]

    def days_until_rotation(self, now: datetime | None = None) -> float | None:
        """Days before the OLDEST missing session's transcript is pruned by
        Claude Code — i.e. how long the gap stays recoverable. ``None`` when
        nothing is missing or no timestamps were readable."""
        oldest = self.oldest_missing
        if oldest is None:
            return None
        age = oldest.age_days(now)
        if age is None:
            return None
        return CLAUDE_CODE_ROTATION_DAYS - age

    def to_dict(self) -> dict:
        return {
            "root": str(self.root),
            "since": self.since.isoformat() if self.since else None,
            "files_scanned": self.files_scanned,
            "files_unreadable": self.files_unreadable,
            "disk_sessions": self.disk_sessions,
            "ingested_sessions": self.ingested_sessions,
            "missing_count": self.missing_count,
            "skipped_empty_count": self.skipped_empty_count,
            "days_until_rotation": self.days_until_rotation(),
            "missing": [
                {
                    "session_id": s.session_id,
                    "project": s.project,
                    "started_at": s.started_at.isoformat() if s.started_at else None,
                    "file_count": s.file_count,
                }
                for s in self.missing
            ],
        }


def _parse_ts(value: object) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _header_fields(path: Path) -> tuple[str | None, datetime | None]:
    """Read the internal ``sessionId`` + first timestamp out of a transcript's
    leading records WITHOUT parsing the whole file.

    Deriving the id from content is the entire point: a nested
    ``subagents/agent-*.jsonl`` file has its own filename but its parent's
    ``sessionId``, so keying on the filename double-counts it as a separate
    session.
    """
    session_id: str | None = None
    started_at: datetime | None = None
    try:
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            for index, line in enumerate(fh):
                if index >= _HEADER_SCAN_LINES:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except (ValueError, TypeError):
                    continue
                if not isinstance(record, dict):
                    continue
                if session_id is None:
                    candidate = record.get("sessionId")
                    if isinstance(candidate, str) and candidate:
                        session_id = candidate
                if started_at is None:
                    started_at = _parse_ts(record.get("timestamp"))
                if session_id is not None and started_at is not None:
                    break
    except OSError as exc:
        logger.debug("could not read transcript header %s: %s", path, exc)
        return None, None
    return session_id, started_at


def scan_disk_sessions(
    root: Path, since: datetime | None = None,
) -> tuple[dict[str, DiskSession], int, int]:
    """Walk ``root`` and return ``(sessions_by_id, files_scanned, files_unreadable)``.

    Sessions are keyed by the internal ``sessionId`` read from file content, so
    the several files belonging to one session (main thread plus its
    ``subagents/`` children) collapse into a single entry.

    ``since`` applies the same cheap ``mtime`` pre-filter
    ``iter_claude_code_sessions`` uses, so a scheduled catch-up over a short
    window only opens the handful of files that actually changed.
    """
    sessions: dict[str, DiskSession] = {}
    scanned = 0
    unreadable = 0
    if not root.exists() or not root.is_dir():
        return sessions, scanned, unreadable

    for path in sorted(root.rglob("*.jsonl")):
        if since is not None:
            try:
                mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
            except OSError:
                continue
            if mtime < since:
                continue
        scanned += 1
        session_id, started_at = _header_fields(path)
        if not session_id:
            unreadable += 1
            continue
        try:
            project = path.relative_to(root).parts[0]
        except ValueError:  # pragma: no cover - defensive
            project = path.parent.name
        existing = sessions.get(session_id)
        if existing is None:
            sessions[session_id] = DiskSession(
                session_id=session_id, started_at=started_at, project=project,
                files=(path,),
            )
            continue
        # Keep the earliest timestamp across the session's files and append the
        # extra file rather than creating a second entry.
        earliest = existing.started_at
        if started_at is not None and (earliest is None or started_at < earliest):
            earliest = started_at
        sessions[session_id] = DiskSession(
            session_id=session_id,
            started_at=earliest,
            project=existing.project,
            files=existing.files + (path,),
        )
    return sessions, scanned, unreadable


def _ingested_session_ids(conn, session_ids: list[str]) -> set[str]:
    """Subset of ``session_ids`` already present in the ``sessions`` table.

    Chunked for the same reason ``backfill._existing_span_ids`` is: a large
    history can otherwise blow past DuckDB's bind-parameter ceiling.
    """
    found: set[str] = set()
    if not session_ids:
        return found
    chunk = 5000
    for start in range(0, len(session_ids), chunk):
        batch = session_ids[start:start + chunk]
        placeholders = ",".join(f"${i + 1}" for i in range(len(batch)))
        rows = conn.execute(
            f"SELECT session_id FROM sessions WHERE session_id IN ({placeholders})",
            batch,
        ).fetchall()
        found.update(row[0] for row in rows)
    return found


def _has_billable_turns(session: DiskSession) -> bool:
    """True when at least one of the session's files parses into >=1 span.

    A session that ended before its first model call has no usage to record;
    backfill deliberately drops it. Classifying those separately keeps the
    "missing" number honest — otherwise a healthy install reports a permanent
    phantom gap that no amount of ingesting can close.
    """
    from tokenjam.core.backfill import parse_claude_code_session

    for path in session.files:
        try:
            parsed = parse_claude_code_session(path)
        except Exception:  # a single bad file must not break the report
            continue
        if parsed is not None and parsed.spans:
            return True
    return False


def reconcile_claude_code(
    db,
    root: Path | None = None,
    since: datetime | None = None,
    classify_empty: bool = True,
) -> SyncReport:
    """Compare on-disk Claude Code sessions against ingested ones.

    ``classify_empty`` re-parses each candidate to split genuinely-missing
    sessions from ones that carry no assistant turn. It costs one parse per
    candidate (not per file on disk), so it stays cheap on a healthy install
    where the candidate set is small; pass ``False`` to skip it when you only
    need the raw anti-join.
    """
    from tokenjam.core.transcript import resolve_projects_root

    base = Path(root) if root is not None else resolve_projects_root()
    report = SyncReport(root=base, since=since)

    sessions, scanned, unreadable = scan_disk_sessions(base, since=since)
    report.files_scanned = scanned
    report.files_unreadable = unreadable
    report.disk_sessions = len(sessions)

    disk_ids = sorted(sessions)
    conn = getattr(db, "conn", None)
    if conn is not None:
        ingested = _ingested_session_ids(conn, disk_ids)
    else:
        # `tj serve` holds the DB write-lock, so the CLI got a connection-less
        # HTTP shim (`ApiBackend`). Route the anti-join through the daemon that
        # owns the connection rather than bailing to `0 already ingested` on a
        # fully-ingested install (#642).
        fetch = getattr(db, "fetch_ingested_session_ids", None)
        if fetch is None:
            logger.debug(
                "reconcile: backend exposes no connection and no HTTP shim; "
                "nothing to compare"
            )
            return report
        try:
            ingested = fetch(disk_ids)
        except Exception:
            logger.warning("reconcile: daemon ingested-id lookup failed", exc_info=True)
            return report
    report.ingested_sessions = len(ingested)

    candidates = [sessions[sid] for sid in disk_ids if sid not in ingested]
    if not classify_empty:
        report.missing = candidates
        return report

    # `scan_disk_sessions` already recorded each session's files, so the
    # classification re-parses ONLY the candidates — never a second walk of the
    # whole tree, which on a real history is the expensive part.
    for candidate in candidates:
        if _has_billable_turns(candidate):
            report.missing.append(candidate)
        else:
            report.skipped_empty.append(candidate)
    return report


def run_catch_up(
    db,
    config=None,
    root: Path | None = None,
    lookback: timedelta | None = None,
):
    """Re-run the Claude Code backfill over a bounded recent window.

    This is the scheduled counterpart to ``tj backfill claude-code``. It exists
    because the live OTLP path drops any session whose shell lacked the telemetry
    env vars or whose daemon was down/unreachable, and nothing else ever notices.
    Backfill is idempotent, so the only cost of an overlapping re-run is
    re-parsing files it has already seen.

    ``lookback=None`` means no window at all (a full history pass) — used by
    ``tj backfill claude-code``, not by the scheduled job.
    """
    from tokenjam.core.backfill import ingest_claude_code
    from tokenjam.core.transcript import resolve_projects_root

    base = Path(root) if root is not None else resolve_projects_root()
    since = None
    if lookback is not None:
        since = datetime.now(tz=timezone.utc) - lookback
    return ingest_claude_code(db, root=base, since=since, config=config)


def sweep_stale_active_sessions(
    db,
    root: Path | str | None = None,
    idle_threshold: timedelta | None = None,
) -> int:
    """Correct raw ``status='active'`` rows that are actually dead zombies.

    ``SessionRecord.status_at`` / ``effective_status`` already COMPUTE a stale
    'active' row as "stale" at read time once its last activity is older than
    ``idle_threshold``, but nothing ever writes that back — so every consumer
    that filters the RAW ``status`` column instead of the computed one (MCP
    tools, ``tj status``, the sessions route, ``relearn_apply``) keeps counting
    a zombie as active forever. A common cause: a single dropped OTLP span
    creates a session row that defaults to 'active' and never receives a
    closing span or an explicit ``/sessions/close`` — nothing else ever
    reclassifies it.

    Reuses the SAME staleness definitions already in play elsewhere — no third
    one is introduced here:
      - ``SESSION_IDLE_THRESHOLD`` (or an explicit override) for "how old is
        too old", identical to ``effective_status``'s default window.
      - ``SessionRecord.status_with_transcript_mtime`` for the live-rescue
        check, the exact function the read-time ``/status`` route already
        calls (``_live_status``) — a Claude Code session whose on-disk
        transcript is still being appended to is left alone even if its
        spans look stale, matching FIX 1's mtime-based decision in
        ``session_record_from_parsed``.

    Conservative by construction: a row is corrected ONLY when both signals
    (span recency AND, for CC sessions, transcript mtime) agree it has gone
    fully quiet. Marks corrected rows 'completed' (never 'closed' — nobody
    explicitly closed them) and never touches ``ended_at``, tokens, or cost —
    status only. Returns the number of rows corrected.
    """
    from tokenjam.core.models import SESSION_IDLE_THRESHOLD, SessionRecord
    from tokenjam.core.transcript import resolve_projects_root, session_transcript_mtime

    conn = getattr(db, "conn", None)
    if conn is None:
        return 0
    threshold = idle_threshold or SESSION_IDLE_THRESHOLD
    projects_root = resolve_projects_root(root)

    rows = conn.execute(
        "SELECT session_id, started_at, ended_at FROM sessions WHERE status = 'active'"
    ).fetchall()

    stale_ids: list[str] = []
    for session_id, started_at, ended_at in rows:
        probe = SessionRecord(
            session_id=session_id, agent_id="", started_at=started_at,
            ended_at=ended_at, status="active",
        )
        mtime = session_transcript_mtime(session_id, projects_root)
        if probe.status_with_transcript_mtime(mtime, threshold) == "stale":
            stale_ids.append(session_id)

    mark = getattr(db, "mark_sessions_completed", None)
    if stale_ids and mark is not None:
        mark(stale_ids)
    return len(stale_ids)


def start_catch_up(
    db_factory,
    config=None,
    root: Path | None = None,
    lookback: timedelta | None = None,
    on_done=None,
) -> threading.Thread:
    """Run :func:`run_catch_up` on a daemon thread with its OWN backend.

    A separate connection per run is the same discipline the retention / relearn
    / cost-proposal jobs already follow in ``cmd_serve`` — it keeps the ingest
    off the request path and away from uvicorn's worker threads. Errors are
    logged, never raised: a failed catch-up must not take the daemon down.

    Folds the zombie-sweep (:func:`sweep_stale_active_sessions`) into this same
    scheduled job so it runs on the same cadence, the same connection, and the
    same thread as the ingest catch-up — no separate scheduler entry needed.
    """

    def _run() -> None:
        backend = None
        try:
            backend = db_factory()
            result = run_catch_up(backend, config=config, root=root, lookback=lookback)
            if result.sessions_new:
                logger.info(
                    "transcript catch-up ingested %d new session(s), %d span(s)",
                    result.sessions_new, result.spans_ingested,
                )
            try:
                swept = sweep_stale_active_sessions(backend, root=root)
                if swept:
                    logger.info(
                        "transcript catch-up corrected %d stale-active session(s)",
                        swept,
                    )
            except Exception:
                # Best-effort: a sweep failure must not affect the ingest
                # result callers already received, nor take the daemon down.
                logger.warning("stale-active session sweep failed", exc_info=True)
            if on_done is not None:
                on_done(result)
        except Exception:
            logger.warning("transcript catch-up failed", exc_info=True)
        finally:
            if backend is not None:
                try:
                    backend.close()
                except Exception:
                    pass

    thread = threading.Thread(target=_run, name="tj-transcript-catchup", daemon=True)
    thread.start()
    return thread
