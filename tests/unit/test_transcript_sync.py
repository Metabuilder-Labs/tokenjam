"""Unit tests for continuous transcript ingestion (reconcile + catch-up).

The behaviour under test is the one that was structurally absent before: a
transcript sitting on disk that the DB has never seen must be *detected*, and a
scheduled catch-up must close the gap without duplicating anything.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from click.testing import CliRunner

from tokenjam.cli import cmd_backfill as cmd_backfill_module
from tokenjam.core.backfill import ingest_claude_code
from tokenjam.core.db import DuckDBBackend
from tokenjam.core.config import StorageConfig
from tokenjam.core.transcript_sync import (
    CLAUDE_CODE_ROTATION_DAYS,
    reconcile_claude_code,
    run_catch_up,
    scan_disk_sessions,
)


def _assistant_record(session_id: str, cwd: str, uuid: str, message_id: str,
                      timestamp: str) -> dict:
    return {
        "type": "assistant",
        "uuid": uuid,
        "sessionId": session_id,
        "cwd": cwd,
        "timestamp": timestamp,
        "message": {
            "id": message_id,
            "model": "claude-sonnet-4-5-20250929",
            "content": [{"type": "text", "text": "ok"}],
            "usage": {
                "input_tokens": 100,
                "output_tokens": 20,
                "cache_read_input_tokens": 0,
                "cache_creation_input_tokens": 0,
            },
        },
    }


def _write_transcript(root: Path, project: str, filename: str,
                      records: list[dict]) -> Path:
    project_dir = root / project
    project_dir.mkdir(parents=True, exist_ok=True)
    path = project_dir / f"{filename}.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in records))
    return path


# The default "recent" timestamp is a time bomb if pinned to an absolute
# literal: `run_catch_up`'s lookback window is computed from wall-clock
# now(), so a fixed literal eventually falls outside any lookback window and
# every test relying on the default starts failing with no code change
# involved (see the `test_onboard_backfill_scope.py` fix for the same class
# of bug). Anchor it relative to test-execution time instead.
_DEFAULT_RECENT_TS = (
    datetime.now(tz=timezone.utc) - timedelta(hours=2)
).strftime("%Y-%m-%dT%H:%M:%S.000Z")


def _write_session(root: Path, session_id: str, project: str = "-tmp-proj",
                   ts: str = _DEFAULT_RECENT_TS,
                   filename: str | None = None) -> Path:
    return _write_transcript(
        root, project, filename or session_id,
        [_assistant_record(session_id, "/tmp/proj", f"{session_id}-u1",
                           f"msg_{session_id}", ts)],
    )


@pytest.fixture()
def db(tmp_path: Path) -> DuckDBBackend:
    backend = DuckDBBackend(StorageConfig(path=str(tmp_path / "t.duckdb")))
    yield backend
    backend.close()


def test_scan_dedupes_subagent_files_onto_parent_session_id(tmp_path: Path) -> None:
    """A nested `subagents/` file carries its PARENT's sessionId internally.

    Counting files (or trusting filenames) is exactly how two prior measurements
    of the same ingest gap disagreed by an order of magnitude.
    """
    root = tmp_path / "projects"
    _write_session(root, "sess-parent")
    # Same internal sessionId, different filename, nested folder — the shape
    # that inflates a filename-derived count.
    _write_transcript(
        root, "-tmp-proj/subagents", "agent-abc123",
        [_assistant_record("sess-parent", "/tmp/proj", "u2", "msg_child",
                           "2026-07-25T10:05:00.000Z")],
    )

    sessions, scanned, unreadable = scan_disk_sessions(root)

    assert scanned == 2
    assert unreadable == 0
    assert list(sessions) == ["sess-parent"]
    assert sessions["sess-parent"].file_count == 2


def test_scan_keeps_earliest_timestamp_across_a_sessions_files(tmp_path: Path) -> None:
    root = tmp_path / "projects"
    _write_session(root, "sess-a", ts="2026-07-25T12:00:00.000Z")
    _write_transcript(
        root, "-tmp-proj/subagents", "agent-early",
        [_assistant_record("sess-a", "/tmp/proj", "u2", "msg_early",
                           "2026-07-25T09:00:00.000Z")],
    )

    sessions, _, _ = scan_disk_sessions(root)

    assert sessions["sess-a"].started_at == datetime(
        2026, 7, 25, 9, 0, tzinfo=timezone.utc
    )


def test_scan_counts_a_file_with_no_session_id_as_unreadable(tmp_path: Path) -> None:
    root = tmp_path / "projects"
    _write_transcript(root, "-tmp-proj", "broken", [{"type": "summary"}])

    sessions, scanned, unreadable = scan_disk_sessions(root)

    assert scanned == 1
    assert unreadable == 1
    assert sessions == {}


def test_scan_honours_the_since_mtime_prefilter(tmp_path: Path) -> None:
    root = tmp_path / "projects"
    old = _write_session(root, "sess-old")
    _write_session(root, "sess-new")
    import os
    stale = (datetime.now(tz=timezone.utc) - timedelta(days=10)).timestamp()
    os.utime(old, (stale, stale))

    sessions, scanned, _ = scan_disk_sessions(
        root, since=datetime.now(tz=timezone.utc) - timedelta(days=1)
    )

    assert scanned == 1
    assert list(sessions) == ["sess-new"]


def test_reconcile_detects_a_transcript_on_disk_but_absent_from_the_db(
    tmp_path: Path, db: DuckDBBackend,
) -> None:
    """The core regression: this is precisely what nothing could see before."""
    root = tmp_path / "projects"
    _write_session(root, "sess-ingested")
    ingest_claude_code(db, root=root)

    # A session that arrives after the (only) ingest — the live-OTLP miss.
    _write_session(root, "sess-dropped")

    report = reconcile_claude_code(db, root=root)

    assert report.disk_sessions == 2
    assert report.ingested_sessions == 1
    assert [s.session_id for s in report.missing] == ["sess-dropped"]
    assert report.skipped_empty_count == 0


def test_reconcile_reports_a_clean_install_as_complete(
    tmp_path: Path, db: DuckDBBackend,
) -> None:
    root = tmp_path / "projects"
    _write_session(root, "sess-a")
    _write_session(root, "sess-b")
    ingest_claude_code(db, root=root)

    report = reconcile_claude_code(db, root=root)

    assert report.missing_count == 0
    assert report.ingested_sessions == 2


def test_reconcile_separates_sessions_with_no_assistant_turn(
    tmp_path: Path, db: DuckDBBackend,
) -> None:
    """A session that ended before its first model call has no usage to record.

    Backfill deliberately drops it; counting it as "missing" would leave a
    healthy install reporting a phantom gap that no ingest can ever close.
    """
    root = tmp_path / "projects"
    _write_transcript(
        root, "-tmp-proj", "sess-empty",
        [{"type": "user", "sessionId": "sess-empty", "cwd": "/tmp/proj",
          "timestamp": "2026-07-25T10:00:00.000Z",
          "message": {"role": "user", "content": "hi"}}],
    )

    report = reconcile_claude_code(db, root=root)

    assert report.missing_count == 0
    assert [s.session_id for s in report.skipped_empty] == ["sess-empty"]


def test_days_until_rotation_counts_down_from_the_oldest_missing_session(
    tmp_path: Path, db: DuckDBBackend,
) -> None:
    root = tmp_path / "projects"
    old_ts = (datetime.now(tz=timezone.utc) - timedelta(days=27)).isoformat()
    _write_session(root, "sess-old", ts=old_ts)
    _write_session(root, "sess-recent")

    report = reconcile_claude_code(db, root=root)
    days_left = report.days_until_rotation()

    assert report.missing_count == 2
    assert report.oldest_missing.session_id == "sess-old"
    assert days_left is not None
    assert 2 <= days_left <= 4  # ~30d rotation horizon minus a 27d-old session
    assert CLAUDE_CODE_ROTATION_DAYS == 30


def test_catch_up_ingests_the_missing_session_and_is_idempotent(
    tmp_path: Path, db: DuckDBBackend,
) -> None:
    root = tmp_path / "projects"
    _write_session(root, "sess-ingested")
    ingest_claude_code(db, root=root)
    _write_session(root, "sess-dropped")

    assert reconcile_claude_code(db, root=root).missing_count == 1

    first = run_catch_up(db, root=root, lookback=timedelta(days=1))
    assert first.sessions_new == 1
    assert reconcile_claude_code(db, root=root).missing_count == 0

    # Re-running the scheduled pass must not double-insert: span ids are
    # deterministic and every run anti-joins against what is already stored.
    before = db.conn.execute("SELECT COUNT(*) FROM spans").fetchone()[0]
    second = run_catch_up(db, root=root, lookback=timedelta(days=1))
    after = db.conn.execute("SELECT COUNT(*) FROM spans").fetchone()[0]

    assert second.sessions_new == 0
    assert after == before


def test_catch_up_lookback_window_skips_files_older_than_the_window(
    tmp_path: Path, db: DuckDBBackend,
) -> None:
    root = tmp_path / "projects"
    old = _write_session(root, "sess-old")
    _write_session(root, "sess-new")
    import os
    stale = (datetime.now(tz=timezone.utc) - timedelta(days=10)).timestamp()
    os.utime(old, (stale, stale))

    result = run_catch_up(db, root=root, lookback=timedelta(days=2))

    assert result.new_session_ids == {"sess-new"}


def test_windowed_catch_up_never_purges_an_out_of_window_sibling_file(
    tmp_path: Path, db: DuckDBBackend,
) -> None:
    """A scheduled catch-up must not destroy spans it isn't looking at.

    The stale-scheme reconciliation DELETEs any backfill-sourced span for a
    session that isn't in the run's keep-set. On a windowed run the keep-set is
    only the session's IN-WINDOW files, so a long-running conversation whose main
    transcript is still being appended while its `subagents/` files finished days
    ago would have those siblings' already-ingested spans deleted as if they were
    stale-scheme orphans. With a job running every few minutes that is continuous
    data loss, so the reconciliation is now unbounded-pass only.
    """
    import os

    root = tmp_path / "projects"
    now = datetime.now(tz=timezone.utc)
    _write_session(root, "sess-long", ts=now.isoformat())
    sibling = _write_transcript(
        root, "-tmp-proj/subagents", "agent-old",
        [_assistant_record("sess-long", "/tmp/proj", "u-old", "msg_old",
                           (now - timedelta(days=5)).isoformat())],
    )
    ingest_claude_code(db, root=root)
    assert db.conn.execute("SELECT COUNT(*) FROM spans").fetchone()[0] == 2

    # Age the sibling out of the catch-up window; the main file stays in it.
    stale = (now - timedelta(days=5)).timestamp()
    os.utime(sibling, (stale, stale))

    result = run_catch_up(db, root=root, lookback=timedelta(days=2))

    assert result.spans_stale_purged == 0
    assert db.conn.execute("SELECT COUNT(*) FROM spans").fetchone()[0] == 2


def test_unbounded_backfill_still_self_heals_stale_scheme_spans(
    tmp_path: Path, db: DuckDBBackend,
) -> None:
    """The windowed guard must not disable the #294/#300 self-heal outright —
    an unwindowed pass still purges a span the transcript no longer accounts for.
    """
    root = tmp_path / "projects"
    _write_session(root, "sess-a")
    ingest_claude_code(db, root=root)

    row = db.conn.execute(
        "SELECT * FROM spans LIMIT 1"
    ).fetchone()
    assert row is not None
    # Forge a stale-scheme orphan: same session + source tag, an id no current
    # parse produces.
    db.conn.execute(
        "INSERT INTO spans SELECT * REPLACE ('stale-orphan-id' AS span_id) "
        "FROM spans LIMIT 1"
    )
    assert db.conn.execute("SELECT COUNT(*) FROM spans").fetchone()[0] == 2

    result = ingest_claude_code(db, root=root)

    assert result.spans_stale_purged == 1
    assert db.conn.execute(
        "SELECT COUNT(*) FROM spans WHERE span_id = 'stale-orphan-id'"
    ).fetchone()[0] == 0


def test_reconcile_counts_ingested_sessions_when_daemon_holds_the_lock(
    tmp_path: Path,
) -> None:
    """Regression for #642: `tj backfill status` reported `0 already ingested`
    on a fully-ingested install whenever `tj serve` held the DB write-lock.

    The daemon owns the DuckDB connection, so the CLI falls back to the HTTP
    shim (`ApiBackend`), which has no `.conn`. `reconcile_claude_code` used to
    read `getattr(db, "conn", None)`, see `None`, and bail with
    `ingested_sessions == 0` and an empty `missing` list — so the three buckets
    summed to 0 and it still printed "✓ Every on-disk session is ingested".

    Before the fix `ingested_sessions` is 0 (and disk_sessions == 3), which
    this asserts against; the fix routes the anti-join through the daemon.
    """
    import threading
    import time
    from contextlib import contextmanager

    import uvicorn

    from tokenjam.api.app import create_app
    from tokenjam.core.api_backend import ApiBackend
    from tokenjam.core.config import (
        ApiAuthConfig,
        ApiConfig,
        SecurityConfig,
        TjConfig,
    )
    from tokenjam.core.ingest import IngestPipeline

    root = tmp_path / "projects"
    # Three on-disk sessions, all ingested into the DB.
    for sid in ("sess-a", "sess-b", "sess-c"):
        _write_session(root, sid)

    db = DuckDBBackend(StorageConfig(path=str(tmp_path / "daemon.duckdb")))
    ingest_claude_code(db, root=root)
    assert reconcile_claude_code(db, root=root).ingested_sessions == 3

    config = TjConfig(
        version="1",
        security=SecurityConfig(ingest_secret="s"),
        api=ApiConfig(auth=ApiAuthConfig(enabled=False)),
    )
    pipeline = IngestPipeline(db=db, config=config)
    app = create_app(config=config, db=db, ingest_pipeline=pipeline)

    @contextmanager
    def _live_server(app):
        server = uvicorn.Server(uvicorn.Config(
            app, host="127.0.0.1", port=0, log_level="error",
        ))
        thread = threading.Thread(target=server.run, daemon=True)
        thread.start()
        deadline = time.monotonic() + 10.0
        while not server.started:
            if time.monotonic() > deadline:
                raise RuntimeError("live test server failed to start")
            time.sleep(0.02)
        try:
            port = server.servers[0].sockets[0].getsockname()[1]
            yield f"http://127.0.0.1:{port}"
        finally:
            server.should_exit = True
            thread.join(timeout=5)

    with _live_server(app) as base_url:
        shim = ApiBackend(base_url)
        try:
            report = reconcile_claude_code(shim, root=root)
        finally:
            shim.close()

    # The fix: the ingested count reflects reality through the daemon...
    assert report.ingested_sessions == 3
    # ...and the buckets reconcile transparently to the distinct-session count.
    assert report.disk_sessions == 3
    assert (
        report.ingested_sessions
        + report.missing_count
        + report.skipped_empty_count
        == report.disk_sessions
    )
    db.close()


def test_reconcile_marks_unverified_when_daemon_lookup_fails(
    tmp_path: Path,
) -> None:
    """Greptile P1 / #642: when the daemon ingested-id lookup raises
    (auth/connectivity/server/decode), `reconcile_claude_code` must NOT return a
    confident all-zero report — it flags `verified=False` so the caller can be
    honest instead of reprinting "0 ingested · everything ingested".
    """
    root = tmp_path / "projects"
    for sid in ("sess-a", "sess-b"):
        _write_session(root, sid)

    class _FailingShim:
        """No `.conn` (like `ApiBackend`) and a fetch that always raises."""

        def fetch_ingested_session_ids(self, session_ids):
            raise RuntimeError("daemon unreachable")

    report = reconcile_claude_code(_FailingShim(), root=root)

    assert report.verified is False
    # Disk scan still worked; only the DB comparison could not run.
    assert report.disk_sessions == 2
    assert report.ingested_sessions == 0
    assert report.missing_count == 0
    assert report.to_dict()["verified"] is False


def test_reconcile_marks_unverified_when_no_conn_and_no_shim(
    tmp_path: Path,
) -> None:
    """A backend with neither `.conn` nor a fetch method cannot verify either."""
    root = tmp_path / "projects"
    _write_session(root, "sess-a")

    class _OpaqueBackend:
        pass

    report = reconcile_claude_code(_OpaqueBackend(), root=root)
    assert report.verified is False


def test_backfill_status_command_reports_could_not_verify(
    tmp_path: Path,
) -> None:
    """The status command must surface an honest "couldn't verify" message on an
    unverified report — never "0 already ingested" or "Every session ingested".
    """
    root = tmp_path / "projects"
    _write_session(root, "sess-a")

    class _FailingShim:
        def fetch_ingested_session_ids(self, session_ids):
            raise RuntimeError("daemon unreachable")

    result = CliRunner().invoke(
        cmd_backfill_module.cmd_backfill,
        ["status", "--root", str(root)],
        obj={"db": _FailingShim(), "config": None},
    )

    assert result.exit_code == 0, result.output
    assert "Couldn't verify" in result.output
    assert "already ingested" not in result.output
    assert "Every on-disk session" not in result.output


def test_backfill_status_command_lists_the_missing_session(
    tmp_path: Path, db: DuckDBBackend,
) -> None:
    root = tmp_path / "projects"
    _write_session(root, "sess-ingested")
    ingest_claude_code(db, root=root)
    _write_session(root, "sess-dropped")

    result = CliRunner().invoke(
        cmd_backfill_module.cmd_backfill,
        ["status", "--root", str(root)],
        obj={"db": db, "config": None},
    )

    assert result.exit_code == 0, result.output
    assert "1 missing" in result.output
    assert "sess-dro" in result.output


def test_backfill_status_json_reports_the_gap_machine_readably(
    tmp_path: Path, db: DuckDBBackend,
) -> None:
    root = tmp_path / "projects"
    _write_session(root, "sess-dropped")

    result = CliRunner().invoke(
        cmd_backfill_module.cmd_backfill,
        ["status", "--root", str(root), "--json"],
        obj={"db": db, "config": None},
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["missing_count"] == 1
    assert payload["disk_sessions"] == 1
    assert payload["missing"][0]["session_id"] == "sess-dropped"
