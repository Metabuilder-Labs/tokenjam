"""The CLI paths issue #770 found broken while `tj serve` held the DuckDB
write lock, each proved against a REAL in-process uvicorn daemon and the
serve-mode HTTP shim (`ApiBackend`) the CLI gets in that state:

* `tj backfill claude-code` wrote nothing and then failed with
  `'ApiBackend' object has no attribute 'upsert_session'` (fix 3): the run
  is now handed to the daemon, which owns the connection and already runs
  this job as its transcript catch-up.
* the transcript context refill needs one session write, which the shim
  now carries (`POST /api/v1/sessions/upsert`; parity-covered in
  `test_storage_backend_parity.py`).
"""
from __future__ import annotations

import threading
import time
from contextlib import contextmanager

import pytest
import uvicorn
from click.testing import CliRunner

from tests.ledger_fixtures import (
    REMOTE,
    contextless_session as _contextless,
    git_repo,
    session_context_row as _context,
    write_transcript as _write_session,
)
from tokenjam.api.app import create_app
from tokenjam.cli import cmd_backfill as cmd_backfill_module
from tokenjam.core.api_backend import ApiBackend
from tokenjam.core.backfill import BackfillUnavailable, ingest_claude_code
from tokenjam.core.config import ApiAuthConfig, ApiConfig, SecurityConfig, StorageConfig, TjConfig
from tokenjam.core.db import DuckDBBackend
from tokenjam.core.ingest import IngestPipeline
from tokenjam.core.transcript_sync import refill_session_context

repo = pytest.fixture(git_repo)


@contextmanager
def _live_server(app):
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=0, log_level="error"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10.0
    while not server.started:
        if time.monotonic() > deadline:
            raise RuntimeError("live test server failed to start")
        time.sleep(0.02)
    try:
        yield f"http://127.0.0.1:{server.servers[0].sockets[0].getsockname()[1]}"
    finally:
        server.should_exit = True
        thread.join(timeout=5)


@contextmanager
def _daemon(tmp_path):
    """A daemon over its own store, and the shim a locked CLI would get."""
    config = TjConfig(
        version="1", security=SecurityConfig(ingest_secret="s"),
        api=ApiConfig(auth=ApiAuthConfig(enabled=False)),
        storage=StorageConfig(path=str(tmp_path / "daemon.duckdb")),
    )
    db = DuckDBBackend(config.storage)
    app = create_app(config=config, db=db, ingest_pipeline=IngestPipeline(db=db, config=config))
    try:
        with _live_server(app) as base_url:
            shim = ApiBackend(base_url)
            try:
                yield db, shim, config
            finally:
                shim.close()
    finally:
        db.close()


def _wait_until(predicate, timeout=15.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return predicate()


# --- Fix 3: the shim cannot ingest, so the daemon does -----------------------------------

def test_ingest_refuses_a_backend_that_cannot_store_spans():
    class Shim:  # no conn, no insert_span: what a locked CLI is handed
        pass

    with pytest.raises(BackfillUnavailable):
        ingest_claude_code(Shim())


def test_backfill_claude_code_under_the_daemon_is_run_by_the_daemon(tmp_path, repo):
    root = tmp_path / "projects"
    _write_session(root, "on-disk", str(repo))
    with _daemon(tmp_path) as (db, shim, config):
        assert db.get_session("on-disk") is None
        result = CliRunner().invoke(
            cmd_backfill_module.cmd_backfill,
            ["claude-code", "--root", str(root), "--since", "30d"],
            obj={"db": shim, "config": config},
        )
        assert result.exit_code == 0, result.output
        assert "tj serve" in result.output and "running this backfill" in result.output
        assert "AttributeError" not in result.output and "upsert_session" not in result.output
        # The daemon ingested it on its own thread, with its own backend,
        # and the row landed WITH repo context.
        assert _wait_until(lambda: db.get_session("on-disk") is not None)
        assert _wait_until(lambda: _context(db, "on-disk")[0] == REMOTE)
        # A second request while nothing runs starts another pass; the
        # answer is never an error.
        again = shim.request_claude_code_backfill(since=None, root=str(root))
        assert again["started"] or again["running"]

        from tests.unit.test_advertised_commands_are_invocable import (
            advertised_commands,
            assert_invocable,
        )
        for command in advertised_commands(result.output):
            assert_invocable(command)


def test_refill_runs_through_the_shim(tmp_path, repo):
    """The refill's two backend needs (the missing-context lookup and one
    fill-null-only session write) both cross the daemon."""
    root = tmp_path / "projects"
    _write_session(root, "live-1", str(repo))
    with _daemon(tmp_path) as (db, shim, config):
        _contextless(db, "live-1", cost=3.0)
        before = _context(db, "live-1")
        assert before[0] is None

        report = refill_session_context(shim, root=root)

        assert (report.candidates, report.filled, report.verified) == (1, 1, True)
        after = _context(db, "live-1")
        assert after[:2] == (REMOTE, str(repo))
        assert (after[5], after[6]) == (3.0, 200)
        # Read back through the shim, the context is what the DB holds.
        assert shim.get_session("live-1").repo_remote == REMOTE
        assert shim.fetch_ingested_session_ids(["live-1"], missing_context=True) == set()


def test_refill_reports_unverified_when_the_daemon_lookup_fails(tmp_path, repo):
    root = tmp_path / "projects"
    _write_session(root, "live-1", str(repo))

    class Broken:
        def fetch_ingested_session_ids(self, ids, **kw):
            raise RuntimeError("daemon down")

    report = refill_session_context(Broken(), root=root)
    assert report.verified is False and report.candidates == 0
