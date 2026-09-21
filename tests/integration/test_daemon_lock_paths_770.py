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
* the session -> commit matcher could not run at all (fix 4): the CLI now
  asks the daemon to run it (`POST /api/v1/shipped/match`).
"""
from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from datetime import timedelta

import pytest
import uvicorn
from click.testing import CliRunner

from tests.ledger_fixtures import (
    REMOTE,
    T0,
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


# --- Fix 4: the matcher runs on the daemon when the CLI cannot ------------------------------

def test_the_matcher_can_be_asked_for_through_the_daemon(tmp_path, repo, monkeypatch):
    import subprocess
    from dataclasses import replace

    from tests.factories import make_session, make_tool_span

    def _git(*args):
        return subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True,
                              check=True).stdout.strip()

    with _daemon(tmp_path) as (db, shim, config):
        db.upsert_session(replace(
            make_session(session_id="s1", agent_id="claude-code-widgets", started_at=T0,
                         status="completed", total_cost_usd=1.0),
            ended_at=T0 + timedelta(minutes=10), repo_remote=REMOTE, repo_root=str(repo),
            branch_start="main", branch_end="main",
        ))
        db.insert_span(make_tool_span(agent_id="claude-code-widgets", tool_name="Bash",
                                      session_id="s1", start_time=T0 + timedelta(minutes=5),
                                      tool_input={"command": "git commit -m feat"}))
        (repo / "a.py").write_text("1\n")
        _git("add", "a.py")
        stamp = str(int((T0 + timedelta(minutes=5, seconds=2)).timestamp()))
        subprocess.run(["git", "commit", "-q", "-m", "feat"], cwd=repo, check=True,
                       env={**__import__("os").environ, "GIT_AUTHOR_DATE": stamp,
                            "GIT_COMMITTER_DATE": stamp})
        sha = _git("rev-parse", "HEAD")

        answer = shim.request_commit_match(wait_s=20)
        assert answer["completed"] is True, answer
        assert answer["rows_written"] == 1
        rows = db.conn.execute(
            "SELECT commit_sha, confidence, source FROM session_commits WHERE session_id = 's1'"
        ).fetchall()
        assert rows == [(sha, "deterministic", "tool_span_git_log")]
        # Idempotent, and never an error on a repeat.
        assert shim.request_commit_match(wait_s=20)["completed"] is True


def _commit_from_session(db, repo, *, session_id: str = "s1"):
    """A session whose Bash tool ran `git commit`, and the commit it made."""
    import os
    import subprocess
    from dataclasses import replace

    from tests.factories import make_llm_span, make_session, make_tool_span

    db.upsert_session(replace(
        make_session(session_id=session_id, agent_id="claude-code-widgets", started_at=T0,
                     status="completed", total_cost_usd=1.0, input_tokens=10, output_tokens=2),
        ended_at=T0 + timedelta(minutes=10), repo_remote=REMOTE, repo_root=str(repo),
        branch_start="main", branch_end="main",
    ))
    db.insert_span(make_llm_span(agent_id="claude-code-widgets", session_id=session_id,
                                 start_time=T0, cost_usd=1.0, input_tokens=10, output_tokens=2))
    db.insert_span(make_tool_span(agent_id="claude-code-widgets", tool_name="Bash",
                                  session_id=session_id, start_time=T0 + timedelta(minutes=5),
                                  tool_input={"command": "git commit -m feat"}))
    (repo / "a.py").write_text("1\n")
    subprocess.run(["git", "add", "a.py"], cwd=repo, check=True)
    stamp = str(int((T0 + timedelta(minutes=5, seconds=2)).timestamp()))
    subprocess.run(["git", "commit", "-q", "-m", "feat"], cwd=repo, check=True,
                   env={**os.environ, "GIT_AUTHOR_DATE": stamp, "GIT_COMMITTER_DATE": stamp})
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True,
                          text=True, check=True).stdout.strip()


def test_tj_optimize_shipped_under_the_daemon_runs_the_matcher_and_renders_the_fresh_join(
    tmp_path, repo, monkeypatch,
):
    """The real run: `tj optimize shipped` with the daemon up failed outright.
    Now the CLI asks the daemon for the pass, and the card it renders is
    the join that pass just made, not the stored report's stale one."""
    from tokenjam.cli.cmd_optimize import cmd_optimize
    from tokenjam.core.optimize import report_store

    with _daemon(tmp_path) as (db, shim, config):
        sha = _commit_from_session(db, repo)
        # The daemon's last scan ran BEFORE the commit was joined: its
        # stored finding says nothing shipped. `recompute_now` is the real
        # store writer (persona reports and all), on this thread.
        stored = report_store.recompute_now(db, config, window_days=30)
        assert stored is not None
        assert report_store.stored_report_dict(config)["findings"]["shipped"]["sessions_shipped"] == 0

        result = CliRunner().invoke(cmd_optimize, ["shipped"], obj={"db": shim, "config": config})

        assert result.exit_code == 0, result.output
        assert "ConnectionException" not in result.output
        rows = db.conn.execute(
            "SELECT commit_sha, confidence, source FROM session_commits"
        ).fetchall()
        assert rows == [(sha, "deterministic", "tool_span_git_log")]
        flat = " ".join(result.output.split())
        assert "Shipped 1 of 1" in flat, flat


def test_tj_optimize_shipped_under_the_daemon_with_a_cold_store_still_runs_the_matcher(
    tmp_path, repo,
):
    from tokenjam.cli.cmd_optimize import cmd_optimize

    with _daemon(tmp_path) as (db, shim, config):
        sha = _commit_from_session(db, repo)
        result = CliRunner().invoke(cmd_optimize, ["shipped"], obj={"db": shim, "config": config})
        assert result.exit_code == 0, result.output
        assert "ConnectionException" not in result.output
        assert db.conn.execute("SELECT commit_sha FROM session_commits").fetchall() == [(sha,)]
        # The store is honestly cold; the join is nonetheless current for
        # the daemon's next pass and every other surface.
        assert "No analyzer report has been computed yet" in result.output
