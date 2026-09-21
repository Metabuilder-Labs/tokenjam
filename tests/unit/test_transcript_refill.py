"""The transcript context refill (`transcript_sync.refill_session_context`;
issue #770, fix 1).

A session ingested WITHOUT repo context (the live OTLP path, or a build that
predates the ledger columns) used to stay that way forever: W1 derived the
columns only at backfill time and nothing re-ran it, so the first
`tj init --cloud` push on a real machine carried ten sessions with no repo
and the session -> commit join had nothing to join on. The refill goes back
for exactly those sessions, through the same fill-null-only write the
backfill uses, and the daemon's catch-up runs it on its own schedule.

Every transcript here points its `cwd` at a REAL git repository under
`tmp_path`, so the context is resolved the way production resolves it.
"""
from __future__ import annotations

import os
from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from tests.factories import make_session
from tests.ledger_fixtures import (
    EMAIL,
    REMOTE,
    T0,
    contextless_session as _contextless,
    git_repo,
    session_context_row as _context,
    write_transcript as _write_session,
)
from tokenjam.core import repo_context
from tokenjam.core.backfill import ingest_claude_code
from tokenjam.core.config import StorageConfig
from tokenjam.core.db import DuckDBBackend
from tokenjam.core.transcript_sync import (
    REFILL_MAX_SESSIONS,
    SESSION_CONTEXT_MISSING_SQL,
    ingested_session_ids,
    refill_session_context,
    run_catch_up,
)

repo = pytest.fixture(git_repo)


@pytest.fixture
def db(tmp_path) -> DuckDBBackend:
    backend = DuckDBBackend(StorageConfig(path=str(tmp_path / "t.duckdb")))
    yield backend
    backend.close()


# --- The refill -----------------------------------------------------------------------

def test_refill_fills_context_and_touches_nothing_else(repo, db, tmp_path):
    root = tmp_path / "projects"
    _write_session(root, "live-1", str(repo))
    _contextless(db, "live-1", cost=3.0)
    before = _context(db, "live-1")
    assert before[:5] == (None, None, None, None, None)

    report = refill_session_context(db, root=root)

    assert (report.candidates, report.filled, report.unresolved) == (1, 1, 0)
    after = _context(db, "live-1")
    assert after[:5] == (REMOTE, str(repo), "main", EMAIL, repo_context.developer_id_for(EMAIL))
    # The row's own figures are untouched: the refill writes ZERO totals
    # through the accumulating upsert, never the transcript's.
    assert (after[5], after[6]) == (3.0, 200)
    # ...and `updated_at` moved, which is what re-sends the row to Cloud.
    assert after[7] > before[7]


def test_refill_is_idempotent_and_skips_sessions_that_already_have_context(repo, db, tmp_path):
    root = tmp_path / "projects"
    _write_session(root, "live-1", str(repo))
    _contextless(db, "live-1")
    refill_session_context(db, root=root)
    stamped = _context(db, "live-1")[7]

    second = refill_session_context(db, root=root)
    assert (second.candidates, second.filled) == (0, 0)
    assert _context(db, "live-1")[7] == stamped


def test_refill_never_inserts_a_session_that_was_not_ingested(repo, db, tmp_path):
    """A transcript on disk with no row is the catch-up's job, not the
    refill's: the fill-null-only write would otherwise INSERT a spanless row."""
    root = tmp_path / "projects"
    _write_session(root, "on-disk-only", str(repo))
    report = refill_session_context(db, root=root)
    assert report.candidates == 0
    assert db.get_session("on-disk-only") is None


def test_refill_reports_an_unresolvable_session_and_retries_it_next_pass(db, tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    (tmp_path / "home").mkdir()
    monkeypatch.setattr(repo_context, "_is_temp_cwd", lambda _cwd: False)
    # No global git identity either, so nothing at all can be derived.
    monkeypatch.setattr(repo_context, "_global_email", lambda: None)
    repo_context.clear_caches()
    root = tmp_path / "projects"
    _write_session(root, "gone", str(tmp_path / "deleted-worktree"))
    _contextless(db, "gone")

    first = refill_session_context(db, root=root)
    assert (first.candidates, first.filled, first.unresolved) == (1, 0, 1)
    remote, root_col, branch, email, dev_id, *_ = _context(db, "gone")
    # The transcript's own branch still lands; nothing git-derived does.
    assert (remote, root_col, email, dev_id) == (None, None, None, None)
    assert branch == "main"
    # Still a candidate: the worktree may come back, and the pass is cheap.
    assert refill_session_context(db, root=root).candidates == 1


def test_refill_is_bounded_per_pass_newest_first(repo, db, tmp_path):
    root = tmp_path / "projects"
    for i in range(3):
        _write_session(root, f"s{i}", str(repo), at=T0 + timedelta(minutes=i))
        _contextless(db, f"s{i}")
    report = refill_session_context(db, root=root, max_sessions=2)
    assert (report.candidates, report.filled, report.deferred) == (3, 2, 1)
    # The newest two were filled; the oldest waits for the next pass.
    assert _context(db, "s0")[0] is None
    assert _context(db, "s2")[0] == REMOTE
    assert REFILL_MAX_SESSIONS >= 1


def test_refill_mtime_window_matches_the_catch_up(repo, db, tmp_path):
    root = tmp_path / "projects"
    old = _write_session(root, "old", str(repo))
    _write_session(root, "new", str(repo))
    _contextless(db, "old")
    _contextless(db, "new")
    stale = (datetime.now(tz=timezone.utc) - timedelta(days=10)).timestamp()
    os.utime(old, (stale, stale))

    report = refill_session_context(
        db, root=root, since=datetime.now(tz=timezone.utc) - timedelta(days=2),
    )
    assert (report.candidates, report.filled) == (1, 1)
    assert _context(db, "new")[0] == REMOTE
    assert _context(db, "old")[0] is None
    # A full pass reaches it.
    assert refill_session_context(db, root=root).filled == 1


def test_missing_context_predicate_and_lookup(repo, db, tmp_path):
    _contextless(db, "a")
    db.upsert_session(replace(make_session(session_id="b", started_at=T0), repo_remote=REMOTE))
    # A transcript branch or a global email alone does not count as resolved.
    db.upsert_session(replace(make_session(session_id="c", started_at=T0),
                              branch_start="main", user_email=EMAIL))
    assert ingested_session_ids(db.conn, ["a", "b", "c", "zzz"]) == {"a", "b", "c"}
    assert ingested_session_ids(db.conn, ["a", "b", "c", "zzz"], missing_context=True) == {"a", "c"}
    assert "repo_remote IS NULL" in SESSION_CONTEXT_MISSING_SQL


# --- The daemon's catch-up ----------------------------------------------------------------

def test_run_catch_up_refills_existing_sessions_not_only_missing_ones(repo, db, tmp_path):
    """Issue #770 fix 1, the daemon half: the scheduled pass must heal a
    contextless row, not only ingest absent sessions."""
    root = tmp_path / "projects"
    _write_session(root, "live-1", str(repo))
    _contextless(db, "live-1")

    run_catch_up(db, root=root, lookback=timedelta(days=1))

    assert _context(db, "live-1")[:2] == (REMOTE, str(repo))


def test_run_catch_up_refill_is_reached_even_when_the_ingest_did_not_fill(
    repo, db, tmp_path, monkeypatch,
):
    """Critical Rule 36: the ingest's own per-file upsert also fills context
    when it re-parses the file, which would hide a refill that never ran.
    With the ingest's session write neutralised, the row still heals, so the
    refill is a mechanism of its own and not a side effect."""
    import tokenjam.core.backfill as backfill_mod

    root = tmp_path / "projects"
    _write_session(root, "live-1", str(repo))
    _contextless(db, "live-1")
    monkeypatch.setattr(backfill_mod, "session_totals_delta",
                        lambda parsed, plan_tier, inserted: replace(
                            backfill_mod.session_record_from_parsed(parsed, plan_tier),
                            repo_remote=None, repo_root=None, branch_start=None,
                            branch_end=None, developer_id=None, user_email=None,
                            total_cost_usd=0.0, input_tokens=0, output_tokens=0,
                            cache_tokens=0, cache_write_tokens=0, tool_call_count=0,
                            error_count=0))

    ingest_claude_code(db, root=root)
    assert _context(db, "live-1")[0] is None, "the ingest's fill is neutralised"

    run_catch_up(db, root=root, lookback=timedelta(days=1))
    assert _context(db, "live-1")[:2] == (REMOTE, str(repo))
