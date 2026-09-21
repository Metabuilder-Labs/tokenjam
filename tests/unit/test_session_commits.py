"""Session -> commit join + the `shipped` analyzer (ledger W2; contracts §1,
§2, §4, §5 read side).

Every join here is proved against a REAL git repository in `tmp_path` with
real commits, through the path production reaches: sessions and tool spans
written to a backend, then `match_sessions_to_commits` run over it, then the
read side (`shipped_summary`, the analyzer, the routes, the CLI) read back.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from click.testing import CliRunner

from tokenjam.core import repo_context, shipped
from tokenjam.core.config import StorageConfig, TjConfig
from tokenjam.core.db import EXPECTED_ADDITIVE_COLUMNS, EXPECTED_TABLES, DuckDBBackend, InMemoryBackend
from tokenjam.core.models import SessionCommit
from tokenjam.core.shipped import (
    SHIPPED_CAVEAT,
    STATE_COMMITTED,
    STATE_NO_REPO,
    STATE_REVERTED,
    STATE_SHIPPED,
    STATE_UNSHIPPED,
    bridge_suffix,
    is_git_commit_command,
    match_sessions_to_commits,
    parse_trailers,
    reverted_sha,
    session_shipped_state,
    shipped_summary,
)
from tokenjam.utils.time_parse import utcnow

from tests.factories import make_llm_span, make_session, make_tool_span

EMAIL = "dev@example.com"
REMOTE = "https://github.com/Acme/widgets"
T0 = datetime(2026, 9, 1, 10, 0, tzinfo=timezone.utc)


def _git(repo: Path, *args: str, env: dict | None = None, at: datetime | None = None) -> str:
    e = dict(os.environ)
    e.update({"GIT_AUTHOR_EMAIL": EMAIL, "GIT_COMMITTER_EMAIL": EMAIL,
              "GIT_AUTHOR_NAME": "Dev", "GIT_COMMITTER_NAME": "Dev"})
    if at is not None:
        stamp = str(int(at.timestamp()))
        e["GIT_AUTHOR_DATE"] = stamp
        e["GIT_COMMITTER_DATE"] = stamp
    if env:
        e.update(env)
    return subprocess.run(
        ["git", *args], cwd=repo, capture_output=True, text=True, check=True, env=e,
    ).stdout.strip()


def _commit(repo: Path, path: str, text: str, message: str, at: datetime, *, amend: bool = False) -> str:
    (repo / path).write_text(text)
    _git(repo, "add", path)
    args = ["commit", "-q", "-m", message]
    if amend:
        args.insert(2, "--amend")
    _git(repo, *args, at=at)
    return _git(repo, "rev-parse", "HEAD")


@pytest.fixture
def repo(tmp_path, monkeypatch) -> Path:
    """A real checkout with an `origin/HEAD` pointing at main."""
    if shutil.which("git") is None:
        pytest.skip("git not available")
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    (tmp_path / "home").mkdir()
    monkeypatch.setattr(repo_context, "_is_temp_cwd", lambda _cwd: False)
    repo_context.clear_caches()
    path = tmp_path / "widgets"
    path.mkdir()
    _git(path, "init", "-q", "-b", "main")
    _git(path, "config", "user.email", EMAIL)
    _git(path, "config", "user.name", "Dev")
    _git(path, "remote", "add", "origin", REMOTE + ".git")
    _commit(path, "README", "hello\n", "init", T0 - timedelta(days=1))
    # A bare "origin" so origin/HEAD resolves the way a clone's does.
    origin = tmp_path / "origin.git"
    _git(path, "init", "-q", "--bare", "-b", "main", str(origin))
    _git(path, "remote", "set-url", "origin", str(origin))
    _git(path, "push", "-q", "origin", "main")
    _git(path, "remote", "set-head", "origin", "main")
    # `origin` stays the bare repo so tests can push; the sessions carry the
    # normalised GitHub remote explicitly (`_session`), as a real backfill would.
    yield path
    repo_context.clear_caches()


def _session(db, sid: str, repo: Path, start: datetime, end: datetime, *, branch: str = "main",
             cost: float = 1.0, bridge: str | None = None, root: str | None = "__repo__",
             agent_id: str = "claude-code-widgets"):
    rec = replace(
        make_session(session_id=sid, agent_id=agent_id, started_at=start, status="completed",
                     total_cost_usd=cost, input_tokens=100, output_tokens=10, tool_call_count=1),
        ended_at=end, repo_remote=REMOTE, repo_root=(str(repo) if root == "__repo__" else root),
        branch_start=branch, branch_end=branch, user_email=EMAIL, bridge_session_id=bridge,
    )
    db.upsert_session(rec)
    return rec


def _bash(db, sid: str, at: datetime, command: str, *, agent_id: str = "claude-code-widgets",
          parent_cost: float | None = None):
    parent = None
    if parent_cost is not None:
        parent = make_llm_span(agent_id=agent_id, session_id=sid, start_time=at, cost_usd=parent_cost,
                               input_tokens=10, output_tokens=5)
        db.insert_span(parent)
    span = make_tool_span(agent_id=agent_id, tool_name="Bash", session_id=sid, start_time=at,
                          tool_input={"command": command})
    if parent is not None:
        span = replace(span, parent_span_id=parent.span_id, trace_id=parent.trace_id)
    db.insert_span(span)
    return span


def _rows(db, sid: str | None = None) -> list[tuple]:
    sql = "SELECT session_id, commit_sha, confidence, source, match_delta_s FROM session_commits"
    if sid:
        return db.conn.execute(sql + " WHERE session_id = $1 ORDER BY commit_sha", [sid]).fetchall()
    return db.conn.execute(sql + " ORDER BY session_id, commit_sha").fetchall()


# --- Schema ------------------------------------------------------------------------

def test_migration_24_is_declared_for_self_heal():
    assert ("sessions", "bridge_session_id") in {(t, c) for t, c, _ in EXPECTED_ADDITIVE_COLUMNS}
    for table in ("session_commits", "repo_commits", "ledger_repo_state",
                  "session_commit_scans", "commit_file_stats"):
        assert table in EXPECTED_TABLES


def test_session_commits_columns_match_the_contract(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    db = DuckDBBackend(StorageConfig(path=str(tmp_path / "t.duckdb")))
    try:
        cols = [r[0] for r in db.conn.execute("DESCRIBE session_commits").fetchall()]
    finally:
        db.close()
    assert cols == ["session_id", "commit_sha", "repo_remote", "confidence", "source",
                    "author_email", "committed_at", "matched_at", "match_delta_s"]


def test_bridge_session_id_round_trips_through_the_backfill(repo, tmp_path):
    from tokenjam.core.backfill import ingest_claude_code

    root = tmp_path / "projects" / "proj"
    root.mkdir(parents=True)
    records = [
        {"type": "bridge-session", "sessionId": "cc-b", "bridgeSessionId": "cse_01ABC", "lastSequenceNum": 0},
        {"type": "assistant", "uuid": "m1", "timestamp": "2026-09-01T10:00:00.000Z", "sessionId": "cc-b",
         "cwd": str(repo), "gitBranch": "main",
         "message": {"model": "claude-opus-4-7", "content": [{"type": "text", "text": "ok"}],
                     "usage": {"input_tokens": 100, "output_tokens": 20}}},
    ]
    (root / "cc-b.jsonl").write_text("\n".join(json.dumps(r) for r in records) + "\n")
    db = InMemoryBackend()
    try:
        ingest_claude_code(db, root=root.parent)
        assert db.get_session("cc-b").bridge_session_id == "cse_01ABC"
    finally:
        db.close()


# --- Pure helpers ------------------------------------------------------------------

@pytest.mark.parametrize("cmd,expected", [
    ("git commit -m 'x'", True),
    ("git -C /repo commit -am x", True),
    ("git add -A && git commit -q -m 'x' --no-verify", True),
    ("git commit --amend --no-edit", True),
    ("git commit --dry-run", False),
    ("git log --oneline | grep commit", False),
    ("echo commit", False),
    (None, False),
])
def test_git_commit_command_detection(cmd, expected):
    assert is_git_commit_command(cmd) is expected


def test_bridge_suffix_matches_both_spellings():
    assert bridge_suffix("cse_01AbC") == "01AbC" == bridge_suffix("session_01AbC")
    assert bridge_suffix(None) is None


def test_trailer_parsing():
    body = ("Subject\n\nCo-Authored-By: Claude Opus 5 <noreply@anthropic.com>\n"
            "Claude-Session: https://claude.ai/code/session_01AbC\nTokenJam-Session: local-1\n")
    t = parse_trailers(body)
    assert t == {"tokenjam_sessions": ["local-1"], "claude_sessions": ["01AbC"], "ai": True}
    assert parse_trailers("plain")["ai"] is False


def test_revert_detection():
    assert reverted_sha('Revert "x"', "This reverts commit abcdef1234567.\n") == "abcdef1234567"
    assert reverted_sha("Revert abcdef1", "") == "abcdef1"
    assert reverted_sha("Fix abcdef1", "This reverts commit abcdef1.") is None


# --- The matcher -------------------------------------------------------------------

def test_tool_span_git_log_matches_the_commit_at_deterministic(repo):
    db = InMemoryBackend()
    try:
        _session(db, "s1", repo, T0, T0 + timedelta(minutes=10))
        _bash(db, "s1", T0 + timedelta(minutes=5), "git add -A && git commit -m 'feat'")
        sha = _commit(repo, "a.py", "print(1)\n", "feat", T0 + timedelta(minutes=5, seconds=4))
        # A commit in the window with NO tool call within 30s: never matched.
        stray = _commit(repo, "b.py", "print(2)\n", "stray", T0 + timedelta(minutes=8))
        result = match_sessions_to_commits(db)
        assert result.sessions_scanned == 1
        rows = _rows(db, "s1")
        assert [(r[1], r[2], r[3]) for r in rows] == [(sha, "deterministic", "tool_span_git_log")]
        assert rows[0][4] == pytest.approx(-4.0)
        assert stray not in [r[1] for r in rows]
    finally:
        db.close()


def test_out_of_window_commit_is_not_matched(repo):
    db = InMemoryBackend()
    try:
        _session(db, "s1", repo, T0, T0 + timedelta(minutes=10))
        _bash(db, "s1", T0 + timedelta(minutes=5), "git commit -m x")
        _commit(repo, "a.py", "1\n", "late", T0 + timedelta(hours=2))
        match_sessions_to_commits(db)
        assert _rows(db, "s1") == []
        assert session_shipped_state(db.conn, "s1") == STATE_UNSHIPPED
    finally:
        db.close()


def test_amend_matches_the_amended_commit(repo):
    db = InMemoryBackend()
    try:
        _session(db, "s1", repo, T0, T0 + timedelta(minutes=10))
        first = _commit(repo, "a.py", "1\n", "feat", T0 + timedelta(minutes=2))
        _bash(db, "s1", T0 + timedelta(minutes=6), "git commit --amend --no-edit")
        amended = _commit(repo, "a.py", "2\n", "feat", T0 + timedelta(minutes=6, seconds=2), amend=True)
        match_sessions_to_commits(db)
        shas = [r[1] for r in _rows(db, "s1")]
        assert amended in shas and first not in shas
    finally:
        db.close()


def test_claude_session_trailer_resolves_through_the_bridge_id(repo):
    db = InMemoryBackend()
    try:
        _session(db, "s1", repo, T0, T0 + timedelta(minutes=10), bridge="cse_01AbC")
        sha = _commit(repo, "a.py", "1\n",
                      "feat\n\nCo-Authored-By: Claude <noreply@anthropic.com>\n"
                      "Claude-Session: https://claude.ai/code/session_01AbC",
                      T0 + timedelta(minutes=3))
        match_sessions_to_commits(db)
        assert [(r[1], r[2], r[3]) for r in _rows(db, "s1")] == [(sha, "deterministic", "trailer_session")]
    finally:
        db.close()


def test_tokenjam_session_trailer_resolves_to_the_named_session(repo):
    db = InMemoryBackend()
    try:
        _session(db, "s1", repo, T0, T0 + timedelta(minutes=10))
        sha = _commit(repo, "a.py", "1\n", "feat\n\nTokenJam-Session: s1", T0 + timedelta(minutes=3))
        match_sessions_to_commits(db)
        assert [(r[1], r[2], r[3]) for r in _rows(db, "s1")] == [(sha, "deterministic", "trailer_session")]
    finally:
        db.close()


def test_git_note_naming_an_ingested_session_joins_at_deterministic(repo):
    db = InMemoryBackend()
    try:
        _session(db, "s1", repo, T0, T0 + timedelta(minutes=10))
        sha = _commit(repo, "a.py", "1\n", "feat", T0 + timedelta(minutes=3))
        _git(repo, "notes", "--ref=ai", "add", "-m", json.dumps({"session": "s1", "agent": "x"}), sha)
        match_sessions_to_commits(db)
        assert [(r[1], r[2], r[3]) for r in _rows(db, "s1")] == [(sha, "deterministic", "git_note")]
        # Never written: the notes ref is exactly what it was.
        assert _git(repo, "notes", "--ref=ai", "show", sha) == json.dumps({"session": "s1", "agent": "x"})
    finally:
        db.close()


def test_ai_trailer_on_the_session_branch_is_inferred_and_a_later_pass_upgrades_never_downgrades(repo):
    db = InMemoryBackend()
    try:
        _session(db, "s1", repo, T0, T0 + timedelta(minutes=10))
        sha = _commit(repo, "a.py", "1\n", "feat\n\nCo-Authored-By: Claude <noreply@anthropic.com>",
                      T0 + timedelta(minutes=3))
        match_sessions_to_commits(db)
        assert [(r[1], r[2], r[3]) for r in _rows(db, "s1")] == [(sha, "inferred", "trailer_window")]
        # Stronger evidence arrives (a tool span) and the session changes: upgrade.
        _bash(db, "s1", T0 + timedelta(minutes=3), "git commit -m feat")
        db.upsert_session(replace(db.get_session("s1"), ended_at=T0 + timedelta(minutes=12)))
        match_sessions_to_commits(db)
        assert [(r[2], r[3]) for r in _rows(db, "s1")] == [("deterministic", "tool_span_git_log")]
        # An `inferred` write can never take a deterministic row back down.
        db.upsert_session_commits([SessionCommit("s1", sha, "inferred", "trailer_window")])
        assert [(r[2], r[3]) for r in _rows(db, "s1")] == [("deterministic", "tool_span_git_log")]
    finally:
        db.close()


def test_a_parents_trailer_on_a_commit_a_subagent_ran_ranks_below_the_tool_span(repo):
    """Issue #770 fix 6. A subagent spawned by a session inherits the parent's
    `CLAUDE_CODE_SESSION_ID`, so a commit the subagent's own Bash tool ran
    carries the PARENT's `TokenJam-Session:` trailer. The subagent produced
    it (`tool_span_git_log`, deterministic); the parent's row is inherited
    evidence and is written at `inferred`, whichever of the two sessions the
    pass scans first."""
    db = InMemoryBackend()
    try:
        _session(db, "parent", repo, T0, T0 + timedelta(minutes=30))
        _session(db, "worker", repo, T0 + timedelta(minutes=1), T0 + timedelta(minutes=20))
        _bash(db, "worker", T0 + timedelta(minutes=5), "git commit -m 'feat'")
        sha = _commit(repo, "a.py", "1\n", "feat\n\nTokenJam-Session: parent",
                      T0 + timedelta(minutes=5, seconds=3))
        match_sessions_to_commits(db)
        assert _rows(db, "worker") == [("worker", sha, "deterministic", "tool_span_git_log", -3.0)]
        assert _rows(db, "parent") == [("parent", sha, "inferred", "trailer_session", None)]
        # A second pass changes nothing (idempotent, and no upgrade is offered).
        match_sessions_to_commits(db)
        assert [(r[2], r[3]) for r in _rows(db, "parent")] == [("inferred", "trailer_session")]
    finally:
        db.close()


def test_a_trailer_naming_the_session_that_ran_the_commit_stays_deterministic(repo):
    """The control for the rule above: the trailer agrees with the tool span,
    so nothing is demoted, and the `Claude-Session:` spelling resolves through
    the bridge id the same way."""
    db = InMemoryBackend()
    try:
        _session(db, "s1", repo, T0, T0 + timedelta(minutes=10), bridge="cse_01AbC")
        _session(db, "other", repo, T0 + timedelta(hours=2), T0 + timedelta(hours=3))
        _bash(db, "s1", T0 + timedelta(minutes=3), "git commit -m feat")
        sha = _commit(repo, "a.py", "1\n",
                      "feat\n\nClaude-Session: https://claude.ai/code/session_01AbC",
                      T0 + timedelta(minutes=3, seconds=1))
        # A commit with ONLY a trailer, and nobody's tool span near it: still
        # deterministic, because no other producer is known.
        alone = _commit(repo, "b.py", "2\n", "docs\n\nTokenJam-Session: other",
                        T0 + timedelta(hours=2, minutes=30))
        match_sessions_to_commits(db)
        assert [(r[1], r[2], r[3]) for r in _rows(db, "s1")] == [(sha, "deterministic", "tool_span_git_log")]
        assert [(r[1], r[2], r[3]) for r in _rows(db, "other")] == [(alone, "deterministic", "trailer_session")]
    finally:
        db.close()


def test_trailer_rank_is_pure():
    assert shipped.trailer_rank("p", set()) == ("deterministic", "trailer_session")
    assert shipped.trailer_rank("p", {"p"}) == ("deterministic", "trailer_session")
    assert shipped.trailer_rank("p", {"w"}) == ("inferred", "trailer_session")
    assert shipped.trailer_rank("p", {"p", "w"}) == ("deterministic", "trailer_session")


def test_commit_tool_index_scopes_producers_to_the_repo():
    at = T0
    index = shipped.CommitToolIndex(
        {"a": [at], "b": [at + timedelta(seconds=10)], "far": [at + timedelta(minutes=5)],
         "local": [at + timedelta(seconds=2)], "unknown": [at + timedelta(seconds=3)]},
        repos={"a": (REMOTE, "/srv/w"), "b": ("https://github.com/Other/repo", "/srv/o"),
               "far": (REMOTE, "/srv/w"),
               # No origin: a local-only checkout is known by its root alone.
               "local": (None, "/srv/local"),
               # Neither known: never evidence about anyone's commit.
               "unknown": (None, None)},
    )
    assert index.producers(at) == {"a", "b", "local", "unknown"}
    assert index.producers(at, REMOTE, "/srv/w") == {"a"}
    # A remote-less session is a producer only for ITS root, never a
    # wildcard over every remote (Greptile P2 on #771).
    assert index.producers(at, REMOTE, "/srv/elsewhere") == {"a"}
    assert index.producers(at, None, "/srv/local") == {"local"}
    assert index.best_delta("a", at + timedelta(seconds=4)) == -4.0
    assert index.best_delta("far", at) is None


def test_tool_span_index_is_bounded_to_the_candidates_window(repo):
    """The producers index reads Bash spans inside the candidate sessions'
    combined window, not the whole retained history (Greptile P2 on #771):
    a commit call a year before any candidate is never loaded."""
    db = InMemoryBackend()
    try:
        _session(db, "s1", repo, T0, T0 + timedelta(minutes=10))
        _bash(db, "s1", T0 + timedelta(minutes=5), "git commit -m feat")
        _session(db, "ancient", repo, T0 - timedelta(days=400), T0 - timedelta(days=400, minutes=-10))
        _bash(db, "ancient", T0 - timedelta(days=400, minutes=-5), "git commit -m old")
        # Mark `ancient` as scanned so only s1 is a candidate.
        db.conn.execute(
            "INSERT INTO session_commit_scans (session_id, ended_at, scanned_at) VALUES ($1,$2,$3)",
            ["ancient", T0 - timedelta(days=400, minutes=-10), utcnow()],
        )
        lo, hi = shipped._Session("s1", str(repo), REMOTE, "main", "main", EMAIL, None,
                                  T0, T0 + timedelta(minutes=10)).window
        by_session, repos = shipped._commit_tool_spans(db.conn, lo, hi)
        assert set(by_session) == {"s1"}
        assert repos["s1"] == (REMOTE, str(repo))
        whole, _ = shipped._commit_tool_spans(db.conn, T0 - timedelta(days=401), hi)
        assert set(whole) == {"s1", "ancient"}
    finally:
        db.close()


def test_a_plain_commit_in_the_window_with_no_evidence_is_not_joined(repo):
    db = InMemoryBackend()
    try:
        _session(db, "s1", repo, T0, T0 + timedelta(minutes=10))
        _commit(repo, "a.py", "1\n", "human commit", T0 + timedelta(minutes=3))
        match_sessions_to_commits(db)
        assert _rows(db, "s1") == []
    finally:
        db.close()


def test_re_run_is_idempotent_and_bounded_by_the_scan_watermark(repo, monkeypatch):
    db = InMemoryBackend()
    try:
        _session(db, "s1", repo, T0, T0 + timedelta(minutes=10))
        _bash(db, "s1", T0 + timedelta(minutes=5), "git commit -m feat")
        _commit(repo, "a.py", "1\n", "feat", T0 + timedelta(minutes=5))
        first = match_sessions_to_commits(db)
        assert first.rows_written == 1 and first.sessions_scanned == 1
        before = _rows(db)
        second = match_sessions_to_commits(db)
        assert second.sessions_scanned == 0 and second.rows_written == 0
        assert _rows(db) == before
        # A session ingested AFTER the pass (older history) has no scan row, so
        # it is picked up regardless of any timestamp.
        _session(db, "s0", repo, T0 - timedelta(days=3), T0 - timedelta(days=3, hours=-1))
        third = match_sessions_to_commits(db)
        assert third.sessions_scanned == 1
    finally:
        db.close()


def test_a_missing_repo_root_is_skipped_unless_a_live_root_shares_its_remote(repo, tmp_path):
    db = InMemoryBackend()
    try:
        gone = str(tmp_path / "deleted-worktree")
        _session(db, "s1", repo, T0, T0 + timedelta(minutes=10), root=gone)
        _bash(db, "s1", T0 + timedelta(minutes=5), "git commit -m feat")
        sha = _commit(repo, "a.py", "1\n", "feat", T0 + timedelta(minutes=5))
        result = match_sessions_to_commits(db)
        assert result.repos_missing == 1 and _rows(db, "s1") == []
        # Another session with a live root on the same remote makes it joinable.
        _session(db, "s2", repo, T0 + timedelta(days=1), T0 + timedelta(days=1, minutes=1))
        result = match_sessions_to_commits(db)
        assert result.repos_missing == 0
        assert [r[1] for r in _rows(db, "s1")] == [sha]
    finally:
        db.close()


# --- Shipped state -----------------------------------------------------------------

def test_shipped_state_derives_from_default_branch_membership_and_reverts(repo):
    db = InMemoryBackend()
    try:
        # s1: commit on main -> shipped.
        _session(db, "s1", repo, T0, T0 + timedelta(minutes=10))
        _bash(db, "s1", T0 + timedelta(minutes=5), "git commit -m one")
        one = _commit(repo, "a.py", "1\n", "one", T0 + timedelta(minutes=5))
        # s2: commit on a feature branch not merged -> committed.
        _git(repo, "checkout", "-q", "-b", "feat/x")
        _session(db, "s2", repo, T0 + timedelta(hours=1), T0 + timedelta(hours=1, minutes=10), branch="feat/x")
        _bash(db, "s2", T0 + timedelta(hours=1, minutes=5), "git commit -m two")
        _commit(repo, "b.py", "2\n", "two", T0 + timedelta(hours=1, minutes=5))
        _git(repo, "checkout", "-q", "main")
        # s3: commit on main, then reverted -> reverted.
        _session(db, "s3", repo, T0 + timedelta(hours=2), T0 + timedelta(hours=2, minutes=10))
        _bash(db, "s3", T0 + timedelta(hours=2, minutes=5), "git commit -m three")
        three = _commit(repo, "c.py", "3\n", "three", T0 + timedelta(hours=2, minutes=5))
        _git(repo, "revert", "--no-edit", three, at=T0 + timedelta(hours=3))
        # s4: no commit at all -> unshipped. s5: no repo -> no_repo.
        _session(db, "s4", repo, T0 + timedelta(hours=4), T0 + timedelta(hours=4, minutes=10))
        db.upsert_session(make_session(session_id="s5", agent_id="claude-code-w", started_at=T0 + timedelta(hours=5)))
        match_sessions_to_commits(db)
        assert session_shipped_state(db.conn, "s1") == STATE_SHIPPED
        assert session_shipped_state(db.conn, "s2") == STATE_COMMITTED
        assert session_shipped_state(db.conn, "s3") == STATE_REVERTED
        assert session_shipped_state(db.conn, "s4") == STATE_UNSHIPPED
        assert session_shipped_state(db.conn, "s5") == STATE_NO_REPO
        states = shipped.shipped_states(db.conn, ["s1", "s2", "s3", "s4", "s5"])
        assert states["s1"] == {"shipped_state": STATE_SHIPPED, "commit_count": 1}
        assert states["s5"]["shipped_state"] == STATE_NO_REPO
        assert one in [r[1] for r in _rows(db, "s1")]

        summary = shipped_summary(db.conn, T0 - timedelta(hours=1), T0 + timedelta(days=1))
        assert (summary.sessions_total, summary.sessions_shipped, summary.sessions_committed,
                summary.sessions_unshipped, summary.sessions_no_repo) == (4, 1, 1, 2, 1)
        assert summary.cost_unshipped_usd == pytest.approx(2.0)   # s3 (reverted) + s4
        assert summary.cost_shipped_usd == pytest.approx(1.0)
        assert {r["session_id"] for r in summary.top_unshipped} == {"s3", "s4"}
        assert summary.caveat == SHIPPED_CAVEAT
        # Coverage: default-branch commits by this author in the window are
        # `one`, `three`; both joined (the revert commit itself is excluded).
        assert summary.coverage == pytest.approx(1.0)
        # A merge of feat/x later flips s2 to shipped on the next pass.
        _git(repo, "merge", "-q", "--no-ff", "feat/x", "-m", "merge", at=T0 + timedelta(hours=6))
        db.upsert_session(replace(db.get_session("s2"), ended_at=T0 + timedelta(hours=1, minutes=11)))
        match_sessions_to_commits(db)
        assert session_shipped_state(db.conn, "s2") == STATE_SHIPPED
    finally:
        db.close()


def test_a_surviving_commit_beside_a_reverted_one_still_ships(repo):
    """Review finding: `on_default` already excludes reverted commits, so a
    session with one surviving default-branch commit and one reverted commit
    is shipped, not committed."""
    db = InMemoryBackend()
    try:
        _session(db, "s1", repo, T0, T0 + timedelta(minutes=10))
        _bash(db, "s1", T0 + timedelta(minutes=2), "git commit -m keep")
        _commit(repo, "keep.py", "1\n", "keep", T0 + timedelta(minutes=2))
        _bash(db, "s1", T0 + timedelta(minutes=5), "git commit -m drop")
        drop = _commit(repo, "drop.py", "1\n", "drop", T0 + timedelta(minutes=5))
        _git(repo, "revert", "--no-edit", drop, at=T0 + timedelta(hours=1))
        match_sessions_to_commits(db)
        assert session_shipped_state(db.conn, "s1") == STATE_SHIPPED
        assert shipped.shipped_states(db.conn, ["s1"])["s1"]["commit_count"] == 2
    finally:
        db.close()


def test_a_rewritten_default_branch_drops_the_commits_it_no_longer_holds(repo):
    """Review finding: a force-pushed-away commit must not stay "shipped"."""
    db = InMemoryBackend()
    try:
        _session(db, "s1", repo, T0, T0 + timedelta(minutes=10))
        _bash(db, "s1", T0 + timedelta(minutes=5), "git commit -m feat")
        sha = _commit(repo, "a.py", "1\n", "feat", T0 + timedelta(minutes=5))
        _git(repo, "push", "-q", "origin", "main")
        match_sessions_to_commits(db)
        assert session_shipped_state(db.conn, "s1") == STATE_SHIPPED
        # History rewritten: main and origin/main no longer contain the commit.
        _git(repo, "reset", "-q", "--hard", "HEAD~1")
        _git(repo, "push", "-q", "--force", "origin", "main")
        db.upsert_session(replace(db.get_session("s1"), ended_at=T0 + timedelta(minutes=11)))
        match_sessions_to_commits(db)
        assert db.conn.execute(
            "SELECT COUNT(*) FROM repo_commits WHERE commit_sha = $1", [sha]).fetchone()[0] == 0
        assert session_shipped_state(db.conn, "s1") == STATE_COMMITTED
    finally:
        db.close()


def test_a_commit_only_on_the_remote_default_branch_counts_as_shipped(repo):
    """Review finding: tj never fetches, so a stale local `main` behind
    `origin/main` must not hide an upstream merge. Both refs are indexed."""
    db = InMemoryBackend()
    try:
        _session(db, "s1", repo, T0, T0 + timedelta(minutes=10))
        _bash(db, "s1", T0 + timedelta(minutes=5), "git commit -m feat")
        _commit(repo, "a.py", "1\n", "feat", T0 + timedelta(minutes=5))
        _git(repo, "push", "-q", "origin", "main")
        # Local main falls behind the remote (the checkout moves elsewhere).
        _git(repo, "reset", "-q", "--hard", "HEAD~1")
        match_sessions_to_commits(db)
        assert session_shipped_state(db.conn, "s1") == STATE_SHIPPED
    finally:
        db.close()


def test_late_evidence_is_picked_up_by_the_periodic_rescan(repo):
    """Review finding: a note attached after the first scan does not move
    `ended_at`; a recent session is re-scanned once its scan is a day old."""
    db = InMemoryBackend()
    try:
        _session(db, "s1", repo, T0, T0 + timedelta(minutes=10))
        sha = _commit(repo, "a.py", "1\n", "feat", T0 + timedelta(minutes=3))
        match_sessions_to_commits(db, now=T0 + timedelta(hours=1))
        assert _rows(db, "s1") == []
        _git(repo, "notes", "--ref=ai", "add", "-m", json.dumps({"session": "s1"}), sha)
        # Same day: the watermark holds.
        assert match_sessions_to_commits(db, now=T0 + timedelta(hours=2)).sessions_scanned == 0
        # A day later: re-scanned, and the note is found.
        assert match_sessions_to_commits(db, now=T0 + timedelta(days=1, hours=2)).sessions_scanned == 1
        assert [(r[2], r[3]) for r in _rows(db, "s1")] == [("deterministic", "git_note")]
        # Past the horizon the session is left alone.
        db.conn.execute("UPDATE session_commit_scans SET scanned_at = $1", [T0])
        assert match_sessions_to_commits(db, now=T0 + timedelta(days=40)).sessions_scanned == 0
    finally:
        db.close()


def test_the_same_remote_indexed_under_two_roots_does_not_double_count(repo, tmp_path):
    """Review finding: a clone and a worktree of one remote both land in
    `repo_commits`; a session's commit must still count once."""
    db = InMemoryBackend()
    try:
        _session(db, "s1", repo, T0, T0 + timedelta(minutes=10))
        _bash(db, "s1", T0 + timedelta(minutes=5), "git commit -m feat")
        _commit(repo, "a.py", "1\n", "feat", T0 + timedelta(minutes=5))
        _git(repo, "push", "-q", "origin", "main")
        clone = tmp_path / "clone"
        _git(repo, "clone", "-q", str(tmp_path / "origin.git"), str(clone))
        _session(db, "s2", clone, T0 + timedelta(hours=1), T0 + timedelta(hours=1, minutes=10))
        match_sessions_to_commits(db)
        assert db.conn.execute("SELECT COUNT(DISTINCT repo_root) FROM repo_commits").fetchone()[0] == 2
        states = shipped.shipped_states(db.conn, ["s1"])
        assert states["s1"] == {"shipped_state": STATE_SHIPPED, "commit_count": 1}
        summary = shipped_summary(db.conn, T0 - timedelta(hours=1), T0 + timedelta(days=1))
        assert summary.commits_joined == 1 and summary.commits_on_default == 1
    finally:
        db.close()


# --- Rework + loop cost ------------------------------------------------------------

def test_rework_cost_counts_sessions_whose_added_lines_were_mostly_deleted_later(repo):
    db = InMemoryBackend()
    try:
        _session(db, "s1", repo, T0, T0 + timedelta(minutes=10), cost=5.0)
        _bash(db, "s1", T0 + timedelta(minutes=5), "git commit -m add")
        _commit(repo, "mod.py", "".join(f"line {i}\n" for i in range(10)), "add", T0 + timedelta(minutes=5))
        # Another session two days later rewrites 8 of the 10 lines.
        _session(db, "s2", repo, T0 + timedelta(days=2), T0 + timedelta(days=2, minutes=10), cost=1.0)
        _bash(db, "s2", T0 + timedelta(days=2, minutes=5), "git commit -m rewrite")
        _commit(repo, "mod.py", "line 0\nline 1\n" + "".join(f"new {i}\n" for i in range(8)),
                "rewrite", T0 + timedelta(days=2, minutes=5))
        match_sessions_to_commits(db, now=T0 + timedelta(days=30))
        summary = shipped_summary(db.conn, T0 - timedelta(hours=1), T0 + timedelta(days=3))
        assert summary.cost_rework_usd == pytest.approx(5.0)
        assert summary.top_reworked == [{"path": "mod.py", "sessions": 1, "cost_usd": 5.0}]
        assert "14 days" in summary.rework_basis
        rows = db.conn.execute(
            "SELECT additions, later_deletions, horizon_closed FROM commit_file_stats WHERE path = 'mod.py' "
            "ORDER BY additions DESC").fetchall()
        assert rows[0] == (10, 8, True)
    finally:
        db.close()


def test_loop_cost_prices_repeated_identical_edits(repo):
    db = InMemoryBackend()
    try:
        sid = "s1"
        _session(db, sid, repo, T0, T0 + timedelta(minutes=10))

        def edit(at, new, cost):
            parent = make_llm_span(agent_id="claude-code-widgets", session_id=sid, start_time=at,
                                   cost_usd=cost, input_tokens=10, output_tokens=5)
            db.insert_span(parent)
            span = make_tool_span(agent_id="claude-code-widgets", tool_name="Edit", session_id=sid,
                                  start_time=at, tool_input={
                                      "file_path": "/w/a.py", "old_string": "x", "new_string": new})
            db.insert_span(replace(span, parent_span_id=parent.span_id, trace_id=parent.trace_id))

        edit(T0 + timedelta(minutes=1), "A", 0.10)
        edit(T0 + timedelta(minutes=2), "A", 0.20)   # identical: a loop
        edit(T0 + timedelta(minutes=3), "B", 0.30)   # changed: not a loop
        summary = shipped_summary(db.conn, T0 - timedelta(hours=1), T0 + timedelta(days=1))
        assert summary.cost_loop_usd == pytest.approx(0.20)
        assert "1 repeat edit" in summary.loop_basis
    finally:
        db.close()


# --- Analyzer + surfaces -----------------------------------------------------------

def test_shipped_is_registered_ordered_rebuildable_and_a_click_choice():
    from tokenjam.cli.cmd_optimize import _FINDING_RENDERERS, _UNPRICED_IN_SCOREBOARD
    from tokenjam.core.optimize import ANALYZER_ORDER, ANALYZER_REGISTRY
    from tokenjam.core.optimize.rank import CARD_FINDING_NAMES
    from tokenjam.core.optimize.runner import finding_class_names

    assert "shipped" in ANALYZER_REGISTRY and "shipped" in ANALYZER_ORDER
    assert "shipped" in finding_class_names()
    assert "shipped" in CARD_FINDING_NAMES and "shipped" in _FINDING_RENDERERS
    assert "shipped" in _UNPRICED_IN_SCOREBOARD
    from tokenjam.cli.main import cli
    result = CliRunner().invoke(cli, ["optimize", "--help"])
    assert result.exit_code == 0 and "shipped" in result.output


def test_shipped_finding_carries_the_caveat_as_a_default():
    from tokenjam.core.optimize.analyzers.shipped import ShippedFinding

    assert ShippedFinding().caveat == SHIPPED_CAVEAT
    assert "past_overspend_usd" not in ShippedFinding().__dataclass_fields__


def test_tj_optimize_shipped_human_and_json(repo, monkeypatch):
    from unittest.mock import patch

    from tokenjam.cli.main import cli

    db = InMemoryBackend()
    try:
        now = utcnow()
        _session(db, "s1", repo, now - timedelta(days=2), now - timedelta(days=2) + timedelta(minutes=10), cost=3.0)
        _bash(db, "s1", now - timedelta(days=2, minutes=-5), "git commit -m feat")
        _commit(repo, "a.py", "1\n", "feat", now - timedelta(days=2, minutes=-5))
        _session(db, "s2", repo, now - timedelta(days=1), now - timedelta(days=1) + timedelta(minutes=10), cost=7.0)
        db.insert_span(make_llm_span(agent_id="claude-code-widgets", session_id="s2",
                                     start_time=now - timedelta(days=1), cost_usd=7.0))
        config = TjConfig(version="1")
        with patch("tokenjam.cli.main.load_config", return_value=config), \
             patch("tokenjam.cli.main.open_db", return_value=db):
            human = CliRunner().invoke(cli, ["optimize", "shipped", "--since", "7d"])
            assert human.exit_code == 0, human.output
            assert "Shipped" in human.output and "1 of 2" in human.output
            assert "Review before" in human.output and "measured" in human.output
            as_json = CliRunner().invoke(cli, ["--json", "optimize", "shipped", "--since", "7d"])
            assert as_json.exit_code == 0, as_json.output
            f = json.loads(as_json.output)["findings"]["shipped"]
            assert f["sessions_shipped"] == 1 and f["sessions_unshipped"] == 1
            assert f["cost_unshipped_usd"] == pytest.approx(7.0)
            assert f["caveat"] == SHIPPED_CAVEAT
            assert f["top_unshipped"][0]["session_id"] == "s2"
    finally:
        db.close()


def test_shipped_api_and_session_detail_shape(repo):
    import asyncio

    import httpx

    from tokenjam.api.app import create_app
    from tokenjam.core.ingest import IngestPipeline

    db = InMemoryBackend()
    try:
        now = utcnow()
        _session(db, "s1", repo, now - timedelta(days=2), now - timedelta(days=2) + timedelta(minutes=10))
        _bash(db, "s1", now - timedelta(days=2, minutes=-5), "git commit -m feat")
        sha = _commit(repo, "a.py", "1\n", "feat", now - timedelta(days=2, minutes=-5))
        _session(db, "s2", repo, now - timedelta(days=1), now - timedelta(days=1) + timedelta(minutes=10))
        match_sessions_to_commits(db)
        config = TjConfig(version="1")
        app = create_app(config=config, db=db, ingest_pipeline=IngestPipeline(db=db, config=config))

        async def go():
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
                return (
                    (await c.get("/api/v1/shipped", params={"since": "7d"})).json(),
                    (await c.get("/api/v1/sessions/s1")).json(),
                    (await c.get("/api/v1/sessions/s2")).json(),
                    (await c.get("/api/v1/sessions")).json(),
                    (await c.get("/api/v1/status")).json(),
                )

        shipped_payload, s1, s2, listing, status = asyncio.run(go())
        for key in ("window_days", "sessions_total", "sessions_shipped", "sessions_unshipped",
                    "cost_shipped_usd", "cost_unshipped_usd", "cost_rework_usd", "cost_loop_usd",
                    "coverage", "top_unshipped", "top_reworked", "caveat", "framing"):
            assert key in shipped_payload, key
        assert shipped_payload["sessions_shipped"] == 1 and shipped_payload["caveat"] == SHIPPED_CAVEAT
        assert s1["session"]["shipped_state"] == STATE_SHIPPED
        assert s1["commits"] == [{"sha": sha, "confidence": "deterministic", "source": "tool_span_git_log",
                                  "committed_at": s1["commits"][0]["committed_at"],
                                  "author_email": EMAIL,
                                  "match_delta_s": s1["commits"][0]["match_delta_s"]}]
        assert s2["session"]["shipped_state"] == STATE_UNSHIPPED and s2["commits"] == []
        by_id = {r["session_id"]: r for r in listing["sessions"]}
        assert by_id["s1"]["shipped_state"] == STATE_SHIPPED and by_id["s1"]["commit_count"] == 1
        archived = {r["session_id"]: r for r in status["archived"]}
        assert archived["s2"]["shipped_state"] == STATE_UNSHIPPED
    finally:
        db.close()


def test_tj_status_agent_card_shows_shipped_line(repo):
    from unittest.mock import patch

    from tokenjam.cli.main import cli

    db = InMemoryBackend()
    try:
        now = utcnow()
        _session(db, "s1", repo, now - timedelta(days=2), now - timedelta(days=2) + timedelta(minutes=10), cost=2.0)
        _bash(db, "s1", now - timedelta(days=2, minutes=-5), "git commit -m feat")
        _commit(repo, "a.py", "1\n", "feat", now - timedelta(days=2, minutes=-5))
        _session(db, "s2", repo, now - timedelta(days=1), now - timedelta(days=1) + timedelta(minutes=10), cost=4.5)
        match_sessions_to_commits(db)
        config = TjConfig(version="1")
        with patch("tokenjam.cli.main.load_config", return_value=config), \
             patch("tokenjam.cli.main.open_db", return_value=db):
            human = CliRunner().invoke(cli, ["status", "--agent", "claude-code-widgets"])
            assert human.exit_code == 0, human.output
            assert "Shipped 1 of 2 sessions" in human.output and "$4.50 unshipped" in human.output
            payload = json.loads(CliRunner().invoke(cli, ["--json", "status", "--agent", "claude-code-widgets"]).output)
            assert payload["agents"][0]["shipped"] == {
                "sessions_total": 2, "sessions_shipped": 1, "cost_unshipped_usd": 4.5, "window_days": 30,
            }
    finally:
        db.close()


def test_lens_shipped_helpers_execute_under_node():
    if shutil.which("node") is None:
        pytest.skip("node not available for JS evaluation")
    ui = Path(__file__).parent.parent.parent / "tokenjam" / "ui" / "index.html"
    html = ui.read_text(encoding="utf-8")
    start = html.index("const CONFIDENCE_GLYPH")
    end = html.index("\n}\n", html.index("function shippedFilterKeep", start)) + 3
    cases = [
        [{"shipped_state": "shipped"}, "shipped"],
        [{"shipped_state": "unshipped"}, "unshipped"],
        [{"shipped_state": "reverted"}, "unshipped"],
        [{"shipped_state": "committed"}, "shipped"],
        [{"shipped_state": "no_repo"}, "unshipped"],
        [{"shipped_state": "no_repo"}, ""],
        [{}, "unshipped"],
    ]
    script = (
        html[start:end]
        + "\nconsole.log(JSON.stringify(["
        + "['deterministic','inferred','estimated','x'].map(confidenceGlyph),"
        + json.dumps(cases) + ".map(([r, f]) => shippedFilterKeep(r, f))]));"
    )
    proc = subprocess.run(["node", "--input-type=module", "-e", script],
                          capture_output=True, text=True, check=True)
    glyphs, keeps = json.loads(proc.stdout)
    assert len(set(glyphs[:3])) == 3 and glyphs[3] == ""
    assert keeps == [True, True, True, False, False, True, False]
