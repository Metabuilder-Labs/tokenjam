"""Unit tests for the zero-install / zero-config first-run (`tj quickstart`, #6).

The contract under test: a user with NO prior setup runs one command and sees
quota composition + a session timeline straight from on-disk Claude Code JSONL —
with no daemon, no onboarding, and crucially **no on-disk DB** (the command uses
a transient in-memory backend and must never call `open_db`).
"""
from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from tokenjam.core.db import InMemoryBackend
from tokenjam.core.session_timeline import (
    compute_session_timeline,
    timeline_to_dict,
)


def _make_session_file(root: Path, session_id: str, cwd: str,
                       records: list[dict]) -> Path:
    project_dir = root / cwd.replace("/", "-")
    project_dir.mkdir(parents=True, exist_ok=True)
    path = project_dir / f"{session_id}.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in records))
    return path


def _assistant(uuid: str, session_id: str, cwd: str, ts: str, *,
               input_tokens: int = 500, output_tokens: int = 200,
               cache_read: int = 8000, cache_creation: int = 0) -> dict:
    return {
        "type": "assistant",
        "uuid": uuid,
        "timestamp": ts,
        "sessionId": session_id,
        "cwd": cwd,
        "message": {
            "model": "claude-sonnet-4-5-20250929",
            "content": [{"type": "text", "text": "ok"}],
            "usage": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cache_read_input_tokens": cache_read,
                "cache_creation_input_tokens": cache_creation,
            },
        },
    }


def _fixture_root(tmp_path: Path) -> Path:
    root = tmp_path / "projects"
    # Two sessions across two projects, recent timestamps.
    _make_session_file(root, "sess-a", "/Users/me/projA", [
        _assistant("a1", "sess-a", "/Users/me/projA", "2026-06-20T10:00:00.000Z"),
        _assistant("a2", "sess-a", "/Users/me/projA", "2026-06-20T10:05:00.000Z"),
    ])
    _make_session_file(root, "sess-b", "/Users/me/projB", [
        _assistant("b1", "sess-b", "/Users/me/projB", "2026-06-21T11:00:00.000Z",
                   cache_read=50000),
    ])
    return root


# ── Session-timeline core (pure logic over an in-memory DB) ──────────────────

def test_timeline_summarizes_backfilled_sessions(tmp_path):
    from tokenjam.core.backfill import ingest_claude_code

    root = _fixture_root(tmp_path)
    db = InMemoryBackend()
    ingest_claude_code(db, root=root)

    timeline = compute_session_timeline(db.conn)

    assert timeline.has_data
    assert timeline.total_sessions == 2
    assert timeline.project_count == 2
    # Most-recent first.
    assert timeline.sessions[0].started_at >= timeline.sessions[-1].started_at
    # Project label is derived from the claude-code-<name> agent_id.
    projects = {s.project for s in timeline.sessions}
    assert "proja" in projects and "projb" in projects


def test_timeline_reread_share_reflects_cache_reads(tmp_path):
    from tokenjam.core.backfill import ingest_claude_code

    root = _fixture_root(tmp_path)
    db = InMemoryBackend()
    ingest_claude_code(db, root=root)

    timeline = compute_session_timeline(db.conn)
    for s in timeline.sessions:
        # Every fixture turn has cache reads, so re-read share is > 0.
        assert s.reread_share > 0
        assert s.total_tokens >= s.cache_tokens


def test_timeline_to_dict_is_json_serialisable(tmp_path):
    from tokenjam.core.backfill import ingest_claude_code

    root = _fixture_root(tmp_path)
    db = InMemoryBackend()
    ingest_claude_code(db, root=root)

    payload = timeline_to_dict(compute_session_timeline(db.conn))
    # Round-trips through json without error.
    round_tripped = json.loads(json.dumps(payload, default=str))
    assert round_tripped["total_sessions"] == 2
    assert len(round_tripped["sessions"]) == 2


def test_timeline_empty_db_has_no_data():
    db = InMemoryBackend()
    timeline = compute_session_timeline(db.conn)
    assert not timeline.has_data
    assert timeline.total_sessions == 0


# ── CLI: the zero-setup first run, with NO on-disk DB ────────────────────────

def _invoke_quickstart(args):
    """Run `tj quickstart` with `open_db` patched to blow up if ever called.

    The whole point of quickstart is that it never opens the on-disk DB or
    contacts the daemon — it manages its own transient in-memory backend.
    """
    import unittest.mock as mock

    from tokenjam.cli.main import cli

    with mock.patch(
        "tokenjam.cli.main.open_db",
        side_effect=AssertionError("quickstart must NOT open the on-disk DB"),
    ):
        return CliRunner().invoke(cli, ["quickstart", *args])


def test_quickstart_renders_without_daemon_or_ondisk_db(tmp_path):
    root = _fixture_root(tmp_path)
    result = _invoke_quickstart(["--root", str(root), "--since", "90d"])

    assert result.exit_code == 0, result.output
    # Leads with the ccusage-parity framing.
    assert "ccusage" in result.output
    assert "~/.claude/projects" in result.output
    # Both halves of the first-run value are present.
    assert "quota" in result.output.lower()
    assert "Session timeline" in result.output
    # The opt-in "go deeper" pointer keeps the daemon path discoverable.
    assert "tj onboard" in result.output


def test_quickstart_json_emits_both_views(tmp_path):
    root = _fixture_root(tmp_path)
    result = _invoke_quickstart(["--root", str(root), "--since", "90d", "--json"])

    assert result.exit_code == 0, result.output
    # The JSON line is the last line (Rich logging may precede it on stderr).
    payload = json.loads(result.output.strip().splitlines()[-1])
    assert "quota_composition" in payload
    assert "session_timeline" in payload
    assert payload["session_timeline"]["total_sessions"] == 2
    assert payload["backfill"]["sessions_ingested"] == 2


def test_quickstart_no_logs_is_graceful(tmp_path):
    missing = tmp_path / "does-not-exist"
    result = _invoke_quickstart(["--root", str(missing)])
    assert result.exit_code == 0, result.output
    assert "No Claude Code logs" in result.output


def test_bare_tj_routes_to_quickstart(tmp_path, monkeypatch):
    """`tj` with no subcommand IS the zero-install first run (one command)."""
    import unittest.mock as mock

    from tokenjam.cli.main import cli

    # Point the default root at an empty dir so the run is fast + deterministic
    # (the no-logs branch), and prove the on-disk DB is never opened.
    missing = tmp_path / "no-cc-logs"
    monkeypatch.setattr(
        "tokenjam.cli.cmd_quickstart.CLAUDE_CODE_PROJECTS_ROOT", missing
    )
    with mock.patch(
        "tokenjam.cli.main.open_db",
        side_effect=AssertionError("bare tj must NOT open the on-disk DB"),
    ):
        result = CliRunner().invoke(cli, [])

    assert result.exit_code == 0, result.output
    # Routed to quickstart's no-logs branch (which names ccusage parity).
    assert "ccusage" in result.output
