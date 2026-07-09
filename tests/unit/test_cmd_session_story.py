"""Unit tests for `tj session-story` (cli/cmd_session_story.py).

Exercises the full CLI path — session resolution (--session / default-to-most-
recent-substantial), story loading (live transcript, honest "no transcript"
fallback), rendering, and --json — via CliRunner + InMemoryBackend, mirroring
test_cmd_cost.py's invocation pattern. Transcript fixtures are hand-written
Claude Code JSONL records, mirroring test_transcript.py's fixture-builder style
(these are on-disk transcript records, not NormalizedSpans, so the span
factories don't apply).
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from click.testing import CliRunner

from tokenjam.cli.main import cli
from tokenjam.core.config import ApiAuthConfig, ApiConfig, TjConfig
from tokenjam.core.db import InMemoryBackend
from tests.factories import make_session


# --- Transcript fixture builders (mirrors test_transcript.py) ---------------

def _user_prompt(text: str) -> dict:
    return {"type": "user", "message": {"role": "user", "content": text}}


def _tool_result(tool_use_id: str, is_error: bool = False) -> dict:
    block: dict = {
        "type": "tool_result", "tool_use_id": tool_use_id, "content": "(omitted)",
    }
    if is_error:
        block["is_error"] = True
    return {"type": "user", "message": {"role": "user", "content": [block]}}


def _assistant(
    text: str | None, tools: list[dict] | None = None,
    model: str = "claude-opus-4-8", ts: str = "2026-06-15T09:11:36.133Z",
) -> dict:
    content: list[dict] = []
    if text is not None:
        content.append({"type": "text", "text": text})
    for t in tools or []:
        content.append({
            "type": "tool_use", "id": t["id"], "name": t["name"],
            "input": t.get("input", {}),
        })
    return {
        "type": "assistant", "timestamp": ts,
        "message": {"role": "assistant", "model": model, "content": content},
    }


def _write_transcript(projects_root: Path, session_id: str, records: list[dict]) -> Path:
    project_dir = projects_root / "-Users-test-project"
    project_dir.mkdir(parents=True, exist_ok=True)
    path = project_dir / f"{session_id}.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")
    return path


def _agent_tool_result(tool_use_id: str, agent_id: str) -> dict:
    text = f"Agent (agentId: {agent_id}) finished. See results above."
    block = {"type": "tool_result", "tool_use_id": tool_use_id, "content": text}
    return {"type": "user", "message": {"role": "user", "content": [block]}}


def _task_turn(text: str, tool_id: str, name: str = "do work") -> dict:
    return _assistant(text, tools=[{"id": tool_id, "name": "Task", "input": {
        "description": name, "subagent_type": "general-purpose", "prompt": "do the work",
    }}])


def _write_subagent(
    projects_root: Path, root_session_id: str, agent_id: str,
    records: list[dict], name: str | None = None,
) -> None:
    subdir = projects_root / "-Users-test-project" / root_session_id / "subagents"
    subdir.mkdir(parents=True, exist_ok=True)
    (subdir / f"agent-{agent_id}.jsonl").write_text(
        "\n".join(json.dumps(r) for r in records), encoding="utf-8"
    )
    if name is not None:
        meta = {"agentType": "general-purpose", "name": name,
                "description": name, "toolUseId": "toolu_x"}
        (subdir / f"agent-{agent_id}.meta.json").write_text(json.dumps(meta), encoding="utf-8")


def _simple_session_records(task: str, outcome: str) -> list[dict]:
    return [
        _user_prompt(task),
        _assistant("Let me look at that.", tools=[
            {"id": "t1", "name": "Read", "input": {"file_path": "src/app.py"}}]),
        _tool_result("t1"),
        _assistant(outcome),
    ]


# --- Test scaffolding ---------------------------------------------------------

def _config() -> TjConfig:
    return TjConfig(version="1", api=ApiConfig(auth=ApiAuthConfig(enabled=False)))


def _invoke(db, args):
    runner = CliRunner()
    with patch("tokenjam.cli.main.load_config", return_value=_config()), \
         patch("tokenjam.cli.main.open_db", return_value=db):
        return runner.invoke(cli, args)


def _seed_session(db, *, session_id: str, tool_call_count: int = 3,
                  started_at=None) -> None:
    db.upsert_session(make_session(
        agent_id="a", session_id=session_id, tool_call_count=tool_call_count,
        started_at=started_at,
    ))


def db_fixture():
    backend = InMemoryBackend()
    return backend


# --- Tests --------------------------------------------------------------------

def test_no_sessions_shows_honest_message_not_a_crash():
    db = db_fixture()
    result = _invoke(db, ["session-story"])
    assert result.exit_code == 0, result.output
    assert "no sessions found" in result.output.lower()


def test_default_picks_most_recent_substantial_session(tmp_path, monkeypatch):
    monkeypatch.setenv("TJ_CLAUDE_PROJECTS_ROOT", str(tmp_path))
    db = db_fixture()

    now = datetime.now(timezone.utc)
    _seed_session(db, session_id="sess-old", started_at=now - timedelta(days=2))
    _seed_session(db, session_id="sess-recent", started_at=now)
    _write_transcript(tmp_path, "sess-old", _simple_session_records(
        "Old task.", "Old outcome."))
    _write_transcript(tmp_path, "sess-recent", _simple_session_records(
        "Fix the failing auth test.", "All tests pass now."))

    result = _invoke(db, ["session-story"])
    assert result.exit_code == 0, result.output
    assert "sess-recent" in result.output
    assert "Fix the failing auth test" in result.output
    assert "Old task" not in result.output


def test_explicit_session_flag_overrides_default(tmp_path, monkeypatch):
    monkeypatch.setenv("TJ_CLAUDE_PROJECTS_ROOT", str(tmp_path))
    db = db_fixture()

    now = datetime.now(timezone.utc)
    _seed_session(db, session_id="sess-old", started_at=now - timedelta(days=2))
    _seed_session(db, session_id="sess-recent", started_at=now)
    _write_transcript(tmp_path, "sess-old", _simple_session_records(
        "Old task.", "Old outcome."))
    _write_transcript(tmp_path, "sess-recent", _simple_session_records(
        "Recent task.", "Recent outcome."))

    result = _invoke(db, ["session-story", "--session", "sess-old"])
    assert result.exit_code == 0, result.output
    assert "sess-old" in result.output
    assert "Old task" in result.output
    assert "Recent task" not in result.output


def test_session_with_no_transcript_and_no_snapshot_is_honest(tmp_path, monkeypatch):
    monkeypatch.setenv("TJ_CLAUDE_PROJECTS_ROOT", str(tmp_path))
    db = db_fixture()
    _seed_session(db, session_id="sess-gone")
    # No transcript written, no session_story snapshot row.

    result = _invoke(db, ["session-story", "--session", "sess-gone"])
    assert result.exit_code == 0, result.output
    assert "no on-disk transcript" in result.output.lower()


def test_json_output_shape(tmp_path, monkeypatch):
    monkeypatch.setenv("TJ_CLAUDE_PROJECTS_ROOT", str(tmp_path))
    db = db_fixture()
    _seed_session(db, session_id="sess-json")
    _write_transcript(tmp_path, "sess-json", _simple_session_records(
        "Fix the bug.", "Bug fixed."))

    result = _invoke(db, ["session-story", "--session", "sess-json", "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["available"] is True
    assert payload["session_id"] == "sess-json"
    assert payload["task"] == "Fix the bug."
    assert payload["outcome"] == "Bug fixed."
    assert isinstance(payload["spine"], list)
    assert payload["spine"][0]["kind"] == "act"
    assert payload["from_snapshot"] is False


def test_json_output_for_unavailable_session_is_honest(tmp_path, monkeypatch):
    monkeypatch.setenv("TJ_CLAUDE_PROJECTS_ROOT", str(tmp_path))
    db = db_fixture()
    _seed_session(db, session_id="sess-gone")

    result = _invoke(db, ["session-story", "--session", "sess-gone", "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["available"] is False
    assert "reason" in payload


def test_subagent_delegation_renders_with_tool_tally(tmp_path, monkeypatch):
    monkeypatch.setenv("TJ_CLAUDE_PROJECTS_ROOT", str(tmp_path))
    db = db_fixture()
    _seed_session(db, session_id="sess-deleg")

    child = "abc123def456abc78"  # 17 hex, matches the agentId shape
    parent = [
        _user_prompt("Fix the failing auth test and make CI green."),
        _task_turn("Delegating the fix to a worker.", "tt1", name="fix-auth"),
        _agent_tool_result("tt1", child),
        _assistant("All tests pass now. CI is green."),
    ]
    _write_transcript(tmp_path, "sess-deleg", parent)
    _write_subagent(tmp_path, "sess-deleg", child, [
        _user_prompt("Implement the fix."),
        _assistant("Reading the auth module.", tools=[
            {"id": "c1", "name": "Read", "input": {"file_path": "src/auth.py"}}]),
        _tool_result("c1"),
        _assistant("Editing the fix in.", tools=[
            {"id": "c2", "name": "Edit", "input": {"file_path": "src/auth.py"}}]),
        _tool_result("c2"),
        _assistant("Done implementing."),
    ], name="fix-auth")

    result = _invoke(db, ["session-story", "--session", "sess-deleg"])
    assert result.exit_code == 0, result.output
    assert "delegate" in result.output
    assert "fix-auth" in result.output
    # Factual per-category tool tally over the subagent's own moves — never a
    # value judgment ("wasted"/"good"/"bad"), per Critical Rule 14.
    assert "1 read" in result.output
    assert "1 edit" in result.output
    assert "wasted" not in result.output.lower()
    assert "good" not in result.output.lower()
