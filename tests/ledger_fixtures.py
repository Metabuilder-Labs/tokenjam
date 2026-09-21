"""Shared building blocks for the shipped-value-ledger tests: a REAL git
checkout to resolve repo context against, Claude Code transcripts pointing
at it, and a contextless session row (what the live OTLP path writes).

Imported by name (`repo = pytest.fixture(git_repo)`) rather than as a
fixture module, so each test file owns its fixture and ruff never sees an
imported fixture redefined by a parameter.
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

from tests.factories import make_llm_span, make_session
from tokenjam.core import repo_context
from tokenjam.core.db import DuckDBBackend

EMAIL = "dev@example.com"
REMOTE = "https://github.com/Acme/widgets"
T0 = datetime.now(tz=timezone.utc) - timedelta(hours=2)


def git(repo: Path, *args: str) -> str:
    env = dict(os.environ, GIT_AUTHOR_EMAIL=EMAIL, GIT_COMMITTER_EMAIL=EMAIL,
               GIT_AUTHOR_NAME="Dev", GIT_COMMITTER_NAME="Dev")
    return subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True,
                          check=True, env=env).stdout.strip()


def git_repo(tmp_path, monkeypatch) -> Path:
    """A checkout under `tmp_path` with one commit on `main` and a GitHub
    remote, with the temp-dir skip patched off (contracts §3 change log 1.1)."""
    if shutil.which("git") is None:
        pytest.skip("git not available")
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    (tmp_path / "home").mkdir(exist_ok=True)
    monkeypatch.setattr(repo_context, "_is_temp_cwd", lambda _cwd: False)
    repo_context.clear_caches()
    path = tmp_path / "widgets"
    path.mkdir()
    git(path, "init", "-q", "-b", "main")
    git(path, "config", "user.email", EMAIL)
    git(path, "config", "user.name", "Dev")
    git(path, "remote", "add", "origin", REMOTE + ".git")
    (path / "README").write_text("hi\n")
    git(path, "add", "README")
    git(path, "commit", "-q", "-m", "init")
    yield path
    repo_context.clear_caches()


def transcript_record(session_id: str, cwd: str, uuid: str, at: datetime, *,
                      branch: str = "main") -> dict:
    return {
        "type": "assistant", "uuid": uuid, "sessionId": session_id, "cwd": cwd,
        "gitBranch": branch, "timestamp": at.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
        "message": {
            "id": f"msg_{uuid}", "model": "claude-sonnet-4-5-20250929",
            "content": [{"type": "text", "text": "ok"}],
            "usage": {"input_tokens": 100, "output_tokens": 20,
                      "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0},
        },
    }


def write_transcript(root: Path, session_id: str, cwd: str, *, at: datetime = T0,
                     branch: str = "main", project: str = "-srv-widgets") -> Path:
    """A two-turn Claude Code transcript for `session_id` run in `cwd`."""
    project_dir = root / project
    project_dir.mkdir(parents=True, exist_ok=True)
    path = project_dir / f"{session_id}.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in (
        transcript_record(session_id, cwd, f"{session_id}-u1", at, branch=branch),
        transcript_record(session_id, cwd, f"{session_id}-u2", at + timedelta(minutes=1),
                          branch=branch),
    )))
    return path


def contextless_session(db: DuckDBBackend, session_id: str, *, cost: float = 3.0) -> None:
    """A session the live OTLP path would have written: totals, no context."""
    db.upsert_session(replace(
        make_session(session_id=session_id, agent_id="claude-code-widgets", started_at=T0,
                     total_cost_usd=cost, input_tokens=200, output_tokens=40, tool_call_count=1),
        ended_at=T0 + timedelta(minutes=1), source="claude-code",
    ))
    db.insert_span(make_llm_span(agent_id="claude-code-widgets", session_id=session_id,
                                 start_time=T0, cost_usd=cost, input_tokens=200, output_tokens=40))


def session_context_row(db: DuckDBBackend, session_id: str) -> tuple:
    """`(repo_remote, repo_root, branch_start, user_email, developer_id,
    total_cost_usd, input_tokens, updated_at)` for one session."""
    return db.conn.execute(
        "SELECT repo_remote, repo_root, branch_start, user_email, developer_id, "
        "total_cost_usd, input_tokens, updated_at FROM sessions WHERE session_id = $1",
        [session_id],
    ).fetchone()
