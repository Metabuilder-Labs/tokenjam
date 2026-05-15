"""Unit tests for the backfill parser + ingest path."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from tokenjam.core.backfill import (
    ingest_claude_code,
    iter_claude_code_sessions,
    parse_claude_code_session,
)
from tokenjam.core.db import InMemoryBackend


def _make_session_file(tmp_path: Path, session_id: str, cwd: str,
                        records: list[dict]) -> Path:
    project_dir = tmp_path / cwd.replace("/", "-")
    project_dir.mkdir(parents=True, exist_ok=True)
    path = project_dir / f"{session_id}.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in records))
    return path


def _assistant_record(uuid: str, model: str, input_tokens: int, output_tokens: int,
                       timestamp: str, session_id: str, cwd: str,
                       tool_uses: list[tuple[str, str]] | None = None) -> dict:
    content: list[dict] = [{"type": "text", "text": "ok"}]
    if tool_uses:
        for tu_id, tu_name in tool_uses:
            content.append({"type": "tool_use", "id": tu_id, "name": tu_name})
    return {
        "type": "assistant",
        "uuid": uuid,
        "timestamp": timestamp,
        "sessionId": session_id,
        "cwd": cwd,
        "message": {
            "model": model,
            "content": content,
            "usage": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cache_read_input_tokens": 0,
                "cache_creation_input_tokens": 0,
            },
        },
    }


def test_parse_extracts_assistant_turns_and_tool_uses(tmp_path):
    path = _make_session_file(
        tmp_path,
        session_id="sess-1",
        cwd="/Users/me/proj",
        records=[
            {"type": "user", "message": {"role": "user", "content": "hi"}},
            _assistant_record(
                "msg-1", "claude-opus-4-7", 1000, 200,
                "2026-04-01T10:00:00.000Z", "sess-1", "/Users/me/proj",
                tool_uses=[("tu-1", "Read"), ("tu-2", "Edit")],
            ),
            _assistant_record(
                "msg-2", "claude-opus-4-7", 500, 100,
                "2026-04-01T10:00:05.000Z", "sess-1", "/Users/me/proj",
            ),
        ],
    )
    parsed = parse_claude_code_session(path)
    assert parsed is not None
    assert parsed.session_id == "sess-1"
    assert parsed.agent_id == "claude-code-proj"
    # 2 LLM spans + 2 tool spans
    assert len(parsed.spans) == 4
    assert parsed.tool_call_count == 2
    assert parsed.total_input_tokens == 1500
    assert parsed.total_output_tokens == 300
    # Cost is recomputed from pricing — must be > 0 for Opus
    assert parsed.total_cost_usd > 0


def test_parse_returns_none_for_file_with_no_assistant_turns(tmp_path):
    path = _make_session_file(
        tmp_path,
        session_id="sess-empty",
        cwd="/Users/me/proj",
        records=[
            {"type": "user", "message": {"role": "user", "content": "hi"}},
        ],
    )
    assert parse_claude_code_session(path) is None


def test_iter_walks_root(tmp_path):
    _make_session_file(
        tmp_path,
        session_id="sess-a",
        cwd="/Users/me/proj-a",
        records=[_assistant_record(
            "msg-a", "claude-sonnet-4-6", 1000, 100,
            "2026-04-01T10:00:00.000Z", "sess-a", "/Users/me/proj-a",
        )],
    )
    _make_session_file(
        tmp_path,
        session_id="sess-b",
        cwd="/Users/me/proj-b",
        records=[_assistant_record(
            "msg-b", "claude-sonnet-4-6", 1000, 100,
            "2026-04-02T10:00:00.000Z", "sess-b", "/Users/me/proj-b",
        )],
    )
    sessions = list(iter_claude_code_sessions(root=tmp_path))
    assert {s.session_id for s in sessions} == {"sess-a", "sess-b"}


def test_ingest_is_idempotent(tmp_path):
    _make_session_file(
        tmp_path,
        session_id="sess-i",
        cwd="/Users/me/proj",
        records=[_assistant_record(
            "msg-i", "claude-haiku-4-5", 1000, 100,
            "2026-04-01T10:00:00.000Z", "sess-i", "/Users/me/proj",
            tool_uses=[("tu-i", "Read")],
        )],
    )
    db = InMemoryBackend()
    try:
        r1 = ingest_claude_code(db, root=tmp_path)
        assert r1.spans_ingested == 2  # 1 LLM + 1 tool
        # Re-run: no new spans
        r2 = ingest_claude_code(db, root=tmp_path)
        assert r2.spans_ingested == 0
        assert r2.spans_skipped_existing == 2
    finally:
        db.close()


def test_ingest_writes_session_record(tmp_path):
    _make_session_file(
        tmp_path,
        session_id="sess-w",
        cwd="/Users/me/proj",
        records=[_assistant_record(
            "msg-w", "claude-haiku-4-5", 800, 150,
            "2026-04-01T10:00:00.000Z", "sess-w", "/Users/me/proj",
        )],
    )
    db = InMemoryBackend()
    try:
        ingest_claude_code(db, root=tmp_path)
        sess = db.get_session("sess-w")
        assert sess is not None
        assert sess.agent_id == "claude-code-proj"
        assert sess.input_tokens == 800
    finally:
        db.close()


def test_iter_skips_files_before_since(tmp_path):
    p = _make_session_file(
        tmp_path,
        session_id="sess-old",
        cwd="/Users/me/proj",
        records=[_assistant_record(
            "msg-old", "claude-haiku-4-5", 100, 50,
            "2026-04-01T10:00:00.000Z", "sess-old", "/Users/me/proj",
        )],
    )
    # Force mtime far in the past
    import os
    old = datetime(2020, 1, 1, tzinfo=timezone.utc).timestamp()
    os.utime(p, (old, old))
    cutoff = datetime(2025, 1, 1, tzinfo=timezone.utc)
    sessions = list(iter_claude_code_sessions(root=tmp_path, since=cutoff))
    assert sessions == []
