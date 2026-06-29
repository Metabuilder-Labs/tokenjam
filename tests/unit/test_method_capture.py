"""Unit tests for core/method_capture.py (M1 — persist a method snapshot).

These drive capture/load against an InMemoryBackend, building a minimal Claude
Code JSONL transcript on disk in a tmp projects dir (the same shape the
transcript parser tests use). The point of the feature: a snapshot persisted at
session close lets a killed agent's method survive Claude Code pruning the
on-disk transcript.
"""
from __future__ import annotations

import json
from pathlib import Path

from tokenjam.core.db import InMemoryBackend
from tokenjam.core.method_capture import (
    SNAPSHOT_SCHEMA_VERSION,
    capture_session_method,
    load_session_method,
)
from tokenjam.core.transcript import build_session_story


# --- Fixtures ----------------------------------------------------------------

def _write_transcript(
    projects_root: Path, session_id: str, *, task: str = "Build the thing."
) -> Path:
    """Write a minimal CC transcript under <root>/<project>/<session_id>.jsonl."""
    project_dir = projects_root / "-Users-test-project"
    project_dir.mkdir(parents=True, exist_ok=True)
    records = [
        {"type": "user", "message": {"role": "user", "content": task}},
        {
            "type": "assistant",
            "timestamp": "2026-06-15T09:11:36.133Z",
            "message": {
                "role": "assistant",
                "model": "claude-opus-4-8",
                "content": [
                    {"type": "text", "text": "Reading the file."},
                    {
                        "type": "tool_use",
                        "id": "t1",
                        "name": "Read",
                        "input": {"file_path": "src/app.py"},
                    },
                ],
            },
        },
        {
            "type": "user",
            "message": {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "t1", "content": "..."}
                ],
            },
        },
        {
            "type": "assistant",
            "timestamp": "2026-06-15T09:12:00.000Z",
            "message": {
                "role": "assistant",
                "model": "claude-opus-4-8",
                "content": [{"type": "text", "text": "Done — it works."}],
            },
        },
    ]
    path = project_dir / f"{session_id}.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")
    return path


# --- Tests -------------------------------------------------------------------

def test_capture_writes_snapshot(tmp_path):
    db = InMemoryBackend()
    try:
        _write_transcript(tmp_path, "sess-1")

        wrote = capture_session_method(db, "sess-1", projects_dir=tmp_path)
        assert wrote is True

        row = db.conn.execute(
            "SELECT session_id, source, schema_version FROM session_story "
            "WHERE session_id = $1",
            ["sess-1"],
        ).fetchone()
        assert row is not None
        assert row[0] == "sess-1"
        assert row[1] == "live-transcript"
        assert row[2] == SNAPSHOT_SCHEMA_VERSION
    finally:
        db.close()


def test_load_round_trips_the_story(tmp_path):
    db = InMemoryBackend()
    try:
        _write_transcript(tmp_path, "sess-1", task="Fix the auth bug.")
        assert capture_session_method(db, "sess-1", projects_dir=tmp_path) is True

        snapshot = load_session_method(db, "sess-1")
        assert snapshot is not None
        # Composite snapshot: both the recursive Story and the ask-segmented story.
        assert snapshot["story"]["task"] == "Fix the auth bug."
        assert snapshot["story"]["step_count"] == 2
        assert snapshot["story"]["outcome"] == "Done — it works."
        assert snapshot["asks"]["asks"][0]["prompt"] == "Fix the auth bug."
        # It round-trips the live reconstruction byte-for-byte.
        live = build_session_story("sess-1", projects_root=tmp_path)
        assert snapshot["story"] == live
    finally:
        db.close()


def test_source_argument_is_persisted(tmp_path):
    db = InMemoryBackend()
    try:
        _write_transcript(tmp_path, "sess-1")
        assert capture_session_method(
            db, "sess-1", projects_dir=tmp_path, source="backfill"
        ) is True
        row = db.conn.execute(
            "SELECT source FROM session_story WHERE session_id = $1", ["sess-1"]
        ).fetchone()
        assert row[0] == "backfill"
    finally:
        db.close()


def test_recapture_overwrites_idempotently(tmp_path):
    db = InMemoryBackend()
    try:
        _write_transcript(tmp_path, "sess-1", task="First task.")
        assert capture_session_method(db, "sess-1", projects_dir=tmp_path) is True

        # The transcript grew / changed; re-capturing must overwrite, not dup.
        _write_transcript(tmp_path, "sess-1", task="Updated task.")
        assert capture_session_method(db, "sess-1", projects_dir=tmp_path) is True

        count = db.conn.execute(
            "SELECT COUNT(*) FROM session_story WHERE session_id = $1", ["sess-1"]
        ).fetchone()[0]
        assert count == 1

        snapshot = load_session_method(db, "sess-1")
        assert snapshot["story"]["task"] == "Updated task."
    finally:
        db.close()


def test_returns_false_and_no_raise_when_no_transcript(tmp_path):
    db = InMemoryBackend()
    try:
        # Empty projects dir -> no transcript for this session.
        wrote = capture_session_method(db, "ghost-sess", projects_dir=tmp_path)
        assert wrote is False
        count = db.conn.execute(
            "SELECT COUNT(*) FROM session_story WHERE session_id = $1", ["ghost-sess"]
        ).fetchone()[0]
        assert count == 0
    finally:
        db.close()


def test_load_returns_none_when_absent():
    db = InMemoryBackend()
    try:
        assert load_session_method(db, "nope") is None
    finally:
        db.close()


def test_read_through_fallback_serves_snapshot_after_prune(tmp_path):
    """The core of the feature: once the transcript is gone, live reconstruction
    yields None, but the persisted snapshot still serves the method."""
    db = InMemoryBackend()
    try:
        path = _write_transcript(tmp_path, "sess-1", task="Survive the prune.")
        assert capture_session_method(db, "sess-1", projects_dir=tmp_path) is True

        # Claude Code prunes the on-disk transcript.
        path.unlink()

        # Live reconstruction can no longer find it...
        assert build_session_story("sess-1", projects_root=tmp_path) is None
        # ...but the snapshot is the read-through fallback.
        snapshot = load_session_method(db, "sess-1")
        assert snapshot is not None
        assert snapshot["story"]["task"] == "Survive the prune."
    finally:
        db.close()
