"""Unit tests for the transcript run-id scanner (core.runlink)."""
from __future__ import annotations

import json

from tokenjam.core.runlink import scan_transcript_run_ids


def _write(projects_root, session_id: str, text: str) -> None:
    project_dir = projects_root / "-Users-test-project"
    project_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "type": "assistant",
        "message": {"role": "assistant", "content": [{"type": "text", "text": text}]},
    }
    (project_dir / f"{session_id}.jsonl").write_text(
        json.dumps(record), encoding="utf-8"
    )


def test_scan_finds_announced_run_id(tmp_path):
    _write(tmp_path, "s1", "TokenJam run id: `gov-20260623T093359Z-11694` started.")
    assert scan_transcript_run_ids("s1", tmp_path) == ["gov-20260623T093359Z-11694"]


def test_scan_finds_attribute_form(tmp_path):
    _write(tmp_path, "s2", "every worker tagged tokenjam.run_id=run-abc123 ok")
    assert scan_transcript_run_ids("s2", tmp_path) == ["run-abc123"]


def test_scan_dedupes_preserving_order(tmp_path):
    _write(
        tmp_path, "s3",
        "run id: gov-20260101T000000Z-1 ... and again gov-20260101T000000Z-1 ...",
    )
    assert scan_transcript_run_ids("s3", tmp_path) == ["gov-20260101T000000Z-1"]


def test_scan_rejects_ellipsis_placeholder(tmp_path):
    # A doc-style mention with an ellipsis is not a real id.
    _write(tmp_path, "s4", "tokenjam.run_id=gov-… (every worker tagged)")
    assert scan_transcript_run_ids("s4", tmp_path) == []


def test_scan_missing_transcript_returns_empty(tmp_path):
    assert scan_transcript_run_ids("nope", tmp_path) == []


def test_scan_no_announcement_returns_empty(tmp_path):
    _write(tmp_path, "s5", "Just doing some work, nothing to announce.")
    assert scan_transcript_run_ids("s5", tmp_path) == []
