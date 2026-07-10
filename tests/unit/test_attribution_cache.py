"""Unit tests for the statusline's cheap on-disk attribution cache.

`refresh_attribution_cache` is the write side (called by `ingest_claude_code`
after a backfill); `read_attribution_cache` is the read side the statusline
uses. Both take an explicit `path` so no test ever touches the real
`~/.local/share/tj/attribution_cache.json`.
"""
from __future__ import annotations

import json
from datetime import timedelta

import pytest

from tokenjam.core.attribution_cache import (
    format_driver_label,
    read_attribution_cache,
    refresh_attribution_cache,
    write_attribution_cache,
)
from tokenjam.core.config import CaptureConfig
from tokenjam.core.db import InMemoryBackend
from tokenjam.otel.semconv import GenAIAttributes
from tokenjam.utils.time_parse import utcnow
from tests.factories import make_llm_span, make_session, make_tool_span

BASE = utcnow() - timedelta(hours=1)


@pytest.fixture
def db():
    backend = InMemoryBackend()
    yield backend
    backend.close()


def _seed_recurring_file_read(db, path: str = "CLAUDE.md") -> None:
    """Three sessions all Read-ing the same file — a recurring inclusion.

    Each session also needs at least one LLM-call span:
    `compute_context_diagnostic` only computes recurring inclusions once
    `load_turn_compositions` finds turns (a tool-only DB has none).
    """
    for i, sid in enumerate(("sess-a", "sess-b", "sess-c")):
        db.upsert_session(make_session(session_id=sid, plan_tier="max_5x"))
        llm = make_llm_span(
            model="claude-sonnet-4-5", input_tokens=100, output_tokens=50,
            cache_tokens=100, cost_usd=0.01, session_id=sid,
        )
        llm.start_time = BASE + timedelta(seconds=i)
        db.insert_span(llm)
        tool = make_tool_span(tool_name="Read")
        tool.session_id = sid
        tool.start_time = BASE + timedelta(seconds=i, milliseconds=500)
        tool.attributes = {GenAIAttributes.TOOL_INPUT: {"file_path": path}}
        db.insert_span(tool)


# --- read/write round trip --------------------------------------------------


def test_write_then_read_round_trips(tmp_path):
    path = tmp_path / "cache.json"
    write_attribution_cache("CLAUDE.md", 14, 3, path=path)
    cached = read_attribution_cache(path=path)
    assert cached["top_label"] == "CLAUDE.md"
    assert cached["occurrences"] == 14
    assert cached["sessions"] == 3


def test_read_missing_file_returns_none(tmp_path):
    assert read_attribution_cache(path=tmp_path / "missing.json") is None


def test_read_corrupt_json_returns_none(tmp_path):
    path = tmp_path / "cache.json"
    path.write_text("{not json")
    assert read_attribution_cache(path=path) is None


def test_read_non_dict_json_returns_none(tmp_path):
    path = tmp_path / "cache.json"
    path.write_text(json.dumps([1, 2, 3]))
    assert read_attribution_cache(path=path) is None


def test_read_missing_fields_returns_none(tmp_path):
    path = tmp_path / "cache.json"
    path.write_text(json.dumps({"computed_at": utcnow().isoformat()}))
    assert read_attribution_cache(path=path) is None


def test_read_missing_computed_at_expires(tmp_path):
    # A complete entry with NO computed_at can't be proven fresh, so it must
    # expire (return None) rather than show a stale driver forever.
    path = tmp_path / "cache.json"
    path.write_text(json.dumps({
        "top_label": "CLAUDE.md", "occurrences": 14, "sessions": 3,
    }))
    assert read_attribution_cache(path=path) is None


def test_read_unparseable_computed_at_expires(tmp_path):
    # A non-string / unparseable timestamp is treated the same as aged-out.
    path = tmp_path / "cache.json"
    path.write_text(json.dumps({
        "top_label": "CLAUDE.md", "occurrences": 14, "sessions": 3,
        "computed_at": "not-a-timestamp",
    }))
    assert read_attribution_cache(path=path) is None


def test_read_stale_entry_returns_none(tmp_path):
    path = tmp_path / "cache.json"
    stale = utcnow() - timedelta(days=30)
    path.write_text(json.dumps({
        "top_label": "CLAUDE.md", "occurrences": 5, "sessions": 2,
        "computed_at": stale.isoformat(),
    }))
    assert read_attribution_cache(path=path, max_age_seconds=7 * 24 * 60 * 60) is None


def test_read_fresh_entry_within_max_age_returns_data(tmp_path):
    path = tmp_path / "cache.json"
    recent = utcnow() - timedelta(hours=1)
    path.write_text(json.dumps({
        "top_label": "CLAUDE.md", "occurrences": 5, "sessions": 2,
        "computed_at": recent.isoformat(),
    }))
    cached = read_attribution_cache(path=path, max_age_seconds=7 * 24 * 60 * 60)
    assert cached is not None
    assert cached["top_label"] == "CLAUDE.md"


# --- refresh_attribution_cache (the ingest-time write) ----------------------


def test_refresh_writes_top_driver_when_capture_on(db, tmp_path):
    _seed_recurring_file_read(db, path="CLAUDE.md")
    cache_path = tmp_path / "cache.json"
    refresh_attribution_cache(
        db.conn, CaptureConfig(tool_inputs=True), path=cache_path
    )
    cached = read_attribution_cache(path=cache_path)
    assert cached is not None
    assert cached["top_label"] == "CLAUDE.md"
    assert cached["occurrences"] == 3
    assert cached["sessions"] == 3


def test_refresh_no_op_when_capture_off(db, tmp_path):
    _seed_recurring_file_read(db, path="CLAUDE.md")
    cache_path = tmp_path / "cache.json"
    refresh_attribution_cache(db.conn, CaptureConfig(), path=cache_path)
    assert not cache_path.exists()


def test_refresh_no_op_when_no_recurring_inclusions(db, tmp_path):
    # A single Read in one session never clears RECURRING_MIN_SESSIONS.
    db.upsert_session(make_session(session_id="sess-a", plan_tier="max_5x"))
    tool = make_tool_span(tool_name="Read")
    tool.session_id = "sess-a"
    tool.start_time = BASE
    tool.attributes = {GenAIAttributes.TOOL_INPUT: {"file_path": "CLAUDE.md"}}
    db.insert_span(tool)

    cache_path = tmp_path / "cache.json"
    refresh_attribution_cache(
        db.conn, CaptureConfig(tool_inputs=True), path=cache_path
    )
    assert not cache_path.exists()


def test_refresh_never_raises_on_bad_connection(tmp_path):
    cache_path = tmp_path / "cache.json"
    refresh_attribution_cache(
        object(), CaptureConfig(tool_inputs=True), path=cache_path
    )
    assert not cache_path.exists()


# --- format_driver_label (the single display reader) ------------------------


def test_format_driver_label_formats_label_and_count(tmp_path):
    path = tmp_path / "cache.json"
    write_attribution_cache("CLAUDE.md", 14, 3, path=path)
    assert format_driver_label(path=path) == "CLAUDE.md ×14"


def test_format_driver_label_none_when_no_cache(tmp_path):
    assert format_driver_label(path=tmp_path / "missing.json") is None


def test_format_driver_label_none_when_stale(tmp_path):
    path = tmp_path / "cache.json"
    stale = utcnow() - timedelta(days=30)
    path.write_text(json.dumps({
        "top_label": "CLAUDE.md", "occurrences": 5, "sessions": 2,
        "computed_at": stale.isoformat(),
    }))
    assert format_driver_label(path=path) is None
