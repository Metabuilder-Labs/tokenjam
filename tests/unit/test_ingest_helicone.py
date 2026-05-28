"""Unit tests for the Helicone ingest adapter."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from tokenjam.core.db import InMemoryBackend
from tokenjam.core.ingest_adapters.helicone import (
    _provider_to_billing_account,
    _record_to_span,
    ingest_helicone,
)
from tokenjam.core.models import SpanStatus


FIXTURE_PATH = Path(__file__).parent.parent / "fixtures" / "helicone_real_response.json"


@pytest.fixture
def db():
    backend = InMemoryBackend()
    yield backend
    backend.close()


# -- Pure-function tests --

def test_provider_to_billing_account_anthropic():
    assert _provider_to_billing_account("ANTHROPIC") == ("anthropic", "anthropic")
    assert _provider_to_billing_account("anthropic") == ("anthropic", "anthropic")


def test_provider_to_billing_account_openai_aliases():
    assert _provider_to_billing_account("OPENAI") == ("openai", "openai")
    assert _provider_to_billing_account("AZURE_OPENAI") == ("openai", "openai")


def test_provider_to_billing_account_google():
    assert _provider_to_billing_account("GOOGLE") == ("google", "google")
    assert _provider_to_billing_account("VERTEX") == ("google", "google")


def test_provider_to_billing_account_unknown_returns_none():
    assert _provider_to_billing_account("WEIRDPROVIDER") == (None, None)
    assert _provider_to_billing_account(None) == (None, None)


def test_record_to_span_basic_anthropic():
    record = {
        "request": {
            "id": "req-1", "model": "claude-sonnet-4-6", "provider": "ANTHROPIC",
            "created_at": "2026-04-01T10:00:00Z", "prompt_tokens": 1000,
        },
        "response": {"completion_tokens": 200, "delay_ms": 1500, "status": 200},
        "cost_usd": 0.003,
    }
    span = _record_to_span(record)
    assert span is not None
    assert span.model == "claude-sonnet-4-6"
    assert span.provider == "anthropic"
    assert span.billing_account == "anthropic"
    assert span.input_tokens == 1000
    assert span.output_tokens == 200
    assert span.cost_usd == 0.003
    assert span.duration_ms == 1500.0


def test_record_to_span_session_property_becomes_conversation_id():
    record = {
        "request": {
            "id": "req-1", "model": "claude-sonnet-4-6", "provider": "ANTHROPIC",
            "created_at": "2026-04-01T10:00:00Z",
            "properties": {"Helicone-Property-Session": "my-session"},
        },
    }
    span = _record_to_span(record)
    assert span is not None
    assert span.conversation_id == "my-session"


def test_record_to_span_error_status_marks_error():
    record = {
        "request": {"id": "req-x", "created_at": "2026-04-01T10:00:00Z"},
        "response": {"status": 500, "error": "rate limited"},
    }
    span = _record_to_span(record)
    assert span is not None
    assert span.status_code == SpanStatus.ERROR
    assert span.status_message == "rate limited"


def test_record_to_span_deterministic_ids():
    """Same request id produces the same TokenJam span_id."""
    record = {
        "request": {"id": "req-deterministic", "created_at": "2026-04-01T10:00:00Z"},
    }
    a = _record_to_span(record)
    b = _record_to_span(record)
    assert a is not None and b is not None
    assert a.span_id == b.span_id
    assert a.trace_id == b.trace_id


def test_record_to_span_missing_required_fields():
    assert _record_to_span({"request": {"id": "x"}}) is None  # missing created_at
    assert _record_to_span({"request": {"created_at": "2026-04-01T10:00:00Z"}}) is None  # missing id


def test_record_to_span_handles_flat_envelope():
    """Some Helicone exports flatten the record. The adapter should tolerate that."""
    record = {
        "id": "req-flat",
        "created_at": "2026-04-01T10:00:00Z",
        "model": "claude-sonnet-4-6",
        "provider": "ANTHROPIC",
        "prompt_tokens": 500,
        "completion_tokens": 100,
        "cost_usd": 0.001,
    }
    span = _record_to_span(record)
    assert span is not None
    assert span.model == "claude-sonnet-4-6"
    assert span.input_tokens == 500
    assert span.output_tokens == 100


# -- Integration tests against the fixture --

def test_ingest_fixture_writes_spans(db):
    result = ingest_helicone(db, source_file=str(FIXTURE_PATH))
    assert result["records_read"] == 4
    assert result["spans_written"] == 4
    assert result["spans_skipped"] == 0


def test_ingest_fixture_is_idempotent(db):
    """Re-running is a no-op (deterministic span IDs)."""
    first = ingest_helicone(db, source_file=str(FIXTURE_PATH))
    second = ingest_helicone(db, source_file=str(FIXTURE_PATH))
    assert first["spans_written"] == 4
    assert second["spans_written"] == 0
    assert second["spans_skipped"] == 4


def test_ingest_fixture_maps_billing_accounts(db):
    ingest_helicone(db, source_file=str(FIXTURE_PATH))
    rows = db.conn.execute(
        "SELECT model, billing_account FROM spans WHERE model IS NOT NULL "
        "ORDER BY start_time"
    ).fetchall()
    by_model = {r[0]: r[1] for r in rows}
    assert by_model["claude-sonnet-4-6"] == "anthropic"
    assert by_model["gpt-4o-mini"] == "openai"
    assert by_model["gemini-2-5-flash"] == "google"
    assert by_model["claude-haiku-4-5"] == "anthropic"


def test_ingest_since_filter(db):
    """--since filters out records older than the cutoff."""
    since = datetime(2026, 4, 2, 0, 0, tzinfo=timezone.utc)
    result = ingest_helicone(db, source_file=str(FIXTURE_PATH), since=since)
    # Only the 2 records on 2026-04-02 survive.
    assert result["spans_written"] == 2


def test_ingest_handles_bare_list(db, tmp_path):
    """A bare JSON array (no {data: [...]} envelope) is accepted."""
    bare = [{
        "request": {
            "id": "req-bare",
            "created_at": "2026-04-01T10:00:00Z",
            "model": "claude-haiku-4-5",
            "provider": "ANTHROPIC",
            "prompt_tokens": 200,
        },
    }]
    path = tmp_path / "bare.json"
    path.write_text(json.dumps(bare))
    result = ingest_helicone(db, source_file=str(path))
    assert result["records_read"] == 1
    assert result["spans_written"] == 1


def test_ingest_rejects_both_sources(db):
    with pytest.raises(ValueError, match="exactly one"):
        ingest_helicone(db, source_url="http://x", source_file="/y")


def test_ingest_rejects_neither_source(db):
    with pytest.raises(ValueError, match="exactly one"):
        ingest_helicone(db)
