"""Unit tests for the OTLP ingest adapter."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from tokenjam.core.db import InMemoryBackend
from tokenjam.core.ingest_adapters.otlp import ingest_otlp


FIXTURE_PATH = Path(__file__).parent.parent / "fixtures" / "otlp_sample.json"


@pytest.fixture
def db():
    backend = InMemoryBackend()
    yield backend
    backend.close()


# -- Fixture-based integration tests --

def test_ingest_fixture_writes_spans(db):
    """The committed fixture has 3 spans across 2 resourceSpans."""
    result = ingest_otlp(db, source_file=str(FIXTURE_PATH))
    assert result["spans_seen"] == 3
    assert result["spans_written"] == 3
    assert result["spans_skipped"] == 0
    assert result["spans_rejected"] == 0


def test_ingest_fixture_is_idempotent(db):
    """Re-running on the same fixture is a no-op (PK on spans.span_id)."""
    first = ingest_otlp(db, source_file=str(FIXTURE_PATH))
    second = ingest_otlp(db, source_file=str(FIXTURE_PATH))
    assert first["spans_written"] == 3
    assert second["spans_written"] == 0
    assert second["spans_skipped"] == 3


def test_ingest_fixture_extracts_billing_account(db):
    """Provider in span attrs maps to billing_account."""
    ingest_otlp(db, source_file=str(FIXTURE_PATH))
    rows = db.conn.execute(
        "SELECT provider, billing_account FROM spans WHERE provider IS NOT NULL"
    ).fetchall()
    pairs = {(r[0], r[1]) for r in rows}
    assert ("anthropic", "anthropic") in pairs
    assert ("openai", "openai") in pairs


def test_ingest_fixture_extracts_indexed_fields(db):
    """Token counts and model are extracted from span attributes."""
    ingest_otlp(db, source_file=str(FIXTURE_PATH))
    row = db.conn.execute(
        "SELECT model, input_tokens, output_tokens, cost_usd FROM spans "
        "WHERE model = 'claude-sonnet-4-6'"
    ).fetchone()
    assert row is not None
    assert row[1] == 1500
    assert row[2] == 320
    assert row[3] == pytest.approx(0.00513)


def test_ingest_fixture_extracts_service_name_as_agent(db):
    """service.name from the resource section becomes agent_id when explicit gen_ai.agent.id is absent."""
    ingest_otlp(db, source_file=str(FIXTURE_PATH))
    rows = db.conn.execute(
        "SELECT DISTINCT agent_id FROM spans ORDER BY agent_id"
    ).fetchall()
    agent_ids = [r[0] for r in rows]
    assert "test-agent" in agent_ids
    assert "test-agent-2" in agent_ids


def test_ingest_since_filter(db):
    """--since filters out spans older than the cutoff."""
    # Fixture timestamps: first resource 2025-06-01 (1748736000s), second
    # 2025-06-02 (1748822400s). Setting since to 2025-06-02 keeps only the
    # OpenAI span in the second resourceSpans.
    since = datetime(2025, 6, 2, 0, 0, tzinfo=timezone.utc)
    result = ingest_otlp(db, source_file=str(FIXTURE_PATH), since=since)
    assert result["spans_written"] == 1


def test_ingest_handles_empty_resource_spans(db, tmp_path):
    """Empty resourceSpans list is accepted, writes zero spans."""
    path = tmp_path / "empty.json"
    path.write_text(json.dumps({"resourceSpans": []}))
    result = ingest_otlp(db, source_file=str(path))
    assert result["spans_seen"] == 0
    assert result["spans_written"] == 0


def test_ingest_handles_ndjson(db, tmp_path):
    """NDJSON with one envelope per line is merged into a single ingest."""
    envelope = {
        "resourceSpans": [{
            "resource": {"attributes": [
                {"key": "service.name", "value": {"stringValue": "nd-agent"}}
            ]},
            "scopeSpans": [{
                "spans": [{
                    "traceId": "0000000000000000000000000000ffff",
                    "spanId": "00000000ffffffff",
                    "name": "gen_ai.llm.call",
                    "kind": 3,
                    "startTimeUnixNano": "1748736000000000000",
                    "endTimeUnixNano": "1748736000500000000",
                    "status": {"code": 1},
                    "attributes": [
                        {"key": "gen_ai.provider.name", "value": {"stringValue": "anthropic"}},
                        {"key": "gen_ai.request.model", "value": {"stringValue": "claude-haiku-4-5"}}
                    ]
                }]
            }]
        }]
    }
    path = tmp_path / "stream.ndjson"
    path.write_text(json.dumps(envelope) + "\n" + json.dumps(envelope) + "\n")
    result = ingest_otlp(db, source_file=str(path))
    # NDJSON: two envelopes each with one span. Both have the same span_id —
    # second is skipped via the PK guard.
    assert result["spans_seen"] == 2
    assert result["spans_written"] == 1
    assert result["spans_skipped"] == 1


def test_ingest_rejects_spans_missing_required_fields(db, tmp_path):
    """Spans without spanId or traceId are rejected, not crashed on."""
    path = tmp_path / "broken.json"
    path.write_text(json.dumps({
        "resourceSpans": [{
            "scopeSpans": [{
                "spans": [{"name": "no-ids", "kind": 1}]
            }]
        }]
    }))
    result = ingest_otlp(db, source_file=str(path))
    assert result["spans_seen"] == 1
    assert result["spans_rejected"] == 1
    assert result["spans_written"] == 0


def test_ingest_rejects_both_sources(db):
    with pytest.raises(ValueError, match="exactly one"):
        ingest_otlp(db, source_url="http://x", source_file="/y")


def test_ingest_rejects_neither_source(db):
    with pytest.raises(ValueError, match="exactly one"):
        ingest_otlp(db)
