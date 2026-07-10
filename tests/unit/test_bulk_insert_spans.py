"""Parity + idempotency tests for `DuckDBBackend.bulk_insert_spans`.

The columnar bulk-append (newline-delimited JSON + DuckDB `read_json`) is the
backfill hot path — it must land rows byte-for-byte identical to the per-row
`insert_span`, and must be idempotent (skip span_ids already present).
"""
from __future__ import annotations

import dataclasses
import json

from tokenjam.core.db import InMemoryBackend
from tokenjam.core.models import NormalizedSpan

from tests.factories import make_llm_span, make_tool_span

# The columns we compare; JSON columns are pulled as text and re-parsed so the
# comparison is on semantic JSON, not DuckDB's internal whitespace.
_COMPARE_SQL = (
    "SELECT span_id, trace_id, parent_span_id, session_id, agent_id, name, kind, "
    "status_code, status_message, start_time, end_time, duration_ms, "
    "attributes::VARCHAR, provider, model, tool_name, input_tokens, output_tokens, "
    "cache_tokens, cost_usd, request_type, conversation_id, events::VARCHAR, "
    "billing_account, cache_write_tokens, request_params::VARCHAR, "
    "request_tools::VARCHAR, sub_agent_id "
    "FROM spans ORDER BY span_id"
)
_JSON_COL_IDX = (12, 22, 25, 26)  # attributes, events, request_params, request_tools


def _snapshot(db) -> list[tuple]:
    rows = db.conn.execute(_COMPARE_SQL).fetchall()
    normalized = []
    for row in rows:
        row = list(row)
        for i in _JSON_COL_IDX:
            row[i] = json.loads(row[i]) if row[i] is not None else None
        normalized.append(tuple(row))
    return normalized


def _sample_spans() -> list[NormalizedSpan]:
    llm = make_llm_span(
        span_id="s-1", session_id="sess", conversation_id="sess",
        input_tokens=1000, output_tokens=200, cache_tokens=42, cost_usd=0.0123456789,
        model="claude-opus-4-7",
        extra_attributes={"source": "backfill.claude_code",
                          "gen_ai.prompt.content": "unicode: héllo 世界 \" \\ /",
                          "nested": {"a": [1, 2, {"b": True}]}},
    )
    tool = dataclasses.replace(
        make_tool_span(session_id="sess", conversation_id="sess", tool_name="Read",
                       tool_input={"file_path": "/etc/x", "深": "value"}),
        span_id="s-2", parent_span_id="s-1",
    )
    # A span exercising request_params/request_tools + NULL end_time/duration_ms.
    rich = dataclasses.replace(
        llm, span_id="s-3", parent_span_id="s-1", sub_agent_id="ag-9",
        end_time=None, duration_ms=None,
        request_params={"temperature": 0.7, "stop": ["\n", "END"]},
        request_tools={"tools": [{"name": "Read"}, {"name": "Edit"}]},
    )
    return [llm, tool, rich]


def test_bulk_insert_matches_per_row_insert_span():
    spans = _sample_spans()

    ref = InMemoryBackend()
    bulk = InMemoryBackend()
    try:
        for span in spans:
            ref.insert_span(span)
        bulk.bulk_insert_spans(spans)
        assert _snapshot(bulk) == _snapshot(ref)
    finally:
        ref.close()
        bulk.close()


def test_bulk_insert_is_idempotent_on_span_id():
    spans = _sample_spans()
    db = InMemoryBackend()
    try:
        db.bulk_insert_spans(spans)
        first = _snapshot(db)
        # Re-appending the same spans (plus a fresh one) inserts only the new id;
        # the existing three are skipped by the anti-join, not duplicated.
        extra = dataclasses.replace(spans[0], span_id="s-4")
        db.bulk_insert_spans([*spans, extra])
        assert db.conn.execute("SELECT COUNT(*) FROM spans").fetchone()[0] == 4
        # The originally-inserted rows are unchanged.
        assert _snapshot(db)[:3] == first
    finally:
        db.close()


def test_bulk_insert_empty_is_noop():
    db = InMemoryBackend()
    try:
        db.bulk_insert_spans([])
        assert db.conn.execute("SELECT COUNT(*) FROM spans").fetchone()[0] == 0
    finally:
        db.close()
