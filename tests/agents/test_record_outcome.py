"""
Tests for the record_outcome SDK helper.

record_outcome emits one OTel span carrying the emerging gen_ai outcome-event
attributes (OTel semconv issue #2665) that TokenJam Cloud's ROI ingest keys off
(roi.is_outcome_event / write_outcome_from_span). These tests assert the span
carries the exact attribute names/shape the Cloud parses, and that argument
validation mirrors the Cloud OutcomeIn validator (at least one of
workflow_id / session_id required).

Uses a single module-level TracerProvider with a collecting exporter, mirroring
tests/agents/test_mock_scenarios.py (Critical Rule 11).
"""
from __future__ import annotations

import threading
from typing import Sequence

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import (
    SimpleSpanProcessor,
    SpanExporter,
    SpanExportResult,
)

from tokenjam.otel.semconv import GenAIAttributes, TjAttributes
from tokenjam.sdk.agent import AgentSession, record_outcome


class _CollectingExporter(SpanExporter):
    """In-memory span exporter that collects finished spans for assertions."""

    def __init__(self) -> None:
        self._spans: list[ReadableSpan] = []
        self._lock = threading.Lock()

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        with self._lock:
            self._spans.extend(spans)
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        pass

    def get_finished_spans(self) -> list[ReadableSpan]:
        with self._lock:
            return list(self._spans)

    def clear(self) -> None:
        with self._lock:
            self._spans.clear()


# Module-level setup: set the tracer provider once (Critical Rule 11).
_exporter = _CollectingExporter()
_provider = TracerProvider()
_provider.add_span_processor(SimpleSpanProcessor(_exporter))
trace.set_tracer_provider(_provider)

import tokenjam.sdk.agent as _agent_mod  # noqa: E402


@pytest.fixture(autouse=True)
def otel_exporter():
    # Bind the SDK's tracer to OUR provider instance (not the global) for the
    # duration of each test, so spans land in our collecting exporter even when
    # another module won set_tracer_provider first (set-once — Critical Rule 11).
    # Restore afterwards so we don't leak our exporter into other test modules.
    original = _agent_mod._tracer
    _agent_mod._tracer = _provider.get_tracer("tokenjam.sdk")
    _exporter.clear()
    try:
        yield _exporter
    finally:
        _agent_mod._tracer = original


def _outcome_spans(exporter: _CollectingExporter) -> list[ReadableSpan]:
    return [
        s for s in exporter.get_finished_spans()
        if s.name == GenAIAttributes.SPAN_OUTCOME
    ]


# ── Marker attributes the Cloud ROI ingest recognises ───────────────────────

def test_record_outcome_emits_marker_attributes(otel_exporter):
    record_outcome("ticket_resolved", workflow_id="wf-123")

    spans = _outcome_spans(otel_exporter)
    assert len(spans) == 1
    attrs = spans[0].attributes

    # event.name + the outcome-type marker (roi.is_outcome_event keys off these)
    assert attrs[GenAIAttributes.EVENT_NAME] == GenAIAttributes.OUTCOME_EVENT_NAME
    assert attrs[GenAIAttributes.OUTCOME_TYPE] == "ticket_resolved"
    # success defaults to True
    assert attrs[GenAIAttributes.OUTCOME_SUCCESS] is True
    # explicit workflow key rides tokenjam.workflow_id
    assert attrs[TjAttributes.WORKFLOW_ID] == "wf-123"
    # no value declared -> value_usd attribute absent (never fabricated)
    assert GenAIAttributes.OUTCOME_VALUE_USD not in attrs


def test_record_outcome_uses_canonical_attribute_names():
    """The attribute STRINGS must match what the Cloud roi.py constants expect."""
    assert GenAIAttributes.OUTCOME_EVENT_NAME == "gen_ai.outcome"
    assert GenAIAttributes.EVENT_NAME == "event.name"
    assert GenAIAttributes.OUTCOME_TYPE == "gen_ai.outcome.type"
    assert GenAIAttributes.OUTCOME_SUCCESS == "gen_ai.outcome.success"
    assert GenAIAttributes.OUTCOME_VALUE_USD == "gen_ai.outcome.value_usd"
    assert TjAttributes.WORKFLOW_ID == "tokenjam.workflow_id"
    # session_id rides "session.id" — the key the canonical OTLP parser reads.
    assert TjAttributes.SESSION_ID == "session.id"


# ── value_usd (self-reported) ───────────────────────────────────────────────

def test_record_outcome_declared_value(otel_exporter):
    record_outcome(
        "lead_qualified", session_id="sess-9", value_usd=42.5, success=True
    )
    attrs = _outcome_spans(otel_exporter)[0].attributes
    assert attrs[GenAIAttributes.OUTCOME_VALUE_USD] == 42.5
    assert attrs[TjAttributes.SESSION_ID] == "sess-9"


def test_record_outcome_failure_success_false(otel_exporter):
    record_outcome("checkout", workflow_id="wf-1", success=False)
    attrs = _outcome_spans(otel_exporter)[0].attributes
    assert attrs[GenAIAttributes.OUTCOME_SUCCESS] is False


def test_record_outcome_extra_attributes(otel_exporter):
    record_outcome(
        "ticket_resolved",
        workflow_id="wf-1",
        attributes={"customer_tier": "enterprise"},
    )
    attrs = _outcome_spans(otel_exporter)[0].attributes
    assert attrs["customer_tier"] == "enterprise"


# ── Session inheritance from the active @watch()/AgentSession span ──────────

def test_record_outcome_inherits_active_session(otel_exporter):
    with AgentSession(agent_id="support-bot") as session:
        # No explicit session_id/workflow_id — inherited from the active span.
        record_outcome("ticket_resolved")

    attrs = _outcome_spans(otel_exporter)[0].attributes
    # The active session's conversation_id is stamped as session.id.
    assert attrs[TjAttributes.SESSION_ID] == session.conversation_id
    assert attrs[GenAIAttributes.AGENT_ID] == "support-bot"


# ── Argument validation (mirrors the Cloud OutcomeIn validator) ─────────────

def test_record_outcome_requires_outcome_type():
    with pytest.raises(ValueError, match="outcome_type"):
        record_outcome("", workflow_id="wf-1")


def test_record_outcome_requires_workflow_or_session():
    # No active session, no workflow_id, no session_id -> reject.
    with pytest.raises(ValueError, match="workflow_id or session_id"):
        record_outcome("ticket_resolved")


def test_record_outcome_rejected_emits_no_span(otel_exporter):
    with pytest.raises(ValueError):
        record_outcome("ticket_resolved")
    # A rejected call must not leak a half-built outcome span.
    assert _outcome_spans(otel_exporter) == []
