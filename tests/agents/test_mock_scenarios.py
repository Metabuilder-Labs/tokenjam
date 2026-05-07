"""
Mock agent scenario tests. Verifies @watch(), record_llm_call,
record_tool_call, and AgentSession behavior using OTel's SimpleSpanProcessor
with a collecting exporter.

Uses a single TracerProvider for the entire module to avoid OTel's
"Overriding of current TracerProvider is not allowed" warning.
"""
from __future__ import annotations

import threading
from typing import Sequence

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider, ReadableSpan
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter, SpanExportResult

from tj.sdk.agent import watch, AgentSession, record_llm_call, record_tool_call
from tj.otel.semconv import GenAIAttributes


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


# Module-level setup: set the tracer provider once
_exporter = _CollectingExporter()
_provider = TracerProvider()
_provider.add_span_processor(SimpleSpanProcessor(_exporter))
trace.set_tracer_provider(_provider)

# Re-bind the SDK's _tracer to use the new provider
import tj.sdk.agent as _agent_mod
_agent_mod._tracer = trace.get_tracer("tj.sdk")


@pytest.fixture(autouse=True)
def otel_exporter():
    """Clear collected spans before each test."""
    _exporter.clear()
    # Re-bind in case it was overwritten
    _agent_mod._tracer = trace.get_tracer("tj.sdk")
    yield _exporter


# ── @watch() creates session span ──────────────────────────────────────────

def test_watch_alone_creates_session_span(otel_exporter):
    """@watch() must create a session span (invoke_agent) even without provider patches."""

    @watch(agent_id="session-only-agent")
    def my_agent():
        pass

    my_agent()
    spans = otel_exporter.get_finished_spans()

    assert len(spans) == 1
    session_span = spans[0]
    assert session_span.name == GenAIAttributes.SPAN_INVOKE_AGENT
    assert session_span.attributes[GenAIAttributes.AGENT_ID] == "session-only-agent"


def test_watch_without_provider_patch_creates_session_not_llm_spans(otel_exporter):
    """
    IMPORTANT: @watch() alone must NOT create LLM call spans.
    """

    @watch(agent_id="no-patch-agent")
    def my_agent():
        x = 1 + 1
        return x

    my_agent()
    spans = otel_exporter.get_finished_spans()

    assert len(spans) == 1
    assert spans[0].name == GenAIAttributes.SPAN_INVOKE_AGENT
    llm_spans = [s for s in spans if s.name == GenAIAttributes.SPAN_LLM_CALL]
    assert len(llm_spans) == 0


# ── record_llm_call creates LLM span ──────────────────────────────────────

def test_record_llm_call_creates_llm_span(otel_exporter):

    @watch(agent_id="test-agent")
    def my_agent():
        record_llm_call("claude-haiku-4-5", "anthropic", 100, 20)

    my_agent()
    spans = otel_exporter.get_finished_spans()

    llm_spans = [s for s in spans if s.name == GenAIAttributes.SPAN_LLM_CALL]
    assert len(llm_spans) == 1
    llm = llm_spans[0]
    assert llm.attributes[GenAIAttributes.REQUEST_MODEL] == "claude-haiku-4-5"
    assert llm.attributes[GenAIAttributes.PROVIDER_NAME] == "anthropic"
    assert llm.attributes[GenAIAttributes.INPUT_TOKENS] == 100
    assert llm.attributes[GenAIAttributes.OUTPUT_TOKENS] == 20


# ── record_tool_call creates tool span ─────────────────────────────────────

def test_record_tool_call_creates_tool_span(otel_exporter):

    @watch(agent_id="test-agent")
    def my_agent():
        record_tool_call("send_email", tool_output={"status": "sent"})

    my_agent()
    spans = otel_exporter.get_finished_spans()

    tool_spans = [s for s in spans if s.name == GenAIAttributes.SPAN_TOOL_CALL]
    assert len(tool_spans) == 1
    assert tool_spans[0].attributes[GenAIAttributes.TOOL_NAME] == "send_email"


# ── Normal session creates session record ──────────────────────────────────

def test_normal_session_creates_session_record(otel_exporter):
    """A @watch() session should produce a span with OK status on success."""

    @watch(agent_id="test-agent", agent_name="Test Agent", agent_version="1.0")
    def my_agent():
        record_llm_call("claude-haiku-4-5", "anthropic", 100, 20)
        return "done"

    result = my_agent()
    assert result == "done"

    spans = otel_exporter.get_finished_spans()
    session_spans = [s for s in spans if s.name == GenAIAttributes.SPAN_INVOKE_AGENT]
    assert len(session_spans) == 1
    session = session_spans[0]
    assert session.attributes[GenAIAttributes.AGENT_ID] == "test-agent"
    assert session.attributes[GenAIAttributes.AGENT_NAME] == "Test Agent"
    assert session.attributes[GenAIAttributes.AGENT_VERSION] == "1.0"
    assert session.status.status_code == trace.StatusCode.OK


def test_normal_session_records_llm_span(otel_exporter):
    """LLM spans recorded inside @watch() should be captured."""

    @watch(agent_id="test-agent")
    def my_agent():
        record_llm_call("claude-haiku-4-5", "anthropic", 200, 50)
        record_llm_call("claude-haiku-4-5", "anthropic", 300, 60)

    my_agent()
    spans = otel_exporter.get_finished_spans()
    llm_spans = [s for s in spans if s.name == GenAIAttributes.SPAN_LLM_CALL]
    assert len(llm_spans) == 2


# ── Exception handling ─────────────────────────────────────────────────────

def test_exception_in_agent_marks_session_as_error(otel_exporter):

    @watch(agent_id="error-agent")
    def failing_agent():
        raise ValueError("something went wrong")

    with pytest.raises(ValueError, match="something went wrong"):
        failing_agent()

    spans = otel_exporter.get_finished_spans()
    session_spans = [s for s in spans if s.name == GenAIAttributes.SPAN_INVOKE_AGENT]
    assert len(session_spans) == 1
    assert session_spans[0].status.status_code == trace.StatusCode.ERROR


# ── Safety when not configured ─────────────────────────────────────────────

def test_watch_is_safe_when_not_configured(otel_exporter):
    """@watch() should not crash even if OTel is minimally configured."""
    @watch(agent_id="unconfigured-agent")
    def my_agent():
        return 42

    result = my_agent()
    assert result == 42


# ── AgentSession context manager ──────────────────────────────────────────

def test_agent_session_context_manager(otel_exporter):

    with AgentSession(agent_id="ctx-agent", agent_name="Ctx") as session:
        record_llm_call("gpt-4o", "openai", 500, 100)
        assert session.conversation_id is not None

    spans = otel_exporter.get_finished_spans()
    session_spans = [s for s in spans if s.name == GenAIAttributes.SPAN_INVOKE_AGENT]
    assert len(session_spans) == 1
    assert session_spans[0].attributes[GenAIAttributes.AGENT_ID] == "ctx-agent"


def test_agent_session_preserves_conversation_id(otel_exporter):

    with AgentSession(agent_id="conv-agent", conversation_id="my-conv-123"):
        pass

    spans = otel_exporter.get_finished_spans()
    assert len(spans) >= 1
    session = [s for s in spans if s.name == GenAIAttributes.SPAN_INVOKE_AGENT][0]
    assert session.attributes[GenAIAttributes.CONVERSATION_ID] == "my-conv-123"
