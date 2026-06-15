"""
Full pipeline integration tests.

Wires the complete path: SDK (@watch + record_*) -> OTel SimpleSpanProcessor ->
TjSpanExporter -> IngestPipeline -> DuckDB (InMemoryBackend) with cost, alert,
and schema validation hooks.

No real LLM calls — uses manual record_llm_call / record_tool_call.

Uses a module-level TracerProvider with a swappable exporter to avoid OTel's
"Overriding of current TracerProvider is not allowed" warning.
"""
from __future__ import annotations

import json
import threading
from typing import Sequence

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider, ReadableSpan
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter, SpanExportResult

from tokenjam.core.alerts import AlertEngine
from tokenjam.core.config import (
    AgentConfig,
    BudgetConfig,
    CaptureConfig,
    TjConfig,
    SecurityConfig,
)
from tokenjam.core.cost import CostEngine
from tokenjam.core.db import InMemoryBackend
from tokenjam.core.ingest import IngestPipeline
from tokenjam.core.models import AgentRecord, NormalizedSpan, SpanKind, SpanStatus
from tokenjam.core.schema_validator import SchemaValidator
from tokenjam.otel.provider import TjSpanExporter, convert_otel_span
from tokenjam.otel.semconv import GenAIAttributes
from tokenjam.sdk.agent import watch, AgentSession, record_llm_call, record_tool_call
from tokenjam.utils.time_parse import utcnow
import tokenjam.sdk.agent as agent_mod




# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _all_spans(db: InMemoryBackend) -> list[NormalizedSpan]:
    """Query all spans from the InMemoryBackend's DuckDB."""
    rows = db.conn.execute("SELECT * FROM spans ORDER BY start_time").fetchall()
    cols = [d[0] for d in db.conn.description]
    result = []
    for row in rows:
        d = dict(zip(cols, row))
        result.append(NormalizedSpan(
            span_id=d["span_id"],
            trace_id=d["trace_id"],
            parent_span_id=d.get("parent_span_id"),
            session_id=d.get("session_id"),
            agent_id=d.get("agent_id"),
            name=d["name"],
            kind=SpanKind(d["kind"]),
            status_code=SpanStatus(d["status_code"]),
            status_message=d.get("status_message"),
            start_time=d["start_time"],
            end_time=d.get("end_time"),
            duration_ms=d.get("duration_ms"),
            attributes=json.loads(d["attributes"]) if d.get("attributes") else {},
            provider=d.get("provider"),
            model=d.get("model"),
            tool_name=d.get("tool_name"),
            input_tokens=d.get("input_tokens"),
            output_tokens=d.get("output_tokens"),
            cache_tokens=d.get("cache_tokens"),
            cost_usd=d.get("cost_usd"),
            request_type=d.get("request_type"),
            conversation_id=d.get("conversation_id"),
            events=json.loads(d["events"]) if d.get("events") else [],
        ))
    return result


def _all_sessions(db: InMemoryBackend) -> list[dict]:
    """Query all sessions from the InMemoryBackend."""
    rows = db.conn.execute("SELECT * FROM sessions").fetchall()
    cols = [d[0] for d in db.conn.description]
    return [dict(zip(cols, row)) for row in rows]


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def full_stack():
    """
    Wire up the full stack: DB -> engines -> pipeline -> TjSpanExporter.
    Swaps the delegating exporter's target for this test.
    """
    db = InMemoryBackend()
    config = TjConfig(
        version="1",
        security=SecurityConfig(ingest_secret="test"),
        capture=CaptureConfig(
            prompts=True,
            completions=True,
            tool_inputs=True,
            tool_outputs=True,
        ),
        agents={
            "test-agent": AgentConfig(
                budget=BudgetConfig(daily_usd=10.0, session_usd=5.0),
            ),
            "test-email-agent": AgentConfig(
                budget=BudgetConfig(daily_usd=10.0, session_usd=5.0),
            ),
        },
    )

    cost_engine = CostEngine(db=db)
    alert_engine = AlertEngine(db=db, config=config)
    schema_validator = SchemaValidator(db=db, alert_engine=alert_engine, config=config)

    pipeline = IngestPipeline(
        db=db,
        config=config,
        cost_engine=cost_engine,
        alert_engine=alert_engine,
        schema_validator=schema_validator,
    )

    ocw_exporter = TjSpanExporter(pipeline)

    # Create a local TracerProvider (not global) and bind the SDK tracer to it
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(ocw_exporter))
    agent_mod._tracer = provider.get_tracer("tokenjam.sdk")

    # Seed agent records
    now = utcnow()
    db.upsert_agent(AgentRecord(
        agent_id="test-agent", first_seen=now, last_seen=now, provider="anthropic",
    ))
    db.upsert_agent(AgentRecord(
        agent_id="test-email-agent", first_seen=now, last_seen=now, provider="anthropic",
    ))

    class _Stack:
        pass

    stack = _Stack()
    stack.db = db

    yield stack

    provider.shutdown()
    db.close()


# ── OTel ReadableSpan -> NormalizedSpan ──────────────────────────────────


def test_convert_otel_span_extracts_cache_read_and_write_tokens():
    """convert_otel_span indexes both cache-read and cache-creation tokens.

    Regression: provider previously read only CACHE_READ_TOKENS, dropping
    cache-creation tokens so cache-write cost was never charged on this path.
    """
    collected: list[ReadableSpan] = []

    class _Collector(SpanExporter):
        def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
            collected.extend(spans)
            return SpanExportResult.SUCCESS

        def shutdown(self) -> None:
            pass

    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(_Collector()))
    tracer = provider.get_tracer("test")

    with tracer.start_as_current_span("gen_ai.llm.call") as span:
        span.set_attribute(GenAIAttributes.REQUEST_MODEL, "claude-haiku-4-5")
        span.set_attribute(GenAIAttributes.CACHE_READ_TOKENS, 1000)
        span.set_attribute(GenAIAttributes.CACHE_CREATE_TOKENS, 2000)

    assert len(collected) == 1
    normalized = convert_otel_span(collected[0])
    assert normalized.cache_tokens == 1000
    assert normalized.cache_write_tokens == 2000


# ── SDK -> Pipeline -> DB ─────────────────────────────────────────────────


def test_watch_and_record_llm_call_flows_to_db(full_stack):
    """@watch() + record_llm_call() should produce spans in the DB."""

    @watch(agent_id="test-agent")
    def my_agent():
        record_llm_call("claude-haiku-4-5", "anthropic", 500, 100)

    my_agent()

    spans = _all_spans(full_stack.db)
    assert len(spans) >= 2  # session + LLM call

    llm_spans = [s for s in spans if s.name == GenAIAttributes.SPAN_LLM_CALL]
    assert len(llm_spans) == 1
    assert llm_spans[0].model == "claude-haiku-4-5"
    assert llm_spans[0].input_tokens == 500
    assert llm_spans[0].output_tokens == 100


def test_session_created_in_db(full_stack):
    """A @watch() session should create a SessionRecord in the DB."""

    @watch(agent_id="test-agent")
    def my_agent():
        record_llm_call("claude-haiku-4-5", "anthropic", 200, 50)

    my_agent()

    sessions = _all_sessions(full_stack.db)
    assert len(sessions) >= 1
    # The session span carries agent_id; verify at least one span has it
    spans = _all_spans(full_stack.db)
    agent_ids = {s.agent_id for s in spans if s.agent_id}
    assert "test-agent" in agent_ids


def test_cost_calculated_for_llm_spans(full_stack):
    """CostEngine should calculate and record cost_usd for LLM spans."""

    @watch(agent_id="test-agent")
    def my_agent():
        record_llm_call("claude-haiku-4-5", "anthropic", 1000, 200)

    my_agent()

    spans = _all_spans(full_stack.db)
    llm_spans = [s for s in spans if s.name == GenAIAttributes.SPAN_LLM_CALL]
    assert len(llm_spans) == 1
    assert llm_spans[0].cost_usd is not None
    assert llm_spans[0].cost_usd > 0


def test_multiple_llm_calls_accumulate_in_session(full_stack):
    """Multiple LLM calls should accumulate tokens in the session."""

    @watch(agent_id="test-agent")
    def my_agent():
        for _ in range(3):
            record_llm_call("claude-haiku-4-5", "anthropic", 100, 20)

    my_agent()

    spans = _all_spans(full_stack.db)
    llm_spans = [s for s in spans if s.name == GenAIAttributes.SPAN_LLM_CALL]
    assert len(llm_spans) == 3


def test_tool_call_flows_to_db(full_stack):
    """record_tool_call() should produce a tool span in the DB."""

    @watch(agent_id="test-agent")
    def my_agent():
        record_llm_call("claude-haiku-4-5", "anthropic", 100, 20)
        record_tool_call("send_email", tool_output={"status": "sent"})

    my_agent()

    spans = _all_spans(full_stack.db)
    tool_spans = [s for s in spans if s.name == GenAIAttributes.SPAN_TOOL_CALL]
    assert len(tool_spans) == 1
    assert tool_spans[0].tool_name == "send_email"


def test_agent_session_context_manager_flows_to_db(full_stack):
    """AgentSession used directly should also produce spans in DB."""

    with AgentSession(agent_id="test-agent", agent_name="Test"):
        record_llm_call("claude-haiku-4-5", "anthropic", 300, 60)

    spans = _all_spans(full_stack.db)
    session_spans = [s for s in spans if s.name == GenAIAttributes.SPAN_INVOKE_AGENT]
    assert len(session_spans) >= 1


def test_exception_records_error_in_db(full_stack):
    """An exception inside @watch() should create an error session span."""

    @watch(agent_id="test-agent")
    def failing_agent():
        record_llm_call("claude-haiku-4-5", "anthropic", 100, 20)
        raise ValueError("intentional error")

    with pytest.raises(ValueError):
        failing_agent()

    spans = _all_spans(full_stack.db)
    session_spans = [s for s in spans if s.name == GenAIAttributes.SPAN_INVOKE_AGENT]
    assert len(session_spans) >= 1
    error_sessions = [s for s in session_spans if s.status_code == SpanStatus.ERROR]
    assert len(error_sessions) >= 1


def test_conversation_id_propagated_through_pipeline(full_stack):
    """conversation_id should flow from SDK through to spans in DB."""

    with AgentSession(
        agent_id="test-agent",
        conversation_id="my-conv-42",
    ):
        record_llm_call("claude-haiku-4-5", "anthropic", 100, 20)

    spans = _all_spans(full_stack.db)
    conv_spans = [s for s in spans if s.conversation_id == "my-conv-42"]
    assert len(conv_spans) >= 1


# ── Mock agent scenario integration ──────────────────────────────────────


def test_mock_normal_agent_produces_expected_spans(full_stack):
    """The normal email agent scenario should produce session + LLM + tool spans."""
    from tests.agents.email_agent_normal import run

    run("Send test email")

    spans = _all_spans(full_stack.db)
    session_spans = [s for s in spans if s.name == GenAIAttributes.SPAN_INVOKE_AGENT]
    llm_spans = [s for s in spans if s.name == GenAIAttributes.SPAN_LLM_CALL]
    tool_spans = [s for s in spans if s.name == GenAIAttributes.SPAN_TOOL_CALL]

    assert len(session_spans) >= 1
    assert len(llm_spans) == 2   # email_agent_normal does 2 LLM calls
    assert len(tool_spans) == 1  # 1 tool call (send_email)


def test_mock_loop_agent_produces_retry_spans(full_stack):
    """The retry loop agent should produce 5 LLM + 5 tool call spans."""
    from tests.agents.email_agent_loop import run

    run("Send test email")

    spans = _all_spans(full_stack.db)
    llm_spans = [s for s in spans if s.name == GenAIAttributes.SPAN_LLM_CALL]
    tool_spans = [s for s in spans if s.name == GenAIAttributes.SPAN_TOOL_CALL]

    assert len(llm_spans) == 5
    assert len(tool_spans) == 5
