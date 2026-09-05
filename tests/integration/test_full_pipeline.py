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
from datetime import timedelta
from typing import Sequence

import pytest
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
from tests.factories import make_invoke_agent_span, make_llm_span




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
    # for the duration of this test. Restored in teardown below so a stale
    # tracer bound to this test's (soon-to-be-shutdown) provider and closed DB
    # doesn't leak into tests that run later in the same session (#615).
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(ocw_exporter))
    original_tracer = agent_mod._tracer
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
    stack.pipeline = pipeline

    yield stack

    agent_mod._tracer = original_tracer
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


def test_llm_only_session_enforces_session_budget_while_still_active(full_stack):
    """A stream without invoke_agent still reaches session-scoped alerts."""
    full_stack.pipeline.process(
        make_llm_span(
            agent_id="test-agent",
            session_id="llm-only-budget",
            input_tokens=30_000_000,
            output_tokens=0,
        )
    )
    full_stack.pipeline.process(
        make_llm_span(
            agent_id="test-agent",
            session_id="llm-only-budget",
            input_tokens=30_000_000,
            output_tokens=0,
        )
    )
    full_stack.pipeline.process(
        make_llm_span(
            agent_id="test-agent",
            session_id="another-llm-only-budget",
            input_tokens=30_000_000,
            output_tokens=0,
        )
    )

    rows = full_stack.db.conn.execute(
        "SELECT type, suppressed FROM alerts ORDER BY fired_at"
    ).fetchall()
    assert [(row[0], row[1]) for row in rows] == [
        ("cost_budget_session", False),
        ("cost_budget_session", False),
    ]


def test_llm_only_session_enforces_duration_limit_while_still_active(full_stack):
    """Duration is checked from the running session's observed span bounds."""
    start = utcnow() - timedelta(seconds=4001)
    full_stack.pipeline.process(
        make_llm_span(
            agent_id="test-agent",
            session_id="llm-only-duration",
            start_time=start,
            cost_usd=0.01,
        )
    )
    full_stack.pipeline.process(
        make_llm_span(
            agent_id="test-agent",
            session_id="llm-only-duration",
            cost_usd=0.01,
        )
    )

    rows = full_stack.db.conn.execute(
        "SELECT type FROM alerts WHERE session_id = ?", ["llm-only-duration"]
    ).fetchall()
    assert [row[0] for row in rows] == ["session_duration"]


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


def test_real_pipeline_leaves_shared_trace_cost_unattributed(full_stack):
    """A shared trace must not charge trace-only cost to its first marker."""
    trace_id = "real-shared-trace"
    full_stack.pipeline.process(make_invoke_agent_span(session_id="w1", trace_id=trace_id))
    full_stack.pipeline.process(make_invoke_agent_span(session_id="w2", trace_id=trace_id))

    cost_span = make_llm_span(trace_id=trace_id, input_tokens=123, output_tokens=45)
    cost_span.session_id = None
    cost_span.conversation_id = None
    full_stack.pipeline.process(cost_span)

    stored = full_stack.db.get_trace_spans(trace_id)
    cost_rows = [s for s in stored if s.name == GenAIAttributes.SPAN_LLM_CALL]
    assert len(cost_rows) == 1
    assert cost_rows[0].session_id is None
    assert full_stack.db.get_session("w1").input_tokens == 0
    assert full_stack.db.get_session("w2").input_tokens == 0


def test_real_pipeline_uses_parent_for_shared_trace_cost(full_stack):
    """A stored parent identifies a child despite another trace marker."""
    trace_id = "real-parent-trace"
    first = make_invoke_agent_span(session_id="w2", trace_id=trace_id)
    full_stack.pipeline.process(first)
    parent = make_invoke_agent_span(session_id="w1", trace_id=trace_id)
    full_stack.pipeline.process(parent)

    cost_span = make_llm_span(trace_id=trace_id, input_tokens=20, output_tokens=5)
    cost_span.session_id = None
    cost_span.conversation_id = None
    cost_span.parent_span_id = parent.span_id
    full_stack.pipeline.process(cost_span)

    stored = full_stack.db.get_trace_spans(trace_id)
    cost_row = next(s for s in stored if s.name == GenAIAttributes.SPAN_LLM_CALL)
    assert cost_row.session_id == "w1"
    assert full_stack.db.get_session("w1").input_tokens == 20
    assert full_stack.db.get_session("w2").input_tokens == 0


def test_real_pipeline_reparents_reverse_arrival_and_reconciles_totals(full_stack):
    """A late marker moves provisional cost and removes its empty session."""
    trace_id = "real-reverse-trace"
    cost_span = make_llm_span(trace_id=trace_id, input_tokens=77, output_tokens=11)
    cost_span.session_id = None
    cost_span.conversation_id = None
    full_stack.pipeline.process(cost_span)
    provisional = full_stack.db.get_trace_spans(trace_id)[0]
    provisional_session_id = provisional.session_id
    assert provisional_session_id is not None
    expected_cost = provisional.cost_usd

    full_stack.pipeline.process(make_invoke_agent_span(
        session_id="late-marker", trace_id=trace_id,
    ))

    stored = full_stack.db.get_trace_spans(trace_id)
    assert {s.session_id for s in stored} == {"late-marker"}
    assert full_stack.db.get_session(provisional_session_id) is None
    session = full_stack.db.get_session("late-marker")
    assert session is not None
    assert session.input_tokens == 77
    assert session.output_tokens == 11
    assert session.total_cost_usd == expected_cost


def test_real_pipeline_reconciliation_rolls_back_on_refresh_failure(full_stack, monkeypatch):
    """A failed aggregate refresh must not leave attribution partially moved."""
    trace_id = "real-reconcile-rollback"
    cost_span = make_llm_span(trace_id=trace_id, input_tokens=77, output_tokens=11)
    cost_span.session_id = None
    cost_span.conversation_id = None
    full_stack.pipeline.process(cost_span)
    provisional = full_stack.db.get_trace_spans(trace_id)[0]
    provisional_session_id = provisional.session_id
    assert provisional_session_id is not None

    def fail_refresh(_session_ids):
        raise RuntimeError("refresh failed")

    monkeypatch.setattr(full_stack.db, "recompute_session_totals_from_spans", fail_refresh)
    with pytest.raises(RuntimeError, match="refresh failed"):
        full_stack.pipeline.process(make_invoke_agent_span(
            session_id="rollback-marker", trace_id=trace_id,
        ))

    stored = full_stack.db.get_trace_spans(trace_id)
    cost_row = next(s for s in stored if s.name == GenAIAttributes.SPAN_LLM_CALL)
    assert cost_row.session_id == provisional_session_id


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
