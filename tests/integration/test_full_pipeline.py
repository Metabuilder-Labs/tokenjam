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
from tests.factories import make_invoke_agent_span, make_llm_span, make_session




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
    unattributed = full_stack.db.get_unattributed_spend()
    assert unattributed["span_count"] == 1
    assert unattributed["trace_count"] == 1
    assert unattributed["cost_usd"] == pytest.approx(cost_rows[0].cost_usd)


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
    """A late marker moves provisional cost and supersedes its provisional session (#749)."""
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
    # Reference safety: provisional session is superseded, never deleted from DB
    provisional_session = full_stack.db.get_session(provisional_session_id)
    assert provisional_session is not None
    assert provisional_session.status == "superseded"
    assert provisional_session.total_cost_usd == 0.0
    assert provisional_session.input_tokens == 0
    session = full_stack.db.get_session("late-marker")
    assert session is not None
    assert session.input_tokens == 77
    assert session.output_tokens == 11
    assert session.total_cost_usd == expected_cost


def test_real_pipeline_generic_custom_marker_sole_marker(full_stack):
    """A sole generic OTLP marker with custom name attributes child spans via step2_marker."""
    trace_id = "real-generic-sole-trace"
    m = make_invoke_agent_span(session_id="wf-1", trace_id=trace_id)
    m.name = "workflow"
    full_stack.pipeline.process(m)

    cost_span = make_llm_span(trace_id=trace_id, input_tokens=30, output_tokens=10)
    cost_span.session_id = None
    cost_span.conversation_id = None
    full_stack.pipeline.process(cost_span)

    stored = full_stack.db.get_trace_spans(trace_id)
    cost_row = next(s for s in stored if s.name == GenAIAttributes.SPAN_LLM_CALL)
    assert cost_row.session_id == "wf-1"
    assert cost_row.attribution_step == "step2_marker"
    assert full_stack.db.get_session("wf-1").input_tokens == 30
    assert full_stack.db.get_session("wf-1").total_cost_usd == cost_row.cost_usd


def test_real_pipeline_generic_custom_markers_multi_marker_unattributed(full_stack):
    """Generic OTLP traces with multiple custom markers leave unparented cost unattributed."""
    trace_id = "real-generic-multi-trace"
    m1 = make_invoke_agent_span(session_id="wf-1", trace_id=trace_id)
    m1.name = "workflow"
    full_stack.pipeline.process(m1)
    m2 = make_invoke_agent_span(session_id="wf-2", trace_id=trace_id)
    m2.name = "agent.run"
    full_stack.pipeline.process(m2)

    cost_span = make_llm_span(trace_id=trace_id, input_tokens=100, output_tokens=50)
    cost_span.session_id = None
    cost_span.conversation_id = None
    full_stack.pipeline.process(cost_span)

    stored = full_stack.db.get_trace_spans(trace_id)
    cost_row = next(s for s in stored if s.name == GenAIAttributes.SPAN_LLM_CALL)
    assert cost_row.session_id is None
    assert cost_row.attribution_step == "step3_unattributed"
    assert full_stack.db.get_session("wf-1").input_tokens == 0
    assert full_stack.db.get_session("wf-2").input_tokens == 0
    unattributed = full_stack.db.get_unattributed_spend()
    assert unattributed["span_count"] == 1
    assert unattributed["trace_count"] == 1
    assert unattributed["cost_usd"] == pytest.approx(cost_row.cost_usd)


def test_real_pipeline_generic_custom_marker_reverse_arrival_reconciliation(full_stack):
    """Late arrival of custom-named marker reconciles provisional cost in DuckDB."""
    trace_id = "real-generic-reverse-trace"
    cost_span = make_llm_span(trace_id=trace_id, input_tokens=60, output_tokens=20)
    cost_span.session_id = None
    cost_span.conversation_id = None
    full_stack.pipeline.process(cost_span)

    provisional = full_stack.db.get_trace_spans(trace_id)[0]
    provisional_id = provisional.session_id
    assert provisional_id is not None
    assert provisional.attribution_step == "provisional"
    expected_cost = provisional.cost_usd

    marker = make_invoke_agent_span(session_id="wf-late", trace_id=trace_id)
    marker.name = "workflow"
    full_stack.pipeline.process(marker)

    stored = full_stack.db.get_trace_spans(trace_id)
    cost_row = next(s for s in stored if s.name == GenAIAttributes.SPAN_LLM_CALL)
    assert cost_row.session_id == "wf-late"
    assert cost_row.attribution_step == "step2_marker"

    # Provisional session superseded and zeroed
    prov_sess = full_stack.db.get_session(provisional_id)
    assert prov_sess.status == "superseded"
    assert prov_sess.total_cost_usd == 0.0
    assert prov_sess.input_tokens == 0

    # New session has totals
    wf_sess = full_stack.db.get_session("wf-late")
    assert wf_sess.total_cost_usd == expected_cost
    assert wf_sess.input_tokens == 60


def test_real_pipeline_generic_custom_markers_nested_hierarchy_forward_and_reverse(full_stack):
    """Nested custom markers (workflow -> agent.run) in real DuckDB pipeline attribute child spans
    to the immediate subagent marker session via step1_parent in both forward and reverse arrival (#749).
    """
    # 1. Forward arrival
    trace_fwd = "real-generic-nested-fwd"
    root_fwd = make_invoke_agent_span(session_id="wf-root-fwd", trace_id=trace_fwd)
    root_fwd.span_id = "root-fwd"
    root_fwd.name = "workflow"
    full_stack.pipeline.process(root_fwd)

    sub_fwd = make_invoke_agent_span(session_id="sub-fwd", trace_id=trace_fwd)
    sub_fwd.span_id = "sub-fwd"
    sub_fwd.parent_span_id = "root-fwd"
    sub_fwd.name = "agent.run"
    full_stack.pipeline.process(sub_fwd)

    child_fwd = make_llm_span(trace_id=trace_fwd, input_tokens=40, output_tokens=15)
    child_fwd.span_id = "child-fwd"
    child_fwd.parent_span_id = "sub-fwd"
    child_fwd.session_id = None
    child_fwd.conversation_id = None
    full_stack.pipeline.process(child_fwd)

    stored_fwd = next(s for s in full_stack.db.get_trace_spans(trace_fwd) if s.span_id == "child-fwd")
    assert stored_fwd.session_id == "sub-fwd"
    assert stored_fwd.attribution_step == "step1_parent"
    assert full_stack.db.get_session("sub-fwd").input_tokens == 40
    assert full_stack.db.get_session("wf-root-fwd").input_tokens == 0

    # 2. Reverse arrival
    trace_rev = "real-generic-nested-rev"
    child_rev = make_llm_span(trace_id=trace_rev, input_tokens=50, output_tokens=25)
    child_rev.span_id = "child-rev"
    child_rev.parent_span_id = "sub-rev"
    child_rev.session_id = None
    child_rev.conversation_id = None
    full_stack.pipeline.process(child_rev)

    root_rev = make_invoke_agent_span(session_id="wf-root-rev", trace_id=trace_rev)
    root_rev.span_id = "root-rev"
    root_rev.name = "workflow"
    full_stack.pipeline.process(root_rev)

    sub_rev = make_invoke_agent_span(session_id="sub-rev", trace_id=trace_rev)
    sub_rev.span_id = "sub-rev"
    sub_rev.parent_span_id = "root-rev"
    sub_rev.name = "agent.run"
    full_stack.pipeline.process(sub_rev)

    stored_rev = next(s for s in full_stack.db.get_trace_spans(trace_rev) if s.span_id == "child-rev")
    assert stored_rev.session_id == "sub-rev"
    assert stored_rev.attribution_step == "step1_parent"
    assert full_stack.db.get_session("sub-rev").input_tokens == 50
    assert full_stack.db.get_session("wf-root-rev").input_tokens == 0


def test_real_pipeline_generic_custom_markers_same_name_same_session(full_stack):
    """Multiple spans with the same custom name and same session_id in DuckDB
    are recognized as a sole marker, resolving unparented spans to step2_marker (#749).
    """
    trace_id = "real-generic-same-name-same-sess"
    m1 = make_invoke_agent_span(session_id="wf-sole-dup", trace_id=trace_id)
    m1.name = "workflow"
    full_stack.pipeline.process(m1)

    m2 = make_invoke_agent_span(session_id="wf-sole-dup", trace_id=trace_id)
    m2.name = "workflow"
    full_stack.pipeline.process(m2)

    cost_span = make_llm_span(trace_id=trace_id, input_tokens=80, output_tokens=30)
    cost_span.session_id = None
    cost_span.conversation_id = None
    full_stack.pipeline.process(cost_span)

    stored = full_stack.db.get_trace_spans(trace_id)
    cost_row = next(s for s in stored if s.name == GenAIAttributes.SPAN_LLM_CALL)
    assert cost_row.session_id == "wf-sole-dup"
    assert cost_row.attribution_step == "step2_marker"
    assert full_stack.db.get_session("wf-sole-dup").input_tokens == 80


def test_real_pipeline_child_spans_with_known_conversation_resolve_by_conversation(full_stack):
    """A known conversation remains the explicit owner even when a parent agrees."""
    trace_id = "real-pipe-conv-inherit"
    conv_id = "real-conv-42"

    root = make_invoke_agent_span(conversation_id=conv_id, trace_id=trace_id)
    root.session_id = None
    full_stack.pipeline.process(root)

    root_stored = full_stack.db.get_trace_spans(trace_id)[0]
    assert root_stored.attribution_step == "conversation"
    assert full_stack.pipeline._is_session_marker(root_stored) is True

    # Cache should contain root span
    assert (trace_id, root.span_id) in full_stack.pipeline._parent_cache

    child = make_llm_span(conversation_id=conv_id, trace_id=trace_id, input_tokens=55, output_tokens=22)
    child.session_id = None
    child.parent_span_id = root.span_id
    full_stack.pipeline.process(child)

    child_stored = next(s for s in full_stack.db.get_trace_spans(trace_id) if s.span_id == child.span_id)
    assert child_stored.session_id == root_stored.session_id
    assert child_stored.attribution_step == "conversation"

    # Parent cache must remain populated
    assert (trace_id, root.span_id) in full_stack.pipeline._parent_cache
    assert (trace_id, child.span_id) in full_stack.pipeline._parent_cache


def test_real_pipeline_known_conversation_overrides_different_parent_session(full_stack):
    """An explicit conversation must not be billed to an inferred parent session."""
    conversation_id = "conversation-owner"
    conversation_marker = make_invoke_agent_span(conversation_id=conversation_id)
    conversation_marker.session_id = None
    full_stack.pipeline.process(conversation_marker)
    conversation_session_id = full_stack.db.get_trace_spans(conversation_marker.trace_id)[0].session_id
    assert conversation_session_id is not None

    trace_id = "conversation-vs-parent"
    parent = make_invoke_agent_span(session_id="parent-owner", trace_id=trace_id)
    full_stack.pipeline.process(parent)

    child = make_llm_span(
        trace_id=trace_id,
        conversation_id=conversation_id,
        input_tokens=55,
        output_tokens=22,
    )
    child.session_id = None
    child.parent_span_id = parent.span_id
    full_stack.pipeline.process(child)

    stored_child = next(s for s in full_stack.db.get_trace_spans(trace_id) if s.span_id == child.span_id)
    assert stored_child.session_id == conversation_session_id
    assert stored_child.attribution_step == "conversation"
    assert full_stack.db.get_session("parent-owner").input_tokens == 0


def test_real_pipeline_reconciles_repeated_marker_session_with_late_parent(full_stack):
    """A repeated session marker can make an earlier child parent-resolvable."""
    trace_id = "repeated-marker-late-parent"

    first_marker = make_invoke_agent_span(session_id="session-a", trace_id=trace_id)
    first_marker.span_id = "marker-a"
    full_stack.pipeline.process(first_marker)

    second_marker = make_invoke_agent_span(session_id="session-b", trace_id=trace_id)
    second_marker.span_id = "marker-b"
    full_stack.pipeline.process(second_marker)

    child = make_llm_span(
        trace_id=trace_id,
        input_tokens=33,
        output_tokens=11,
    )
    child.span_id = "orphan-child"
    child.parent_span_id = "late-parent"
    child.session_id = None
    child.conversation_id = None
    full_stack.pipeline.process(child)

    before = next(s for s in full_stack.db.get_trace_spans(trace_id) if s.span_id == child.span_id)
    assert before.session_id is None
    assert before.attribution_step == "step3_unattributed"

    late_parent = make_invoke_agent_span(session_id="session-a", trace_id=trace_id)
    late_parent.span_id = "late-parent"
    full_stack.pipeline.process(late_parent)

    after = next(s for s in full_stack.db.get_trace_spans(trace_id) if s.span_id == child.span_id)
    assert after.session_id == "session-a"
    assert after.attribution_step == "step1_parent"
    assert full_stack.db.get_session("session-a").input_tokens == 33
    assert full_stack.db.get_session("session-b").input_tokens == 0


def test_real_pipeline_reconciliation_rolls_back_on_refresh_failure(full_stack, monkeypatch):
    """A failed aggregate refresh rolls back reconciliation without rejecting the marker span (#749)."""
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
    # Reconciliation failure must NOT reject the marker span; ingest must succeed
    full_stack.pipeline.process(make_invoke_agent_span(
        session_id="rollback-marker", trace_id=trace_id,
    ))

    stored = full_stack.db.get_trace_spans(trace_id)
    # The marker span is durable
    assert any(s.session_id == "rollback-marker" for s in stored)
    # The cost span was rolled back and stays on provisional session
    cost_row = next(s for s in stored if s.name == GenAIAttributes.SPAN_LLM_CALL)
    assert cost_row.session_id == provisional_session_id


def test_real_pipeline_session_attribution_invariant(full_stack):
    """Invariant: sum(session totals) + unattributed_spend == total span spend."""
    # 1. Standalone provisional trace
    span1 = make_llm_span(trace_id="tr-1", cost_usd=0.05, input_tokens=100, output_tokens=50)
    span1.session_id = None
    span1.conversation_id = None
    full_stack.pipeline.process(span1)

    # 2. Shared trace with 2 markers (ambiguous -> unattributed)
    m1 = make_invoke_agent_span(session_id="marker-1", trace_id="tr-shared")
    m2 = make_invoke_agent_span(session_id="marker-2", trace_id="tr-shared")
    full_stack.pipeline.process(m1)
    full_stack.pipeline.process(m2)
    unatt_cost = make_llm_span(trace_id="tr-shared", cost_usd=0.15, input_tokens=200, output_tokens=100)
    unatt_cost.session_id = None
    unatt_cost.conversation_id = None
    full_stack.pipeline.process(unatt_cost)

    # 3. Explicit session trace
    m3 = make_invoke_agent_span(session_id="marker-3", trace_id="tr-single")
    full_stack.pipeline.process(m3)
    span3 = make_llm_span(trace_id="tr-single", cost_usd=0.25, input_tokens=300, output_tokens=150)
    span3.session_id = None
    span3.conversation_id = None
    full_stack.pipeline.process(span3)

    # Query all spans cost
    cur = full_stack.db.conn.execute("SELECT SUM(cost_usd) FROM spans")
    total_spans_cost = float(cur.fetchone()[0] or 0.0)

    # Query sessions total (excluding superseded)
    cur = full_stack.db.conn.execute(
        "SELECT SUM(total_cost_usd) FROM sessions "
        "WHERE (status IS NULL OR status != 'superseded')"
    )
    active_sessions_cost = float(cur.fetchone()[0] or 0.0)

    # Query unattributed spend
    unatt_data = full_stack.db.get_unattributed_spend()
    unatt_cost_val = float(unatt_data["cost_usd"])

    assert pytest.approx(total_spans_cost, rel=1e-5) == active_sessions_cost + unatt_cost_val


def test_real_pipeline_multi_marker_reverse_arrival_preserves_parentage(full_stack):
    """When multiple markers arrive on a shared trace, spans that arrived before
    their marker must resolve via parentage (Step 1) rather than falling to unattributed (Step 3).
    """
    trace_id = "real-multi-marker-reverse"

    # 1. Child 1 arrives first (parent is m1, not yet in DB)
    c1 = make_llm_span(trace_id=trace_id, input_tokens=100, output_tokens=50, cost_usd=1.0)
    c1.span_id = "c1"
    c1.parent_span_id = "m1"
    c1.session_id = None
    c1.conversation_id = None
    full_stack.pipeline.process(c1)

    # 2. Unparented span arrives second
    u = make_llm_span(trace_id=trace_id, input_tokens=40, output_tokens=20, cost_usd=0.5)
    u.span_id = "u"
    u.parent_span_id = None
    u.session_id = None
    u.conversation_id = None
    full_stack.pipeline.process(u)

    # 3. Marker 1 arrives
    m1 = make_invoke_agent_span(session_id="agent-1", trace_id=trace_id)
    m1.span_id = "m1"
    full_stack.pipeline.process(m1)

    # 4. Marker 2 arrives
    m2 = make_invoke_agent_span(session_id="agent-2", trace_id=trace_id)
    m2.span_id = "m2"
    full_stack.pipeline.process(m2)

    # 5. Child 2 arrives after its marker (forward arrival)
    c2 = make_llm_span(trace_id=trace_id, input_tokens=200, output_tokens=100, cost_usd=2.0)
    c2.span_id = "c2"
    c2.parent_span_id = "m2"
    c2.session_id = None
    c2.conversation_id = None
    full_stack.pipeline.process(c2)

    stored = {s.span_id: s for s in full_stack.db.get_trace_spans(trace_id)}

    # c1 resolved to agent-1 via Step 1
    assert stored["c1"].session_id == "agent-1"
    assert stored["c1"].attribution_step == "step1_parent"

    # c2 resolved to agent-2 via Step 1
    assert stored["c2"].session_id == "agent-2"
    assert stored["c2"].attribution_step == "step1_parent"

    # unparented span on 2-marker trace resolved to unattributed (Step 3)
    assert stored["u"].session_id is None
    assert stored["u"].attribution_step == "step3_unattributed"

    # Check session totals
    sess1 = full_stack.db.get_session("agent-1")
    sess2 = full_stack.db.get_session("agent-2")
    assert sess1 is not None and sess1.input_tokens == 100
    assert sess2 is not None and sess2.input_tokens == 200
    assert pytest.approx(sess1.total_cost_usd) == stored["c1"].cost_usd
    assert pytest.approx(sess2.total_cost_usd) == stored["c2"].cost_usd

    # Check unattributed spend
    unatt = full_stack.db.get_unattributed_spend()
    assert pytest.approx(unatt["cost_usd"]) == stored["u"].cost_usd
    assert unatt["span_count"] == 1


def test_real_pipeline_five_trace_only_spans_mint_one_session(full_stack):
    """Preserve #326 behavior in real DuckDB pipeline: 5 trace-only cost spans without marker
    must share exactly one minted provisional session.
    """
    trace_id = "real-pipe-trace-five-spans-no-marker"

    for _ in range(5):
        span = make_llm_span(
            trace_id=trace_id,
            input_tokens=100,
            output_tokens=50,
            cost_usd=0.10,
        )
        span.session_id = None
        span.conversation_id = None
        full_stack.pipeline.process(span)

    spans = full_stack.db.get_trace_spans(trace_id)
    assert len(spans) == 5

    session_ids = {s.session_id for s in spans}
    assert len(session_ids) == 1
    assert None not in session_ids

    single_session_id = list(session_ids)[0]
    sess = full_stack.db.get_session(single_session_id)
    assert sess is not None
    assert sess.input_tokens == 500
    assert sess.output_tokens == 250
    assert pytest.approx(sess.total_cost_usd) == sum(s.cost_usd or 0.0 for s in spans)
    assert sess.total_cost_usd > 0.0

    cur = full_stack.db.conn.execute(
        "SELECT COUNT(*) FROM sessions WHERE (status IS NULL OR status != 'superseded')"
    )
    assert cur.fetchone()[0] == 1


def test_real_pipeline_reconciliation_idempotency(full_stack):
    """Calling reconcile_trace_session_attribution multiple times must be idempotent."""
    trace_id = "real-pipe-idempotent-trace"

    # Step 1: Ingest 3 child spans before marker
    for i in range(3):
        c = make_llm_span(
            trace_id=trace_id,
            input_tokens=100,
            output_tokens=50,
            cost_usd=0.20,
        )
        c.span_id = f"c_{i}"
        c.parent_span_id = "marker_root"
        c.session_id = None
        c.conversation_id = None
        full_stack.pipeline.process(c)

    # Ingest 1 unparented span
    u = make_llm_span(
        trace_id=trace_id,
        input_tokens=50,
        output_tokens=25,
        cost_usd=0.10,
    )
    u.span_id = "unparented"
    u.parent_span_id = None
    u.session_id = None
    u.conversation_id = None
    full_stack.pipeline.process(u)

    # Ingest 2 markers -> multi-marker trace:
    # children of marker_root should resolve to marker_root's session
    # unparented should resolve to unattributed (step3_unattributed)
    m1 = make_invoke_agent_span(session_id="session-root", trace_id=trace_id)
    m1.span_id = "marker_root"
    full_stack.pipeline.process(m1)

    m2 = make_invoke_agent_span(session_id="session-other", trace_id=trace_id)
    m2.span_id = "marker_other"
    full_stack.pipeline.process(m2)

    def _snapshot():
        spans_snap = sorted(
            (s.span_id, s.session_id, s.attribution_step, s.cost_usd)
            for s in full_stack.db.get_trace_spans(trace_id)
        )
        sessions_snap = sorted(
            (s["session_id"], s["status"], s["total_cost_usd"], s["input_tokens"])
            for s in _all_sessions(full_stack.db)
        )
        unatt_snap = full_stack.db.get_unattributed_spend()
        return spans_snap, sessions_snap, unatt_snap

    snap1_spans, snap1_sessions, snap1_unatt = _snapshot()

    assert {
        span_id: (session_id, attribution_step)
        for span_id, session_id, attribution_step, _ in snap1_spans
    } == {
        "c_0": ("session-root", "step1_parent"),
        "c_1": ("session-root", "step1_parent"),
        "c_2": ("session-root", "step1_parent"),
        "marker_other": ("session-other", "explicit"),
        "marker_root": ("session-root", "explicit"),
        "unparented": (None, "step3_unattributed"),
    }
    assert full_stack.db.get_session("session-root").input_tokens == 300
    assert full_stack.db.get_session("session-other").input_tokens == 0
    unparented_cost = next(cost for span_id, _, _, cost in snap1_spans if span_id == "unparented")
    assert snap1_unatt["cost_usd"] == pytest.approx(unparented_cost)
    assert snap1_unatt["span_count"] == 1
    assert snap1_unatt["trace_count"] == 1

    # Reconcile again explicitly (second time)
    full_stack.db.reconcile_trace_session_attribution(trace_id)
    snap2_spans, snap2_sessions, snap2_unatt = _snapshot()

    assert snap1_spans == snap2_spans
    assert snap1_sessions == snap2_sessions
    assert snap1_unatt == snap2_unatt

    # Reconcile again explicitly (third time)
    full_stack.db.reconcile_trace_session_attribution(trace_id)
    snap3_spans, snap3_sessions, snap3_unatt = _snapshot()

    assert snap1_spans == snap3_spans
    assert snap1_sessions == snap3_sessions
    assert snap1_unatt == snap3_unatt


def test_real_pipeline_reconciliation_scale_benchmark(full_stack):
    """A real pipeline marker reconciles 1,000 DuckDB spans under 2.0s (#749)."""
    import time
    trace_id = "scale-bench-trace"
    now = utcnow()

    # Pre-insert 1,000 spans on the same trace
    spans = [
        make_llm_span(
            agent_id="bench-agent",
            trace_id=trace_id,
            cost_usd=0.01,
            input_tokens=10,
            output_tokens=5,
            start_time=now + timedelta(milliseconds=i),
        )
        for i in range(1000)
    ]
    for s in spans:
        s.session_id = "provisional-bench-session"
        s.attribution_step = "provisional"

    # Create the provisional session row
    full_stack.db.upsert_session(
        make_session(
            session_id="provisional-bench-session",
            agent_id="bench-agent",
            total_cost_usd=10.0,
            input_tokens=10000,
            output_tokens=5000,
        )
    )
    full_stack.db.bulk_insert_spans(spans)

    # Ingest the marker through the production pipeline. The timed path must
    # include marker resolution, persistence, session upsert, and reconciliation
    # so an O(n²) trace hydration regression cannot hide behind direct DB setup.
    marker = make_invoke_agent_span(
        session_id="canonical-bench-session",
        agent_id="bench-agent",
        trace_id=trace_id,
    )

    start = time.perf_counter()
    full_stack.pipeline.process(marker)
    duration = time.perf_counter() - start

    assert duration < 2.0, f"Reconciliation of 1,000 spans took {duration:.3f}s (must be < 2.0s)"

    # Verify reconciliation correctness: all 1,000 spans moved to canonical-bench-session
    trace_spans = full_stack.db.get_trace_spans(trace_id)
    llm_spans = [s for s in trace_spans if s.name == GenAIAttributes.SPAN_LLM_CALL]
    assert len(llm_spans) == 1000
    for s in llm_spans:
        assert s.session_id == "canonical-bench-session"
        assert s.attribution_step == "step2_marker"

    # Verify canonical session got totals recomputed
    canonical = full_stack.db.get_session("canonical-bench-session")
    assert canonical is not None
    assert canonical.input_tokens == 10000
    assert canonical.output_tokens == 5000
    assert pytest.approx(canonical.total_cost_usd) == 10.0

    # Verify provisional session was marked superseded
    provisional = full_stack.db.get_session("provisional-bench-session")
    assert provisional is not None
    assert provisional.status == "superseded"
    assert provisional.total_cost_usd == 0.0


def test_real_pipeline_explicit_session_trace_stays_near_trace_only_path(
    full_stack, monkeypatch,
):
    """Explicit-session traces must not reconcile the whole trace per span."""
    import time

    span_count = 2_000
    explicit_trace = "explicit-session-hot-path"
    trace_only_trace = "trace-only-hot-path"
    pipeline = IngestPipeline(full_stack.db, full_stack.pipeline.config)
    reconcile_calls: list[str] = []
    reconcile = full_stack.db.reconcile_trace_session_attribution

    def counted_reconcile(trace_id: str) -> None:
        reconcile_calls.append(trace_id)
        reconcile(trace_id)

    monkeypatch.setattr(
        full_stack.db,
        "reconcile_trace_session_attribution",
        counted_reconcile,
    )

    started = time.perf_counter()
    for i in range(span_count):
        span = make_llm_span(
            agent_id="bench-agent",
            trace_id=explicit_trace,
            span_id=f"explicit-{i}",
            session_id="explicit-session",
            input_tokens=10,
            output_tokens=5,
            cost_usd=0.01,
        )
        pipeline.process(span)
    explicit_duration = time.perf_counter() - started

    started = time.perf_counter()
    for i in range(span_count):
        span = make_llm_span(
            agent_id="bench-agent",
            trace_id=trace_only_trace,
            span_id=f"trace-only-{i}",
            input_tokens=10,
            output_tokens=5,
            cost_usd=0.01,
        )
        span.session_id = None
        span.conversation_id = None
        pipeline.process(span)
    trace_only_duration = time.perf_counter() - started

    assert reconcile_calls == [explicit_trace]
    assert explicit_duration < trace_only_duration * 8 + 1.0, (
        f"explicit-session ingest took {explicit_duration / span_count * 1000:.2f} ms/span; "
        f"trace-only ingest took {trace_only_duration / span_count * 1000:.2f} ms/span"
    )


def test_real_pipeline_multi_marker_all_unparented_spans_unattributed(full_stack):
    """When multiple markers exist and non-marker spans have no parentage,
    all eligible spans must batch-reconcile to session_id=NULL, step3_unattributed (Issue #749).
    """
    trace_id = "real-pipe-multi-marker-all-unatt"

    # 1. Ingest 3 unparented spans before any marker (minting a provisional session)
    for i in range(3):
        span = make_llm_span(
            trace_id=trace_id,
            input_tokens=100,
            output_tokens=50,
            cost_usd=0.30,
        )
        span.span_id = f"unparented_{i}"
        span.parent_span_id = None
        span.session_id = None
        span.conversation_id = None
        full_stack.pipeline.process(span)

    # All 3 spans minted and shared a provisional session
    prov_spans = full_stack.db.get_trace_spans(trace_id)
    assert len(prov_spans) == 3
    prov_sess_id = prov_spans[0].session_id
    assert prov_sess_id is not None

    # 2. Ingest two explicit markers on this trace -> multi-marker trace
    m1 = make_invoke_agent_span(session_id="marker-sess-1", trace_id=trace_id)
    m1.span_id = "marker_1"
    full_stack.pipeline.process(m1)

    m2 = make_invoke_agent_span(session_id="marker-sess-2", trace_id=trace_id)
    m2.span_id = "marker_2"
    full_stack.pipeline.process(m2)

    # 3. Verify all 3 unparented spans resolved to unattributed (session_id is NULL)
    trace_spans = {s.span_id: s for s in full_stack.db.get_trace_spans(trace_id)}
    assert len(trace_spans) == 5
    for i in range(3):
        s = trace_spans[f"unparented_{i}"]
        assert s.session_id is None
        assert s.attribution_step == "step3_unattributed"

    # Markers retain their own session_ids
    assert trace_spans["marker_1"].session_id == "marker-sess-1"
    assert trace_spans["marker_2"].session_id == "marker-sess-2"

    # Provisional session must be marked superseded
    prov_sess = full_stack.db.get_session(prov_sess_id)
    assert prov_sess is not None
    assert prov_sess.status == "superseded"
    assert prov_sess.total_cost_usd == 0.0

    # Unattributed spend should report all 3 spans
    unatt = full_stack.db.get_unattributed_spend()
    expected_cost = sum(trace_spans[f"unparented_{i}"].cost_usd or 0.0 for i in range(3))
    assert pytest.approx(expected_cost) == unatt["cost_usd"]
    assert unatt["span_count"] == 3
    assert unatt["trace_count"] == 1


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
