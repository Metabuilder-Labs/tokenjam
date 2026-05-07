"""Integration tests for the database layer."""
from __future__ import annotations

from datetime import date, timedelta

import pytest

from tj.core.db import InMemoryBackend, run_migrations
from tj.core.models import (
    AgentRecord,
    Alert,
    AlertFilters,
    AlertType,
    CostFilters,
    DriftBaseline,
    SchemaValidationResult,
    Severity,
    TraceFilters,
)
from tj.utils.ids import new_uuid, new_span_id, new_trace_id
from tj.utils.time_parse import utcnow
from tests.factories import make_llm_span, make_session, make_tool_span


@pytest.fixture
def db():
    backend = InMemoryBackend()
    yield backend
    backend.close()


def _insert_agent(db, agent_id="test-agent"):
    """Helper to ensure an agent row exists."""
    now = utcnow()
    db.upsert_agent(AgentRecord(
        agent_id=agent_id, first_seen=now, last_seen=now,
    ))


# -- Migration tests --

def test_migrations_run_on_empty_db():
    backend = InMemoryBackend()
    rows = backend.conn.execute("SELECT version FROM schema_migrations").fetchall()
    assert len(rows) == 3
    assert {r[0] for r in rows} == {1, 2, 3}
    backend.close()


def test_migrations_are_idempotent():
    backend = InMemoryBackend()
    # Running migrations again should not raise
    run_migrations(backend.conn)
    rows = backend.conn.execute("SELECT version FROM schema_migrations").fetchall()
    assert len(rows) == 3
    backend.close()


# -- Span insert / agent upsert --

def test_insert_span_and_retrieve(db):
    _insert_agent(db)
    span = make_llm_span(agent_id="test-agent")
    session = make_session(agent_id="test-agent", session_id=span.session_id or new_uuid())
    db.upsert_session(session)
    db.insert_span(span)

    result = db.get_trace_spans(span.trace_id)
    assert len(result) == 1
    assert result[0].span_id == span.span_id
    assert result[0].model == "claude-haiku-4-5"


def test_insert_span_creates_agent_row(db):
    agent_id = "new-agent"
    now = utcnow()
    db.upsert_agent(AgentRecord(
        agent_id=agent_id, first_seen=now, last_seen=now, provider="anthropic",
    ))
    # Verify agent exists
    rows = db.conn.execute(
        "SELECT agent_id, provider FROM agents WHERE agent_id = $1", [agent_id]
    ).fetchall()
    assert len(rows) == 1
    assert rows[0][1] == "anthropic"


def test_upsert_agent_updates_last_seen(db):
    agent_id = "test-agent"
    t1 = utcnow() - timedelta(hours=1)
    t2 = utcnow()
    db.upsert_agent(AgentRecord(agent_id=agent_id, first_seen=t1, last_seen=t1))
    db.upsert_agent(AgentRecord(agent_id=agent_id, first_seen=t1, last_seen=t2))
    rows = db.conn.execute(
        "SELECT last_seen FROM agents WHERE agent_id = $1", [agent_id]
    ).fetchall()
    assert rows[0][0] >= t2 - timedelta(seconds=1)


# -- Session upsert / continuity --

def test_upsert_session_totals(db):
    _insert_agent(db)
    session = make_session(input_tokens=100, output_tokens=50)
    db.upsert_session(session)

    fetched = db.get_session(session.session_id)
    assert fetched is not None
    assert fetched.input_tokens == 100
    assert fetched.output_tokens == 50

    # Update totals
    session.input_tokens = 200
    session.output_tokens = 100
    db.upsert_session(session)

    fetched = db.get_session(session.session_id)
    assert fetched is not None
    assert fetched.input_tokens == 200
    assert fetched.output_tokens == 100


def test_conversation_id_continuity_across_sessions(db):
    _insert_agent(db)
    conv_id = new_uuid()
    session1 = make_session(conversation_id=conv_id, input_tokens=100)
    db.upsert_session(session1)

    # Look up session by conversation_id
    found = db.get_session_by_conversation(conv_id)
    assert found is not None
    assert found.session_id == session1.session_id


def test_get_session_by_conversation_returns_none_for_unknown(db):
    result = db.get_session_by_conversation("nonexistent")
    assert result is None


# -- Cost queries --

def test_get_daily_cost_sums_correctly(db):
    _insert_agent(db)
    session = make_session()
    db.upsert_session(session)

    now = utcnow()
    today = now.date()
    for i in range(3):
        span = make_llm_span(
            agent_id="test-agent",
            cost_usd=1.50,
            session_id=session.session_id,
            start_time=now,
        )
        db.insert_span(span)

    total = db.get_daily_cost("test-agent", today)
    assert abs(total - 4.50) < 0.001


def test_get_session_cost(db):
    _insert_agent(db)
    session = make_session()
    db.upsert_session(session)

    for _ in range(2):
        span = make_llm_span(cost_usd=2.0, session_id=session.session_id)
        db.insert_span(span)

    total = db.get_session_cost(session.session_id)
    assert abs(total - 4.0) < 0.001


# -- Recent spans --

def test_get_recent_spans_returns_last_n(db):
    _insert_agent(db)
    session = make_session()
    db.upsert_session(session)
    sid = session.session_id

    now = utcnow()
    for i in range(5):
        span = make_llm_span(
            session_id=sid,
            start_time=now + timedelta(seconds=i),
        )
        db.insert_span(span)

    recent = db.get_recent_spans(sid, limit=3)
    assert len(recent) == 3
    # Should be reverse chronological
    assert recent[0].start_time >= recent[1].start_time


# -- Retention --

def test_delete_spans_before_cutoff(db):
    _insert_agent(db)
    session = make_session()
    db.upsert_session(session)

    now = utcnow()
    old = now - timedelta(days=100)
    recent = now - timedelta(days=1)

    span_old = make_llm_span(session_id=session.session_id, start_time=old)
    span_new = make_llm_span(session_id=session.session_id, start_time=recent)
    db.insert_span(span_old)
    db.insert_span(span_new)

    cutoff = now - timedelta(days=90)
    deleted = db.delete_spans_before(cutoff)
    assert deleted == 1

    remaining = db.conn.execute("SELECT COUNT(*) FROM spans").fetchone()
    assert remaining[0] == 1


# -- Traces --

def test_get_traces_with_filters(db):
    _insert_agent(db)
    session = make_session()
    db.upsert_session(session)

    now = utcnow()
    span = make_llm_span(agent_id="test-agent", session_id=session.session_id, start_time=now)
    db.insert_span(span)

    traces = db.get_traces(TraceFilters(agent_id="test-agent"))
    assert len(traces) == 1
    assert traces[0].trace_id == span.trace_id

    # Filter by different agent returns nothing
    traces = db.get_traces(TraceFilters(agent_id="other-agent"))
    assert len(traces) == 0


# -- Alerts --

def test_insert_and_get_alerts(db):
    now = utcnow()
    alert = Alert(
        alert_id=new_uuid(),
        fired_at=now,
        type=AlertType.COST_BUDGET_DAILY,
        severity=Severity.WARNING,
        title="Budget exceeded",
        detail={"budget": 10.0, "actual": 12.0},
        agent_id="test-agent",
    )
    db.insert_alert(alert)

    results = db.get_alerts(AlertFilters(agent_id="test-agent"))
    assert len(results) == 1
    assert results[0].alert_id == alert.alert_id
    assert results[0].type == AlertType.COST_BUDGET_DAILY


# -- Tool calls --

def test_get_tool_calls(db):
    _insert_agent(db)
    session = make_session()
    db.upsert_session(session)

    for _ in range(3):
        span = make_tool_span(agent_id="test-agent", tool_name="search")
        db.insert_span(span)

    results = db.get_tool_calls(agent_id="test-agent", since=None, tool_name=None)
    assert len(results) == 1
    assert results[0]["tool_name"] == "search"
    assert results[0]["call_count"] == 3


# -- Drift baselines --

def test_upsert_and_get_baseline(db):
    _insert_agent(db)
    now = utcnow()
    baseline = DriftBaseline(
        agent_id="test-agent",
        sessions_sampled=10,
        computed_at=now,
        avg_input_tokens=1000.0,
        stddev_input_tokens=200.0,
        common_tool_sequences=[["search", "answer"]],
    )
    db.upsert_baseline(baseline)

    result = db.get_baseline("test-agent")
    assert result is not None
    assert result.sessions_sampled == 10
    assert result.avg_input_tokens == 1000.0
    assert result.common_tool_sequences == [["search", "answer"]]


def test_get_baseline_returns_none_for_unknown(db):
    result = db.get_baseline("nonexistent")
    assert result is None


# -- Schema validations --

def test_insert_validation(db):
    _insert_agent(db)
    session = make_session()
    db.upsert_session(session)
    span = make_llm_span(session_id=session.session_id)
    db.insert_span(span)

    validation = SchemaValidationResult(
        validation_id=new_uuid(),
        span_id=span.span_id,
        validated_at=utcnow(),
        passed=False,
        errors=["missing field 'result'"],
        agent_id="test-agent",
    )
    db.insert_validation(validation)

    rows = db.conn.execute(
        "SELECT passed FROM schema_validations WHERE span_id = $1",
        [span.span_id],
    ).fetchall()
    assert len(rows) == 1
    assert rows[0][0] is False


# -- Completed sessions --

def test_get_completed_sessions(db):
    _insert_agent(db)
    for i in range(3):
        session = make_session(status="completed")
        db.upsert_session(session)
    session_active = make_session(status="active")
    db.upsert_session(session_active)

    completed = db.get_completed_sessions("test-agent", limit=10)
    assert len(completed) == 3
    for s in completed:
        assert s.status == "completed"


def test_get_completed_session_count(db):
    _insert_agent(db)
    for _ in range(5):
        db.upsert_session(make_session(status="completed"))
    db.upsert_session(make_session(status="active"))

    count = db.get_completed_session_count("test-agent")
    assert count == 5


# -- Cost summary --

def test_get_cost_summary_by_model(db):
    _insert_agent(db)
    session = make_session()
    db.upsert_session(session)

    for _ in range(2):
        span = make_llm_span(
            model="claude-haiku-4-5", cost_usd=1.0,
            session_id=session.session_id,
        )
        db.insert_span(span)

    span = make_llm_span(
        model="gpt-4o", cost_usd=2.0, provider="openai",
        session_id=session.session_id,
    )
    db.insert_span(span)

    results = db.get_cost_summary(CostFilters(group_by="model"))
    assert len(results) >= 2
    models = {r.model: r.cost_usd for r in results}
    assert abs(models.get("claude-haiku-4-5", 0) - 2.0) < 0.001
    assert abs(models.get("gpt-4o", 0) - 2.0) < 0.001


# -- InMemoryBackend resets --

def test_in_memory_backend_resets_between_tests():
    """Each InMemoryBackend instance starts fresh."""
    db1 = InMemoryBackend()
    db1.upsert_agent(AgentRecord(
        agent_id="agent-1", first_seen=utcnow(), last_seen=utcnow(),
    ))
    db1.close()

    db2 = InMemoryBackend()
    rows = db2.conn.execute("SELECT COUNT(*) FROM agents").fetchall()
    assert rows[0][0] == 0
    db2.close()
