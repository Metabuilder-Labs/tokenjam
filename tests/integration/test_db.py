"""Integration tests for the database layer."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from tokenjam.core.db import MIGRATIONS, InMemoryBackend, run_migrations
from tokenjam.core.models import (
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
from tokenjam.utils.ids import new_uuid
from tokenjam.utils.time_parse import utcnow
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
    expected_versions = {v for v, _ in MIGRATIONS}
    assert len(rows) == len(MIGRATIONS)
    assert {r[0] for r in rows} == expected_versions
    backend.close()


def test_migrations_are_idempotent():
    backend = InMemoryBackend()
    # Running migrations again should not raise
    run_migrations(backend.conn)
    rows = backend.conn.execute("SELECT version FROM schema_migrations").fetchall()
    assert len(rows) == len(MIGRATIONS)
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
    spans_deleted, sessions_deleted = db.delete_spans_before(cutoff)
    assert spans_deleted == 1

    remaining = db.conn.execute("SELECT COUNT(*) FROM spans").fetchone()
    assert remaining[0] == 1
    # The session straddles the cutoff and still has a live span, so it stays:
    # a session is orphaned by the delete only once it has NO spans left.
    assert sessions_deleted == 0
    assert db.conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 1


def test_delete_spans_before_cutoff_takes_the_sessions_it_orphans(db):
    """Deleting only spans left the parent sessions asserting a day had data.

    `core/data_span` unions `sessions.started_at` into the day set it measures
    the available span from, so every orphan went on claiming a day carried data
    after the data for that day was destroyed — the deletion skewed the measure
    of what survived it.
    """
    _insert_agent(db)
    now = utcnow()
    old = now - timedelta(days=100)

    aged_out = make_session(started_at=old, ended_at=old + timedelta(minutes=1))
    db.upsert_session(aged_out)
    db.insert_span(make_llm_span(session_id=aged_out.session_id, start_time=old))

    live = make_session()
    db.upsert_session(live)
    db.insert_span(make_llm_span(
        session_id=live.session_id, start_time=now - timedelta(days=1),
    ))

    spans_deleted, sessions_deleted = db.delete_spans_before(now - timedelta(days=90))
    assert (spans_deleted, sessions_deleted) == (1, 1)

    surviving = [
        r[0] for r in db.conn.execute("SELECT session_id FROM sessions").fetchall()
    ]
    assert surviving == [live.session_id]


def test_delete_spans_before_cutoff_takes_an_aged_out_session_with_no_spans(db):
    """A pre-cutoff session with no spans is aged-out history like any other.

    Left behind it would go on asserting, to `core/data_span`, that a day
    beyond the retention horizon carried data.
    """
    _insert_agent(db)
    now = utcnow()
    empty = make_session(
        started_at=now - timedelta(days=100),
        ended_at=now - timedelta(days=100) + timedelta(minutes=1),
    )
    db.upsert_session(empty)

    spans_deleted, sessions_deleted = db.delete_spans_before(now - timedelta(days=90))
    assert (spans_deleted, sessions_deleted) == (0, 1)
    assert db.conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 0


def test_delete_spans_before_cutoff_keeps_a_live_session_with_no_spans_yet(db):
    """An open session that has not written a span is not aged out."""
    _insert_agent(db)
    now = utcnow()
    db.upsert_session(make_session(started_at=now, ended_at=None))

    spans_deleted, sessions_deleted = db.delete_spans_before(now - timedelta(days=90))
    assert (spans_deleted, sessions_deleted) == (0, 0)
    assert db.conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 1


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


# -- Trace cost ranking --

def test_get_traces_sort_cost_orders_highest_first(db):
    _insert_agent(db)
    now = utcnow()
    cheap = make_llm_span(agent_id="test-agent", cost_usd=0.01, trace_id="cheap", start_time=now)
    pricey = make_llm_span(
        agent_id="test-agent", cost_usd=5.0, trace_id="pricey",
        start_time=now - timedelta(minutes=1),  # older, so "recent" sort would put it 2nd
    )
    db.insert_span(cheap)
    db.insert_span(pricey)

    traces = db.get_traces(TraceFilters(sort="cost"))
    assert [t.trace_id for t in traces] == ["pricey", "cheap"]

    # Default ("recent") sort is unaffected — reverse-chronological.
    traces = db.get_traces(TraceFilters())
    assert [t.trace_id for t in traces] == ["cheap", "pricey"]


def test_get_traces_min_cost_usd_filters_and_matches_count(db):
    _insert_agent(db)
    now = utcnow()
    for i, cost in enumerate([0.01, 1.0, 10.0]):
        db.insert_span(make_llm_span(
            agent_id="test-agent", cost_usd=cost, trace_id=f"t{i}", start_time=now,
        ))

    traces = db.get_traces(TraceFilters(min_cost_usd=1.0))
    assert {t.trace_id for t in traces} == {"t1", "t2"}
    assert db.count_traces(TraceFilters(min_cost_usd=1.0)) == 2
    assert db.count_traces(TraceFilters()) == 3


def test_get_traces_flags_statistical_cost_outlier(db):
    """Tukey's-fence rule: Q3 + 1.5*IQR over priced traces in the window."""
    _insert_agent(db)
    now = utcnow()
    # 7 traces at $1 + 1 trace at $50 -> the $50 trace is a clear outlier once
    # there are enough priced traces to trust the quartiles.
    for i in range(7):
        db.insert_span(make_llm_span(
            agent_id="test-agent", cost_usd=1.0, trace_id=f"cheap-{i}", start_time=now,
        ))
    db.insert_span(make_llm_span(
        agent_id="test-agent", cost_usd=50.0, trace_id="spike", start_time=now,
    ))

    traces = {t.trace_id: t for t in db.get_traces(TraceFilters())}
    assert traces["spike"].is_outlier is True
    assert all(not t.is_outlier for tid, t in traces.items() if tid != "spike")

    stats = db.get_trace_cost_stats(TraceFilters())
    assert stats.sample_size == 8
    assert stats.threshold_usd is not None
    assert stats.q3_usd is not None and stats.q1_usd is not None


def test_get_traces_outlier_requires_minimum_sample(db):
    """Below MIN_OUTLIER_SAMPLE priced traces, nothing is flagged — a handful
    of traces can't support a reliable quartile-based rule."""
    _insert_agent(db)
    now = utcnow()
    db.insert_span(make_llm_span(agent_id="test-agent", cost_usd=1.0, trace_id="a", start_time=now))
    db.insert_span(make_llm_span(agent_id="test-agent", cost_usd=100.0, trace_id="b", start_time=now))

    traces = db.get_traces(TraceFilters())
    assert all(not t.is_outlier for t in traces)

    stats = db.get_trace_cost_stats(TraceFilters())
    assert stats.sample_size == 2
    assert stats.threshold_usd is None


def test_get_traces_zero_cost_trace_never_flagged_outlier(db):
    """A $0/unpriced trace is not an 'outlier' even in a window with a real
    spike — is_outlier only ever fires on a trace with positive cost."""
    _insert_agent(db)
    now = utcnow()
    for i in range(7):
        db.insert_span(make_llm_span(
            agent_id="test-agent", cost_usd=1.0, trace_id=f"cheap-{i}", start_time=now,
        ))
    db.insert_span(make_llm_span(agent_id="test-agent", cost_usd=50.0, trace_id="spike", start_time=now))
    db.insert_span(make_llm_span(agent_id="test-agent", cost_usd=0.0, trace_id="free", start_time=now))

    traces = {t.trace_id: t for t in db.get_traces(TraceFilters())}
    assert traces["free"].is_outlier is False


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


def test_get_completed_sessions_orders_by_last_activity(db):
    """A long session must rank ahead of a short fragment that started later.

    Regression: the status tile showed a 40s fragment instead of the real
    multi-hour session because ordering was by started_at, not last activity.
    """
    _insert_agent(db)
    base = utcnow()
    # Long session: started first, stayed active for ~4.5h.
    long_session = make_session(
        status="completed",
        started_at=base - timedelta(hours=4, minutes=27),
        ended_at=base,
    )
    # Short fragment: started 3 min AFTER the long one, ended 40s later.
    short_fragment = make_session(
        status="completed",
        started_at=base - timedelta(hours=4, minutes=24),
        ended_at=base - timedelta(hours=4, minutes=23, seconds=20),
    )
    db.upsert_session(long_session)
    db.upsert_session(short_fragment)

    latest = db.get_completed_sessions("test-agent", limit=1)
    assert latest[0].session_id == long_session.session_id
    assert latest[0].duration_seconds > 3600  # the multi-hour one, not the 40s blip


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


def test_get_cost_summary_by_tool(db):
    """`--group-by tool` must attribute call counts to each real tool name,
    not collapse everything into one 'None' bucket.

    Tool-call spans (``gen_ai.tool.call``) never carry a ``model`` — that
    attribute lives on the LLM completion span, a separate row (see
    otel/otlp_parsing.py). ``get_cost_summary`` used to hardcode
    ``model IS NOT NULL`` for every grouping, which silently dropped every
    tool span and left `tool` grouping with nothing but LLM spans (whose
    tool_name is NULL) — one bogus `TOOL: None` row.
    """
    _insert_agent(db)
    session = make_session()
    db.upsert_session(session)

    for _ in range(3):
        db.insert_span(make_tool_span(tool_name="Read", session_id=session.session_id))
    for _ in range(2):
        db.insert_span(make_tool_span(tool_name="Bash", session_id=session.session_id))
    # An LLM span alongside the tool spans must not leak into the tool
    # grouping (and confirms the bug's "TOOL: None" bucket is gone).
    db.insert_span(make_llm_span(model="claude-haiku-4-5", cost_usd=1.0,
                                  session_id=session.session_id))

    results = db.get_cost_summary(CostFilters(group_by="tool"))
    groups = {r.group: r.call_count for r in results}
    assert groups == {"Read": 3, "Bash": 2}
    assert "None" not in groups


# -- SDK cost-attribution dimensions (#SDK dashboard shape) --

def test_get_cost_summary_by_tenant(db):
    """`group_by="tenant"` sums spend per tenant_id, biggest spender first, and
    excludes spans that never set tenant_id (degrade-honestly contract — an
    unattributed span must not fold into a misleading bucket)."""
    _insert_agent(db)
    session = make_session()
    db.upsert_session(session)

    db.insert_span(make_llm_span(
        model="claude-haiku-4-5", cost_usd=5.0, tenant_id="acme-corp",
        session_id=session.session_id,
    ))
    db.insert_span(make_llm_span(
        model="claude-haiku-4-5", cost_usd=1.0, tenant_id="small-co",
        session_id=session.session_id,
    ))
    # No tenant_id set — must be excluded from the grouping, not folded into
    # a "(none)" bucket.
    db.insert_span(make_llm_span(
        model="claude-haiku-4-5", cost_usd=100.0,
        session_id=session.session_id,
    ))

    results = db.get_cost_summary(CostFilters(group_by="tenant"))
    groups = {r.group: r.cost_usd for r in results}
    assert set(groups) == {"acme-corp", "small-co"}
    assert abs(groups["acme-corp"] - 5.0) < 0.001
    assert abs(groups["small-co"] - 1.0) < 0.001
    # Biggest spender first.
    assert results[0].group == "acme-corp"


def test_get_cost_summary_by_feature_environment_prompt_version(db):
    _insert_agent(db)
    session = make_session()
    db.upsert_session(session)

    db.insert_span(make_llm_span(
        model="claude-haiku-4-5", cost_usd=2.0,
        feature="support-triage", environment="production",
        prompt_template_version="3",
        session_id=session.session_id,
    ))
    db.insert_span(make_llm_span(
        model="claude-haiku-4-5", cost_usd=1.0,
        feature="onboarding", environment="staging",
        prompt_template_version="1",
        session_id=session.session_id,
    ))

    by_feature = {r.group: r.cost_usd for r in db.get_cost_summary(CostFilters(group_by="feature"))}
    assert abs(by_feature["support-triage"] - 2.0) < 0.001
    assert abs(by_feature["onboarding"] - 1.0) < 0.001

    by_env = {r.group: r.cost_usd for r in db.get_cost_summary(CostFilters(group_by="environment"))}
    assert abs(by_env["production"] - 2.0) < 0.001
    assert abs(by_env["staging"] - 1.0) < 0.001

    by_version = {r.group: r.cost_usd for r in db.get_cost_summary(CostFilters(group_by="prompt_version"))}
    assert abs(by_version["3"] - 2.0) < 0.001
    assert abs(by_version["1"] - 1.0) < 0.001


def test_get_cost_summary_tenant_equality_filter(db):
    """The tenant_id/feature/environment/prompt_version equality filters scope
    the summary independently of group_by (e.g. a per-model breakdown for one
    tenant's spend)."""
    _insert_agent(db)
    session = make_session()
    db.upsert_session(session)

    db.insert_span(make_llm_span(
        model="claude-haiku-4-5", cost_usd=3.0, tenant_id="acme-corp",
        session_id=session.session_id,
    ))
    db.insert_span(make_llm_span(
        model="claude-haiku-4-5", cost_usd=7.0, tenant_id="other-co",
        session_id=session.session_id,
    ))

    results = db.get_cost_summary(CostFilters(group_by="model", tenant_id="acme-corp"))
    assert len(results) == 1
    assert abs(results[0].cost_usd - 3.0) < 0.001
    assert results[0].model == "claude-haiku-4-5"


@pytest.mark.parametrize("db_timezone", ["UTC", "Asia/Kolkata", "America/Los_Angeles"])
def test_get_cost_summary_by_day_buckets_on_the_utc_date(db, db_timezone):
    """``--group-by day`` must bucket by the UTC date, not the session's local
    timezone.

    A bare ``CAST(start_time AS DATE)`` resolves a TIMESTAMPTZ through the
    connection's local timezone before truncating, so a span logged late in
    the UTC day gets stamped with tomorrow's date on any machine running
    ahead of UTC (e.g. Asia/Kolkata, +05:30) — the CLI's per-day total would
    then disagree with anything computed on a UTC basis.
    """
    db.conn.execute(f"SET TimeZone='{db_timezone}'")
    _insert_agent(db)
    session = make_session()
    db.upsert_session(session)

    late_utc = datetime(2026, 3, 14, 23, 30, tzinfo=timezone.utc)
    db.insert_span(make_llm_span(
        model="claude-haiku-4-5", cost_usd=5.0,
        session_id=session.session_id, start_time=late_utc,
    ))

    results = db.get_cost_summary(CostFilters(group_by="day"))
    assert len(results) == 1
    assert results[0].group == "2026-03-14"


def test_get_cost_summary_returns_empty_when_dimension_never_set(db):
    """An attribution dim with zero data in the window returns an EMPTY list —
    never a misleading '(none)' bucket — so the API/UI can render an honest
    empty state distinct from 'zero spend'."""
    _insert_agent(db)
    session = make_session()
    db.upsert_session(session)
    db.insert_span(make_llm_span(
        model="claude-haiku-4-5", cost_usd=9.0, session_id=session.session_id,
    ))

    assert db.get_cost_summary(CostFilters(group_by="tenant")) == []


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
