"""Unit tests for small db.py query helpers."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from tokenjam.core.db import InMemoryBackend, session_active_seconds
from tests.factories import make_llm_span, make_session


@pytest.fixture
def db():
    backend = InMemoryBackend()
    yield backend
    backend.close()


def test_active_seconds_sums_span_durations(db):
    sess = make_session(agent_id="a1", session_id="s1", status="completed")
    db.upsert_session(sess)
    for ms in (1000.0, 2500.0, 500.0):
        sp = make_llm_span(agent_id="a1", duration_ms=ms)
        sp.session_id = "s1"
        db.insert_span(sp)
    # 4000 ms total → 4.0 s of active (compute) time.
    assert session_active_seconds(db.conn, "s1") == pytest.approx(4.0)


def test_active_seconds_none_when_no_spans(db):
    sess = make_session(agent_id="a1", session_id="s-empty", status="completed")
    db.upsert_session(sess)
    assert session_active_seconds(db.conn, "s-empty") is None


def test_active_seconds_none_for_unknown_session(db):
    assert session_active_seconds(db.conn, "does-not-exist") is None


# ---------------------------------------------------------------------------
# Issue #309: StorageBackend methods that used to be direct db.conn access in
# CostEngine / cmd_status / cost-compare. Exercised here against InMemoryBackend.
# ---------------------------------------------------------------------------

def test_update_span_cost_and_increment_session_cost(db):
    sess = make_session(session_id="s1", total_cost_usd=None)
    db.upsert_session(sess)
    span = make_llm_span(session_id="s1")
    db.insert_span(span)

    db.update_span_cost(span.span_id, 0.5)
    db.increment_session_cost("s1", 0.5)
    db.increment_session_cost("s1", 0.25)

    stored = [s for s in db.get_trace_spans(span.trace_id) if s.span_id == span.span_id][0]
    assert stored.cost_usd == pytest.approx(0.5)
    assert db.get_session("s1").total_cost_usd == pytest.approx(0.75)


def test_get_distinct_agent_ids_sorted_and_deduped(db):
    for aid in ("zeta", "alpha", "alpha", "mid"):
        db.upsert_session(make_session(agent_id=aid))
    assert db.get_distinct_agent_ids() == ["alpha", "mid", "zeta"]


def test_get_active_session_prefers_active_over_completed(db):
    db.upsert_session(make_session(agent_id="a1", session_id="done", status="completed"))
    db.upsert_session(make_session(agent_id="a1", session_id="live", status="active"))

    active = db.get_active_session("a1")
    assert active is not None
    assert active.session_id == "live"
    # No active session for an unknown agent.
    assert db.get_active_session("nobody") is None


def test_get_session_active_seconds_matches_helper(db):
    db.upsert_session(make_session(agent_id="a1", session_id="s1"))
    for ms in (1000.0, 1000.0):
        sp = make_llm_span(agent_id="a1", duration_ms=ms)
        sp.session_id = "s1"
        db.insert_span(sp)
    assert db.get_session_active_seconds("s1") == pytest.approx(2.0)
    assert db.get_session_active_seconds("missing") is None


def test_count_unknown_plan_tier_sessions(db):
    db.upsert_session(make_session(session_id="api1", plan_tier="api"))
    db.upsert_session(make_session(session_id="unk1", plan_tier="unknown"))
    db.upsert_session(make_session(session_id="unk2", plan_tier="unknown"))
    assert db.count_unknown_plan_tier_sessions() == 2


def test_get_window_cost_totals_aggregates_and_scopes(db):
    base = datetime(2026, 5, 10, tzinfo=timezone.utc)
    for i in range(2):
        db.upsert_session(make_session(session_id=f"s{i}", agent_id="a1"))
        db.insert_span(make_llm_span(
            session_id=f"s{i}", agent_id="a1",
            input_tokens=1000, output_tokens=200, cache_tokens=50, cost_usd=0.01,
            start_time=base + timedelta(minutes=i),
        ))
    # An out-of-window span must be excluded.
    db.upsert_session(make_session(session_id="old", agent_id="a1"))
    db.insert_span(make_llm_span(
        session_id="old", agent_id="a1", input_tokens=9999, cost_usd=9.0,
        start_time=base - timedelta(days=5),
    ))

    sessions, in_tok, out_tok, cache_tok, cost = db.get_window_cost_totals(
        base - timedelta(hours=1), base + timedelta(hours=1),
    )
    assert (sessions, in_tok, out_tok, cache_tok) == (2, 2000, 400, 100)
    assert cost == pytest.approx(0.02)


def test_get_cost_delta_by_group_ranks_shifts(db):
    now = datetime(2026, 5, 15, tzinfo=timezone.utc)
    cur_since, cur_until = now - timedelta(days=3), now
    prev_since, prev_until = now - timedelta(days=10), now - timedelta(days=7)

    db.upsert_session(make_session(session_id="p"))
    db.insert_span(make_llm_span(
        session_id="p", agent_id="a1", cost_usd=0.01,
        start_time=prev_since + timedelta(hours=1),
    ))
    db.upsert_session(make_session(session_id="c"))
    db.insert_span(make_llm_span(
        session_id="c", agent_id="a1", cost_usd=0.05,
        start_time=cur_since + timedelta(hours=1),
    ))

    rows = db.get_cost_delta_by_group(
        "agent_id", cur_since, cur_until, prev_since, prev_until, top_n=5,
    )
    assert rows[0]["group"] == "a1"
    assert rows[0]["current_cost"] == pytest.approx(0.05)
    assert rows[0]["previous_cost"] == pytest.approx(0.01)
    assert rows[0]["delta"] == pytest.approx(0.04)


def test_get_cost_delta_by_group_rejects_unknown_column(db):
    with pytest.raises(ValueError):
        db.get_cost_delta_by_group(
            "drop_table", datetime.now(timezone.utc), datetime.now(timezone.utc),
            datetime.now(timezone.utc), datetime.now(timezone.utc), top_n=5,
        )
