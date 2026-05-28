"""Unit tests for tj cost --compare / tj optimize --compare."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from tokenjam.core.cost import (
    compute_cost_diff,
    compute_window_totals,
    parse_compare_window,
)
from tokenjam.core.db import InMemoryBackend
from tests.factories import make_llm_span, make_session


@pytest.fixture
def db():
    backend = InMemoryBackend()
    yield backend
    backend.close()


# -- parse_compare_window --

def test_parse_compare_previous_equal_length():
    """`previous` returns the equal-length window immediately before `since`."""
    cur_since = datetime(2026, 5, 8, tzinfo=timezone.utc)
    cur_until = datetime(2026, 5, 15, tzinfo=timezone.utc)
    prev_since, prev_until = parse_compare_window("previous", cur_since, cur_until)
    assert prev_until == cur_since
    assert prev_until - prev_since == cur_until - cur_since


def test_parse_compare_last_week_same_as_previous_for_7d():
    """For a 7-day current window, last-week == previous."""
    cur_since = datetime(2026, 5, 8, tzinfo=timezone.utc)
    cur_until = datetime(2026, 5, 15, tzinfo=timezone.utc)
    a = parse_compare_window("previous", cur_since, cur_until)
    b = parse_compare_window("last-week", cur_since, cur_until)
    assert a == b


def test_parse_compare_last_month_uses_30d():
    """`last-month` references a 30-day prior window relative to `until`."""
    cur_since = datetime(2026, 5, 8, tzinfo=timezone.utc)
    cur_until = datetime(2026, 5, 15, tzinfo=timezone.utc)
    prev_since, prev_until = parse_compare_window("last-month", cur_since, cur_until)
    # last-month: prev_until = current_since; prev_since = current_until - 30d - length
    assert prev_until == cur_since
    expected_prev_since = cur_until - timedelta(days=30) - (cur_until - cur_since)
    assert prev_since == expected_prev_since


def test_parse_compare_explicit_date_range():
    cur_since = datetime(2026, 5, 8, tzinfo=timezone.utc)
    cur_until = datetime(2026, 5, 15, tzinfo=timezone.utc)
    prev_since, prev_until = parse_compare_window(
        "2026-04-01:2026-04-30", cur_since, cur_until,
    )
    assert prev_since == datetime(2026, 4, 1, tzinfo=timezone.utc)
    assert prev_until == datetime(2026, 4, 30, tzinfo=timezone.utc)


def test_parse_compare_explicit_range_end_before_start_rejected():
    cur_since = datetime(2026, 5, 8, tzinfo=timezone.utc)
    cur_until = datetime(2026, 5, 15, tzinfo=timezone.utc)
    with pytest.raises(ValueError, match="end must be after start"):
        parse_compare_window("2026-05-30:2026-05-01", cur_since, cur_until)


def test_parse_compare_unknown_keyword_rejected():
    cur_since = datetime(2026, 5, 8, tzinfo=timezone.utc)
    cur_until = datetime(2026, 5, 15, tzinfo=timezone.utc)
    with pytest.raises(ValueError, match="Unknown --compare"):
        parse_compare_window("yesterday", cur_since, cur_until)


# -- compute_window_totals --

def test_compute_window_totals_aggregates(db):
    """compute_window_totals sums tokens and cost across spans in scope."""
    base = datetime(2026, 5, 10, tzinfo=timezone.utc)
    for i in range(3):
        sess = make_session(session_id=f"s{i}", plan_tier="api")
        db.upsert_session(sess)
        span = make_llm_span(
            session_id=f"s{i}",
            input_tokens=1000, output_tokens=200, cost_usd=0.01,
            start_time=base + timedelta(minutes=i),
        )
        db.insert_span(span)

    totals = compute_window_totals(
        db.conn,
        since=base - timedelta(hours=1),
        until=base + timedelta(hours=1),
    )
    assert totals.sessions == 3
    assert totals.input_tokens == 3000
    assert totals.output_tokens == 600
    assert totals.total_cost_usd == pytest.approx(0.03)


# -- compute_cost_diff --

def test_compute_cost_diff_reports_increase(db):
    """Spend in current window > previous window produces a positive cost delta."""
    now = datetime(2026, 5, 15, tzinfo=timezone.utc)

    # Previous week: $0.05 total
    for i in range(5):
        sess = make_session(session_id=f"prev-{i}", plan_tier="api")
        db.upsert_session(sess)
        db.insert_span(make_llm_span(
            session_id=f"prev-{i}",
            input_tokens=500, output_tokens=100, cost_usd=0.01,
            start_time=now - timedelta(days=10) + timedelta(hours=i),
        ))

    # Current week: $0.20 total — 4x increase
    for i in range(5):
        sess = make_session(session_id=f"cur-{i}", plan_tier="api")
        db.upsert_session(sess)
        db.insert_span(make_llm_span(
            session_id=f"cur-{i}",
            input_tokens=2000, output_tokens=400, cost_usd=0.04,
            start_time=now - timedelta(days=3) + timedelta(hours=i),
        ))

    diff = compute_cost_diff(
        db,
        current_since=now - timedelta(days=7),
        current_until=now,
        compare="previous",
    )
    assert diff.previous.total_cost_usd == pytest.approx(0.05)
    assert diff.current.total_cost_usd == pytest.approx(0.20)
    assert diff.cost_delta_usd == pytest.approx(0.15)
    assert diff.cost_delta_pct == pytest.approx(300.0)
    assert diff.tokens_delta == 12000 - 3000


def test_compute_cost_diff_empty_previous(db):
    """When the prior window has no data, percentages return None (no zero division)."""
    now = datetime(2026, 5, 15, tzinfo=timezone.utc)
    sess = make_session(session_id="cur", plan_tier="api")
    db.upsert_session(sess)
    db.insert_span(make_llm_span(
        session_id="cur",
        input_tokens=1000, output_tokens=200, cost_usd=0.01,
        start_time=now - timedelta(days=3),
    ))

    diff = compute_cost_diff(
        db,
        current_since=now - timedelta(days=7),
        current_until=now,
        compare="previous",
    )
    assert diff.previous.total_cost_usd == 0.0
    assert diff.cost_delta_pct is None
    assert diff.tokens_delta_pct is None
