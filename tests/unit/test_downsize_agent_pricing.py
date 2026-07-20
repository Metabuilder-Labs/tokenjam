"""Per-agent price arithmetic on the downsize card.

The numbers here are the card's headline, so they are asserted against hand
calculations off the real pricing table rather than against themselves.
"""
from __future__ import annotations

from datetime import timedelta

import pytest

from tokenjam.core.db import InMemoryBackend
from tokenjam.core.optimize.analyzers.downsize_agents import (
    build_agent_price_rows,
    price_tokens,
    thinking_share,
)
from tokenjam.core.optimize.analyzers.model_downgrade import analyze_model_downgrade
from tokenjam.core.pricing import get_rates
from tokenjam.utils.time_parse import utcnow
from tests.factories import make_llm_span

WINDOW_DAYS = 30.0


@pytest.fixture
def db():
    backend = InMemoryBackend()
    yield backend
    backend.close()


def _window():
    return utcnow() - timedelta(days=WINDOW_DAYS), utcnow() + timedelta(hours=1)


def _cheap_shaped_session(db, *, agent_id, session_id, model="claude-opus-4-8",
                          cache_tokens=0, cache_write_tokens=0, attributes=None):
    """One session matching the downsize shape heuristic (small input, small
    output, no tool calls)."""
    db.insert_span(make_llm_span(
        agent_id=agent_id, model=model, provider="anthropic",
        input_tokens=1_000, output_tokens=200,
        cache_tokens=cache_tokens, cache_write_tokens=cache_write_tokens,
        cost_usd=0.03, session_id=session_id,
        start_time=utcnow() - timedelta(days=2),
        extra_attributes=attributes,
    ))


# --------------------------------------------------------------------------- #
# Detection: positive, negative, threshold edge
# --------------------------------------------------------------------------- #

def test_per_agent_rows_priced_from_the_real_rate_table(db):
    # Arrange
    _cheap_shaped_session(db, agent_id="svc-a", session_id="s1",
                          cache_tokens=5_000, cache_write_tokens=2_000)
    since, until = _window()

    # Act
    finding = analyze_model_downgrade(db.conn, since, until, None, WINDOW_DAYS)

    # Assert
    assert finding is not None
    assert len(finding.per_agent) == 1
    row = finding.per_agent[0]
    assert (row.agent_id, row.model, row.alt_model) == (
        "svc-a", "claude-opus-4-8", "claude-haiku-4-5",
    )
    current = get_rates("anthropic", "claude-opus-4-8")
    alt = get_rates("anthropic", "claude-haiku-4-5")
    expected_current = (
        1_000 / 1e6 * current.input_per_mtok
        + 200 / 1e6 * current.output_per_mtok
        + 5_000 / 1e6 * current.cache_read_per_mtok
        + 2_000 / 1e6 * current.cache_write_per_mtok
    )
    expected_alt = (
        1_000 / 1e6 * alt.input_per_mtok
        + 200 / 1e6 * alt.output_per_mtok
        + 5_000 / 1e6 * alt.cache_read_per_mtok
        + 2_000 / 1e6 * alt.cache_write_per_mtok
    )
    assert row.current_cost_usd == pytest.approx(expected_current, abs=1e-6)
    assert row.alternative_cost_usd == pytest.approx(expected_alt, abs=1e-6)
    assert row.delta_usd == pytest.approx(expected_current - expected_alt, abs=1e-6)


def test_no_candidate_sessions_yields_no_rows(db):
    # A session too large for the shape heuristic is not a candidate, so no card
    # and no per-agent arithmetic.
    db.insert_span(make_llm_span(
        agent_id="svc-a", model="claude-opus-4-8", provider="anthropic",
        input_tokens=50_000, output_tokens=9_000, cost_usd=2.0,
        session_id="big", start_time=utcnow() - timedelta(days=1),
    ))
    since, until = _window()
    assert analyze_model_downgrade(db.conn, since, until, None, WINDOW_DAYS) is None


def test_rows_split_per_agent_and_sort_by_delta(db):
    _cheap_shaped_session(db, agent_id="svc-a", session_id="s1")
    _cheap_shaped_session(db, agent_id="svc-a", session_id="s2")
    _cheap_shaped_session(db, agent_id="svc-b", session_id="s3")
    since, until = _window()

    finding = analyze_model_downgrade(db.conn, since, until, None, WINDOW_DAYS)

    rows = {r.agent_id: r for r in finding.per_agent}
    assert set(rows) == {"svc-a", "svc-b"}
    assert rows["svc-a"].sessions == 2
    assert rows["svc-b"].sessions == 1
    assert finding.per_agent[0].agent_id == "svc-a"   # largest delta first


def test_group_without_pricing_for_both_sides_is_dropped():
    # Threshold edge: an unpriced model must never be priced at a default rate,
    # so the row disappears rather than carrying an invented number.
    rows = build_agent_price_rows([{
        "session_id": "s1", "agent_id": "svc-a", "provider": "anthropic",
        "model": "claude-opus-4-8", "alt_model": "totally-made-up-model",
        "input_tokens": 1_000, "output_tokens": 100,
        "cache_tokens": 0, "cache_write_tokens": 0,
    }], WINDOW_DAYS)
    assert rows == []


# --------------------------------------------------------------------------- #
# Token accounting: all four billed types, always
# --------------------------------------------------------------------------- #

def test_pricing_counts_cache_reads_and_cache_writes(db):
    tokens = {
        "input_tokens": 1_000, "output_tokens": 200,
        "cache_tokens": 100_000, "cache_write_tokens": 50_000,
    }
    with_cache = price_tokens("anthropic", "claude-opus-4-8", **tokens)
    without_cache = price_tokens(
        "anthropic", "claude-opus-4-8",
        input_tokens=1_000, output_tokens=200,
        cache_tokens=0, cache_write_tokens=0,
    )
    assert with_cache > without_cache

    # And the row's own token total carries all four types.
    _cheap_shaped_session(db, agent_id="svc-a", session_id="s1",
                          cache_tokens=100_000, cache_write_tokens=50_000)
    since, until = _window()
    row = analyze_model_downgrade(db.conn, since, until, None, WINDOW_DAYS).per_agent[0]
    assert row.total_tokens == 1_000 + 200 + 100_000 + 50_000


def test_projection_scales_the_window_delta_to_thirty_days(db):
    _cheap_shaped_session(db, agent_id="svc-a", session_id="s1")
    since, until = _window()
    row = analyze_model_downgrade(db.conn, since, until, None, 10.0).per_agent[0]
    assert row.projected_30d_delta_usd == pytest.approx(round(row.delta_usd * 3.0, 2))


# --------------------------------------------------------------------------- #
# Thinking share: informational, and omitted when unreported
# --------------------------------------------------------------------------- #

def test_thinking_share_is_none_when_the_runtime_reports_nothing(db):
    _cheap_shaped_session(db, agent_id="svc-a", session_id="s1")
    since, until = _window()
    row = analyze_model_downgrade(db.conn, since, until, None, WINDOW_DAYS).per_agent[0]
    assert row.thinking_tokens is None
    assert row.thinking_share_of_output is None


def test_thinking_share_read_from_span_attributes(db):
    _cheap_shaped_session(
        db, agent_id="svc-a", session_id="s1",
        attributes={"reasoning_token_count": 50},
    )
    since, until = _window()
    row = analyze_model_downgrade(db.conn, since, until, None, WINDOW_DAYS).per_agent[0]
    assert row.thinking_tokens == 50
    assert row.thinking_share_of_output == pytest.approx(0.25)   # 50 of 200 output


def test_thinking_share_helper_guards_zero_output():
    assert thinking_share(None, 100) is None
    assert thinking_share(10, 0) is None
    assert thinking_share(25, 100) == 0.25
