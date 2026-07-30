"""Unit tests for DuckDBBackend.get_daily_cost_for_agents — the summed-daily-
cost-over-a-SET-of-agent_ids query that a coding-tool GROUP budget cap
(e.g. every claude-code-<project> variant) is checked against, generalizing
the existing single-agent get_daily_cost.
"""
from __future__ import annotations

from datetime import timedelta

import pytest

from tokenjam.core.db import InMemoryBackend
from tokenjam.utils.time_parse import utcnow
from tests.factories import make_llm_span


@pytest.fixture
def db():
    backend = InMemoryBackend()
    yield backend
    backend.close()


def _seed(db, agent_id, cost_usd, when=None):
    span = make_llm_span(agent_id=agent_id, cost_usd=cost_usd, start_time=when or utcnow())
    db.insert_span(span)


def test_sums_cost_across_multiple_agent_ids_same_day(db):
    today = utcnow().date()
    _seed(db, "claude-code-proj-a", 3.0)
    _seed(db, "claude-code-proj-b", 4.5)
    _seed(db, "claude-code-proj-c", 0.5)
    total = db.get_daily_cost_for_agents(
        ["claude-code-proj-a", "claude-code-proj-b", "claude-code-proj-c"], today,
    )
    assert total == pytest.approx(8.0)


def test_group_cap_trips_on_summed_spend_where_no_single_member_would(db):
    """The load-bearing case: 20 agents at a small individual spend each,
    none of which alone crosses a per-agent-sized cap, must still sum past a
    group cap sized for the tool as a whole."""
    today = utcnow().date()
    member_ids = [f"claude-code-proj-{i}" for i in range(20)]
    for aid in member_ids:
        _seed(db, aid, 3.0)  # $3 each; no individual agent exceeds e.g. a $10 cap
    total = db.get_daily_cost_for_agents(member_ids, today)
    assert total == pytest.approx(60.0)
    group_cap = 50.0
    assert total > group_cap  # trips the group cap
    assert all(db.get_daily_cost(aid, today) < group_cap for aid in member_ids)  # no individual would


def test_only_sums_the_named_agent_ids_not_others(db):
    today = utcnow().date()
    _seed(db, "claude-code-proj-a", 3.0)
    _seed(db, "some-other-agent", 999.0)
    total = db.get_daily_cost_for_agents(["claude-code-proj-a"], today)
    assert total == pytest.approx(3.0)


def test_excludes_spend_from_a_different_day(db):
    today = utcnow().date()
    yesterday = utcnow() - timedelta(days=1)
    _seed(db, "claude-code-proj-a", 3.0, when=yesterday)
    _seed(db, "claude-code-proj-a", 2.0, when=utcnow())
    total = db.get_daily_cost_for_agents(["claude-code-proj-a"], today)
    assert total == pytest.approx(2.0)


def test_empty_agent_id_list_returns_zero(db):
    assert db.get_daily_cost_for_agents([], utcnow().date()) == 0.0
