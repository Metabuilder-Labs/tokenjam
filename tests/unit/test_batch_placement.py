"""Batch API placement detection (core.optimize.analyzers.batch_placement).

Both conditions are load-bearing and tested independently: a cadence-regular
workload with a person in the loop is not a candidate, and an unattended
workload with scattered start times is not one either.
"""
from __future__ import annotations

from datetime import timedelta

import pytest

from tokenjam.core.db import InMemoryBackend
from tokenjam.core.optimize.analyzers.batch_placement import (
    BATCH_DISCOUNT,
    MAX_START_GAP_CV,
    analyze_batch_placement,
    gap_coefficient_of_variation,
)
from tokenjam.utils.time_parse import utcnow
from tests.factories import make_invoke_agent_span, make_llm_span

WINDOW_DAYS = 30.0
BASE = utcnow() - timedelta(days=10)


@pytest.fixture
def db():
    backend = InMemoryBackend()
    yield backend
    backend.close()


def _window():
    return utcnow() - timedelta(days=WINDOW_DAYS), utcnow() + timedelta(hours=1)


def _cron_sessions(db, *, agent_id="nightly", count=6, gap_hours=6.0,
                   jitter_hours=0.0, cost_usd=1.0):
    """``count`` sessions started every ``gap_hours``, each one model call."""
    starts = []
    for i in range(count):
        drift = jitter_hours * (i % 2)
        start = BASE + timedelta(hours=gap_hours * i + drift)
        starts.append(start)
        db.insert_span(make_llm_span(
            agent_id=agent_id, model="claude-sonnet-4-6", provider="anthropic",
            input_tokens=2_000, output_tokens=500, cache_tokens=100,
            cache_write_tokens=50, cost_usd=cost_usd,
            session_id=f"{agent_id}-{i}", start_time=start,
        ))
    return starts


# --------------------------------------------------------------------------- #
# Positive
# --------------------------------------------------------------------------- #

def test_cadence_regular_unattended_workload_is_a_candidate(db):
    _cron_sessions(db, count=6, cost_usd=1.0)
    since, until = _window()

    finding = analyze_batch_placement(db.conn, since, until, None, 12.0)

    assert finding is not None
    assert [c.agent_id for c in finding.candidates] == ["nightly"]
    candidate = finding.candidates[0]
    assert candidate.sessions == 6
    assert candidate.gap_cv == 0.0
    assert candidate.cost_usd == pytest.approx(6.0)
    # The Batch API is a flat half of standard prices.
    assert candidate.estimated_batch_saving_usd == pytest.approx(6.0 * BATCH_DISCOUNT)
    assert finding.estimated_recoverable_usd == pytest.approx(3.0)
    assert finding.percent_of_window_cost == pytest.approx(50.0)
    # All four billed token types travel with the candidate.
    assert candidate.tokens == 6 * (2_000 + 500 + 100 + 50)


def test_opening_human_turn_does_not_disqualify(db):
    # The prompt that starts an unattended run arrives before the first model
    # call and is not a person sitting in the loop.
    starts = _cron_sessions(db, count=6)
    for i, start in enumerate(starts):
        db.insert_span(make_invoke_agent_span(
            agent_id="nightly", session_id=f"nightly-{i}",
            start_time=start - timedelta(seconds=5),
        ))
    since, until = _window()
    finding = analyze_batch_placement(db.conn, since, until, None, 12.0)
    assert finding is not None
    assert finding.candidates[0].sessions == 6


# --------------------------------------------------------------------------- #
# Negative
# --------------------------------------------------------------------------- #

def test_mid_run_human_turn_disqualifies_the_group(db):
    starts = _cron_sessions(db, count=6)
    db.insert_span(make_invoke_agent_span(
        agent_id="nightly", session_id="nightly-0",
        start_time=starts[0] + timedelta(minutes=5),
    ))
    since, until = _window()
    assert analyze_batch_placement(db.conn, since, until, None, 12.0) is None


def test_irregular_start_times_are_not_a_candidate(db):
    for i, offset in enumerate([0, 1, 9, 11, 40, 41]):
        db.insert_span(make_llm_span(
            agent_id="adhoc", model="claude-sonnet-4-6", provider="anthropic",
            input_tokens=2_000, output_tokens=500, cost_usd=1.0,
            session_id=f"adhoc-{i}", start_time=BASE + timedelta(hours=offset),
        ))
    since, until = _window()
    assert analyze_batch_placement(db.conn, since, until, None, 12.0) is None


def test_too_few_sessions_to_call_a_cadence(db):
    _cron_sessions(db, count=4)
    since, until = _window()
    assert analyze_batch_placement(db.conn, since, until, None, 12.0) is None


def test_trivial_spend_is_not_worth_an_architectural_change(db):
    _cron_sessions(db, count=6, cost_usd=0.01)
    since, until = _window()
    assert analyze_batch_placement(db.conn, since, until, None, 12.0) is None


# --------------------------------------------------------------------------- #
# Threshold edge
# --------------------------------------------------------------------------- #

def test_gap_cv_needs_at_least_three_gaps():
    starts = [BASE + timedelta(hours=6 * i) for i in range(3)]
    assert gap_coefficient_of_variation(starts) is None
    assert gap_coefficient_of_variation(starts + [BASE + timedelta(hours=18)]) == 0.0


def test_jitter_either_side_of_the_cv_threshold(db):
    # Just inside: a small drift on alternate runs stays under the threshold.
    _cron_sessions(db, agent_id="tight", count=8, gap_hours=6.0, jitter_hours=0.25)
    # Well outside: a large alternating drift scatters the gaps.
    _cron_sessions(db, agent_id="loose", count=8, gap_hours=6.0, jitter_hours=3.0)
    since, until = _window()

    finding = analyze_batch_placement(db.conn, since, until, None, 20.0)

    assert finding is not None
    names = [c.agent_id for c in finding.candidates]
    assert "tight" in names
    assert "loose" not in names
    assert finding.candidates[0].gap_cv < MAX_START_GAP_CV


# --------------------------------------------------------------------------- #
# Serialization round-trip (the daemon path)
# --------------------------------------------------------------------------- #

def test_placement_survives_the_report_dict_round_trip(db):
    """`report_from_dict` drops any finding name it has no constructor for, so
    a missing entry loses the whole card over HTTP: the CLI deserialises the
    report a running `tj serve` hands back through exactly this path, while the
    in-process run keeps the dataclass and never notices."""
    from tokenjam.core.optimize.analyzers.batch_placement import BatchPlacementFinding
    from tokenjam.core.optimize.runner import report_from_dict, report_to_dict
    from tokenjam.core.optimize.types import OptimizeReport, WindowSummary

    _cron_sessions(db, count=6, cost_usd=1.0)
    since, until = _window()
    finding = analyze_batch_placement(db.conn, since, until, None, 12.0)
    assert finding is not None

    report = OptimizeReport(
        window=WindowSummary(
            since=since, until=until, days=WINDOW_DAYS, sessions=6, spans=6,
            total_tokens=15_900, total_cost_usd=6.0, thin_data=False,
        ),
        findings={"placement": finding},
    )
    restored = report_from_dict(report_to_dict(report)).findings.get("placement")

    assert isinstance(restored, BatchPlacementFinding)
    assert restored.window_cost_usd == finding.window_cost_usd
    assert restored.candidate_cost_usd == finding.candidate_cost_usd
    assert restored.percent_of_window_cost == finding.percent_of_window_cost
    assert restored.estimated_recoverable_usd == finding.estimated_recoverable_usd
    assert restored.estimate_basis == finding.estimate_basis
    assert restored.friction == finding.friction
    # The nested candidates come back as dataclasses, not dicts.
    assert [c.agent_id for c in restored.candidates] == ["nightly"]
    original = finding.candidates[0]
    candidate = restored.candidates[0]
    assert candidate.sessions == original.sessions
    assert candidate.first_start == original.first_start
    assert candidate.last_start == original.last_start
    assert candidate.median_gap_seconds == original.median_gap_seconds
    assert candidate.gap_cv == original.gap_cv
    assert candidate.cost_usd == original.cost_usd
    assert candidate.tokens == original.tokens
    assert (candidate.estimated_batch_saving_usd
            == original.estimated_batch_saving_usd)
