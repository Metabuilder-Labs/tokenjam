"""Sampling confidence interval on savings estimates (#308).

Covers the bootstrap helper, that the downgrade analyzer populates
n_sessions + ci_low/ci_high, that the fields round-trip through the report dict,
and the load-bearing property: a small-n run produces a visibly WIDER interval
than a large-n run drawn from the same per-session distribution.

All synthetic data is built via tests/factories (Critical Rule 8).
"""
from __future__ import annotations

from datetime import timedelta

import pytest

from tokenjam.core.db import InMemoryBackend
from tokenjam.core.optimize import analyze_model_downgrade
from tokenjam.core.optimize.runner import report_from_dict, report_to_dict
from tokenjam.core.optimize.stats import bootstrap_ci
from tokenjam.core.optimize.types import OptimizeReport, WindowSummary
from tokenjam.utils.time_parse import utcnow
from tests.factories import make_llm_span


@pytest.fixture
def db():
    backend = InMemoryBackend()
    yield backend
    backend.close()


def _small_opus(db, session_id, *, cost=0.030):
    """One Opus session matching the downgrade heuristic (small in/out, no tools)."""
    db.insert_span(make_llm_span(
        agent_id="claude-code-x", model="claude-opus-4-7", provider="anthropic",
        input_tokens=1000, output_tokens=200, cost_usd=cost,
        session_id=session_id, start_time=utcnow() - timedelta(days=2),
    ))


# --- bootstrap_ci (the pure helper) --------------------------------------- #

def test_bootstrap_ci_none_below_two_samples():
    assert bootstrap_ci([]) is None
    assert bootstrap_ci([5.0]) is None


def test_bootstrap_ci_is_deterministic():
    # A report must not jitter between runs — same input, same interval.
    vals = [0.5, 1.0, 1.5, 2.0, 0.8]
    assert bootstrap_ci(vals) == bootstrap_ci(vals)


def test_bootstrap_ci_brackets_the_point_estimate():
    vals = [1.0, 1.2, 0.9, 1.1, 1.0, 1.3]
    point = sum(vals)  # scale=1.0
    lo, hi = bootstrap_ci(vals)  # type: ignore[misc]
    assert lo <= point <= hi
    assert lo >= 0.0  # savings floored at 0


def test_bootstrap_ci_scale_projects_point_estimate():
    vals = [1.0, 1.0, 1.0, 1.0]  # window sum = 4
    lo, hi = bootstrap_ci(vals, scale=7.5)  # type: ignore[misc]
    # All-equal samples → zero spread → interval collapses on 4 * 7.5 = 30.
    assert lo == pytest.approx(30.0)
    assert hi == pytest.approx(30.0)


def test_small_n_interval_wider_than_large_n():
    # THE load-bearing property (#308): same per-session distribution, but a
    # 5-session sample yields a much wider band than a 500-session one. Scale
    # each so the point estimate is identical, isolating sampling spread.
    base = [0.5, 2.0, 1.0, 1.5, 0.8]
    small = bootstrap_ci(base, scale=10.0)              # n=5
    large = bootstrap_ci(base * 100, scale=10.0 / 100)  # n=500, same mean
    assert small is not None and large is not None
    small_width = small[1] - small[0]
    large_width = large[1] - large[0]
    assert small_width > large_width
    # And the point estimate is the same to within rounding.
    assert sum(base) * 10.0 == pytest.approx(sum(base * 100) * (10.0 / 100))


# --- analyzer populates the fields ---------------------------------------- #

def test_downgrade_finding_carries_n_and_ci():
    db = InMemoryBackend()
    try:
        for i in range(6):
            _small_opus(db, f"s{i}", cost=0.03 + i * 0.005)  # varied savings
        since = utcnow() - timedelta(days=30)
        until = utcnow() + timedelta(hours=1)
        f = analyze_model_downgrade(db.conn, since, until, agent_id=None,
                                    window_days=30.0)
        assert f is not None
        assert f.n_sessions == 6
        assert f.n_sessions == f.candidate_sessions
        assert f.ci_low is not None and f.ci_high is not None
        assert f.ci_low <= f.ci_high
        # The monthly projection sits inside its own interval.
        assert f.ci_low <= f.monthly_savings_usd <= f.ci_high
    finally:
        db.close()


def test_single_candidate_session_has_no_interval():
    # One session can't bracket a projection — n surfaces, CI stays None.
    db = InMemoryBackend()
    try:
        _small_opus(db, "only")
        since = utcnow() - timedelta(days=30)
        until = utcnow() + timedelta(hours=1)
        f = analyze_model_downgrade(db.conn, since, until, agent_id=None,
                                    window_days=30.0)
        assert f is not None
        assert f.n_sessions == 1
        assert f.ci_low is None
        assert f.ci_high is None
    finally:
        db.close()


# --- report dict round-trip ----------------------------------------------- #

def test_n_and_ci_round_trip_through_report_dict():
    db = InMemoryBackend()
    try:
        for i in range(5):
            _small_opus(db, f"s{i}", cost=0.03 + i * 0.004)
        since = utcnow() - timedelta(days=30)
        until = utcnow() + timedelta(hours=1)
        finding = analyze_model_downgrade(db.conn, since, until, agent_id=None,
                                          window_days=30.0)
        report = OptimizeReport(
            window=WindowSummary(
                since=since, until=until, days=30.0, sessions=5, spans=5,
                total_tokens=6000, total_cost_usd=0.2, thin_data=False,
            ),
            downgrade=finding,
        )
        d = report_to_dict(report)
        # Present in the serialised dict.
        assert d["downgrade"]["n_sessions"] == finding.n_sessions
        assert d["downgrade"]["ci_low"] == finding.ci_low
        assert d["downgrade"]["ci_high"] == finding.ci_high
        # And survive the deserialisation symmetrically.
        restored = report_from_dict(d)
        assert restored.downgrade is not None
        assert restored.downgrade.n_sessions == finding.n_sessions
        assert restored.downgrade.ci_low == finding.ci_low
        assert restored.downgrade.ci_high == finding.ci_high
    finally:
        db.close()


# --- CLI savings-line suffix (honesty framing) ---------------------------- #

def _finding(n, ci_low, ci_high):
    from tokenjam.core.optimize.types import DowngradeFinding
    return DowngradeFinding(
        candidate_sessions=n, total_sessions=n, actual_cost_usd=1.0,
        alternative_cost_usd=0.5, monthly_savings_usd=15.0,
        percent_of_sessions=100.0, examples=[], suggestions={},
        n_sessions=n, ci_low=ci_low, ci_high=ci_high,
    )


def test_cli_suffix_shows_n_and_interval():
    from tokenjam.cli.cmd_optimize import _sampling_ci_suffix
    s = _sampling_ci_suffix(_finding(42, 8.0, 23.0))
    assert "n=42" in s
    assert "95% CI" in s
    assert "/mo" in s


def test_cli_suffix_reads_as_sampling_not_safety():
    # Honesty (Rule 14): the label must frame sampling confidence, never imply
    # the swap is safe / validated.
    from tokenjam.cli.cmd_optimize import _sampling_ci_suffix
    s = _sampling_ci_suffix(_finding(42, 8.0, 23.0)).lower()
    assert "sampling confidence" in s
    assert "not a safety claim" in s
    for banned in ("safe to", "validated", "guarantee", "preserves quality"):
        assert banned not in s


def test_cli_suffix_thin_sample_surfaces_n_without_interval():
    from tokenjam.cli.cmd_optimize import _sampling_ci_suffix
    s = _sampling_ci_suffix(_finding(1, None, None))
    assert "n=1" in s
    assert "CI" not in s  # no invented interval for a single session


def test_cli_suffix_empty_when_no_sessions():
    from tokenjam.cli.cmd_optimize import _sampling_ci_suffix
    assert _sampling_ci_suffix(_finding(0, None, None)) == ""


def test_round_trip_tolerates_missing_ci_fields():
    # A report produced by an older daemon (no n/ci keys) must still parse,
    # defaulting n_sessions=0 and ci None — forward/backward compatibility.
    d = {
        "window": {"since": utcnow().isoformat(), "until": utcnow().isoformat(),
                   "days": 7.0, "sessions": 1, "spans": 1,
                   "total_tokens": 100, "total_cost_usd": 0.01, "thin_data": True},
        "downgrade": {
            "candidate_sessions": 1, "total_sessions": 1,
            "actual_cost_usd": 0.03, "alternative_cost_usd": 0.01,
            "monthly_savings_usd": 0.08, "percent_of_sessions": 100.0,
            "examples": [], "suggestions": {},
        },
    }
    restored = report_from_dict(d)
    assert restored.downgrade is not None
    assert restored.downgrade.n_sessions == 0
    assert restored.downgrade.ci_low is None
    assert restored.downgrade.ci_high is None
