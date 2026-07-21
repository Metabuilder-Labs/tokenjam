"""Verbosity analyzer (#478) — the output-side lever.

Covers detection against the per-task-shape median baseline (the preferred,
most-defensible signal), the recoverable-savings contract (output above the
baseline priced at output rates), the honest candidate framing (Rule 14), and a
clean report_to_dict/report_from_dict round-trip.

All spans go through tests/factories (Critical Rule 8).
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from tokenjam.core.config import CaptureConfig, OptimizeConfig, TjConfig
from tokenjam.core.db import InMemoryBackend
from tokenjam.core.optimize import ANALYZER_REGISTRY, build_report
from tokenjam.core.optimize.analyzers.output_verbosity import (
    HIGH_VERBOSITY_MULTIPLE,
    MAX_EXAMPLES,
    MIN_COHORT_SESSIONS,
    VERBOSITY_HONESTY_CAVEAT,
    VerbosityFinding,
)
from tokenjam.core.optimize.runner import report_from_dict, report_to_dict
from tokenjam.otel.semconv import GenAIAttributes
from tests.factories import make_llm_span, make_session, make_tool_span


@pytest.fixture
def db():
    backend = InMemoryBackend()
    yield backend
    backend.close()


SINCE = datetime(2026, 5, 1, tzinfo=timezone.utc)
UNTIL = datetime(2026, 5, 30, tzinfo=timezone.utc)
BASE = datetime(2026, 5, 10, tzinfo=timezone.utc)


def _config(tool_inputs: bool = True) -> TjConfig:
    return TjConfig(version="1", capture=CaptureConfig(tool_inputs=tool_inputs))


def _seed_session(db, sid, *, output_tokens, input_tokens=1000, i=0,
                  tool_input=None, model="claude-haiku-4-5"):
    """One session: an LLM span carrying the output tokens + a tool span that
    defines the task shape."""
    sess = make_session(session_id=sid, plan_tier="api", duration_seconds=30.0)
    db.upsert_session(sess)
    llm = make_llm_span(
        model=model, input_tokens=input_tokens, output_tokens=output_tokens,
    )
    llm.session_id = sid
    llm.start_time = BASE + timedelta(minutes=i)
    db.insert_span(llm)
    tool = make_tool_span(tool_name="bash")
    tool.session_id = sid
    tool.start_time = BASE + timedelta(minutes=i, seconds=1)
    tool.attributes = {GenAIAttributes.TOOL_INPUT: tool_input or {"command": "git pull"}}
    db.insert_span(tool)


def _run(db, config=None, agent_id=None):
    report = build_report(
        db=db, config=config or _config(), since=SINCE, until=UNTIL,
        agent_id=agent_id, findings=["verbosity"],
    )
    return report.findings["verbosity"]


# -- Registration --

def test_self_registers():
    assert "verbosity" in ANALYZER_REGISTRY


# -- Detection --

def test_flags_output_above_task_shape_median(db):
    """A cohort of like-shaped sessions with one blatant output outlier flags
    that outlier against the cohort median."""
    # 5 baseline sessions at 200 output tokens each (median = 200) …
    for i in range(MIN_COHORT_SESSIONS):
        _seed_session(db, f"base-{i}", output_tokens=200, i=i)
    # … plus one verbose outlier well above HIGH_VERBOSITY_MULTIPLE × median.
    over = int(200 * HIGH_VERBOSITY_MULTIPLE) + 1000
    _seed_session(db, "verbose", output_tokens=over, i=50)

    finding = _run(db)
    assert isinstance(finding, VerbosityFinding)
    ids = {c.session_id for c in finding.candidates}
    assert "verbose" in ids
    # None of the baseline sessions should be flagged.
    assert not (ids & {f"base-{i}" for i in range(MIN_COHORT_SESSIONS)})

    c = next(c for c in finding.candidates if c.session_id == "verbose")
    assert c.baseline_output_tokens == 200
    assert c.over_baseline_tokens == over - 200
    assert c.over_baseline_multiple >= HIGH_VERBOSITY_MULTIPLE


def test_no_flag_when_output_near_median(db):
    """A cohort where every session is close to the median flags nothing —
    output length is not waste, only clear outliers are candidates."""
    for i in range(MIN_COHORT_SESSIONS + 3):
        _seed_session(db, f"even-{i}", output_tokens=200 + i, i=i)
    finding = _run(db)
    assert finding.candidates == []
    # The cohort still had a usable median.
    assert finding.cohorts_examined == 1


def test_cohort_below_min_size_is_not_a_baseline(db):
    """Below MIN_COHORT_SESSIONS the median is noise — no cohort, no flag even
    with a huge output."""
    _seed_session(db, "solo-a", output_tokens=200, i=0)
    _seed_session(db, "solo-b", output_tokens=100000, i=1)
    finding = _run(db)
    assert finding.cohorts_examined == 0
    assert finding.candidates == []
    assert finding.min_cohort_sessions == MIN_COHORT_SESSIONS


def test_config_lowers_cohort_bar_surfaces_previously_hidden_candidate(db):
    """A 3-session cohort (2 baseline + 1 outlier) is too small to be a
    baseline at the default MIN_COHORT_SESSIONS=5, so nothing is flagged;
    lowering [optimize] min_cohort_sessions to 3 makes the same data a usable
    cohort and flags the outlier."""
    _seed_session(db, "base-0", output_tokens=200, i=0)
    _seed_session(db, "base-1", output_tokens=200, i=1)
    over = int(200 * HIGH_VERBOSITY_MULTIPLE) + 1000
    _seed_session(db, "verbose", output_tokens=over, i=2)

    default_finding = _run(db)
    assert default_finding.cohorts_examined == 0
    assert default_finding.candidates == []

    lowered_config = TjConfig(
        version="1", capture=CaptureConfig(tool_inputs=True),
        optimize=OptimizeConfig(min_cohort_sessions=3),
    )
    lowered_finding = _run(db, config=lowered_config)
    assert lowered_finding.cohorts_examined == 1
    assert lowered_finding.min_cohort_sessions == 3
    ids = {c.session_id for c in lowered_finding.candidates}
    assert "verbose" in ids


def test_output_input_ratio_is_descriptive_not_the_flag(db):
    """A huge output:input ratio alone does NOT flag a session when its output
    is in line with its cohort median — the ratio is the weakest signal."""
    # Every session has a tiny input (10) and identical output (500): ratio 50×,
    # but no output outlier, so nothing is flagged.
    for i in range(MIN_COHORT_SESSIONS + 1):
        _seed_session(db, f"hi-ratio-{i}", output_tokens=500, input_tokens=10, i=i)
    finding = _run(db)
    assert finding.candidates == []


# -- Recoverable-savings contract (#111) --

def test_recoverable_is_output_above_baseline_at_output_rates(db):
    """estimated_recoverable_usd = over-baseline output priced at OUTPUT rates."""
    from tokenjam.core.pricing import get_rates

    for i in range(MIN_COHORT_SESSIONS):
        _seed_session(db, f"b-{i}", output_tokens=200, i=i, model="claude-haiku-4-5")
    over = 5000
    _seed_session(db, "big", output_tokens=200 + over, i=50, model="claude-haiku-4-5")

    finding = _run(db)
    assert finding.estimated_recoverable_tokens == over
    rates = get_rates("anthropic", "claude-haiku-4-5")
    assert rates is not None
    expected = round((over / 1_000_000) * rates.output_per_mtok, 6)
    assert finding.estimated_recoverable_usd == pytest.approx(expected)
    assert finding.estimate_confidence == "heuristic"
    assert finding.estimate_basis  # non-empty basis surfaced


def test_empty_window_contributes_no_recoverable(db):
    """A dead telemetry window yields a finding with no recoverable figure."""
    finding = _run(db)
    assert finding.estimated_recoverable_usd is None
    assert finding.estimated_recoverable_tokens is None
    assert finding.candidates == []


# -- Honest framing (Rule 14) --

def test_caveat_is_conservative_candidate_framing(db):
    """The mandatory caveat frames output as a review candidate, never waste."""
    finding = VerbosityFinding()
    assert finding.caveat == VERBOSITY_HONESTY_CAVEAT
    lowered = VERBOSITY_HONESTY_CAVEAT.lower()
    assert "review" in lowered
    assert "candidate" in lowered
    assert "not waste" in lowered
    # Never asserts wasted spend.
    assert "you are wasting" not in lowered or "never a claim" in lowered


def test_surfaces_remedy_but_does_not_apply(db):
    """A brevity remedy (snippet + suggested max_tokens) is surfaced, not applied."""
    for i in range(MIN_COHORT_SESSIONS):
        _seed_session(db, f"r-{i}", output_tokens=300, i=i)
    _seed_session(db, "loud", output_tokens=3000, i=50)
    finding = _run(db)
    assert finding.remedy_snippet
    assert "concise" in finding.remedy_snippet.lower()
    assert finding.suggested_max_tokens == 300  # the cohort median baseline


# -- Round-trip --

def test_report_round_trips(db):
    for i in range(MIN_COHORT_SESSIONS):
        _seed_session(db, f"rt-{i}", output_tokens=250, i=i)
    _seed_session(db, "rt-loud", output_tokens=9000, i=50)

    report = build_report(db=db, config=_config(), since=SINCE, until=UNTIL,
                          findings=["verbosity"])
    d = report_to_dict(report)
    restored = report_from_dict(d)
    orig = report.findings["verbosity"]
    back = restored.findings["verbosity"]
    assert back.estimated_recoverable_tokens == orig.estimated_recoverable_tokens
    assert back.estimated_recoverable_usd == orig.estimated_recoverable_usd
    assert back.suggested_max_tokens == orig.suggested_max_tokens
    assert len(back.candidates) == len(orig.candidates)
    assert back.candidates[0].session_id == orig.candidates[0].session_id
    assert back.candidates[0].task_shape == orig.candidates[0].task_shape
    assert back.caveat == orig.caveat


# -- CLI wiring (self-registration reaches the renderer + Click choices) --

def test_in_click_choices_and_renderer():
    """verbosity appears in the positional Click choices (auto-derived from the
    registry) and has a human-readable renderer — no edits to cmd_optimize's
    dispatch needed beyond registration wiring."""
    from tokenjam.cli.cmd_optimize import _FINDING_RENDERERS, cmd_optimize

    findings_param = next(
        p for p in cmd_optimize.params if getattr(p, "name", None) == "findings"
    )
    assert "verbosity" in findings_param.type.choices
    assert "verbosity" in _FINDING_RENDERERS


def test_cli_renders_candidate_without_error(db):
    """The finding renders through the CLI dispatch path (regression guard: a
    finding in the registry but missing from _FINDING_RENDERERS would KeyError)."""
    from tokenjam.cli.cmd_optimize import _render_verbosity

    for i in range(MIN_COHORT_SESSIONS):
        _seed_session(db, f"cli-{i}", output_tokens=300, i=i)
    _seed_session(db, "cli-loud", output_tokens=8000, i=50)
    finding = _run(db)
    assert finding.candidates
    # Should not raise for any pricing mode.
    for mode in ("api", "subscription", "local", "unknown"):
        _render_verbosity(finding, pricing_mode=mode, marker="①")


def test_truncates_display_candidates_but_reports_true_total(db):
    """More than MAX_EXAMPLES flagged sessions: `candidates` is capped for
    display, but `total_candidates` carries the real count so the renderer can
    say "top 5 of N" instead of silently under-reporting — and it survives the
    report_to_dict/report_from_dict round-trip (the serve path). (Greptile #479)"""
    # 10 baseline sessions at 100 output tokens (median stays ~100) …
    for i in range(10):
        _seed_session(db, f"low-{i}", output_tokens=100, i=i)
    # … plus MAX_EXAMPLES + 1 verbose sessions well above 2× the median.
    n_high = MAX_EXAMPLES + 1
    for i in range(n_high):
        _seed_session(db, f"hi-{i}", output_tokens=1000, i=20 + i)

    report = build_report(
        db=db, config=_config(), since=SINCE, until=UNTIL, findings=["verbosity"],
    )
    finding = report.findings["verbosity"]
    assert len(finding.candidates) == MAX_EXAMPLES      # display cap
    assert finding.total_candidates == n_high           # true flagged count

    restored = report_from_dict(report_to_dict(report)).findings["verbosity"]
    assert restored.total_candidates == n_high
    assert len(restored.candidates) == MAX_EXAMPLES
