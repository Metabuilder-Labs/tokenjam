"""Unit tests for the context-resend analyzer ("resend")."""
from __future__ import annotations

import statistics
from datetime import datetime, timedelta, timezone

import pytest

from tokenjam.core.config import CaptureConfig, TjConfig
from tokenjam.core.context_diagnostic import TurnComposition
from tokenjam.core.db import InMemoryBackend
from tokenjam.core.optimize import build_report
from tokenjam.core.optimize.analyzers.context_resend import (
    AVOIDABLE_FRACTION_OF_REPEAT,
    MIN_SESSIONS_FOR_SIGNAL,
    MIN_TURNS_FOR_SIGNAL,
    _dominant_provider_model,
    _percentile,
)
from tokenjam.core.pricing import get_rates
from tests.factories import make_llm_span, make_session, make_tool_span

UTC = timezone.utc


@pytest.fixture
def db():
    backend = InMemoryBackend()
    yield backend
    backend.close()


def _config(*, tool_inputs=False, prompts=False, tool_outputs=False) -> TjConfig:
    return TjConfig(version="1", capture=CaptureConfig(
        tool_inputs=tool_inputs, prompts=prompts, tool_outputs=tool_outputs,
    ))


def _seed_session(db, session_id, sizes, *, provider="anthropic",
                   model="claude-haiku-4-5", cache_ratio=0.0,
                   start=None, cost_usd=0.01):
    """Insert one session with `len(sizes)` LLM turns.

    `sizes[i]` is that turn's prompt_size (input_tokens + cache_tokens);
    `cache_ratio` splits it between new input and cache-read tokens
    (0.0 = fully uncached, 1.0 = fully cached).
    """
    start = start or datetime(2026, 5, 10, tzinfo=UTC)
    db.upsert_session(make_session(session_id=session_id, plan_tier="api"))
    for i, size in enumerate(sizes):
        cache_tok = int(size * cache_ratio)
        input_tok = size - cache_tok
        db.insert_span(make_llm_span(
            session_id=session_id, provider=provider, model=model,
            input_tokens=input_tok, cache_tokens=cache_tok, output_tokens=50,
            cost_usd=cost_usd, start_time=start + timedelta(minutes=i),
        ))


def _run(db, config):
    since = datetime(2026, 5, 1, tzinfo=UTC)
    until = datetime(2026, 5, 30, tzinfo=UTC)
    report = build_report(db=db, config=config, since=since, until=until,
                           findings=["resend"])
    return report.findings["resend"]


# --------------------------------------------------------------------------
# Pure-function tests
# --------------------------------------------------------------------------

def test_percentile_single_value():
    assert _percentile([5.0], 0.9) == 5.0


def test_percentile_interpolates():
    assert _percentile([1.0, 2.0, 3.0, 4.0], 0.5) == pytest.approx(2.5)


def test_percentile_empty_list_returns_zero():
    assert _percentile([], 0.9) == 0.0


def _turn(provider, model):
    return TurnComposition(
        session_id="s", sub_agent_id=None, model=model,
        reread_tokens=0, new_input_tokens=1, output_tokens=1,
        cache_write_tokens=0, cost_usd=0.0, provider=provider,
    )


def test_dominant_provider_model_majority():
    turns = [_turn("p1", "a"), _turn("p1", "a"), _turn("p2", "b")]
    assert _dominant_provider_model(turns) == ("p1", "a")


def test_dominant_provider_model_empty_returns_unknown():
    assert _dominant_provider_model([]) == ("unknown", "")


# --------------------------------------------------------------------------
# Empty-state / threshold tests (never a bare "nothing found")
# --------------------------------------------------------------------------

def test_no_llm_turns_notes_reason(db):
    finding = _run(db, _config())
    assert finding.repeat_share is None
    assert finding.notes
    assert "No LLM turns" in finding.notes[0]


def test_too_few_sessions_notes_reason(db):
    """2 sessions, plenty of turns each: still below MIN_SESSIONS_FOR_SIGNAL."""
    assert MIN_SESSIONS_FOR_SIGNAL > 2
    _seed_session(db, "s1", [100, 200, 300])
    _seed_session(db, "s2", [100, 200, 300])
    finding = _run(db, _config())
    assert finding.repeat_share is None
    assert any("too few sessions" in n for n in finding.notes)


def test_too_few_turns_notes_reason(db):
    """3 sessions clears MIN_SESSIONS_FOR_SIGNAL but only 1 turn each, so
    total turns stays below MIN_TURNS_FOR_SIGNAL."""
    assert MIN_SESSIONS_FOR_SIGNAL == 3
    assert MIN_TURNS_FOR_SIGNAL > 3
    _seed_session(db, "s1", [500])
    _seed_session(db, "s2", [500])
    _seed_session(db, "s3", [500])
    finding = _run(db, _config())
    assert finding.repeat_share is None
    assert any("too few turns" in n for n in finding.notes)


# --------------------------------------------------------------------------
# Core metric tests
# --------------------------------------------------------------------------

def test_heavy_repetition_high_share(db):
    """Every turn resends the identical 1000-token prefix: 0.75 for 4 equal
    turns (sum=4000, max=1000)."""
    _seed_session(db, "heavy", [1000, 1000, 1000, 1000])
    _seed_session(db, "pad1", [50, 60, 70])
    _seed_session(db, "pad2", [50, 60, 70])
    finding = _run(db, _config())
    assert finding.repeat_share is not None
    heavy = next(e for e in finding.examples if e.session_id == "heavy")
    assert heavy.repeat_share == pytest.approx(0.75)
    assert heavy.repeat_tokens == 3000


def test_no_repetition_low_share(db):
    """One dominant turn, tiny distinct trailing turns: repeat_share near
    zero (sum=1020, max=1000)."""
    _seed_session(db, "noshare", [1000, 10, 10])
    _seed_session(db, "pad1", [50, 60, 70])
    _seed_session(db, "pad2", [50, 60, 70])
    finding = _run(db, _config())
    noshare = next(e for e in finding.examples if e.session_id == "noshare")
    assert noshare.repeat_share == pytest.approx(20 / 1020, abs=1e-4)
    assert noshare.repeat_tokens == 20


def test_single_turn_session_edge_case(db):
    """A session with exactly one turn cannot structurally repeat: max ==
    sum, repeat_share == 0.0 exactly, no division-by-zero."""
    _seed_session(db, "single", [500])
    _seed_session(db, "multi1", [100, 200, 300])
    _seed_session(db, "multi2", [100, 200, 300])
    finding = _run(db, _config())
    assert finding.repeat_share is not None
    single = next(e for e in finding.examples if e.session_id == "single")
    assert single.turns == 1
    assert single.repeat_share == 0.0
    assert single.repeat_tokens == 0


def test_aggregate_is_token_weighted_not_averaged(db):
    """Aggregate repeat_share = 1 - (sum of maxes / sum of sums), not a naive
    average of per-session shares (benchmarks/RESULTS.md's own definition)."""
    # a: [100, 100] -> sum=200, max=100, share=0.5
    # b: [1000]*4  -> sum=4000, max=1000, share=0.75
    # c: [50, 60]  -> sum=110, max=60, share=60/110... wait share = 1-60/110
    _seed_session(db, "a", [100, 100])
    _seed_session(db, "b", [1000, 1000, 1000, 1000])
    _seed_session(db, "c", [50, 60])
    finding = _run(db, _config())
    total_sum = 200 + 4000 + 110
    total_max = 100 + 1000 + 60
    expected = round(1.0 - (total_max / total_sum), 4)
    assert finding.repeat_share == pytest.approx(expected)
    naive_avg = round(statistics.mean([0.5, 0.75, 1 - 60 / 110]), 4)
    assert finding.repeat_share != pytest.approx(naive_avg)


# --------------------------------------------------------------------------
# Recoverable-estimate tests (honesty discipline: fraction, not full share)
# --------------------------------------------------------------------------

def test_estimated_recoverable_tokens_uses_avoidable_fraction(db):
    _seed_session(db, "heavy", [1000, 1000, 1000, 1000])
    _seed_session(db, "pad1", [500])
    _seed_session(db, "pad2", [500])
    finding = _run(db, _config())
    expected = round(AVOIDABLE_FRACTION_OF_REPEAT * finding.repeat_tokens)
    assert finding.estimated_recoverable_tokens == expected
    # Never the full repeat share: recoverable must be strictly less than
    # repeat_tokens (0.683 < 1.0).
    assert finding.estimated_recoverable_tokens < finding.repeat_tokens


def test_estimated_recoverable_usd_prices_uncached_share_at_rate_delta(db):
    """Fully-uncached session: uncached_fraction == 1.0, so usd is exactly
    repeat_tokens x (input - cache-read rate) x AVOIDABLE_FRACTION_OF_REPEAT."""
    _seed_session(db, "heavy", [1000, 1000, 1000, 1000], cache_ratio=0.0,
                  provider="anthropic", model="claude-haiku-4-5")
    _seed_session(db, "pad1", [500])
    _seed_session(db, "pad2", [500])
    finding = _run(db, _config())

    rates = get_rates("anthropic", "claude-haiku-4-5")
    heavy_repeat_tokens = 3000  # sum=4000, max=1000
    expected_usd = round(
        heavy_repeat_tokens / 1_000_000
        * (rates.input_per_mtok - rates.cache_read_per_mtok)
        * AVOIDABLE_FRACTION_OF_REPEAT,
        6,
    )
    assert finding.estimated_recoverable_usd == pytest.approx(expected_usd)


def test_fully_cached_session_recovers_no_usd_but_still_tokens(db):
    """Same repeat_share/repeat_tokens as the uncached case, but already
    100% cached: the cache_control-adoption dollar opportunity must be
    absent (already captured), even though the compaction token estimate
    still applies. This is what stops the analyzer double-counting
    cache_efficacy's own recoverable figure."""
    _seed_session(db, "cached", [1000, 1000, 1000, 1000], cache_ratio=1.0)
    _seed_session(db, "pad1", [500])  # single-turn: contributes 0 repeat
    _seed_session(db, "pad2", [500])  # single-turn: contributes 0 repeat
    finding = _run(db, _config())
    assert finding.estimated_recoverable_usd is None
    assert finding.estimated_recoverable_tokens > 0


def test_recoverable_usd_none_when_no_priced_model(db):
    _seed_session(db, "heavy", [1000, 1000, 1000, 1000],
                  provider="unknown-provider", model="unknown-model")
    _seed_session(db, "pad1", [500])
    _seed_session(db, "pad2", [500])
    finding = _run(db, _config())
    assert finding.estimated_recoverable_usd is None
    assert finding.estimated_recoverable_tokens is not None


# --------------------------------------------------------------------------
# Fix / evidence surfacing
# --------------------------------------------------------------------------

def test_fix_compaction_always_present(db):
    _seed_session(db, "heavy", [1000, 1000, 1000, 1000])
    _seed_session(db, "pad1", [500])
    _seed_session(db, "pad2", [500])
    finding = _run(db, _config())
    assert finding.fix_compaction


def test_fix_cache_control_snippet_present_for_heaviest_example(db):
    _seed_session(db, "heavy", [1000, 1000, 1000, 1000])
    _seed_session(db, "pad1", [500])
    _seed_session(db, "pad2", [500])
    finding = _run(db, _config())
    assert finding.fix_cache_control
    assert "cache_control" in finding.fix_cache_control


def test_caveat_and_estimate_basis_present(db):
    _seed_session(db, "heavy", [1000, 1000, 1000, 1000])
    _seed_session(db, "pad1", [500])
    _seed_session(db, "pad2", [500])
    finding = _run(db, _config())
    assert "conservative lower bound" in finding.caveat
    assert finding.estimate_basis
    assert "68.3%" in finding.estimate_basis


# --------------------------------------------------------------------------
# Recurring inclusions ("why"): reused from context_diagnostic, capture-gated
# --------------------------------------------------------------------------

def test_capture_off_notes_reason_and_no_recurring_examples(db):
    _seed_session(db, "s1", [100, 200, 300])
    _seed_session(db, "s2", [100, 200, 300])
    _seed_session(db, "s3", [100, 200, 300])
    finding = _run(db, _config())
    assert finding.recurring_examples == []
    assert any("Enable" in n for n in finding.notes)


def test_capture_on_populates_recurring_examples(db):
    base = datetime(2026, 5, 10, tzinfo=UTC)
    for i, sid in enumerate(["s1", "s2", "s3"]):
        _seed_session(db, sid, [100, 200, 300], start=base + timedelta(hours=i))
        ts = make_tool_span(tool_name="Read", tool_input={"file_path": "/repo/schema.py"})
        ts.session_id = sid
        ts.start_time = base + timedelta(hours=i, minutes=1)
        db.insert_span(ts)
    finding = _run(db, _config(tool_inputs=True))
    assert len(finding.recurring_examples) == 1
    assert finding.recurring_examples[0].target == "/repo/schema.py"
    assert not any("Enable" in n for n in finding.notes)


# --------------------------------------------------------------------------
# Round-trip (report_to_dict / report_from_dict) — the daemon-fetch path
# --------------------------------------------------------------------------

def test_finding_round_trips_through_report_dict(db):
    from tokenjam.core.optimize.runner import report_from_dict, report_to_dict

    _seed_session(db, "heavy", [1000, 1000, 1000, 1000])
    _seed_session(db, "pad1", [500])
    _seed_session(db, "pad2", [500])
    finding = _run(db, _config())

    since = datetime(2026, 5, 1, tzinfo=UTC)
    until = datetime(2026, 5, 30, tzinfo=UTC)
    report = build_report(db=db, config=_config(), since=since, until=until,
                           findings=["resend"])
    payload = report_to_dict(report)
    rebuilt = report_from_dict(payload)

    original = report.findings["resend"]
    restored = rebuilt.findings["resend"]
    assert restored.repeat_share == original.repeat_share
    assert restored.estimated_recoverable_tokens == original.estimated_recoverable_tokens
    assert len(restored.examples) == len(original.examples)
    assert restored.examples[0].session_id == original.examples[0].session_id
