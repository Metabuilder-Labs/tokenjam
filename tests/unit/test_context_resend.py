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


def test_cost_of_waste_is_observed_and_never_the_recoverable_figure(db):
    """The gross cost of re-sent context is an OBSERVATION on its own field.

    It used to be absent entirely while `estimated_recoverable_usd` carried the
    cache_control-adoption delta. Now the two are separate quantities: the gross
    is priced per token class at what it really billed (cache reads at the
    cache-read rate, the still-uncached repeat at the input rate) and is never
    the same number as the recoverable one.
    """
    _seed_session(db, "heavy", [1000, 1000, 1000, 1000], cache_ratio=0.0,
                  provider="anthropic", model="claude-haiku-4-5")
    _seed_session(db, "pad1", [500])
    _seed_session(db, "pad2", [500])
    finding = _run(db, _config())

    rates = get_rates("anthropic", "claude-haiku-4-5")
    # Fully uncached, so the whole repeat volume bills at the input rate and
    # there are no cache reads to add.
    heavy_repeat_tokens = 3000  # sum=4000, max=1000
    expected_gross = round(heavy_repeat_tokens / 1_000_000 * rates.input_per_mtok, 6)
    assert finding.cost_of_waste_usd == pytest.approx(expected_gross)
    assert finding.cost_of_waste_basis
    # No subagent anywhere in the window, so nothing measures the offloadable
    # share and no recoverable dollar figure is claimed at all.
    assert finding.offloadable_share is None
    assert finding.estimated_recoverable_usd is None
    assert finding.cost_of_waste_usd != finding.estimated_recoverable_usd


def test_cost_of_waste_prices_cache_reads_at_the_cache_read_rate(db):
    """A fully-cached session still cost real money to re-send — just a tenth
    of the uncached rate. A zero here would read as "re-reading is free"."""
    _seed_session(db, "cached", [1000, 1000, 1000, 1000], cache_ratio=1.0,
                  provider="anthropic", model="claude-haiku-4-5")
    _seed_session(db, "pad1", [500])
    _seed_session(db, "pad2", [500])
    finding = _run(db, _config())
    rates = get_rates("anthropic", "claude-haiku-4-5")
    # Every cache read IS re-sent context: 4 turns x 1000 cached tokens.
    assert finding.cost_of_waste_usd == pytest.approx(
        round(4000 / 1_000_000 * rates.cache_read_per_mtok, 6)
    )
    assert finding.estimated_recoverable_tokens > 0


def test_recoverable_usd_none_when_no_priced_model(db):
    _seed_session(db, "heavy", [1000, 1000, 1000, 1000],
                  provider="unknown-provider", model="unknown-model")
    _seed_session(db, "pad1", [500])
    _seed_session(db, "pad2", [500])
    finding = _run(db, _config())
    assert finding.estimated_recoverable_usd is None
    assert finding.cost_of_waste_usd is None
    assert finding.estimated_recoverable_tokens is not None


def _seed_mixed_model_session(db, session_id, sizes, models, *, cache_ratio=0.0, start=None):
    """Like `_seed_session` but each turn can carry its OWN (provider, model)
    instead of one model for the whole session."""
    start = start or datetime(2026, 5, 10, tzinfo=UTC)
    db.upsert_session(make_session(session_id=session_id, plan_tier="api"))
    for i, (size, (provider, model)) in enumerate(zip(sizes, models)):
        cache_tok = int(size * cache_ratio)
        input_tok = size - cache_tok
        db.insert_span(make_llm_span(
            session_id=session_id, provider=provider, model=model,
            input_tokens=input_tok, cache_tokens=cache_tok, output_tokens=50,
            cost_usd=0.01, start_time=start + timedelta(minutes=i),
        ))


def test_cost_of_waste_prices_each_turn_at_its_own_model_not_the_dominant_one(db):
    """A session that mixes models must price EACH turn's re-sent volume at
    THAT turn's own model's rate, not at whichever model dominated the turn
    count. Two opus turns ($5/MTok) and two haiku turns ($1/MTok), tied on
    turn count, all uncached and equally sized -- the correct figure prices
    each turn's own uncached-repeat share at its own rate."""
    _seed_mixed_model_session(db, "mixed", [1000, 1000, 1000, 1000], [
        ("anthropic", "claude-opus-4-8"), ("anthropic", "claude-opus-4-8"),
        ("anthropic", "claude-haiku-4-5"), ("anthropic", "claude-haiku-4-5"),
    ])
    _seed_session(db, "pad1", [500])
    _seed_session(db, "pad2", [500])
    finding = _run(db, _config())

    opus = get_rates("anthropic", "claude-opus-4-8")
    haiku = get_rates("anthropic", "claude-haiku-4-5")
    # sum=4000, max=1000 (tie) -> repeat_tokens=3000, split evenly across the
    # 4 equally-sized turns: 750 uncached-repeat tokens per turn.
    per_turn_uncached_repeat = 750
    expected = (
        2 * (per_turn_uncached_repeat / 1_000_000 * opus.input_per_mtok)
        + 2 * (per_turn_uncached_repeat / 1_000_000 * haiku.input_per_mtok)
    )
    # What pricing the WHOLE session at a single dominant model (a tie here,
    # but the old code's Counter.most_common(1) picks one) would have given --
    # strictly higher than the correct mixed figure since it prices some
    # cheap-model volume at the expensive rate (or vice versa; either way it
    # cannot equal the correctly split figure for these two distinct rates).
    all_opus_estimate = 3000 / 1_000_000 * opus.input_per_mtok
    all_haiku_estimate = 3000 / 1_000_000 * haiku.input_per_mtok

    assert finding.cost_of_waste_usd == pytest.approx(round(expected, 6))
    assert finding.cost_of_waste_usd != pytest.approx(round(all_opus_estimate, 6))
    assert finding.cost_of_waste_usd != pytest.approx(round(all_haiku_estimate, 6))
    assert all_haiku_estimate < finding.cost_of_waste_usd < all_opus_estimate


# --------------------------------------------------------------------------
# The offload lever: measured from this corpus's own sub_agent_id telemetry
# --------------------------------------------------------------------------

def _seed_offload_corpus(db):
    """One delegating session (which measures the offloadable share) plus one
    context-heavy in-thread session (where the saving is then claimed)."""
    start = datetime(2026, 5, 10, tzinfo=UTC)
    db.upsert_session(make_session(session_id="delegator", plan_tier="api"))
    for i in range(4):
        db.insert_span(make_llm_span(
            session_id="delegator", provider="anthropic", model="claude-opus-4-8",
            input_tokens=1000, cache_tokens=1000, output_tokens=100,
            cost_usd=0.01, start_time=start + timedelta(minutes=i),
        ))
    for i in range(4):
        db.insert_span(make_llm_span(
            session_id="delegator", provider="anthropic", model="claude-opus-4-8",
            input_tokens=1000, cache_tokens=0, output_tokens=100,
            cost_usd=0.01, sub_agent_id="researcher",
            start_time=start + timedelta(minutes=10 + i),
        ))
    # Context-heavy main-thread session: prompt grows past
    # MIN_SESSION_CONTEXT_TOKENS, so offloading has something to remove.
    db.upsert_session(make_session(session_id="inthread", plan_tier="api"))
    for i in range(5):
        db.insert_span(make_llm_span(
            session_id="inthread", provider="anthropic", model="claude-opus-4-8",
            input_tokens=20_000, cache_tokens=20_000 * i, output_tokens=500,
            cost_usd=0.5, start_time=start + timedelta(hours=1, minutes=i),
        ))
    _seed_session(db, "pad1", [500])


def test_offloadable_share_is_measured_from_sub_agent_id(db):
    """The avoidable fraction comes from the user's own delegation behaviour,
    not from the cross-corpus 68.3% constant the token claim still uses."""
    _seed_offload_corpus(db)
    finding = _run(db, _config())
    assert finding.offloadable_share is not None
    assert 0.0 < finding.offloadable_share < 1.0
    assert finding.offloadable_share != pytest.approx(AVOIDABLE_FRACTION_OF_REPEAT)


def test_recoverable_usd_is_offload_plus_rightsize_and_excludes_gross(db):
    _seed_offload_corpus(db)
    finding = _run(db, _config())
    assert finding.offload_recoverable_usd is not None
    assert finding.rightsize_recoverable_usd is not None
    assert finding.estimated_recoverable_usd == pytest.approx(
        round(finding.offload_recoverable_usd + finding.rightsize_recoverable_usd, 6)
    )
    # The whole point of the split: the recoverable claim is a small fraction
    # of what re-sending actually cost, and the gross never leaks into it.
    assert finding.cost_of_waste_usd > finding.estimated_recoverable_usd


def _seed_recoverable_corpus(db, first_turn_model):
    """Same shape as `_seed_offload_corpus`, except the FIRST main-thread turn
    of the in-thread session -- the one with the largest re-read tail, since
    it's the one read back by every later turn -- uses `first_turn_model`
    while every later turn stays on the expensive model."""
    start = datetime(2026, 5, 10, tzinfo=UTC)
    db.upsert_session(make_session(session_id="delegator", plan_tier="api"))
    for i in range(4):
        db.insert_span(make_llm_span(
            session_id="delegator", provider="anthropic", model="claude-opus-4-8",
            input_tokens=1000, cache_tokens=1000, output_tokens=100,
            cost_usd=0.01, start_time=start + timedelta(minutes=i),
        ))
    for i in range(4):
        db.insert_span(make_llm_span(
            session_id="delegator", provider="anthropic", model="claude-opus-4-8",
            input_tokens=1000, cache_tokens=0, output_tokens=100,
            cost_usd=0.01, sub_agent_id="researcher",
            start_time=start + timedelta(minutes=10 + i),
        ))
    db.upsert_session(make_session(session_id="inthread", plan_tier="api"))
    models = [first_turn_model] + ["claude-opus-4-8"] * 4
    for i, model in enumerate(models):
        db.insert_span(make_llm_span(
            session_id="inthread", provider="anthropic", model=model,
            input_tokens=20_000, cache_tokens=20_000 * i, output_tokens=500,
            cost_usd=0.5, start_time=start + timedelta(hours=1, minutes=i),
        ))
    _seed_session(db, "pad1", [500])


def test_offload_recoverable_prices_each_turn_at_its_own_model(db):
    """Changing ONLY the first main-thread turn's model (the one with the
    largest re-read tail) must move the offload-recoverable figure --
    otherwise it is still being priced off whichever model dominates the turn
    count (4 opus turns vs. 1 other), not per turn. Under the old
    dominant-model-only code this comparison would be a no-op: the dominant
    model stays opus in both runs, so the figure would not move at all."""
    _seed_recoverable_corpus(db, "claude-opus-4-8")
    all_opus = _run(db, _config())

    cheap_db = InMemoryBackend()
    _seed_recoverable_corpus(cheap_db, "claude-haiku-4-5")
    mixed = _run(cheap_db, _config())

    assert all_opus.offload_recoverable_usd is not None
    assert mixed.offload_recoverable_usd is not None
    # A cheaper first-turn model must pull the recoverable figure DOWN.
    assert mixed.offload_recoverable_usd < all_opus.offload_recoverable_usd


def test_no_delegating_session_means_no_offload_claim(db):
    """Nothing to measure the share from, so nothing is claimed — never a
    fraction invented to fill the gap."""
    _seed_session(db, "heavy", [1000, 1000, 1000, 1000])
    _seed_session(db, "pad1", [500])
    _seed_session(db, "pad2", [500])
    finding = _run(db, _config())
    assert finding.offloadable_share is None
    assert finding.estimated_recoverable_usd is None
    assert any("offload lever" in n for n in finding.notes)


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
