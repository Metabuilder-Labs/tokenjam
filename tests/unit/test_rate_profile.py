"""Unit tests for `core/optimize/rate_profile.py`'s blended rate derivation.

Each rate must be weighted by its OWN matching token class: the input rate by
observed input-token volume, the cache-read ratio by observed cache-read-token
volume -- never by output or cache-write volume, which have nothing to do with
either priced rate.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from tokenjam.core.db import InMemoryBackend
from tokenjam.core.optimize.rate_profile import blended_rate_profile
from tokenjam.core.pricing import get_rates
from tests.factories import make_llm_span, make_session

UTC = timezone.utc


def _db():
    backend = InMemoryBackend()
    return backend


def _window():
    since = datetime(2026, 5, 1, tzinfo=UTC)
    until = datetime(2026, 5, 30, tzinfo=UTC)
    return since, until


def test_input_rate_weighted_by_input_tokens_not_output_volume():
    """Model A carries almost all the INPUT volume; Model B carries almost all
    the OUTPUT volume (with a much higher input rate). Weighting by a combined
    input+output+cache+cache_write total (the old behavior) would give the two
    models roughly EQUAL weight and drag the blended input rate toward the
    midpoint -- even though nearly none of the actual input volume came from
    Model B. Weighting by input volume alone must keep the blend close to
    Model A's own rate.
    """
    db = _db()
    db.upsert_session(make_session(session_id="s1"))
    start = datetime(2026, 5, 10, tzinfo=UTC)
    # Model A: claude-opus-4-8 ($5/MTok input) — nearly all the input volume.
    db.insert_span(make_llm_span(
        session_id="s1", provider="anthropic", model="claude-opus-4-8",
        input_tokens=1_000_000, output_tokens=1, cache_tokens=0,
        start_time=start,
    ))
    # Model B: claude-fable-5 ($10/MTok input) — nearly all the output volume,
    # negligible input volume.
    db.insert_span(make_llm_span(
        session_id="s1", provider="anthropic", model="claude-fable-5",
        input_tokens=1, output_tokens=1_000_000, cache_tokens=0,
        start_time=start + timedelta(minutes=1),
    ))

    since, until = _window()
    profile = blended_rate_profile(db.conn, since=since, until=until)

    rate_a = get_rates("anthropic", "claude-opus-4-8")
    rate_b = get_rates("anthropic", "claude-fable-5")
    assert profile is not None
    # Must land close to Model A's own rate (its ~1M input tokens dwarf
    # Model B's 1), not the ~7.5 midpoint an output-inclusive weighting would
    # have produced from two roughly-equal combined totals.
    assert abs(profile.input_rate_per_token - rate_a.input_per_mtok / 1_000_000) < (
        0.05 * rate_a.input_per_mtok / 1_000_000
    )
    midpoint = (rate_a.input_per_mtok + rate_b.input_per_mtok) / 2 / 1_000_000
    assert abs(profile.input_rate_per_token - midpoint) > abs(
        profile.input_rate_per_token - rate_a.input_per_mtok / 1_000_000
    )


def test_cache_read_ratio_weighted_by_cache_read_tokens_not_output_volume():
    """Model A carries all the CACHE-READ volume (Anthropic ratio 0.1). Model B
    carries a huge OUTPUT volume but a rate table entry with NO cache-read
    price (defaults to 0.0 -- e.g. a provider row with no cache_read_per_mtok
    set). Weighting the ratio by a combined total would drag it toward 0
    purely because Model B's output volume swamped the combined denominator,
    even though Model B contributed zero cache-read evidence. Weighting by
    cache-read volume alone must keep the observed 0.1 ratio intact.
    """
    db = _db()
    db.upsert_session(make_session(session_id="s1"))
    start = datetime(2026, 5, 10, tzinfo=UTC)
    # Model A: claude-opus-4-8 -- input + cache-read volume, ratio 0.1.
    db.insert_span(make_llm_span(
        session_id="s1", provider="anthropic", model="claude-opus-4-8",
        input_tokens=100_000, output_tokens=1, cache_tokens=100_000,
        start_time=start,
    ))
    # Model B: gemini-2-5-pro -- huge output volume, no cache-read price
    # (defaults to 0.0), no cache reads of its own.
    db.insert_span(make_llm_span(
        session_id="s1", provider="google", model="gemini-2-5-pro",
        input_tokens=1, output_tokens=1_000_000, cache_tokens=0,
        start_time=start + timedelta(minutes=1),
    ))

    since, until = _window()
    profile = blended_rate_profile(db.conn, since=since, until=until)

    assert profile is not None
    # The observed ratio must stay at Model A's own 0.1 -- not dragged toward
    # 0 by Model B's unrelated output volume.
    assert abs(profile.cache_read_ratio - 0.100) < 0.01


def test_cache_read_ratio_falls_back_to_input_weighting_when_no_cache_reads_observed():
    """A window with input volume but literally zero cache-read tokens still
    resolves a ratio (derived from the SAME observed models' own rates,
    weighted by input volume instead) rather than returning None outright."""
    db = _db()
    db.upsert_session(make_session(session_id="s1"))
    start = datetime(2026, 5, 10, tzinfo=UTC)
    db.insert_span(make_llm_span(
        session_id="s1", provider="anthropic", model="claude-opus-4-8",
        input_tokens=100_000, output_tokens=1_000, cache_tokens=0,
        start_time=start,
    ))

    since, until = _window()
    profile = blended_rate_profile(db.conn, since=since, until=until)

    assert profile is not None
    rate_a = get_rates("anthropic", "claude-opus-4-8")
    assert abs(
        profile.cache_read_ratio - rate_a.cache_read_per_mtok / rate_a.input_per_mtok
    ) < 1e-9


def test_single_model_input_rate_and_ratio_match_its_own_rates():
    db = _db()
    db.upsert_session(make_session(session_id="s1"))
    start = datetime(2026, 5, 10, tzinfo=UTC)
    db.insert_span(make_llm_span(
        session_id="s1", provider="anthropic", model="claude-sonnet-5",
        input_tokens=10_000, output_tokens=2_000, cache_tokens=5_000,
        cache_write_tokens=1_000, start_time=start,
    ))

    since, until = _window()
    profile = blended_rate_profile(db.conn, since=since, until=until)

    rates = get_rates("anthropic", "claude-sonnet-5")
    assert profile is not None
    assert profile.input_rate_per_token == rates.input_per_mtok / 1_000_000
    assert abs(
        profile.cache_read_ratio - rates.cache_read_per_mtok / rates.input_per_mtok
    ) < 1e-9
