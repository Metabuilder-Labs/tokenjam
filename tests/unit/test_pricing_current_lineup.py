"""The current Anthropic / OpenAI / Gemini lineups price at their published rates.

Each row added for the 0.7.1 lineup refresh is pinned twice: (1) every id FORM a
real span carries resolves to the row rather than the default fallback, and
(2) a sample token mix costs exactly what the published rates say. As with the
backfilled entries in `test_pricing_lookup_forms.py`, these pin what
`models.toml` says, which is all a test can do; the provenance (source URL and
read date) lives next to each entry there.

Every case passes `at=` explicitly. Several of these models carry dated rows
(gemini-4-carbon / -argon from 2026-12-01, gemini-3.8-flash / -3.6-flash from
2027-01-01), and a case that let the instant float to "now" would start
asserting the wrong row the day each boundary lands, which is the failure the
pinned instant in `tests/unit/test_cost.py` exists to prevent.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from tokenjam.core.cost import calculate_cost
from tokenjam.core.pricing import classify_pricing_source, get_rates

_AT = datetime(2026, 11, 15, tzinfo=timezone.utc)


@pytest.mark.parametrize(
    "provider,model,expected_input",
    [
        # Anthropic: bare API id, the 1M-context tag, and a gateway's dotted,
        # prefixed form.
        ("anthropic", "claude-fable-5-2", 8.00),
        ("anthropic", "claude-mythos-5-2", 8.00),
        ("anthropic", "claude-fable-5-1", 10.00),
        ("anthropic", "claude-mythos-5-1", 10.00),
        ("anthropic", "claude-opus-5-5", 4.00),
        ("anthropic", "claude-opus-5-5[1m]", 4.00),
        ("anthropic", "anthropic/claude-opus-5.5", 4.00),
        ("anthropic", "claude-sonnet-5-5", 2.00),
        ("anthropic", "claude-haiku-5-5", 0.10),
        # OpenAI: the dotted API id, a dashed dated snapshot, and a routing prefix.
        ("openai", "gpt-6-astra", 10.00),
        ("openai", "gpt-6.1-sol", 2.00),
        ("openai", "gpt-6.1-terra", 1.00),
        ("openai", "gpt-6-sol", 2.00),
        ("openai", "gpt-6-luna", 0.10),
        ("openai", "gpt-5.6-terra", 2.00),
        ("openai", "gpt-5.6-luna", 0.20),
        ("openai", "gpt-5.6-luna-2026-07-01", 0.20),
        ("openai", "openai/gpt-5.6-terra", 2.00),
        ("openai", "gpt-5.2", 1.75),
        ("openai", "gpt-5.2-pro", 21.00),
        ("openai", "gpt-5.1", 1.25),
        ("openai", "gpt-5", 1.25),
        ("openai", "gpt-5-mini", 0.25),
        ("openai", "gpt-5-mini-2025-08-07", 0.25),
        ("openai", "gpt-5-nano", 0.05),
        ("openai", "gpt-5-pro", 15.00),
        ("openai", "gpt-5.3-codex", 1.75),
        ("openai", "gpt-4.1-mini", 0.40),
        ("openai", "o3-pro", 20.00),
        # Gemini: the dotted API id (dashed in the table) and the `models/` form
        # the Gemini API returns.
        ("google", "gemini-4-carbon", 2.00),
        ("google", "gemini-4-argon", 2.00),
        ("google", "gemini-4-hydrogen", 1.50),
        ("google", "gemini-3.8-flash", 0.75),
        ("google", "gemini-3.6-flash", 0.75),
        ("google", "gemini-3.5-flash-lite", 0.30),
        ("google", "models/gemini-3.5-flash-lite", 0.30),
        ("google", "gemini-3.1-flash-lite", 0.25),
        ("google", "gemini-3.1-pro-preview", 2.00),
        ("google", "gemini-3-flash-preview", 0.50),
        ("google", "gemini-2.5-flash-lite", 0.10),
    ],
)
def test_current_lineup_id_forms_resolve(provider, model, expected_input):
    rates = get_rates(provider, model, at=_AT)
    assert rates is not None, f"{provider}/{model} fell through to the default rate"
    assert rates.input_per_mtok == expected_input
    assert classify_pricing_source(provider, model) != "default_fallback"


_BEFORE_DEC_1 = datetime(2026, 11, 30, 23, 59, tzinfo=timezone.utc)
_DEC_1 = datetime(2026, 12, 1, tzinfo=timezone.utc)
_BEFORE_JAN_1 = datetime(2026, 12, 31, 23, 59, tzinfo=timezone.utc)
_JAN_1 = datetime(2027, 1, 1, tzinfo=timezone.utc)


# Sample mix: 1M input, 1M output, 1M cache-read, plus 1M cache-write for rows
# that publish a write price (0 otherwise, so an unpublished write price cannot
# leak into the figure). Expected = the sum of the published per-MTok rates.
@pytest.mark.parametrize(
    "provider,model,cache_write_tokens,at,expected",
    [
        ("anthropic", "claude-fable-5-2", 1_000_000, _AT, 8 + 40 + 0.20 + 10),
        ("anthropic", "claude-mythos-5-2", 1_000_000, _AT, 8 + 40 + 0.20 + 10),
        ("anthropic", "claude-fable-5-1", 1_000_000, _AT, 10 + 50 + 0.25 + 12.50),
        ("anthropic", "claude-mythos-5-1", 1_000_000, _AT, 10 + 50 + 0.25 + 12.50),
        ("anthropic", "claude-opus-5-5", 1_000_000, _AT, 4 + 20 + 0.20 + 5),
        ("anthropic", "claude-sonnet-5-5", 1_000_000, _AT, 2 + 10 + 0.10 + 2.50),
        ("anthropic", "claude-haiku-5-5", 1_000_000, _AT, 0.10 + 0.50 + 0.01 + 0.125),
        ("openai", "gpt-6-astra", 1_000_000, _AT, 10 + 50 + 1.00 + 12.50),
        ("openai", "gpt-6.1-sol", 1_000_000, _AT, 2 + 10 + 0.10 + 2.50),
        ("openai", "gpt-6.1-terra", 1_000_000, _AT, 1 + 5 + 0.05 + 1.25),
        ("openai", "gpt-6-sol", 1_000_000, _AT, 2 + 10 + 0.20 + 2.50),
        ("openai", "gpt-6-luna", 1_000_000, _AT, 0.10 + 0.50 + 0.01 + 0.125),
        ("openai", "gpt-5.6-terra", 1_000_000, _AT, 2 + 12 + 0.20 + 2.50),
        ("openai", "gpt-5.6-luna", 1_000_000, _AT, 0.20 + 1.20 + 0.02 + 0.25),
        ("openai", "gpt-5.2", 0, _AT, 1.75 + 14 + 0.175),
        ("openai", "gpt-5-mini", 0, _AT, 0.25 + 2 + 0.025),
        ("openai", "gpt-5-nano", 0, _AT, 0.05 + 0.40 + 0.005),
        ("openai", "gpt-5.3-codex", 0, _AT, 1.75 + 14 + 0.175),
        ("google", "gemini-4-hydrogen", 0, _AT, 1.50 + 7.50 + 0.075),
        ("google", "gemini-3.5-flash-lite", 0, _AT, 0.30 + 2.50 + 0.03),
        ("google", "gemini-3.1-flash-lite", 0, _AT, 0.25 + 1.50 + 0.025),
        ("google", "gemini-3.1-pro-preview", 0, _AT, 2 + 12 + 0.20),
        ("google", "gemini-3-flash-preview", 0, _AT, 0.50 + 3 + 0.05),
        ("google", "gemini-2.5-flash-lite", 0, _AT, 0.10 + 0.40 + 0.01),
        # Dated rows: one case on EACH side of every boundary, each pinned.
        ("google", "gemini-4-carbon", 0, _BEFORE_DEC_1, 2 + 10 + 0.10),
        ("google", "gemini-4-carbon", 0, _DEC_1, 4 + 20 + 0.20),
        ("google", "gemini-4-argon", 0, _BEFORE_DEC_1, 2 + 10 + 0.10),
        ("google", "gemini-4-argon", 0, _DEC_1, 4 + 20 + 0.20),
        ("google", "gemini-3.8-flash", 0, _BEFORE_JAN_1, 0.75 + 3.75 + 0.075),
        ("google", "gemini-3.8-flash", 0, _JAN_1, 1.50 + 7.50 + 0.15),
        ("google", "gemini-3.6-flash", 0, _BEFORE_JAN_1, 0.75 + 3.75 + 0.075),
        ("google", "gemini-3.6-flash", 0, _JAN_1, 1.50 + 7.50 + 0.15),
    ],
)
def test_current_lineup_sample_mix_costs_the_published_rates(
    provider, model, cache_write_tokens, at, expected,
):
    cost = calculate_cost(
        provider, model,
        input_tokens=1_000_000,
        output_tokens=1_000_000,
        cache_read_tokens=1_000_000,
        cache_write_tokens=cache_write_tokens,
        at=at,
    )
    assert cost == pytest.approx(expected)


@pytest.mark.parametrize(
    "model,variant,expected",
    [
        # Fast mode: the page's premium rate with the caching multipliers on top.
        ("claude-opus-5-5", "fast", (8.00, 40.00, 0.40, 10.00)),
        ("claude-fable-5-2", "fast", (16.00, 80.00, 0.40, 20.00)),
        ("claude-mythos-5-2", "fast", (16.00, 80.00, 0.40, 20.00)),
        # Batch is a flat half and the 1-hour write is 2x input, both from
        # `[variants]`; the page's own figures for these models agree.
        ("claude-opus-5-5", "batch", (2.00, 10.00, 0.10, 2.50)),
        ("claude-opus-5-5", "cache-1h", (4.00, 20.00, 0.20, 8.00)),
        ("claude-fable-5-2", "cache-1h", (8.00, 40.00, 0.20, 16.00)),
        ("claude-sonnet-5-5", "cache-1h", (2.00, 10.00, 0.10, 4.00)),
    ],
)
def test_current_anthropic_variants(model, variant, expected):
    rates = get_rates("anthropic", model, at=_AT, variant=variant)
    assert rates is not None
    got = (
        rates.input_per_mtok, rates.output_per_mtok,
        rates.cache_read_per_mtok, rates.cache_write_per_mtok,
    )
    assert got == pytest.approx(expected)


def test_sonnet_5_stays_at_2_10_after_the_cancelled_september_rise():
    """Anthropic cancelled the 2026-09-01 move to $3 / $15; a span from after
    that date must still price at $2 / $10."""
    after = get_rates("anthropic", "claude-sonnet-5", at=datetime(2026, 10, 1, tzinfo=timezone.utc))
    assert after is not None
    assert (after.input_per_mtok, after.output_per_mtok) == (2.00, 10.00)


def test_gpt_5_6_sol_stays_unpriced_until_its_post_promo_rate_is_published():
    """Its only published rate is a promotion ending 2026-11-30 with no stated
    successor, so the table deliberately carries no row; adding one without the
    follow-on rate would misprice every span after that date."""
    assert get_rates("openai", "gpt-5.6-sol", at=_AT) is None


@pytest.mark.parametrize(
    "provider,model,change,before,after",
    [
        # o3's 80% cut; spans from before it keep the rate that billed them.
        ("openai", "o3", datetime(2025, 6, 10, tzinfo=timezone.utc), (10.00, 40.00), (2.00, 8.00)),
        # gemini-2.5-flash preview rate, then the GA rate.
        ("google", "gemini-2-5-flash", datetime(2025, 6, 17, tzinfo=timezone.utc),
         (0.15, 0.60), (0.30, 2.50)),
    ],
)
def test_historic_price_changes_are_dated_not_overwritten(provider, model, change, before, after):
    pre = get_rates(provider, model, at=datetime(change.year, change.month, change.day - 1,
                                                tzinfo=timezone.utc))
    post = get_rates(provider, model, at=change)
    now = get_rates(provider, model, at=_AT)
    assert pre is not None and post is not None and now is not None
    assert (pre.input_per_mtok, pre.output_per_mtok) == before
    assert (post.input_per_mtok, post.output_per_mtok) == after
    assert (now.input_per_mtok, now.output_per_mtok) == after
