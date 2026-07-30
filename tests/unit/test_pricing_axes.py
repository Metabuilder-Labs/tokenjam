"""The pricing table's two extra axes: rate history (time) and variants.

One rate per model can express neither a price change nor a premium way of
buying the same model id. These tests pin both axes, and — just as importantly —
pin that adding them left every existing single-rate row valid unchanged.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

import pytest

from tests.factories import make_llm_span
from tokenjam.core import pricing
from tokenjam.core.cost import calculate_cost, rate_variant_for_span
from tokenjam.core.pricing import STANDARD_VARIANT, get_rates


@pytest.fixture(autouse=True)
def _clean_pricing_cache(monkeypatch, tmp_path):
    """Isolate every test from any real user override file, and reset caches."""
    monkeypatch.delenv(pricing.USER_PRICING_ENV, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    pricing.clear_pricing_cache()
    yield
    pricing.clear_pricing_cache()


def _write_override(monkeypatch, path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    monkeypatch.setenv(pricing.USER_PRICING_ENV, str(path))
    pricing.clear_pricing_cache()


# --- Time axis -------------------------------------------------------------

RATE_HISTORY_TOML = """
[testprovider.dated-model]
input_per_mtok = 2.00
output_per_mtok = 10.00
cache_read_per_mtok = 0.20
cache_write_per_mtok = 2.50

[[testprovider.dated-model.rates]]
valid_from = 2026-09-01
input_per_mtok = 3.00
output_per_mtok = 15.00
"""

BEFORE = datetime(2026, 8, 15, 12, 0, tzinfo=timezone.utc)
AFTER = datetime(2026, 9, 15, 12, 0, tzinfo=timezone.utc)


def test_two_dated_rows_price_each_side_of_the_boundary_at_its_own_rate(
    monkeypatch, tmp_path,
):
    _write_override(monkeypatch, tmp_path / "pricing.toml", RATE_HISTORY_TOML)

    before = get_rates("testprovider", "dated-model", at=BEFORE)
    after = get_rates("testprovider", "dated-model", at=AFTER)

    assert before is not None and after is not None
    assert (before.input_per_mtok, before.output_per_mtok) == (2.00, 10.00)
    assert (after.input_per_mtok, after.output_per_mtok) == (3.00, 15.00)


def test_a_dated_row_inherits_the_fields_it_does_not_restate(monkeypatch, tmp_path):
    """The 2026-09-01 row names only input/output — the cache rates carry over,
    rather than silently collapsing to zero."""
    _write_override(monkeypatch, tmp_path / "pricing.toml", RATE_HISTORY_TOML)

    after = get_rates("testprovider", "dated-model", at=AFTER)

    assert after is not None
    assert after.cache_read_per_mtok == 0.20
    assert after.cache_write_per_mtok == 2.50


def test_the_cost_of_a_span_follows_the_rate_in_effect_when_it_happened(
    monkeypatch, tmp_path,
):
    _write_override(monkeypatch, tmp_path / "pricing.toml", RATE_HISTORY_TOML)

    old = calculate_cost("testprovider", "dated-model", 1_000_000, 0, at=BEFORE)
    new = calculate_cost("testprovider", "dated-model", 1_000_000, 0, at=AFTER)

    assert old == pytest.approx(2.00)
    assert new == pytest.approx(3.00)


def test_a_span_older_than_every_known_rate_prices_at_the_earliest_one(
    monkeypatch, tmp_path,
):
    """An unpriced span is worse than one priced at the oldest rate we hold."""
    _write_override(
        monkeypatch,
        tmp_path / "pricing.toml",
        "[[testprovider.only-dated.rates]]\n"
        "valid_from = 2026-09-01\n"
        "input_per_mtok = 3.00\noutput_per_mtok = 15.00\n",
    )

    rates = get_rates("testprovider", "only-dated", at=BEFORE)

    assert rates is not None
    assert rates.input_per_mtok == 3.00


def test_the_packaged_sonnet_5_rate_change_is_expressed_not_pending():
    """The introductory rate expires 2026-08-31; both sides must be expressible
    at once, so a figure already recorded under the intro rate stays checkable
    against the bill that produced it."""
    intro = get_rates("anthropic", "claude-sonnet-5", at=BEFORE)
    standard = get_rates("anthropic", "claude-sonnet-5", at=AFTER)

    assert intro is not None and standard is not None
    assert (intro.input_per_mtok, intro.output_per_mtok) == (2.00, 10.00)
    assert (standard.input_per_mtok, standard.output_per_mtok) == (3.00, 15.00)


# --- Variant axis ----------------------------------------------------------

def test_fast_mode_traffic_on_claude_opus_5_prices_at_the_fast_rate():
    standard = get_rates("anthropic", "claude-opus-5")
    fast = get_rates("anthropic", "claude-opus-5", variant="fast")

    assert standard is not None and fast is not None
    assert (standard.input_per_mtok, standard.output_per_mtok) == (5.00, 25.00)
    assert (fast.input_per_mtok, fast.output_per_mtok) == (10.00, 50.00)

    priced = calculate_cost("anthropic", "claude-opus-5", 1_000_000, 0, variant="fast")
    assert priced == pytest.approx(10.00)


def test_a_fast_mode_span_is_costed_at_the_fast_rate_end_to_end():
    """The variant comes off the span's own captured request params, so a fast
    call is not silently billed at half its real rate."""
    fast_span = make_llm_span(
        model="claude-opus-5", input_tokens=1_000_000, output_tokens=0,
        request_params={"speed": "fast"},
    )
    standard_span = make_llm_span(
        model="claude-opus-5", input_tokens=1_000_000, output_tokens=0,
    )

    assert rate_variant_for_span(fast_span) == "fast"
    assert rate_variant_for_span(standard_span) == STANDARD_VARIANT

    fast_cost = calculate_cost(
        fast_span.provider, fast_span.model, 1_000_000, 0,
        at=fast_span.start_time, variant=rate_variant_for_span(fast_span),
    )
    standard_cost = calculate_cost(
        standard_span.provider, standard_span.model, 1_000_000, 0,
        at=standard_span.start_time, variant=rate_variant_for_span(standard_span),
    )

    assert fast_cost == pytest.approx(2 * standard_cost)


def test_the_batch_discount_is_rate_data_not_an_analyzer_constant():
    from tokenjam.core.optimize.analyzers.batch_placement import batch_discount

    assert pricing.variant_price_ratio("batch") == pytest.approx(0.50)
    assert batch_discount() == pytest.approx(0.50)

    standard = get_rates("anthropic", "claude-haiku-4-5")
    batch = get_rates("anthropic", "claude-haiku-4-5", variant="batch")
    assert standard is not None and batch is not None
    for field_name in pricing.RATE_FIELDS:
        assert getattr(batch, field_name) == pytest.approx(
            0.50 * getattr(standard, field_name)
        )


def test_the_one_hour_cache_ttl_write_rate_is_rate_data_not_an_analyzer_constant():
    from tokenjam.core.optimize.analyzers.cache_efficacy import ONE_HOUR_TTL_VARIANT

    standard = get_rates("anthropic", "claude-opus-4-8")
    ttl = get_rates("anthropic", "claude-opus-4-8", variant=ONE_HOUR_TTL_VARIANT)

    assert standard is not None and ttl is not None
    # A 1-hour write bills at 2x the model's INPUT rate (not 2x its 5-min write).
    assert ttl.cache_write_per_mtok == pytest.approx(2.0 * standard.input_per_mtok)
    # Everything the variant doesn't name keeps the model's standard rate.
    assert ttl.input_per_mtok == standard.input_per_mtok
    assert ttl.cache_read_per_mtok == standard.cache_read_per_mtok


def test_a_per_model_variant_row_wins_over_a_global_variant_definition(
    monkeypatch, tmp_path,
):
    _write_override(
        monkeypatch,
        tmp_path / "pricing.toml",
        "[testprovider.pinned]\n"
        "input_per_mtok = 4.00\noutput_per_mtok = 20.00\n"
        "\n[[testprovider.pinned.rates]]\n"
        'variant = "batch"\n'
        "input_per_mtok = 1.00\n",
    )

    batch = get_rates("testprovider", "pinned", variant="batch")

    assert batch is not None
    assert batch.input_per_mtok == 1.00           # the per-model row
    assert batch.output_per_mtok == pytest.approx(20.00)  # not restated -> standard


def test_an_unknown_variant_falls_back_to_standard_and_says_so(caplog):
    with caplog.at_level(logging.WARNING, logger="tokenjam.core.pricing"):
        rates = get_rates("anthropic", "claude-haiku-4-5", variant="no-such-variant")

    standard = get_rates("anthropic", "claude-haiku-4-5")
    assert rates == standard
    assert "no-such-variant" in caplog.text


# --- Backward compatibility ------------------------------------------------

def test_every_packaged_single_rate_row_still_resolves_unchanged():
    """The rows in models.toml carry no valid_from and no variant, so they must
    resolve for any instant and for the standard variant — this is the whole
    backward-compatibility guarantee of adding the two axes."""
    rows = pricing.load_pricing_rows()
    assert rows, "packaged pricing table failed to load"

    long_ago = datetime(2020, 1, 1, tzinfo=timezone.utc)
    far_ahead = datetime(2099, 1, 1, tzinfo=timezone.utc)
    for provider, models in rows.items():
        for model in models:
            for at in (long_ago, far_ahead):
                rates = get_rates(provider, model, at=at)
                assert rates is not None, f"{provider}/{model} unresolvable at {at}"
                assert rates.input_per_mtok >= 0
                assert rates.output_per_mtok >= 0


def test_a_flat_row_with_no_rates_list_parses_to_one_always_applied_row(
    monkeypatch, tmp_path,
):
    _write_override(
        monkeypatch,
        tmp_path / "pricing.toml",
        "[testprovider.flat]\ninput_per_mtok = 1.0\noutput_per_mtok = 2.0\n",
    )

    rows = pricing.load_pricing_rows()["testprovider"]["flat"]

    assert len(rows) == 1
    assert rows[0].valid_from is None
    assert rows[0].variant == STANDARD_VARIANT
    assert rows[0].rates.input_per_mtok == 1.0


def test_the_flat_pricing_table_view_keeps_its_model_to_rates_shape():
    """`tj pricing list` and friends read this shape; the axes are additive."""
    table = pricing.load_pricing_table()

    assert isinstance(table, dict)
    haiku = table["anthropic"]["claude-haiku-4-5"]
    assert haiku.input_per_mtok == 1.00
