"""Unit tests for tokenjam.core.cost and tokenjam.core.pricing."""
from __future__ import annotations
import logging

import pytest

from tokenjam.core.cost import calculate_cost
from tokenjam.core.pricing import load_pricing_table, get_rates


def test_calculate_cost_known_model():
    # anthropic/claude-haiku-4-5: input=0.80, output=4.00 per MTok
    # 1000 input, 200 output
    # Expected: (1000/1M * 0.80) + (200/1M * 4.00) = 0.0008 + 0.0008 = 0.0016
    cost = calculate_cost("anthropic", "claude-haiku-4-5", 1000, 200)
    assert cost == 0.0016


def test_calculate_cost_with_cache_tokens():
    # claude-haiku-4-5: cache_read=0.08 per MTok
    cost = calculate_cost(
        "anthropic", "claude-haiku-4-5",
        input_tokens=1000,
        output_tokens=200,
        cache_read_tokens=5000,
    )
    # (1000/1M * 0.80) + (200/1M * 4.00) + (5000/1M * 0.08)
    # = 0.0008 + 0.0008 + 0.0004 = 0.0020
    assert cost == 0.002


def test_calculate_cost_with_cache_write_tokens():
    # claude-haiku-4-5: cache_write=1.00 per MTok
    cost = calculate_cost(
        "anthropic", "claude-haiku-4-5",
        input_tokens=0,
        output_tokens=0,
        cache_read_tokens=0,
        cache_write_tokens=1_000_000,
    )
    # Zero input/output but cache_write tokens present — the early return only
    # fires when ALL token counts are zero, so cache_write cost is still charged.
    # (1_000_000/1M * 1.00) = 1.00
    assert cost == 1.0


def test_calculate_cost_cache_read_only():
    # claude-haiku-4-5: cache_read=0.08 per MTok. A pure cache hit (no new
    # input/output) still costs the cache-read rate and must not be dropped.
    cost = calculate_cost(
        "anthropic", "claude-haiku-4-5",
        input_tokens=0,
        output_tokens=0,
        cache_read_tokens=1_000_000,
    )
    # (1_000_000/1M * 0.08) = 0.08
    assert cost == 0.08


def test_calculate_cost_unknown_model_uses_default(caplog):
    # Use a unique provider/model so the dedupe set doesn't suppress this run.
    with caplog.at_level(logging.WARNING, logger="tokenjam.core.cost"):
        cost = calculate_cost("test_unknown_provider", "test_unknown_model", 1_000_000, 1_000_000)
    # Default rates: 0.50 input, 2.00 output per MTok
    # (1M/1M * 0.50) + (1M/1M * 2.00) = 2.50
    assert cost == 2.5
    assert "No pricing data for test_unknown_provider/test_unknown_model" in caplog.text


def test_calculate_cost_unknown_model_warns_only_once_per_pair(caplog):
    """Backfilling many spans of the same unknown model used to spam the
    warning N times. Now it's emitted once per (provider, model) per
    process. Issue #98."""
    import tokenjam.core.cost as cost_mod
    # Reset dedupe set so this test is isolated.
    cost_mod._UNKNOWN_MODEL_WARNED.clear()

    with caplog.at_level(logging.WARNING, logger="tokenjam.core.cost"):
        for _ in range(5):
            calculate_cost("test_provider_xyz", "test_model_xyz", 1000, 200)

    # Exactly one warning, not five.
    matching = [r for r in caplog.records if "test_provider_xyz/test_model_xyz" in r.message]
    assert len(matching) == 1, f"expected 1 warning, got {len(matching)}"

    # A DIFFERENT unknown model in the same process should still warn (once).
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="tokenjam.core.cost"):
        for _ in range(3):
            calculate_cost("test_provider_xyz", "different_model", 1000, 200)
    matching = [r for r in caplog.records if "different_model" in r.message]
    assert len(matching) == 1


def test_deprecated_anthropic_base_models_are_priced():
    """Dated variants (claude-sonnet-4-20250514, etc.) resolve via the
    YYYYMMDD-stripping fallback to the deprecated base entries we added in
    pricing/models.toml. Issue #98 — was previously falling through to
    defaults and spamming warnings."""
    # Sonnet 4 (deprecated): $3 / $15 per MTok
    cost = calculate_cost("anthropic", "claude-sonnet-4-20250514", 1_000_000, 1_000_000)
    assert cost == pytest.approx(18.0)  # 3 + 15

    # Opus 4 (deprecated): $15 / $75 per MTok
    cost = calculate_cost("anthropic", "claude-opus-4-20250514", 1_000_000, 1_000_000)
    assert cost == pytest.approx(90.0)  # 15 + 75

    # Opus 4.1 (deprecated): $15 / $75 per MTok
    cost = calculate_cost("anthropic", "claude-opus-4-1-20250805", 1_000_000, 1_000_000)
    assert cost == pytest.approx(90.0)

    # Haiku 3.5 (retired): $0.80 / $4 per MTok
    cost = calculate_cost("anthropic", "claude-haiku-3-5-20241022", 1_000_000, 1_000_000)
    assert cost == pytest.approx(4.8)  # 0.8 + 4


def test_calculate_cost_zero_tokens_returns_zero_no_warning(caplog):
    with caplog.at_level(logging.WARNING, logger="tokenjam.core.cost"):
        cost = calculate_cost("anthropic", "claude-haiku-4-5", 0, 0)
    assert cost == 0.0
    assert caplog.text == ""


def test_calculate_cost_rounds_to_8_decimal_places():
    # Use values that would produce more than 8 decimal places
    cost = calculate_cost("anthropic", "claude-haiku-4-5", 1, 1)
    # (1/1M * 0.80) + (1/1M * 4.00) = 0.0000008 + 0.000004 = 0.0000048
    assert cost == 0.0000048
    assert len(str(cost).split(".")[-1]) <= 8


def test_pricing_table_loads_without_error():
    table = load_pricing_table()
    assert isinstance(table, dict)
    assert len(table) > 0


def test_all_models_in_pricing_table_have_required_fields():
    table = load_pricing_table()
    for provider, models in table.items():
        for model_name, rates in models.items():
            assert rates.input_per_mtok >= 0, f"{provider}/{model_name} missing input rate"
            assert rates.output_per_mtok >= 0, f"{provider}/{model_name} missing output rate"


def test_get_rates_returns_none_for_unknown():
    assert get_rates("nonexistent", "model") is None


def test_get_rates_returns_model_rates_for_known():
    rates = get_rates("anthropic", "claude-haiku-4-5")
    assert rates is not None
    assert rates.input_per_mtok == 0.80
    assert rates.output_per_mtok == 4.00
    assert rates.cache_read_per_mtok == 0.08
    assert rates.cache_write_per_mtok == 1.00


def test_calculate_cost_opus_model():
    # claude-opus-4-6: input=5.00, output=25.00
    cost = calculate_cost("anthropic", "claude-opus-4-6", 1_000_000, 1_000_000)
    assert cost == 30.0


def test_calculate_cost_opus_4_8_model():
    # claude-opus-4-8: input=5.00, output=25.00
    cost = calculate_cost("anthropic", "claude-opus-4-8", 1_000_000, 1_000_000)
    assert cost == 30.0


def test_get_rates_opus_4_8():
    rates = get_rates("anthropic", "claude-opus-4-8")
    assert rates is not None
    assert rates.input_per_mtok == 5.00
    assert rates.output_per_mtok == 25.00
    assert rates.cache_read_per_mtok == 0.50
    assert rates.cache_write_per_mtok == 6.25


def test_calculate_cost_opus_4_5_model():
    # claude-opus-4-5: input=5.00, output=25.00 (same tier as 4.6/4.7/4.8)
    cost = calculate_cost("anthropic", "claude-opus-4-5", 1_000_000, 1_000_000)
    assert cost == 30.0


def test_get_rates_opus_4_5():
    rates = get_rates("anthropic", "claude-opus-4-5")
    assert rates is not None
    assert rates.input_per_mtok == 5.00
    assert rates.output_per_mtok == 25.00
    assert rates.cache_read_per_mtok == 0.50
    assert rates.cache_write_per_mtok == 6.25


def test_calculate_cost_openai_model():
    # gpt-4o: input=2.50, output=10.00
    cost = calculate_cost("openai", "gpt-4o", 500_000, 100_000)
    # (500k/1M * 2.50) + (100k/1M * 10.00) = 1.25 + 1.00 = 2.25
    assert cost == 2.25


def test_pricing_file_exists_at_expected_path():
    """Regression: PRICING_FILE must resolve to tokenjam/pricing/models.toml,
    not a path outside the package. A broken path causes $0.00 costs
    when installed via pip (non-editable). See v0.1.7 fix."""
    from tokenjam.core.pricing import PRICING_FILE
    assert PRICING_FILE.exists(), f"Pricing file not found at {PRICING_FILE}"
    assert "tokenjam" in PRICING_FILE.parts, (
        f"PRICING_FILE should be inside the tokenjam package, got {PRICING_FILE}"
    )
