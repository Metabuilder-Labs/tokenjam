"""Unit tests for ocw.core.cost and ocw.core.pricing."""
from __future__ import annotations
import logging

from tj.core.cost import calculate_cost
from tj.core.pricing import load_pricing_table, get_rates


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
    # Zero input/output but cache_write tokens present — still returns 0
    # because input_tokens == 0 and output_tokens == 0 triggers early return
    assert cost == 0.0


def test_calculate_cost_unknown_model_uses_default(caplog):
    with caplog.at_level(logging.WARNING, logger="tj.core.cost"):
        cost = calculate_cost("unknown_provider", "unknown_model", 1_000_000, 1_000_000)
    # Default rates: 0.50 input, 2.00 output per MTok
    # (1M/1M * 0.50) + (1M/1M * 2.00) = 2.50
    assert cost == 2.5
    assert "No pricing data for unknown_provider/unknown_model" in caplog.text


def test_calculate_cost_zero_tokens_returns_zero_no_warning(caplog):
    with caplog.at_level(logging.WARNING, logger="tj.core.cost"):
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
    # claude-opus-4-6: input=15.00, output=75.00
    cost = calculate_cost("anthropic", "claude-opus-4-6", 1_000_000, 1_000_000)
    assert cost == 90.0


def test_calculate_cost_openai_model():
    # gpt-4o: input=2.50, output=10.00
    cost = calculate_cost("openai", "gpt-4o", 500_000, 100_000)
    # (500k/1M * 2.50) + (100k/1M * 10.00) = 1.25 + 1.00 = 2.25
    assert cost == 2.25


def test_pricing_file_exists_at_expected_path():
    """Regression: PRICING_FILE must resolve to ocw/pricing/models.toml,
    not a path outside the package. A broken path causes $0.00 costs
    when installed via pip (non-editable). See v0.1.7 fix."""
    from tj.core.pricing import PRICING_FILE
    assert PRICING_FILE.exists(), f"Pricing file not found at {PRICING_FILE}"
    assert "tj" in PRICING_FILE.parts, (
        f"PRICING_FILE should be inside the tj package, got {PRICING_FILE}"
    )
