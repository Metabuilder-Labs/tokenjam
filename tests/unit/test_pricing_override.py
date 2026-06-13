"""Unit tests for the user pricing override file in tokenjam.core.pricing.

Covers the TJ_PRICING_FILE env var, the default ~/.config/tj/pricing.toml
path, merge semantics over the packaged table, and graceful fallback when
the override is missing or malformed.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pytest

from tokenjam.core import pricing


@pytest.fixture(autouse=True)
def _isolate_pricing(monkeypatch, tmp_path):
    """Clear both pricing lru_caches around every test, point HOME at an empty
    dir so a real ~/.config/tj/pricing.toml can't leak in, and chdir into an
    empty dir so a project-local tj.toml/.tj/config.toml [pricing] section
    can't leak in either (config discovery walks the cwd)."""
    monkeypatch.delenv(pricing.USER_PRICING_ENV, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))          # POSIX
    monkeypatch.setenv("USERPROFILE", str(tmp_path))   # Windows Path.home()
    monkeypatch.chdir(tmp_path)
    pricing.clear_pricing_cache()
    yield
    pricing.clear_pricing_cache()


def _write(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")


def test_no_override_uses_packaged_rates():
    rates = pricing.get_rates("anthropic", "claude-haiku-4-5")
    assert rates is not None
    assert rates.input_per_mtok == 0.80


def test_env_override_adds_new_model(tmp_path, monkeypatch):
    f = tmp_path / "pricing.toml"
    _write(f, "[myprovider.custom-1]\ninput_per_mtok = 1.0\noutput_per_mtok = 2.0\n")
    monkeypatch.setenv(pricing.USER_PRICING_ENV, str(f))
    pricing.load_pricing_table.cache_clear()

    rates = pricing.get_rates("myprovider", "custom-1")
    assert rates is not None
    assert rates.input_per_mtok == 1.0
    assert rates.output_per_mtok == 2.0


def test_env_override_overrides_packaged_rate(tmp_path, monkeypatch):
    f = tmp_path / "pricing.toml"
    _write(
        f,
        "[anthropic.claude-haiku-4-5]\n"
        "input_per_mtok = 99.0\noutput_per_mtok = 100.0\n",
    )
    monkeypatch.setenv(pricing.USER_PRICING_ENV, str(f))
    pricing.load_pricing_table.cache_clear()

    rates = pricing.get_rates("anthropic", "claude-haiku-4-5")
    assert rates is not None
    assert rates.input_per_mtok == 99.0
    assert rates.output_per_mtok == 100.0


def test_env_override_leaves_unrelated_models_intact(tmp_path, monkeypatch):
    f = tmp_path / "pricing.toml"
    _write(f, "[anthropic.claude-haiku-4-5]\ninput_per_mtok = 99.0\noutput_per_mtok = 100.0\n")
    monkeypatch.setenv(pricing.USER_PRICING_ENV, str(f))
    pricing.load_pricing_table.cache_clear()

    # A model not mentioned in the override keeps its packaged rate.
    rates = pricing.get_rates("openai", "gpt-4o")
    assert rates is not None
    assert rates.input_per_mtok == 2.50


def test_default_user_path_is_used_when_present(tmp_path, monkeypatch):
    cfg = tmp_path / ".config" / "tj" / "pricing.toml"
    _write(cfg, "[anthropic.claude-haiku-4-5]\ninput_per_mtok = 0.01\noutput_per_mtok = 0.02\n")
    pricing.load_pricing_table.cache_clear()

    rates = pricing.get_rates("anthropic", "claude-haiku-4-5")
    assert rates is not None
    assert rates.input_per_mtok == 0.01


def test_missing_env_file_warns_and_falls_back(tmp_path, monkeypatch, caplog):
    monkeypatch.setenv(pricing.USER_PRICING_ENV, str(tmp_path / "nope.toml"))
    pricing.load_pricing_table.cache_clear()

    with caplog.at_level(logging.WARNING, logger="tokenjam.core.pricing"):
        rates = pricing.get_rates("anthropic", "claude-haiku-4-5")

    # Packaged rates still load.
    assert rates is not None
    assert rates.input_per_mtok == 0.80
    assert "not found" in caplog.text.lower()


def test_malformed_override_warns_and_falls_back(tmp_path, monkeypatch, caplog):
    f = tmp_path / "pricing.toml"
    _write(f, "this is not = valid = toml [[[")
    monkeypatch.setenv(pricing.USER_PRICING_ENV, str(f))
    pricing.load_pricing_table.cache_clear()

    with caplog.at_level(logging.WARNING, logger="tokenjam.core.pricing"):
        rates = pricing.get_rates("anthropic", "claude-haiku-4-5")

    assert rates is not None
    assert rates.input_per_mtok == 0.80
    assert "could not read" in caplog.text.lower()


# --- Model-keyed (attribution-proof) overrides — #200 ----------------------


def test_model_keyed_override_wins_for_unknown_provider(tmp_path, monkeypatch):
    """A bare-model-name override prices a span whose provider resolved to
    "unknown" — the exact #194 class the packaged [provider.model] table
    can't reach."""
    f = tmp_path / "pricing.toml"
    _write(
        f,
        "[models]\n"
        '"claude-haiku-4-5" = { input_per_mtok = 7.0, output_per_mtok = 9.0, '
        "cache_read_per_mtok = 1.0, cache_write_per_mtok = 2.0 }\n",
    )
    monkeypatch.setenv(pricing.USER_PRICING_ENV, str(f))
    pricing.clear_pricing_cache()

    rates = pricing.get_rates("unknown", "claude-haiku-4-5")
    assert rates is not None
    assert rates.input_per_mtok == 7.0
    assert rates.output_per_mtok == 9.0
    assert rates.cache_read_per_mtok == 1.0
    assert rates.cache_write_per_mtok == 2.0


def test_model_keyed_override_prices_unattributed_span_via_factory(tmp_path, monkeypatch):
    """End-to-end: a factory-built span with provider="unknown" costs out at
    the user-declared model-keyed rate instead of the $0.50/$2.00 default."""
    from tokenjam.core.cost import calculate_cost
    from tests.factories import make_llm_span

    f = tmp_path / "pricing.toml"
    _write(f, '[models]\n"my-local-model" = { input_per_mtok = 3.0, output_per_mtok = 6.0 }\n')
    monkeypatch.setenv(pricing.USER_PRICING_ENV, str(f))
    pricing.clear_pricing_cache()

    span = make_llm_span(
        provider="unknown",
        billing_account=None,
        model="my-local-model",
        input_tokens=1_000_000,
        output_tokens=1_000_000,
        cache_tokens=0,
        cache_write_tokens=0,
    )
    cost = calculate_cost(
        span.provider,
        span.model,
        span.input_tokens,
        span.output_tokens,
        span.cache_tokens,
        span.cache_write_tokens,
    )
    # 1M input @ $3 + 1M output @ $6 = $9.00 — not the default 0.50 + 2.00.
    assert cost == 9.0


def test_model_keyed_override_wins_over_packaged_provider_rate(tmp_path, monkeypatch):
    """Even when the provider is correctly attributed, a user's model-keyed
    declaration outranks the packaged table — the user pays a negotiated rate
    they alone know."""
    f = tmp_path / "pricing.toml"
    _write(f, '[models]\n"claude-haiku-4-5" = { input_per_mtok = 0.01, output_per_mtok = 0.02 }\n')
    monkeypatch.setenv(pricing.USER_PRICING_ENV, str(f))
    pricing.clear_pricing_cache()

    rates = pricing.get_rates("anthropic", "claude-haiku-4-5")
    assert rates is not None
    assert rates.input_per_mtok == 0.01  # packaged is 0.80


def test_model_keyed_override_honors_date_suffix(tmp_path, monkeypatch):
    """A model-keyed override pins the base name and still prices the dated
    `-YYYYMMDD` variant Anthropic/OpenAI ship."""
    f = tmp_path / "pricing.toml"
    _write(f, '[models]\n"claude-haiku-4-5" = { input_per_mtok = 5.0, output_per_mtok = 5.0 }\n')
    monkeypatch.setenv(pricing.USER_PRICING_ENV, str(f))
    pricing.clear_pricing_cache()

    rates = pricing.get_rates("unknown", "claude-haiku-4-5-20251001")
    assert rates is not None
    assert rates.input_per_mtok == 5.0


def test_precedence_model_keyed_then_provider_then_packaged(tmp_path, monkeypatch):
    """Full precedence chain: model-keyed > [provider.model] > packaged."""
    f = tmp_path / "pricing.toml"
    _write(
        f,
        # Provider-keyed override of the packaged anthropic rate.
        "[anthropic]\n"
        "claude-haiku-4-5 = { input_per_mtok = 50.0, output_per_mtok = 50.0 }\n"
        # Model-keyed override (reserved [models] section) — should outrank it.
        # Order doesn't matter: the sections are explicit and disjoint.
        "[models]\n"
        '"claude-haiku-4-5" = { input_per_mtok = 1.0, output_per_mtok = 1.0 }\n',
    )
    monkeypatch.setenv(pricing.USER_PRICING_ENV, str(f))
    pricing.clear_pricing_cache()

    # Known provider: model-keyed (1.0) beats provider-keyed (50.0) beats packaged (0.80).
    assert pricing.get_rates("anthropic", "claude-haiku-4-5").input_per_mtok == 1.0
    # Unknown provider: only the model-keyed entry can match.
    assert pricing.get_rates("unknown", "claude-haiku-4-5").input_per_mtok == 1.0
    # A provider-keyed-only model stays reachable when the provider matches.
    assert pricing.get_rates("openai", "gpt-4o").input_per_mtok == 2.50


# --- [pricing] section in the main config (tj.toml) — #200 -----------------


def _write_main_config(cwd, body: str) -> None:
    """Write a project-local tokenjam.toml (first on the config search path)."""
    (cwd / "tokenjam.toml").write_text('version = "1"\n' + body, encoding="utf-8")


def test_config_pricing_section_model_keyed(tmp_path, monkeypatch):
    """A [pricing] section in tj.toml supports model-keyed overrides too."""
    _write_main_config(
        tmp_path,
        '[pricing.models]\n"claude-haiku-4-5" = { input_per_mtok = 0.05, output_per_mtok = 0.06 }\n',
    )
    pricing.clear_pricing_cache()

    rates = pricing.get_rates("unknown", "claude-haiku-4-5")
    assert rates is not None
    assert rates.input_per_mtok == 0.05


def test_config_pricing_section_provider_keyed(tmp_path, monkeypatch):
    """A [pricing.<provider>] section overrides the packaged provider rate."""
    _write_main_config(
        tmp_path,
        "[pricing.anthropic]\n"
        "claude-haiku-4-5 = { input_per_mtok = 0.07, output_per_mtok = 0.08 }\n",
    )
    pricing.clear_pricing_cache()

    rates = pricing.get_rates("anthropic", "claude-haiku-4-5")
    assert rates is not None
    assert rates.input_per_mtok == 0.07


def test_config_pricing_merges_with_user_file_and_wins(tmp_path, monkeypatch):
    """tj.toml [pricing] and the user pricing file merge; on a shared key the
    project-local tj.toml wins, while each source's unique entries survive."""
    user_file = tmp_path / "pricing.toml"
    _write(
        user_file,
        "[models]\n"
        '"shared-model" = { input_per_mtok = 1.0, output_per_mtok = 1.0 }\n'
        '"file-only-model" = { input_per_mtok = 2.0, output_per_mtok = 2.0 }\n',
    )
    monkeypatch.setenv(pricing.USER_PRICING_ENV, str(user_file))
    _write_main_config(
        tmp_path,
        "[pricing.models]\n"
        '"shared-model" = { input_per_mtok = 9.0, output_per_mtok = 9.0 }\n'
        '"config-only-model" = { input_per_mtok = 3.0, output_per_mtok = 3.0 }\n',
    )
    pricing.clear_pricing_cache()

    # Shared key: tj.toml [pricing] wins over the user file.
    assert pricing.get_rates("unknown", "shared-model").input_per_mtok == 9.0
    # Each source's unique entries survive the merge.
    assert pricing.get_rates("unknown", "file-only-model").input_per_mtok == 2.0
    assert pricing.get_rates("unknown", "config-only-model").input_per_mtok == 3.0
