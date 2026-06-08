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
    """Clear the lru_cache around every test and point HOME at an empty dir
    so a real ~/.config/tj/pricing.toml on the dev machine can't leak in."""
    monkeypatch.delenv(pricing.USER_PRICING_ENV, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))          # POSIX
    monkeypatch.setenv("USERPROFILE", str(tmp_path))   # Windows Path.home()
    pricing.load_pricing_table.cache_clear()
    yield
    pricing.load_pricing_table.cache_clear()


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
