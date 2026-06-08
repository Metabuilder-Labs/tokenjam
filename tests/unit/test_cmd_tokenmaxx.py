"""Unit tests for `tj tokenmaxx`."""
from __future__ import annotations

from tokenjam.cli.cmd_tokenmaxx import (
    Tier,
    _TIERS,
    _classify,
    _config_declared_plan,
    _plan_label_and_fee,
)
from tokenjam.core.config import ProviderBudget, TjConfig


# ───────────────────────────── classification ─────────────────────────────

def test_classify_zero_spend_is_sipper():
    t = _classify(0.0)
    assert t.label == "TokenSipper"


def test_classify_walks_tiers_correctly():
    # Each threshold should map into the next tier exactly at the boundary.
    cases = [
        (49.99,   "TokenSipper"),
        (50.0,    "TokenModerator"),
        (199.99,  "TokenModerator"),
        (200.0,   "TokenMaxxer"),
        (499.99,  "TokenMaxxer"),
        (500.0,   "TokenChad"),
        (1499.99, "TokenChad"),
        (1500.0,  "TokenGigaChad"),
        (50_000,  "TokenGigaChad"),
    ]
    for spend, expected in cases:
        assert _classify(spend).label == expected, f"failed at ${spend}"


def test_every_tier_has_emoji_and_quip():
    # The artifact's social-shareability depends on every tier having
    # readable text — guard against a future blank entry.
    for tier in _TIERS:
        assert isinstance(tier, Tier)
        assert tier.label
        assert tier.emoji
        assert tier.quip


# ───────────────────────────── config helpers ─────────────────────────────

def test_config_declared_plan_none_when_no_budgets():
    cfg = TjConfig(version="1")
    assert _config_declared_plan(cfg) is None


def test_config_declared_plan_returns_subscription_tier():
    cfg = TjConfig(version="1")
    cfg.budgets["anthropic"] = ProviderBudget(plan="max_5x")
    assert _config_declared_plan(cfg) == "max_5x"


def test_config_declared_plan_prefers_first_sorted_provider():
    cfg = TjConfig(version="1")
    cfg.budgets["openai"] = ProviderBudget(plan="plus")
    cfg.budgets["anthropic"] = ProviderBudget(plan="max_20x")
    # alphabetical: anthropic < openai
    assert _config_declared_plan(cfg) == "max_20x"


def test_config_declared_plan_skips_providers_without_plan_field():
    cfg = TjConfig(version="1")
    cfg.budgets["anthropic"] = ProviderBudget(usd=200.0)  # no plan
    cfg.budgets["openai"] = ProviderBudget(plan="plus")
    assert _config_declared_plan(cfg) == "plus"


# ───────────────────────────── plan table ─────────────────────────────────

def test_plan_label_and_fee_known_subscription_tiers():
    assert _plan_label_and_fee("max_5x") == ("Max 5x plan", 100.0)
    assert _plan_label_and_fee("max_20x") == ("Max 20x plan", 200.0)
    assert _plan_label_and_fee("pro") == ("Pro plan", 20.0)
    assert _plan_label_and_fee("plus") == ("ChatGPT Plus", 20.0)


def test_plan_label_and_fee_contract_priced_tiers_have_no_fee():
    label, fee = _plan_label_and_fee("team")
    assert label == "ChatGPT Team"
    assert fee is None


def test_plan_label_and_fee_returns_none_for_unknown_or_api():
    # 'api' and 'local' deliberately not in the table — the renderer
    # doesn't quote a "plan cost multiplier" against pay-per-use plans.
    assert _plan_label_and_fee("api") is None
    assert _plan_label_and_fee("local") is None
    assert _plan_label_and_fee(None) is None
    assert _plan_label_and_fee("not_a_plan") is None
