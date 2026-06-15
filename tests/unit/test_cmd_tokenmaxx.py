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
    # Zero spend → Sipper, regardless of which path.
    assert _classify(0.0).label == "TokenSipper"
    assert _classify(0.0, multiplier=0.0).label == "TokenSipper"


def test_classify_multiplier_path_walks_tier_boundaries():
    # Multiplier-based classification — the primary path for subscription users.
    # Six tiers, boundaries at 1× / 4× / 10× / 20× / 50×.
    cases = [
        (0.99,  "TokenSipper"),
        (1.0,   "TokenModerator"),
        (3.99,  "TokenModerator"),
        (4.0,   "TokenMaxxer"),
        (9.99,  "TokenMaxxer"),
        (10.0,  "TokenSuperMaxxer"),
        (19.99, "TokenSuperMaxxer"),
        (20.0,  "TokenMegaMaxxer"),
        (49.99, "TokenMegaMaxxer"),
        (50.0,  "TokenGigaMaxxer"),
        (200.0, "TokenGigaMaxxer"),
    ]
    for mult, expected in cases:
        assert _classify(0.0, multiplier=mult).label == expected, f"failed at {mult}×"


def test_classify_absolute_path_for_api_users_walks_tier_boundaries():
    # Absolute USD/mo fallback — calibrated against Max-5x = $100/mo so the
    # tier names mean roughly the same thing in both paths.
    cases = [
        (99.99,   "TokenSipper"),
        (100.0,   "TokenModerator"),
        (399.99,  "TokenModerator"),
        (400.0,   "TokenMaxxer"),
        (999.99,  "TokenMaxxer"),
        (1000.0,  "TokenSuperMaxxer"),
        (1999.99, "TokenSuperMaxxer"),
        (2000.0,  "TokenMegaMaxxer"),
        (4999.99, "TokenMegaMaxxer"),
        (5000.0,  "TokenGigaMaxxer"),
        (50_000,  "TokenGigaMaxxer"),
    ]
    for spend, expected in cases:
        assert _classify(spend).label == expected, f"failed at ${spend}"


def test_classify_multiplier_overrides_absolute_when_both_provided():
    # A subscription user at $50/mo on a $20 Pro plan (2.5× their plan) is
    # a TokenModerator, NOT a TokenSipper — the multiplier path wins.
    t = _classify(50.0, multiplier=2.5)
    assert t.label == "TokenModerator"


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


def test_all_anthropic_subscription_plans_produce_a_multiplier():
    # Every Anthropic subscription plan in the onboard wizard must produce
    # a finite multiplier, otherwise the tokenmaxx tweet hook ("Nx your
    # plan") doesn't render for those users. This guards against the table
    # diverging from `cmd_onboard.py::_ANTHROPIC_PLAN_CHOICES`.
    for plan in ("pro", "max_5x", "max_20x"):
        info = _plan_label_and_fee(plan)
        assert info is not None, f"{plan!r} not in plan-fee table"
        label, fee = info
        assert label, f"{plan!r} missing a display label"
        assert fee and fee > 0, f"{plan!r} missing a numeric monthly fee"
        # Sanity-check the multiplier math the renderer does.
        spend = 1000.0
        assert spend / fee > 0
