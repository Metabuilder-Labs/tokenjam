"""Unit tests for the proxy pricing-mode gate — the substrate invariant (#219).

The invariant: resolve pricing mode FIRST; subscription AND unknown (fail-safe)
AND local are forwarded unmodified (observe-only); ONLY api/usage-billed reaches
the policy path. The killswitch forces observe-only regardless.

These configs declare a budget for an UNRELATED provider where the "no plan for
this provider" case is exercised, so ``_declared_budget_plans`` does not fall
back to reading the developer's real global ``~/.config/tj/config.toml`` (#106).
"""
from __future__ import annotations

import pytest

from tokenjam.core.config import ProviderBudget, TjConfig
from tokenjam.proxy.gate import OBSERVE_ONLY, POLICY, classify


def _config(provider_plans: dict[str, str | None]) -> TjConfig:
    budgets = {p: ProviderBudget(plan=plan) for p, plan in provider_plans.items()}
    return TjConfig(version="1", budgets=budgets)


def test_api_plan_reaches_policy_path():
    cfg = _config({"anthropic": "api"})
    d = classify(cfg, "anthropic")
    assert d.pricing_mode == "api"
    assert d.path == POLICY
    assert not d.observe_only


@pytest.mark.parametrize("plan", ["pro", "max_5x", "max_20x", "plus", "team", "enterprise"])
def test_subscription_plans_are_observe_only(plan):
    cfg = _config({"anthropic": plan})
    d = classify(cfg, "anthropic")
    assert d.pricing_mode == "subscription"
    assert d.path == OBSERVE_ONLY  # TOS: never a policy decision
    assert d.observe_only


def test_local_plan_is_observe_only():
    cfg = _config({"anthropic": "local"})
    d = classify(cfg, "anthropic")
    assert d.pricing_mode == "local"
    assert d.path == OBSERVE_ONLY


def test_unknown_is_failsafe_observe_only():
    # The config declares a plan for openai only — so anthropic has no declared
    # plan and resolves to "unknown" WITHOUT hitting the global-config fallback
    # (budgets is non-empty). Unknown must fail safe to observe-only.
    cfg = _config({"openai": "api"})
    d = classify(cfg, "anthropic")
    assert d.plan_tier is None
    assert d.pricing_mode == "unknown"
    assert d.path == OBSERVE_ONLY


def test_unrecognised_provider_is_observe_only():
    cfg = _config({"anthropic": "api"})
    d = classify(cfg, "some-other-provider")
    assert d.pricing_mode == "unknown"
    assert d.path == OBSERVE_ONLY


def test_killswitch_forces_observe_only_even_for_api():
    cfg = _config({"anthropic": "api"})
    d = classify(cfg, "anthropic", killswitch=True)
    assert d.pricing_mode == "api"          # mode is still resolved + inspectable
    assert d.path == OBSERVE_ONLY           # but the path is forced to passthrough
    assert d.killswitch is True
    assert d.reason == "killswitch_passthrough"


def test_decision_is_inspectable():
    # The whole point: the invariant is a plain, inspectable value object.
    cfg = _config({"openai": "api"})
    d = classify(cfg, "openai")
    assert d.provider == "openai"
    assert d.plan_tier == "api"
    assert d.path == POLICY
    assert d.reason == "api_usage_billed"
