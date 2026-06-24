"""Unit tests for the policy engine + envelope (#220) — HTTP-free.

Covers: the engine evaluates a registered policy; the api-only guard (observe-
only traffic never reaches the engine); suggest-mode records but enforces
nothing; the `unvalidated` label is always present; the enforce path is
scaffolded but gated off; and the envelope round-trips.
"""
from __future__ import annotations

import pytest

from tokenjam.core.config import PolicyConfig, ProviderBudget, TjConfig
from tokenjam.proxy.engine import (
    ACTION_NOOP,
    ACTION_WOULD_BLOCK,
    ENFORCE_GATE_OPEN,
    UNVALIDATED_LABEL,
    PolicyEngine,
    PolicyEnvelope,
    PolicyEvaluation,
    PolicyGuardError,
    PolicyOutcome,
    PolicyRequest,
    register_policy,
)
from tokenjam.proxy.gate import OBSERVE_ONLY, POLICY, GateDecision, classify


# --- a stub policy kind the tests register (no concrete logic ships) ---
@register_policy("test_block")
def _test_block(policy, request):
    return PolicyOutcome(
        would_action=ACTION_WOULD_BLOCK,
        reason="stub: would block",
        details={"seen_provider": request.provider},
    )


def _api_config(policies):
    return TjConfig(
        version="1",
        budgets={"openai": ProviderBudget(plan="api"),
                 "anthropic": ProviderBudget(plan="api")},
        policies=policies,
    )


def _api_gate(config, provider="openai") -> GateDecision:
    gd = classify(config, provider)
    assert gd.path == POLICY  # precondition: api/usage-billed
    return gd


def test_engine_evaluates_registered_policy():
    cfg = _api_config([PolicyConfig(name="blocker", kind="test_block")])
    engine = PolicyEngine.from_config(cfg)
    env = engine.evaluate(_api_gate(cfg), PolicyRequest(provider="openai", path="/v1/chat/completions"))

    assert len(env.evaluations) == 1
    ev = env.evaluations[0]
    assert ev.policy_name == "blocker"
    assert ev.kind == "test_block"
    assert ev.would_action == ACTION_WOULD_BLOCK
    assert ev.details["seen_provider"] == "openai"
    # Overall action summarises to the strongest would-action.
    assert env.overall_action == ACTION_WOULD_BLOCK


def test_api_only_guard_rejects_observe_only():
    # The engine must NEVER evaluate observe-only traffic (#219 invariant,
    # belt-and-suspenders with the gate).
    sub_cfg = TjConfig(version="1", budgets={"anthropic": ProviderBudget(plan="max_5x")})
    observe_gd = classify(sub_cfg, "anthropic")
    assert observe_gd.path == OBSERVE_ONLY

    engine = PolicyEngine.from_config(_api_config([PolicyConfig(name="b", kind="test_block")]))
    with pytest.raises(PolicyGuardError):
        engine.evaluate(observe_gd, PolicyRequest(provider="anthropic", path="/v1/messages"))


def test_api_only_guard_rejects_any_non_policy_path():
    # Even a hand-built decision with a bogus path is refused.
    engine = PolicyEngine.from_config(_api_config([]))
    bogus = GateDecision(provider="openai", plan_tier="api", pricing_mode="api",
                         path="something_else", reason="x")
    with pytest.raises(PolicyGuardError):
        engine.evaluate(bogus, PolicyRequest(provider="openai", path="/v1/chat/completions"))


def test_suggest_mode_never_enforces():
    cfg = _api_config([PolicyConfig(name="blocker", kind="test_block", mode="suggest")])
    env = PolicyEngine.from_config(cfg).evaluate(
        _api_gate(cfg), PolicyRequest(provider="openai", path="/v1/chat/completions"))
    # Records what it WOULD do, but enforces nothing.
    assert env.overall_action == ACTION_WOULD_BLOCK
    assert env.enforced is False
    assert env.suggest_only is True


def test_enforce_mode_is_scaffolded_but_gated_off():
    # A mode=enforce policy is recorded as enforcement-eligible but NEVER acts in
    # the OSS rails (the cert gate is closed).
    assert ENFORCE_GATE_OPEN is False
    cfg = _api_config([PolicyConfig(name="blocker", kind="test_block", mode="enforce")])
    env = PolicyEngine.from_config(cfg).evaluate(
        _api_gate(cfg), PolicyRequest(provider="openai", path="/v1/chat/completions"))
    assert env.enforced is False
    assert env.enforcement_gated is True
    assert env.evaluations[0].enforcement_gated is True


def test_unvalidated_label_always_present():
    cfg = _api_config([PolicyConfig(name="b", kind="test_block")])
    env = PolicyEngine.from_config(cfg).evaluate(
        _api_gate(cfg), PolicyRequest(provider="openai", path="/v1/chat/completions"))
    assert env.label == UNVALIDATED_LABEL
    assert env.validated is False


def test_target_provider_filters_applicable_policies():
    # A policy targeting anthropic does not apply to an openai request.
    cfg = _api_config([PolicyConfig(name="anthro-only", kind="test_block",
                                    target_provider="anthropic")])
    env = PolicyEngine.from_config(cfg).evaluate(
        _api_gate(cfg, "openai"), PolicyRequest(provider="openai", path="/v1/chat/completions"))
    assert env.evaluations == []
    assert env.overall_action == ACTION_NOOP


def test_disabled_policy_is_skipped():
    cfg = _api_config([PolicyConfig(name="off", kind="test_block", enabled=False)])
    engine = PolicyEngine.from_config(cfg)
    assert engine.policies == []  # filtered at load


def test_unknown_kind_is_reported_not_raised():
    cfg = _api_config([PolicyConfig(name="mystery", kind="does_not_exist")])
    env = PolicyEngine.from_config(cfg).evaluate(
        _api_gate(cfg), PolicyRequest(provider="openai", path="/v1/chat/completions"))
    assert env.evaluations[0].would_action == "error"
    assert "unknown policy kind" in env.evaluations[0].reason


def test_envelope_round_trip():
    cfg = _api_config([PolicyConfig(name="blocker", kind="test_block", mode="enforce")])
    env = PolicyEngine.from_config(cfg).evaluate(
        _api_gate(cfg), PolicyRequest(provider="openai", path="/v1/chat/completions"))
    restored = PolicyEnvelope.from_dict(env.to_dict())
    assert restored.to_dict() == env.to_dict()
    assert isinstance(restored.evaluations[0], PolicyEvaluation)
    assert restored.label == UNVALIDATED_LABEL


def test_noop_example_kind_ships_and_evaluates():
    # The built-in `noop` reference kind works without tests registering anything.
    cfg = _api_config([PolicyConfig(name="ex", kind="noop")])
    env = PolicyEngine.from_config(cfg).evaluate(
        _api_gate(cfg), PolicyRequest(provider="openai", path="/v1/chat/completions"))
    assert env.evaluations[0].would_action == ACTION_NOOP
    assert env.overall_action == ACTION_NOOP
