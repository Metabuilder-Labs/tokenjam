"""Unit tests for the `tj route` static-export generators (ccr + litellm)."""
from __future__ import annotations

import json

import pytest
import yaml

from tokenjam.core.export.ccr import render_ccr_config
from tokenjam.core.export.litellm import render_litellm_config
from tokenjam.core.optimize.types import (
    MODEL_DOWNGRADE_CAVEAT,
    DowngradeExample,
    DowngradeFinding,
)


def _sample_finding() -> DowngradeFinding:
    return DowngradeFinding(
        candidate_sessions=8,
        total_sessions=17,
        actual_cost_usd=12.34,
        alternative_cost_usd=2.10,
        monthly_savings_usd=42.50,
        percent_of_sessions=47.0,
        examples=[DowngradeExample(
            trace_id="0102030405060708",
            session_id="sess-x",
            model="claude-opus-4-7",
            tool_calls=2,
            duration_seconds=12.3,
            cost_usd=4.0,
        )],
        suggestions={"claude-opus-4-7": "claude-haiku-4-5"},
        candidate_tokens=350_000,
        window_total_tokens=900_000,
        percent_of_tokens=38.9,
        monthly_tokens_in_candidates=1_400_000,
    )


def _strip_jsonc(body: str) -> str:
    return "\n".join(
        line for line in body.splitlines() if not line.lstrip().startswith("//")
    )


# --- validity --------------------------------------------------------------

def test_ccr_emits_valid_jsonc():
    body = render_ccr_config(
        downgrade=_sample_finding(), pricing_mode="api", plan_tier="api",
        since="30d", until="2026-06-22",
    )
    doc = json.loads(_strip_jsonc(body))
    rec = doc["tokenjam"]["routing_recommendations"]
    assert rec["target"] == "claude-code-router"
    assert rec["rules"][0]["match"]["original_model"] == "claude-opus-4-7"
    assert rec["rules"][0]["suggested_model"] == "claude-haiku-4-5"
    assert rec["rules"][0]["evidence"] == "L1"


def test_litellm_emits_valid_yaml():
    body = render_litellm_config(
        downgrade=_sample_finding(), pricing_mode="api", plan_tier="api",
        since="30d", until="2026-06-22",
    )
    doc = yaml.safe_load(body)
    rec = doc["tokenjam_routing_recommendations"]
    assert rec["target"] == "litellm"
    assert rec["rules"][0]["match"]["original_model"] == "claude-opus-4-7"
    assert rec["rules"][0]["suggested_model"] == "claude-haiku-4-5"
    assert rec["rules"][0]["evidence"] == "L1"


# --- honesty discipline (Rule 14) ------------------------------------------

@pytest.mark.parametrize("render", [render_ccr_config, render_litellm_config])
def test_embeds_caveat_evidence_window_and_derived_at(render):
    body = render(
        downgrade=_sample_finding(), pricing_mode="api", plan_tier="api",
        since="30d", until="2026-06-22",
    )
    # MODEL_DOWNGRADE_CAVEAT verbatim
    assert MODEL_DOWNGRADE_CAVEAT in body
    # evidence level per rule + uniform L1
    assert "evidence: L1 — structural match, review before applying" in body
    assert "Evidence level: L1" in body
    # derivation window + derived-at
    assert "Derivation window: 30d -> 2026-06-22" in body
    assert "Derived at:" in body
    # advisory framing, and never the word "safe" except in the "never safe" caveat
    assert "review before applying" in body
    assert "STRUCTURAL HEURISTIC ONLY" in body
    assert "TokenJam does not enforce" in body
    lowered = body.lower()
    # the only permitted occurrence of "safe" is the disclaimer 'never "safe"'
    assert lowered.count("safe") == lowered.count('never "safe"')


@pytest.mark.parametrize("render", [render_ccr_config, render_litellm_config])
def test_never_claims_safe_or_equivalent(render):
    body = render(
        downgrade=_sample_finding(), pricing_mode="api", plan_tier="api",
        since="30d", until="2026-06-22",
    )
    assert "safe to downgrade" not in body.lower()
    assert "would have worked" not in body.lower()
    assert "quality equivalent" not in body.lower()


# --- plan-tier-aware figure selection --------------------------------------

@pytest.mark.parametrize("render", [render_ccr_config, render_litellm_config])
def test_api_user_carries_usd_savings(render):
    body = render(
        downgrade=_sample_finding(), pricing_mode="api", plan_tier="api",
        since="30d", until="2026-06-22",
    )
    assert "estimated_savings_usd_month" in body
    assert "42.5" in body
    assert "estimated_tokens_freed" not in body


@pytest.mark.parametrize("render", [render_ccr_config, render_litellm_config])
def test_subscription_user_carries_tokens_freed(render):
    body = render(
        downgrade=_sample_finding(), pricing_mode="subscription", plan_tier="max_20x",
        since="30d", until="2026-06-22",
    )
    assert "estimated_tokens_freed" in body
    assert "1400000" in body
    assert "estimated_savings_usd_month" not in body


@pytest.mark.parametrize("render", [render_ccr_config, render_litellm_config])
def test_unknown_plan_carries_reconfigure_note(render):
    body = render(
        downgrade=_sample_finding(), pricing_mode="unknown", plan_tier="unknown",
        since="30d", until="2026-06-22",
    )
    assert "tj onboard --reconfigure" in body
    assert "estimated_savings_usd_month" not in body
    assert "estimated_tokens_freed" not in body


# --- empty-finding case -----------------------------------------------------

def test_ccr_empty_finding_still_valid_and_carries_caveat():
    body = render_ccr_config(
        downgrade=None, pricing_mode="api", plan_tier="api",
        since="7d", until="2026-06-22",
    )
    doc = json.loads(_strip_jsonc(body))
    assert doc["tokenjam"]["routing_recommendations"]["rules"] == []
    assert MODEL_DOWNGRADE_CAVEAT in body


def test_litellm_empty_finding_still_valid_and_carries_caveat():
    body = render_litellm_config(
        downgrade=None, pricing_mode="api", plan_tier="api",
        since="7d", until="2026-06-22",
    )
    doc = yaml.safe_load(body)
    assert doc["tokenjam_routing_recommendations"]["rules"] == []
    assert MODEL_DOWNGRADE_CAVEAT in body


@pytest.mark.parametrize("render", [render_ccr_config, render_litellm_config])
def test_agent_scope_recorded(render):
    body = render(
        downgrade=_sample_finding(), pricing_mode="api", plan_tier="api",
        since="30d", until="2026-06-22", agent_id="my-agent",
    )
    assert "agent_id=my-agent" in body
