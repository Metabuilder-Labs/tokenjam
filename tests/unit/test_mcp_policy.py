"""MCP policy tools + self-observation spans (#223).

Covers: the three MCP tools return correct shapes; get_savings_summary is
labeled estimated / unvalidated and never says "saved"; suggest_policies frames
output as suggestions (not validated-safe); the api-only invariant flows through
(observe-only traffic accrues no savings); and the proxy self-observation span
carries `tokenjam.policy.*` attributes from semconv (never hardcoded strings).
"""
from __future__ import annotations

import json

from tokenjam.core.config import PolicyConfig, ProviderBudget, TjConfig
from tokenjam.core.db import InMemoryBackend
from tokenjam.mcp.server import (
    _tool_get_policy_status,
    _tool_get_savings_summary,
    _tool_suggest_policies,
)
from tokenjam.otel.semconv import TjAttributes
from tokenjam.proxy.audit import AuditSink, policy_decision_span
from tokenjam.proxy.engine import (
    ACTION_WOULD_BLOCK,
    PolicyEnvelope,
    PolicyEvaluation,
)
from tokenjam.proxy.gate import OBSERVE_ONLY, POLICY, GateDecision
from tokenjam.proxy.observer import ProxyObserver
from tokenjam.utils.time_parse import utcnow
from tests.factories import make_llm_span, make_session


def _config():
    return TjConfig(
        version="1",
        budgets={"openai": ProviderBudget(plan="api")},
        policies=[PolicyConfig(name="cap", kind="budget_cap")],
    )


def _policy_gate():
    return GateDecision(provider="openai", plan_tier="api", pricing_mode="api",
                        path=POLICY, reason="api_usage_billed")


def _observe_gate():
    return GateDecision(provider="anthropic", plan_tier="max_5x",
                        pricing_mode="subscription", path=OBSERVE_ONLY,
                        reason="subscription_passthrough")


def _envelope(usd=0.0):
    return PolicyEnvelope(
        ts=utcnow().isoformat(), provider="openai", path="/v1/chat/completions",
        agent=None, gate_path=POLICY, overall_action=ACTION_WOULD_BLOCK,
        evaluations=[PolicyEvaluation(
            policy_name="cap", kind="budget_cap", mode="suggest",
            would_action=ACTION_WOULD_BLOCK, reason="over cap",
            enforcement_gated=False,
            details={"estimated_recoverable_usd": usd} if usd else {})],
    )


def _seed(db, *, actual_spend=10.0, recoverable=2.0):
    sess = make_session(agent_id="codegen", total_cost_usd=actual_spend)
    db.upsert_session(sess)
    db.insert_span(make_llm_span(agent_id="codegen", provider="openai",
                                 model="gpt-4o", cost_usd=actual_spend,
                                 session_id=sess.session_id))
    obs = ProxyObserver(sink=AuditSink(db))
    obs.record(method="POST", path="/v1/chat/completions",
               decision=_policy_gate(), envelope=_envelope(recoverable))


# --- get_policy_status ---

def test_policy_status_shape():
    db = InMemoryBackend()
    _seed(db)
    out = _tool_get_policy_status(db, _config())
    assert out["suggest_mode"] is True
    assert out["enforced"] is False
    assert out["label"] == "unvalidated"
    assert len(out["policies"]) == 1
    assert out["policies"][0]["name"] == "cap"
    assert len(out["recent_decisions"]) == 1
    assert out["recent_decisions"][0]["would_action"] == ACTION_WOULD_BLOCK
    db.close()


def test_policy_status_no_config():
    db = InMemoryBackend()
    assert "error" in _tool_get_policy_status(db, None)
    db.close()


# --- get_savings_summary ---

def test_savings_summary_estimated_never_realized():
    db = InMemoryBackend()
    _seed(db, actual_spend=10.0, recoverable=2.0)
    out = _tool_get_savings_summary(db)
    assert out["estimated_recoverable_usd"] == 2.0
    assert out["actual_spend_usd"] == 10.0
    assert out["estimated_recoverable_pct"] == 20.0  # 2/10
    assert out["realized"] is False
    assert out["label"] == "unvalidated"
    # Honesty: never present as money saved.
    non_disclaimer = {k: v for k, v in out.items() if k != "disclaimer"}
    assert "saved" not in json.dumps(non_disclaimer).lower()
    assert "not realized" in out["disclaimer"].lower()


def test_savings_summary_api_only_observe_only_accrues_nothing():
    # The api-only invariant flows through: observe-only (subscription) traffic
    # reaches the audit log but NEVER the savings ledger, so the meter is zero.
    db = InMemoryBackend()
    obs = ProxyObserver(sink=AuditSink(db))
    obs.record(method="POST", path="/v1/messages",
               decision=_observe_gate(), envelope=None)
    out = _tool_get_savings_summary(db)
    assert out["estimated_recoverable_usd"] == 0.0
    assert out["realized"] is False
    db.close()


# --- suggest_policies ---

def test_suggest_policies_frames_as_suggestion_not_validated():
    db = InMemoryBackend()
    _seed(db)
    out = _tool_suggest_policies(db, _config())
    assert out["suggest_mode"] is True
    assert out["label"] == "unvalidated"
    # openai has api spend + no [budget.openai] usd ceiling → a budget_cap suggestion.
    assert len(out["suggestions"]) == 1
    s = out["suggestions"][0]
    assert s["kind"] == "budget_cap"
    assert s["provider"] == "openai"
    assert s["suggested_ceiling_usd"] >= s["observed_cycle_spend_usd"]
    assert "budget_cap" in s["toml"]
    # Honesty: framed as a starting point to review, not validated-safe.
    assert "not validated" in out["note"].lower()


def test_suggest_policies_skips_provider_with_existing_ceiling():
    db = InMemoryBackend()
    _seed(db)
    cfg = TjConfig(version="1",
                   budgets={"openai": ProviderBudget(plan="api", usd=50.0)})
    out = _tool_suggest_policies(db, cfg)
    assert out["suggestions"] == []  # already has a ceiling
    db.close()


def test_suggest_policies_no_dollar_suggestion_for_subscription():
    # Dollars are api-only: a subscription provider gets no dollar ceiling suggestion.
    db = InMemoryBackend()
    sess = make_session(agent_id="a", total_cost_usd=5.0)
    db.upsert_session(sess)
    db.insert_span(make_llm_span(agent_id="a", provider="anthropic",
                                 model="claude-haiku-4-5", cost_usd=5.0,
                                 session_id=sess.session_id))
    cfg = TjConfig(version="1", budgets={"anthropic": ProviderBudget(plan="max_5x")})
    out = _tool_suggest_policies(db, cfg)
    assert out["suggestions"] == []
    db.close()


# --- self-observation span (#223) ---

def test_policy_decision_span_carries_tokenjam_policy_attributes():
    db = InMemoryBackend()
    _seed(db)
    # The span landed in the spans table with the tokenjam.policy.* namespace.
    rows = db.conn.execute(
        "SELECT name, attributes FROM spans WHERE name = ?",
        ["tokenjam.policy.decision"],
    ).fetchall()
    assert len(rows) == 1
    attrs = json.loads(rows[0][1])
    assert attrs[TjAttributes.POLICY_DECISION] == "policy"
    assert attrs[TjAttributes.POLICY_ACTION] == ACTION_WOULD_BLOCK
    assert attrs[TjAttributes.POLICY_MODE] == "suggest"
    assert attrs[TjAttributes.POLICY_LABEL] == "unvalidated"
    assert attrs[TjAttributes.POLICY_REALIZED] is False
    assert attrs[TjAttributes.POLICY_ESTIMATED_RECOVERABLE_USD] == 2.0
    db.close()


def test_policy_decision_span_builder_is_pure_and_uses_semconv():
    # Build the span directly from an observation-like object — no DB needed.
    class _Obs:
        decision = "observe_only"
        provider = "anthropic"
        pricing_mode = "subscription"
        path = "/v1/messages"
        suggest_only = True
        policy = None

    span = policy_decision_span(_Obs())
    assert span.name == "tokenjam.policy.decision"
    a = span.attributes
    assert a[TjAttributes.POLICY_DECISION] == "observe_only"
    assert a[TjAttributes.POLICY_PASSTHROUGH_TOS] is True   # subscription TOS
    assert a[TjAttributes.POLICY_REALIZED] is False
    # observe-only never carries a recoverable estimate (never acted on).
    assert TjAttributes.POLICY_ESTIMATED_RECOVERABLE_USD not in a


def test_policy_span_emitted_through_pipeline_when_provided():
    # When a pipeline is wired, the span flows through it (not a direct insert).
    captured = []

    class _Pipeline:
        def process(self, span):
            captured.append(span)

    db = InMemoryBackend()
    obs = ProxyObserver(sink=AuditSink(db, pipeline=_Pipeline()))
    obs.record(method="POST", path="/v1/chat/completions",
               decision=_policy_gate(), envelope=_envelope(1.0))
    assert len(captured) == 1
    assert captured[0].name == "tokenjam.policy.decision"
    assert captured[0].attributes[TjAttributes.POLICY_ACTION] == ACTION_WOULD_BLOCK
    db.close()
