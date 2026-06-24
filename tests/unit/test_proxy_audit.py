"""Tests for the audit log + savings meter (#221).

Covers: migration 6 creates both tables; the observer sink persists decisions +
the savings ledger; observe-only (subscription) is logged with passthrough_tos
and accrues NO savings; the savings figure is always estimated-recoverable /
would-have-saved and NEVER "saved" (Critical Rule 14); the reconciliation math
vs actual spend reconciles to the same source `tj cost` reads.
"""
from __future__ import annotations

from tokenjam.core.db import InMemoryBackend
from tokenjam.core.models import PolicyDecisionFilters
from tokenjam.proxy.audit import (
    SAVINGS_DISCLAIMER,
    AuditSink,
    reconcile_savings,
)
from tokenjam.proxy.engine import (
    ACTION_WOULD_BLOCK,
    PolicyEnvelope,
    PolicyEvaluation,
)
from tokenjam.proxy.gate import OBSERVE_ONLY, POLICY, GateDecision
from tokenjam.proxy.observer import ProxyObserver
from tokenjam.utils.time_parse import utcnow
from tests.factories import make_llm_span, make_session


def _db():
    return InMemoryBackend()


def _policy_envelope(usd=0.0, tokens=0, basis=""):
    details = {}
    if usd:
        details["estimated_recoverable_usd"] = usd
    if tokens:
        details["estimated_recoverable_tokens"] = tokens
    if basis:
        details["estimate_basis"] = basis
    return PolicyEnvelope(
        ts=utcnow().isoformat(), provider="openai", path="/v1/chat/completions",
        agent=None, gate_path=POLICY, overall_action=ACTION_WOULD_BLOCK,
        evaluations=[PolicyEvaluation(
            policy_name="blocker", kind="test", mode="suggest",
            would_action=ACTION_WOULD_BLOCK, reason="stub",
            enforcement_gated=False, details=details)],
    )


def _policy_gate():
    return GateDecision(provider="openai", plan_tier="api", pricing_mode="api",
                        path=POLICY, reason="api_usage_billed")


def _observe_gate():
    return GateDecision(provider="anthropic", plan_tier="max_5x",
                        pricing_mode="subscription", path=OBSERVE_ONLY,
                        reason="subscription_passthrough")


# --- migration ---

def test_migration_creates_audit_tables():
    db = _db()
    tables = {r[0] for r in db.conn.execute(
        "SELECT table_name FROM information_schema.tables").fetchall()}
    assert "policy_decisions" in tables
    assert "savings_ledger" in tables
    applied = {r[0] for r in db.conn.execute(
        "SELECT version FROM schema_migrations").fetchall()}
    assert 6 in applied
    db.close()


# --- persistence via the observer sink ---

def test_sink_persists_policy_decision_and_savings():
    db = _db()
    observer = ProxyObserver(sink=AuditSink(db))
    observer.record(method="POST", path="/v1/chat/completions",
                    decision=_policy_gate(),
                    envelope=_policy_envelope(usd=0.42, basis="stub"))

    decisions = db.get_policy_decisions(PolicyDecisionFilters())
    assert len(decisions) == 1
    d = decisions[0]
    assert d.gate_decision == "policy"
    assert d.would_action == ACTION_WOULD_BLOCK
    assert d.policy_name == "blocker"
    assert d.label == "unvalidated"
    assert d.envelope["overall_action"] == ACTION_WOULD_BLOCK  # round-tripped JSON

    savings = db.get_savings_entries(PolicyDecisionFilters())
    assert len(savings) == 1
    assert savings[0].estimated_recoverable_usd == 0.42
    assert savings[0].decision_id == d.decision_id
    assert savings[0].realized is False  # suggest mode: never realized
    assert savings[0].label == "unvalidated"
    db.close()


def test_observe_only_logs_passthrough_tos_and_no_savings():
    db = _db()
    observer = ProxyObserver(sink=AuditSink(db))
    # Subscription observe-only — recorded, but never acted on → no savings.
    observer.record(method="POST", path="/v1/messages",
                    decision=_observe_gate(), envelope=None)

    decisions = db.get_policy_decisions(PolicyDecisionFilters())
    assert len(decisions) == 1
    assert decisions[0].gate_decision == "observe_only"
    assert decisions[0].passthrough_tos is True   # "not permitted to act" (TOS)
    assert db.get_savings_entries(PolicyDecisionFilters()) == []
    db.close()


def test_noop_envelope_records_zero_savings():
    db = _db()
    observer = ProxyObserver(sink=AuditSink(db))
    observer.record(method="POST", path="/v1/chat/completions",
                    decision=_policy_gate(), envelope=_policy_envelope())  # no estimate
    s = db.get_savings_entries(PolicyDecisionFilters())
    assert len(s) == 1
    assert s[0].estimated_recoverable_usd == 0.0  # a noop would recover nothing
    db.close()


def test_sink_never_raises_on_bad_db():
    class _BrokenDB:
        def insert_policy_decision(self, *_a, **_k):
            raise RuntimeError("boom")

    # AuditSink swallows persistence errors so the proxy never breaks.
    sink = AuditSink(_BrokenDB())
    observer = ProxyObserver(sink=sink)
    # Should not raise.
    observer.record(method="POST", path="/v1/chat/completions",
                    decision=_policy_gate(), envelope=_policy_envelope(usd=1.0))


# --- savings meter: honesty + reconciliation ---

def test_savings_disclaimer_never_says_saved():
    # The load-bearing honesty check.
    assert "saved" not in SAVINGS_DISCLAIMER.lower().replace("would-have-saved", "")
    assert "estimated recoverable" in SAVINGS_DISCLAIMER.lower()
    assert "not realized" in SAVINGS_DISCLAIMER.lower()


def test_reconcile_savings_reconciles_to_cost_summary():
    db = _db()
    # Seed ACTUAL spend the same way `tj cost` reads it (spans with cost_usd).
    sess = make_session(agent_id="codegen", total_cost_usd=10.0)
    db.upsert_session(sess)
    db.insert_span(make_llm_span(agent_id="codegen", provider="openai",
                                 model="gpt-4o", cost_usd=6.0, session_id=sess.session_id))
    db.insert_span(make_llm_span(agent_id="codegen", provider="openai",
                                 model="gpt-4o", cost_usd=4.0, session_id=sess.session_id))

    # Seed two policy decisions that WOULD have recovered $1.50 + $0.50.
    observer = ProxyObserver(sink=AuditSink(db))
    observer.record(method="POST", path="/v1/chat/completions",
                    decision=_policy_gate(), envelope=_policy_envelope(usd=1.50, basis="a"))
    observer.record(method="POST", path="/v1/chat/completions",
                    decision=_policy_gate(), envelope=_policy_envelope(usd=0.50, basis="b"))

    summary = reconcile_savings(db)
    # Actual spend matches what get_cost_summary (the tj cost source) reports.
    from tokenjam.core.models import CostFilters
    cost_total = sum(r.cost_usd for r in db.get_cost_summary(CostFilters(group_by="day")))
    assert summary.actual_spend_usd == cost_total == 10.0
    # Estimated recoverable is the sum of the would-have figures.
    assert summary.estimated_recoverable_usd == 2.0
    assert summary.estimated_recoverable_pct == 20.0  # 2 / 10 * 100
    assert summary.realized is False
    assert summary.decisions == 2
    db.close()


def test_savings_summary_to_dict_is_estimated_never_saved():
    db = _db()
    observer = ProxyObserver(sink=AuditSink(db))
    observer.record(method="POST", path="/v1/chat/completions",
                    decision=_policy_gate(), envelope=_policy_envelope(usd=3.0))
    d = reconcile_savings(db).to_dict()
    assert "estimated_recoverable_usd" in d
    assert d["realized"] is False
    assert d["label"] == "unvalidated"
    assert "saved" not in str({k: v for k, v in d.items() if k != "disclaimer"}).lower()
    db.close()


def test_zero_spend_window_has_no_pct():
    db = _db()
    observer = ProxyObserver(sink=AuditSink(db))
    observer.record(method="POST", path="/v1/chat/completions",
                    decision=_policy_gate(), envelope=_policy_envelope(usd=1.0))
    summary = reconcile_savings(db)
    assert summary.actual_spend_usd == 0.0
    assert summary.estimated_recoverable_pct is None  # no divide-by-zero claim
    db.close()
