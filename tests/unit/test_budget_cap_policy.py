"""Unit tests for the budget_cap policy (#222) — the first concrete policy on
the #220 engine. HTTP-free.

Covers: fires would_block when current-cycle spend is over the per-provider
ceiling and no-ops under; the soft "approaching ceiling" warn; reads the RIGHT
``[budget.<provider>] usd`` ceiling (provider-scoped); the real DuckDB cycle-
spend path (cycle bounds + provider scoping); graceful no-op when no ceiling /
no spend data; the api-only guard means budget_cap never runs on observe-only
traffic; and the envelope carries the unvalidated label + a would-do action
(never "blocked" — suggest mode).
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from tokenjam.core.config import PolicyConfig, ProviderBudget, TjConfig
from tokenjam.core.db import InMemoryBackend
from tokenjam.proxy.engine import (
    ACTION_NOOP,
    ACTION_WOULD_BLOCK,
    UNVALIDATED_LABEL,
    PolicyContext,
    PolicyEngine,
    PolicyGuardError,
    PolicyRequest,
)
from tokenjam.proxy.gate import OBSERVE_ONLY, POLICY, GateDecision, classify
from tokenjam.utils.time_parse import utcnow
from tests.factories import make_llm_span, make_session


def _cfg(*, ceilings: dict[str, float | None], policy_params: dict | None = None) -> TjConfig:
    budgets = {p: ProviderBudget(plan="api", usd=usd) for p, usd in ceilings.items()}
    return TjConfig(
        version="1",
        budgets=budgets,
        policies=[PolicyConfig(name="cap", kind="budget_cap", params=policy_params or {})],
    )


def _engine(cfg: TjConfig, *, spend, db=None, now_fn=None) -> PolicyEngine:
    # `spend` is a {provider: usd} map (or a single float) injected as the
    # cycle-spend source so the evaluator is testable without a DB.
    if callable(spend):
        spend_fn = spend
    elif isinstance(spend, dict):
        spend_fn = lambda p: spend.get(p)  # noqa: E731
    else:
        spend_fn = lambda p: spend  # noqa: E731
    ctx = PolicyContext(
        config=cfg,
        db=db,
        spend_fn=(None if db is not None else spend_fn),
        now_fn=now_fn,
    )
    return PolicyEngine(list(cfg.policies), context=ctx)



def _api_gate(cfg: TjConfig, provider="openai") -> GateDecision:
    gd = classify(cfg, provider)
    assert gd.path == POLICY  # precondition: api/usage-billed
    return gd


def _req(provider="openai") -> PolicyRequest:
    path = "/v1/messages" if provider == "anthropic" else "/v1/chat/completions"
    return PolicyRequest(provider=provider, path=path)


def test_fires_would_block_when_over_ceiling():
    cfg = _cfg(ceilings={"openai": 100.0})
    env = _engine(cfg, spend=150.0).evaluate(_api_gate(cfg), _req("openai"))
    ev = env.evaluations[0]
    assert ev.would_action == ACTION_WOULD_BLOCK
    assert env.overall_action == ACTION_WOULD_BLOCK
    assert ev.details["over_by_usd"] == pytest.approx(50.0)
    assert ev.details["would_do"]  # carries the WOULD-do action
    # Suggest mode: never claims it acted.
    assert "would block" in ev.reason.lower()
    assert "blocked" not in ev.reason.lower()


def test_noop_when_under_ceiling():
    cfg = _cfg(ceilings={"openai": 100.0})
    env = _engine(cfg, spend=30.0).evaluate(_api_gate(cfg), _req("openai"))
    ev = env.evaluations[0]
    assert ev.would_action == ACTION_NOOP
    assert ev.details["near_ceiling"] is False
    assert env.overall_action == ACTION_NOOP


def test_warns_near_ceiling():
    cfg = _cfg(ceilings={"openai": 100.0})  # default warn_at = 0.8
    env = _engine(cfg, spend=90.0).evaluate(_api_gate(cfg), _req("openai"))
    ev = env.evaluations[0]
    assert ev.would_action == ACTION_NOOP            # a warn is not a block
    assert ev.details["near_ceiling"] is True
    assert ev.details["pct_of_ceiling"] == pytest.approx(90.0)
    assert "approaching" in ev.reason.lower()


def test_warn_at_is_configurable_via_params():
    cfg = _cfg(ceilings={"openai": 100.0}, policy_params={"warn_at": 0.5})
    env = _engine(cfg, spend=60.0).evaluate(_api_gate(cfg), _req("openai"))
    assert env.evaluations[0].details["near_ceiling"] is True  # 60% ≥ 50%


def test_noop_when_no_ceiling_configured():
    # No [budget.<provider>] usd → nothing to cap, even with huge spend.
    cfg = _cfg(ceilings={"openai": None})
    env = _engine(cfg, spend=9999.0).evaluate(_api_gate(cfg), _req("openai"))
    ev = env.evaluations[0]
    assert ev.would_action == ACTION_NOOP
    assert "no [budget.openai] usd ceiling" in ev.reason
    assert ev.details["ceiling_usd"] is None


def test_reads_the_right_provider_ceiling():
    # anthropic has a tight $10 ceiling, openai a loose $1000. The same $50 spend
    # blocks on anthropic but is fine on openai — proving provider-scoped reads.
    cfg = _cfg(ceilings={"anthropic": 10.0, "openai": 1000.0})
    spend = {"anthropic": 50.0, "openai": 50.0}
    anthro = _engine(cfg, spend=spend).evaluate(_api_gate(cfg, "anthropic"), _req("anthropic"))
    openai = _engine(cfg, spend=spend).evaluate(_api_gate(cfg, "openai"), _req("openai"))
    assert anthro.evaluations[0].would_action == ACTION_WOULD_BLOCK
    assert anthro.evaluations[0].details["ceiling_usd"] == pytest.approx(10.0)
    assert openai.evaluations[0].would_action == ACTION_NOOP
    assert openai.evaluations[0].details["ceiling_usd"] == pytest.approx(1000.0)


def test_cycle_spend_read_from_duckdb():
    # The real path: cycle spend summed from the telemetry DB, scoped to the
    # provider + the current billing cycle. Seed current-cycle spans via factory.
    db = InMemoryBackend()
    now = datetime(2026, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
    db.upsert_session(make_session(session_id="s1", plan_tier="api", agent_id="cc"))
    # $7 of anthropic spend this cycle; an openai span must NOT count toward it.
    for i in range(7):
        db.insert_span(make_llm_span(session_id="s1", agent_id="cc", provider="anthropic",
                                     model="claude-opus-4-7", cost_usd=1.0,
                                     start_time=now - timedelta(hours=i + 1)))
    db.insert_span(make_llm_span(session_id="s1", agent_id="cc", provider="openai",
                                 model="gpt-4o", cost_usd=100.0, start_time=now - timedelta(hours=1)))

    over = _cfg(ceilings={"anthropic": 5.0})   # $7 spend > $5 ceiling
    env = _engine(over, spend=None, db=db, now_fn=lambda: now).evaluate(_api_gate(over, "anthropic"), _req("anthropic"))
    ev = env.evaluations[0]
    assert ev.would_action == ACTION_WOULD_BLOCK
    assert ev.details["cycle_spend_usd"] == pytest.approx(7.0)  # openai's $100 excluded

    under = _cfg(ceilings={"anthropic": 20.0})  # $7 spend < $20 ceiling
    env2 = _engine(under, spend=None, db=db, now_fn=lambda: now).evaluate(_api_gate(under, "anthropic"), _req("anthropic"))
    assert env2.evaluations[0].would_action == ACTION_NOOP
    db.close()


def test_old_spend_outside_cycle_is_excluded():
    # Spend from before the current cycle must not count (deterministic ceiling).
    db = InMemoryBackend()
    now = datetime(2026, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
    db.upsert_session(make_session(session_id="s1", plan_tier="api", agent_id="cc"))
    db.insert_span(make_llm_span(session_id="s1", agent_id="cc", provider="anthropic",
                                 model="claude-opus-4-7", cost_usd=500.0,
                                 start_time=now - timedelta(days=45)))  # last cycle
    cfg = _cfg(ceilings={"anthropic": 5.0})
    env = _engine(cfg, spend=None, db=db, now_fn=lambda: now).evaluate(_api_gate(cfg, "anthropic"), _req("anthropic"))
    assert env.evaluations[0].would_action == ACTION_NOOP  # 0 this cycle, under $5
    assert env.evaluations[0].details["cycle_spend_usd"] == pytest.approx(0.0)
    db.close()



def test_noop_when_spend_unavailable():
    # No DB and no injected spend → can't evaluate → graceful no-op (not a guess).
    cfg = _cfg(ceilings={"openai": 100.0})
    ctx = PolicyContext(config=cfg)  # no db, no spend_fn
    env = PolicyEngine(list(cfg.policies), context=ctx).evaluate(_api_gate(cfg), _req("openai"))
    ev = env.evaluations[0]
    assert ev.would_action == ACTION_NOOP
    assert "unavailable" in ev.reason
    assert ev.details["cycle_spend_usd"] is None


def test_only_evaluated_on_policy_path_traffic():
    # budget_cap can never run on observe-only traffic — the engine's api-only
    # guard refuses it before any evaluator is called.
    sub_cfg = TjConfig(
        version="1", budgets={"anthropic": ProviderBudget(plan="max_5x", usd=10.0)},
        policies=[PolicyConfig(name="cap", kind="budget_cap")],
    )
    observe_gd = classify(sub_cfg, "anthropic")
    assert observe_gd.path == OBSERVE_ONLY
    engine = PolicyEngine(list(sub_cfg.policies),
                          context=PolicyContext(config=sub_cfg, spend_fn=lambda p: 9999.0))
    with pytest.raises(PolicyGuardError):
        engine.evaluate(observe_gd, _req("anthropic"))


def test_envelope_unvalidated_and_suggest_only_when_blocking():
    cfg = _cfg(ceilings={"openai": 100.0})
    env = _engine(cfg, spend=150.0).evaluate(_api_gate(cfg), _req("openai"))
    assert env.label == UNVALIDATED_LABEL
    assert env.validated is False
    assert env.enforced is False          # suggest mode: never acts
    assert env.suggest_only is True


def test_from_config_threads_db_into_context():
    # PolicyEngine.from_config(config, db=...) wires the DB so budget_cap can
    # read cycle spend through the engine the proxy actually builds.
    db = InMemoryBackend()
    now = utcnow()
    db.upsert_session(make_session(session_id="s1", plan_tier="api", agent_id="cc"))
    db.insert_span(make_llm_span(session_id="s1", agent_id="cc", provider="openai",
                                 model="gpt-4o", cost_usd=200.0, start_time=now))
    cfg = _cfg(ceilings={"openai": 50.0})
    engine = PolicyEngine.from_config(cfg, db=db)
    env = engine.evaluate(_api_gate(cfg), _req("openai"))
    assert env.evaluations[0].would_action == ACTION_WOULD_BLOCK
    db.close()
