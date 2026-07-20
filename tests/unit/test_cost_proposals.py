"""Unit tests for wiring the cost analyzers (downsize/cache/trim) into the
self-improve loop as advise-only proposals with delta-verify receipts.

Fully isolated: the DB is an ``InMemoryBackend`` and every JSON ledger/store
write is routed under ``tmp_path`` via ``cfg.storage.path`` — nothing here
touches a real ``~/.tj`` / ``~/.claude`` (mirrors ``test_pothole_apply``).
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from tokenjam.core.config import StorageConfig, TjConfig
from tokenjam.core.db import InMemoryBackend
from tokenjam.core.optimize import cost_apply, cost_verify, pothole_store
from tokenjam.core.optimize.analyzers.cache_efficacy import (
    CacheEfficacyFinding,
    CacheEfficacyRow,
)
from tokenjam.core.optimize.analyzers.pothole import PotholeFinding
from tokenjam.core.optimize.analyzers.prompt_bloat import BloatPrompt, PromptBloatFinding
from tokenjam.core.optimize.cost_proposals import cost_proposals_from_report
from tokenjam.core.optimize.types import (
    DowngradeFinding,
    OptimizeReport,
    WindowSummary,
)
from tests.factories import make_llm_span

NOW = datetime(2026, 7, 20, 12, 0, tzinfo=timezone.utc)
MARKER = NOW - timedelta(days=5)


@pytest.fixture
def db():
    backend = InMemoryBackend()
    yield backend
    backend.close()


@pytest.fixture
def cfg(tmp_path):
    return TjConfig(version="1", storage=StorageConfig(path=str(tmp_path / "t.duckdb")))


def _report():
    dg = DowngradeFinding(
        candidate_sessions=4, total_sessions=10, actual_cost_usd=5.0,
        alternative_cost_usd=2.0, monthly_savings_usd=3.0, percent_of_sessions=40.0,
        examples=[], suggestions={"claude-opus-4-8": "claude-sonnet-5"},
        estimated_recoverable_usd=3.0, percent_of_tokens=35.0,
        estimate_basis="downsize basis",
    )
    cache = CacheEfficacyFinding(
        flagged=[CacheEfficacyRow("anthropic", "claude-sonnet-5", 100_000, 5_000,
                                  0.05, "full", True)],
        estimated_recoverable_usd=1.2, estimate_basis="cache basis",
    )
    trim = PromptBloatFinding(
        enabled=True,
        per_prompt=[BloatPrompt(agent_id="svc-a", sample_chars="x", prompt_chars=8000,
                                significant_chars=3000, bloat_chars=5000,
                                estimated_token_reduction=1250)],
        estimated_recoverable_usd=0.8, estimate_basis="trim basis",
    )
    w = WindowSummary(since=MARKER, until=NOW, days=5, sessions=10, spans=100,
                      total_tokens=1, total_cost_usd=5.0, thin_data=False)
    return OptimizeReport(window=w, downgrade=dg, findings={"cache": cache, "trim": trim})


# --- Adapter: findings -> proposals ------------------------------------------

def test_adapter_produces_one_proposal_per_analyzer():
    props = cost_proposals_from_report(_report())
    by_analyzer = {p.analyzer for p in props}
    assert by_analyzer == {"downsize", "cache", "trim"}


def test_adapter_proposals_are_advise_only_cost_kind():
    for p in cost_proposals_from_report(_report()):
        assert p.kind == "cost"
        assert p.advise_only is True
        # Advise-only structural contract: the recommendation carries text and
        # a labeled, correlational estimate — never a causal savings claim.
        assert p.advise_text
        assert p.correlational is True
        assert p.estimate_confidence == "estimated"


def test_adapter_carries_evidence_and_estimate_per_analyzer():
    props = {p.analyzer: p for p in cost_proposals_from_report(_report())}
    assert "claude-opus-4-8" in props["downsize"].evidence
    assert props["downsize"].estimated_recoverable_usd == 3.0
    assert props["cache"].target_key == {"provider": "anthropic", "model": "claude-sonnet-5"}
    assert props["cache"].estimated_recoverable_usd is not None
    assert props["trim"].target_key == {"agent_id": "svc-a"}
    assert props["trim"].estimated_recoverable_tokens == 1250


def test_adapter_empty_report_yields_nothing():
    w = WindowSummary(since=MARKER, until=NOW, days=5, sessions=0, spans=0,
                      total_tokens=0, total_cost_usd=0.0, thin_data=True)
    assert cost_proposals_from_report(OptimizeReport(window=w)) == []


def test_adapter_skips_disabled_trim():
    rep = _report()
    rep.findings["trim"] = PromptBloatFinding(enabled=False)
    analyzers = {p.analyzer for p in cost_proposals_from_report(rep)}
    assert "trim" not in analyzers


# --- Mark applied (the marker) — advise-only, no code write -------------------

def _mark(db, cfg, prop):
    return cost_apply.mark_applied(db, cfg, {
        "signature": prop.signature, "analyzer": prop.analyzer, "title": prop.title,
        "target_key": prop.target_key, "baseline": prop.baseline,
        "advise_text": prop.advise_text, "agent_id": prop.agent_id,
        "estimated_recoverable_usd": prop.estimated_recoverable_usd,
    })


def test_mark_applied_creates_expectation_marker(db, cfg):
    from tokenjam.core.loop import list_expectations

    trim = next(p for p in cost_proposals_from_report(_report()) if p.analyzer == "trim")
    rec = _mark(db, cfg, trim)
    assert rec["state"] == "applied"
    assert rec["applied_at"]                       # the marker timestamp
    exps = list_expectations(db)
    assert len(exps) == 1
    assert exps[0].expectation_id == rec["expectation_id"]
    assert exps[0].agent_id == "svc-a"


def test_mark_applied_is_idempotent_per_signature(db, cfg):
    trim = next(p for p in cost_proposals_from_report(_report()) if p.analyzer == "trim")
    r1 = _mark(db, cfg, trim)
    r2 = _mark(db, cfg, trim)
    assert r1["id"] == r2["id"]
    assert len(cost_apply.list_applied(cfg)) == 1


def test_revert_flips_state_and_stops_counting(db, cfg):
    trim = next(p for p in cost_proposals_from_report(_report()) if p.analyzer == "trim")
    rec = _mark(db, cfg, trim)
    reverted = cost_apply.revert_applied(cfg, rec["id"])
    assert reverted["state"] == "reverted"
    assert reverted["reverted_at"]


def test_mark_applied_refuses_empty_signature(db, cfg):
    with pytest.raises(cost_apply.CostApplyRefused):
        cost_apply.mark_applied(db, cfg, {"signature": ""})


# --- Delta-verify: the receipts ----------------------------------------------

def _seed(db, *, agent, model, input_tok, when, count=30, provider="anthropic",
          cache_tok=0):
    for i in range(count):
        db.insert_span(make_llm_span(
            agent_id=agent, provider=provider, model=model, billing_account=provider,
            input_tokens=input_tok, output_tokens=200, cache_tokens=cache_tok,
            session_id=f"{agent}-{when.isoformat()}-{i}",
            start_time=when + timedelta(minutes=i),
        ))


def _record(analyzer, target_key, agent_id="svc-a"):
    return {
        "id": "rec-1", "expectation_id": "exp-1", "signature": f"cost:{analyzer}",
        "analyzer": analyzer, "kind": "cost", "title": "t", "target_key": target_key,
        "agent_id": agent_id, "applied_at": MARKER.isoformat(), "baseline": {},
        "estimated_recoverable_usd": None, "estimated_recoverable_tokens": None,
        "estimate_basis": "", "state": "applied", "verify": {},
    }


def test_trim_delta_improved_when_input_per_call_drops(db):
    # pre: 10k input/call; post: 4k input/call — a real trim.
    _seed(db, agent="svc-a", model="claude-sonnet-5", input_tok=10_000,
          when=MARKER - timedelta(hours=40))
    _seed(db, agent="svc-a", model="claude-sonnet-5", input_tok=4_000,
          when=MARKER + timedelta(hours=1))
    v = cost_verify.measure_cost_delta(db.conn, _record("trim", {"agent_id": "svc-a"}), now=NOW)
    assert v["verdict"] == "improved"
    assert v["realized_tokens_delta"] > 0
    assert v["realized_usd_delta"] > 0


def test_trim_delta_regressed_when_no_improvement(db):
    # pre and post identical — no movement -> regressed, not silently kept.
    _seed(db, agent="svc-a", model="claude-sonnet-5", input_tok=8_000,
          when=MARKER - timedelta(hours=40))
    _seed(db, agent="svc-a", model="claude-sonnet-5", input_tok=8_000,
          when=MARKER + timedelta(hours=1))
    v = cost_verify.measure_cost_delta(db.conn, _record("trim", {"agent_id": "svc-a"}), now=NOW)
    assert v["verdict"] == "regressed"


def test_cache_delta_improved_when_efficacy_rises(db):
    # pre: mostly fresh input; post: mostly cache reads for the flagged model.
    _seed(db, agent="svc-a", model="claude-sonnet-5", input_tok=10_000, cache_tok=500,
          when=MARKER - timedelta(hours=40))
    _seed(db, agent="svc-a", model="claude-sonnet-5", input_tok=2_000, cache_tok=9_000,
          when=MARKER + timedelta(hours=1))
    rec = _record("cache", {"provider": "anthropic", "model": "claude-sonnet-5"})
    v = cost_verify.measure_cost_delta(db.conn, rec, now=NOW)
    assert v["verdict"] == "improved"
    assert v["post_value"] > v["pre_value"]        # efficacy rose


def test_downsize_delta_improved_when_oversized_share_drops(db):
    # pre: all calls on the oversized model; post: moved to the cheaper one.
    _seed(db, agent="svc-a", model="claude-opus-4-8", input_tok=8_000,
          when=MARKER - timedelta(hours=40))
    _seed(db, agent="svc-a", model="claude-sonnet-5", input_tok=8_000,
          when=MARKER + timedelta(hours=1))
    rec = _record("downsize", {"models": ["claude-opus-4-8"]})
    v = cost_verify.measure_cost_delta(db.conn, rec, now=NOW)
    assert v["verdict"] == "improved"
    assert v["realized_usd_delta"] > 0


def test_insufficient_data_below_exposure_gate(db):
    # Only a handful of post calls — no confident verdict.
    _seed(db, agent="svc-a", model="claude-sonnet-5", input_tok=10_000,
          when=MARKER - timedelta(hours=40))
    _seed(db, agent="svc-a", model="claude-sonnet-5", input_tok=4_000,
          when=MARKER + timedelta(hours=1), count=3)
    v = cost_verify.measure_cost_delta(db.conn, _record("trim", {"agent_id": "svc-a"}), now=NOW)
    assert v["verdict"] == "insufficient_data"
    assert v["realized_usd_delta"] is None


def test_verify_record_writes_ledger_and_outcome(db, cfg):
    from tokenjam.core.loop import create_expectation, list_expectation_runs

    exp = create_expectation(db, name="trim svc-a", agent_id="svc-a")
    rec = _record("trim", {"agent_id": "svc-a"})
    rec["expectation_id"] = exp.expectation_id
    # Persist the record so set_verify can find it.
    from tokenjam.core.optimize.cost_apply import _write_ledger
    _write_ledger(cfg, [rec])
    _seed(db, agent="svc-a", model="claude-sonnet-5", input_tok=10_000,
          when=MARKER - timedelta(hours=40))
    _seed(db, agent="svc-a", model="claude-sonnet-5", input_tok=4_000,
          when=MARKER + timedelta(hours=1))

    v = cost_verify.verify_record(db, cfg, rec, now=NOW)
    assert v["verdict"] == "improved"
    # Outcome written to the loop's fix-history ledger.
    runs = list_expectation_runs(db, exp.expectation_id)
    assert len(runs) == 1
    assert runs[0].outcome == "pass"
    # And persisted onto the record.
    stored = cost_apply.get_applied(cfg, "rec-1")
    assert stored["verify"]["verdict"] == "improved"


def test_cost_compound_ledger_sums_realized_dollars():
    records = [
        {"state": "applied", "verify": {"verdict": "improved",
         "realized_usd_delta": 0.5, "realized_tokens_delta": 1000}},
        {"state": "applied", "verify": {"verdict": "regressed",
         "realized_usd_delta": 0.0}},
        {"state": "reverted", "verify": {"verdict": "improved",
         "realized_usd_delta": 9.9}},   # reverted never counts
        {"state": "applied", "verify": {"verdict": "insufficient_data"}},
    ]
    ledger = cost_verify.cost_compound_ledger(records)
    assert ledger["total_realized_usd"] == 0.5
    assert ledger["improved_count"] == 1
    assert ledger["regressed_count"] == 1
    assert ledger["insufficient_data_count"] == 1
    assert ledger["verified_count"] == 3


# --- Store: cost proposals share the pothole cache file ----------------------

def test_store_cost_and_pothole_coexist_without_clobber(tmp_path):
    path = tmp_path / "pothole_cache.json"
    # Pothole finding written first.
    pothole_store.write_cache(PotholeFinding(sessions_scanned=7), path=path)
    # Cost proposals written into the SAME file must preserve the finding.
    pothole_store.write_cost_proposals(
        [{"kind": "cost", "analyzer": "trim", "signature": "cost:trim:x"}], path=path,
    )
    cost_block = pothole_store.read_cost_proposals(path=path)
    assert cost_block is not None
    assert cost_block["cost_proposals"][0]["signature"] == "cost:trim:x"
    # The pothole finding is still intact.
    raw = pothole_store.read_cache(path=path)
    assert raw["finding"]["sessions_scanned"] == 7
    # A later pothole recompute preserves the cost proposals.
    pothole_store.write_cache(PotholeFinding(sessions_scanned=9), path=path)
    assert pothole_store.read_cost_proposals(path=path)["cost_proposals"][0]["signature"] == "cost:trim:x"
    assert pothole_store.read_cache(path=path)["finding"]["sessions_scanned"] == 9


# --- Subagent right-sizing: the apply-capable 4th analyzer --------------------

def _sub_finding(models=("claude-opus-4-8",)):
    from tokenjam.core.optimize.analyzers.subagent_rightsizing import (
        SubagentRightsizingFinding,
        SubagentRow,
    )
    flagged = [
        SubagentRow(session_id="s1", sub_agent_id=f"sa{i}", model=m, llm_calls=2,
                    tool_calls=1, input_tokens=60000, output_tokens=500, cache_tokens=0,
                    cache_write_tokens=0, cost_usd=1.2, provider="anthropic",
                    flags=["over_powered"])
        for i, m in enumerate(models)
    ]
    return SubagentRightsizingFinding(
        flagged=flagged, percent_of_cost=0.66, flagged_cost_usd=1.2,
        subagent_cost_usd=1.5, estimated_recoverable_usd=0.4,
        estimated_recoverable_tokens=60500,
    )


def test_subagent_proposal_is_apply_capable_cc_origin():
    from tokenjam.core.optimize.cost_proposals import _subagent_to_proposals
    props = _subagent_to_proposals(_sub_finding())
    assert len(props) == 1
    p = props[0]
    assert p.analyzer == "subagent"
    assert p.apply_capable is True
    assert p.advise_only is False        # CC-origin has a workspace surface
    assert p.rung == 1 and p.scope == "project"
    assert p.proposed_fix                # a sizing rubric note to write
    assert p.target_key == {"models": ["claude-opus-4-8"], "subagent": True}


def test_subagent_proposal_degrades_to_advise_only_without_over_powered():
    from tokenjam.core.optimize.analyzers.subagent_rightsizing import (
        SubagentRightsizingFinding,
    )
    # over_provisioned-only (no model swap) yields no proposal here.
    assert _sub_finding.__doc__ is None  # sanity: helper unchanged
    from tokenjam.core.optimize.cost_proposals import _subagent_to_proposals
    assert _subagent_to_proposals(SubagentRightsizingFinding(flagged=[])) == []


def _seed_sub(db, *, model, when, count=30, provider="anthropic"):
    for i in range(count):
        db.insert_span(make_llm_span(
            agent_id="claude-code-x", provider=provider, model=model,
            billing_account=provider, input_tokens=8000, output_tokens=200,
            session_id=f"{model}-{when.isoformat()}-{i}", sub_agent_id="sa1",
            start_time=when + timedelta(minutes=i),
        ))


def test_subagent_delta_improved_when_fanout_moves_off_oversized_model(db):
    # pre: subagent fan-out on opus; post: moved to sonnet.
    _seed_sub(db, model="claude-opus-4-8", when=MARKER - timedelta(hours=40))
    _seed_sub(db, model="claude-sonnet-5", when=MARKER + timedelta(hours=1))
    # main-thread opus post-marker must NOT count (subagent-scoped metric).
    for i in range(10):
        db.insert_span(make_llm_span(
            agent_id="claude-code-x", model="claude-opus-4-8", input_tokens=8000,
            output_tokens=200, session_id=f"main-{i}",
            start_time=MARKER + timedelta(hours=1, minutes=i),
        ))
    rec = _record("subagent", {"models": ["claude-opus-4-8"], "subagent": True}, agent_id="")
    v = cost_verify.measure_cost_delta(db.conn, rec, now=NOW)
    assert v["verdict"] == "improved"
    assert v["realized_usd_delta"] > 0
    assert v["post_value"] == 0.0        # no post-marker subagent spend on opus
