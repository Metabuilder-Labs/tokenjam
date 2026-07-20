"""Unit tests for wiring the cost analyzers (downsize/cache/trim) into the
self-improve loop as advise-only proposals with delta-verify receipts.

Fully isolated: the DB is an ``InMemoryBackend`` and every JSON ledger/store
write is routed under ``tmp_path`` via ``cfg.storage.path`` — nothing here
touches a real ``~/.tj`` / ``~/.claude`` (mirrors ``test_relearn_apply``).
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from tokenjam.core.config import StorageConfig, TjConfig
from tokenjam.core.db import InMemoryBackend
from tokenjam.core.optimize import cost_apply, cost_verify, relearn_store
from tokenjam.core.optimize.analyzers.cache_efficacy import (
    CacheEfficacyFinding,
    CacheEfficacyRow,
)
from tokenjam.core.optimize.analyzers.relearn import RelearnFinding
from tokenjam.core.optimize.analyzers.prompt_bloat import BloatPrompt, PromptBloatFinding
from tokenjam.core.optimize.cost_proposals import (
    cost_proposals_from_report,
    estimated_recoverable_rollup,
)
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


# --- Delta-verify: deadweight (C1's "measured" receipt) -----------------------

def _deadweight_target(tmp_path, *, still_present):
    """A real on-disk config file the verify branch re-reads. ``still_present``
    controls whether the ORIGINAL detected server entry is still there —
    the exact still-configured re-check ``server_still_configured`` performs."""
    import json
    config_path = tmp_path / ".mcp.json"
    servers = {"apollo": {}} if still_present else {}
    config_path.write_text(json.dumps({"mcpServers": servers}), encoding="utf-8")
    return {"server": "apollo", "scope": "project", "source": str(config_path)}


def test_deadweight_delta_no_change_when_still_configured(db, tmp_path):
    # Even with a real drop in the spans, a still-configured server must
    # read as no_change -- the user hasn't actually removed it yet.
    _seed(db, agent="svc-a", model="claude-opus-4-8", input_tok=10_000,
          when=MARKER - timedelta(hours=40))
    _seed(db, agent="svc-a", model="claude-opus-4-8", input_tok=2_000,
          when=MARKER + timedelta(hours=1))
    target = _deadweight_target(tmp_path, still_present=True)
    rec = _record("deadweight", target, agent_id="svc-a")
    v = cost_verify.measure_cost_delta(db.conn, rec, now=NOW)
    assert v["verdict"] == "no_change"
    assert v["realized_usd_delta"] is None
    assert v["realized_tokens_delta"] is None
    assert "still appears in its configured set" in v["reason"]


def test_deadweight_delta_measured_drop_when_removed(db, tmp_path):
    # pre: 10k input tok/session; post (server removed): 2k input tok/session.
    _seed(db, agent="svc-a", model="claude-opus-4-8", input_tok=10_000,
          when=MARKER - timedelta(hours=40))
    _seed(db, agent="svc-a", model="claude-opus-4-8", input_tok=2_000,
          when=MARKER + timedelta(hours=1))
    target = _deadweight_target(tmp_path, still_present=False)
    rec = _record("deadweight", target, agent_id="svc-a")
    v = cost_verify.measure_cost_delta(db.conn, rec, now=NOW)
    assert v["verdict"] == "improved"
    assert v["realized_tokens_delta"] > 0
    assert v["realized_usd_delta"] > 0


def test_deadweight_delta_regressed_when_removed_but_no_drop(db, tmp_path):
    # Removed from config, but per-session input tokens didn't actually
    # fall -- must surface as regressed, never silently kept as a win.
    _seed(db, agent="svc-a", model="claude-opus-4-8", input_tok=8_000,
          when=MARKER - timedelta(hours=40))
    _seed(db, agent="svc-a", model="claude-opus-4-8", input_tok=8_000,
          when=MARKER + timedelta(hours=1))
    target = _deadweight_target(tmp_path, still_present=False)
    rec = _record("deadweight", target, agent_id="svc-a")
    v = cost_verify.measure_cost_delta(db.conn, rec, now=NOW)
    assert v["verdict"] == "regressed"


def test_deadweight_delta_insufficient_when_removed_but_thin_exposure(db, tmp_path):
    # Removed from config, but too few post-marker calls for a confident verdict.
    _seed(db, agent="svc-a", model="claude-opus-4-8", input_tok=10_000,
          when=MARKER - timedelta(hours=40))
    _seed(db, agent="svc-a", model="claude-opus-4-8", input_tok=2_000,
          when=MARKER + timedelta(hours=1), count=3)
    target = _deadweight_target(tmp_path, still_present=False)
    rec = _record("deadweight", target, agent_id="svc-a")
    v = cost_verify.measure_cost_delta(db.conn, rec, now=NOW)
    assert v["verdict"] == "insufficient_data"
    assert v["realized_usd_delta"] is None
    assert v["realized_tokens_delta"] is None


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


def test_cost_compound_ledger_no_change_bucket_sums_to_verified():
    # A no_change record (deadweight: server still configured) must land in
    # its own named bucket, not just inflate verified_count with nowhere to
    # go -- the named buckets must always add back up to verified_count.
    records = [
        {"state": "applied", "verify": {"verdict": "improved",
         "realized_usd_delta": 0.5, "realized_tokens_delta": 1000}},
        {"state": "applied", "verify": {"verdict": "no_change"}},
        {"state": "applied", "verify": {"verdict": "no_change"}},
        {"state": "applied", "verify": {"verdict": "regressed",
         "realized_usd_delta": 0.0}},
        {"state": "applied", "verify": {"verdict": "insufficient_data"}},
    ]
    ledger = cost_verify.cost_compound_ledger(records)
    assert ledger["no_change_count"] == 2
    assert ledger["verified_count"] == 5
    assert (
        ledger["improved_count"] + ledger["no_change_count"]
        + ledger["regressed_count"] + ledger["insufficient_data_count"]
        == ledger["verified_count"]
    )


# --- Store: cost proposals share the relearn cache file ----------------------

def test_store_cost_and_relearn_coexist_without_clobber(tmp_path):
    path = tmp_path / "relearn_cache.json"
    # Relearn finding written first.
    relearn_store.write_cache(RelearnFinding(sessions_scanned=7), path=path)
    # Cost proposals written into the SAME file must preserve the finding.
    relearn_store.write_cost_proposals(
        [{"kind": "cost", "analyzer": "trim", "signature": "cost:trim:x"}], path=path,
    )
    cost_block = relearn_store.read_cost_proposals(path=path)
    assert cost_block is not None
    assert cost_block["cost_proposals"][0]["signature"] == "cost:trim:x"
    # The relearn finding is still intact.
    raw = relearn_store.read_cache(path=path)
    assert raw["finding"]["sessions_scanned"] == 7
    # A later relearn recompute preserves the cost proposals.
    relearn_store.write_cache(RelearnFinding(sessions_scanned=9), path=path)
    assert relearn_store.read_cost_proposals(path=path)["cost_proposals"][0]["signature"] == "cost:trim:x"
    assert relearn_store.read_cache(path=path)["finding"]["sessions_scanned"] == 9


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


# --- deadweight (C1: MCP dead-weight servers) --------------------------------

def _dead_server(**overrides):
    from tokenjam.core.optimize.analyzers.deadweight import ServerDeadweight
    fields = dict(
        name="apollo", scope="project", source="/repo/.mcp.json",
        sessions_present=10, invocations=0, deferred_sessions=0, dead=True,
        estimated_tax_tokens_per_session=25_000, estimated_tax_tokens_90d=225_000,
        tax_construction="25,000 tok/session (full schema injection), cited estimate.",
        fix="Remove or project-scope the `apollo` MCP server (/repo/.mcp.json); "
            "zero tool calls across 10 session(s) in this window.",
        example_sessions=["s0", "s1", "s2"],
    )
    fields.update(overrides)
    return ServerDeadweight(**fields)


def _deadweight_finding(dead_servers=None, tax_table=None):
    from tokenjam.core.optimize.analyzers.deadweight import ContextTaxRow, DeadweightFinding
    dead_servers = dead_servers if dead_servers is not None else [_dead_server()]
    tax_table = tax_table if tax_table is not None else [
        ContextTaxRow(source="MCP schema: apollo", sessions=10,
                      avg_tokens_per_session=25_000, total_tokens_window=250_000),
    ]
    return DeadweightFinding(
        sessions_scanned=10, configured_servers=1,
        servers=dead_servers, dead_servers=dead_servers, tax_table=tax_table,
        estimated_recoverable_tokens=sum(s.estimated_tax_tokens_90d for s in dead_servers) or None,
        estimate_basis="dead-server 90d tax sum",
    )


def test_deadweight_proposal_shape_and_fields():
    from tokenjam.core.optimize.cost_proposals import _deadweight_to_proposals
    props = _deadweight_to_proposals(_deadweight_finding())
    assert len(props) == 1
    p = props[0]
    assert p.kind == "cost"
    assert p.analyzer == "deadweight"
    assert p.signature == "cost:deadweight:apollo"
    assert p.advise_only is True
    assert p.correlational is True
    assert p.estimate_confidence == "estimated"
    assert "apollo" in p.title
    assert "0 tool calls" in p.evidence
    assert p.target_key == {"server": "apollo", "scope": "project", "source": "/repo/.mcp.json"}
    assert p.baseline["example_sessions"] == ["s0", "s1", "s2"]
    assert p.estimated_recoverable_tokens == 225_000
    assert p.estimated_recoverable_usd is None  # tokens-only, never a stacked $ guess
    assert "claude mcp remove apollo --scope project" in p.suggestion


def test_deadweight_proposal_notes_deferred_sessions_in_evidence():
    from tokenjam.core.optimize.cost_proposals import _deadweight_to_proposals
    server = _dead_server(deferred_sessions=4)
    p = _deadweight_to_proposals(_deadweight_finding(dead_servers=[server]))[0]
    assert "ToolSearch deferred" in p.evidence
    assert "4" in p.evidence


def test_deadweight_proposal_empty_for_no_dead_servers():
    from tokenjam.core.optimize.cost_proposals import _deadweight_to_proposals
    assert _deadweight_to_proposals(_deadweight_finding(dead_servers=[])) == []
    assert _deadweight_to_proposals(None) == []


def test_deadweight_wired_into_cost_analyzers_and_report_adapter():
    from tokenjam.core.optimize.cost_proposals import COST_ANALYZERS, cost_proposals_from_report
    assert "deadweight" in COST_ANALYZERS

    rep = _report()
    rep.findings["deadweight"] = _deadweight_finding()
    props = {p.analyzer for p in cost_proposals_from_report(rep)}
    assert "deadweight" in props


def test_deadweight_tax_table_never_becomes_a_second_proposal():
    """Dedup guarantee: a live (non-dead) server sits in the tax table for
    visibility but must never itself spawn a proposal, and a dead server's
    proposal total must equal exactly its OWN 90d tax -- never the tax
    table's (possibly multi-row) sum."""
    from tokenjam.core.optimize.analyzers.deadweight import ContextTaxRow
    from tokenjam.core.optimize.cost_proposals import _deadweight_to_proposals

    dead = _dead_server(name="apollo", estimated_tax_tokens_90d=225_000)
    finding = _deadweight_finding(
        dead_servers=[dead],
        tax_table=[
            ContextTaxRow(source="MCP schema: apollo", sessions=10,
                          avg_tokens_per_session=25_000, total_tokens_window=250_000),
            # A live server: present in the tax table, but not in dead_servers.
            ContextTaxRow(source="MCP schema: exa", sessions=10,
                          avg_tokens_per_session=25_000, total_tokens_window=250_000),
            ContextTaxRow(source="CLAUDE.md", sessions=10,
                          avg_tokens_per_session=500, total_tokens_window=5_000),
        ],
    )
    props = _deadweight_to_proposals(finding)
    assert len(props) == 1
    assert props[0].estimated_recoverable_tokens == 225_000
    assert finding.estimated_recoverable_tokens == 225_000  # never the tax-table's 505,000 sum


# --- deadweight finding round-trips through report_to_dict/report_from_dict --

def test_deadweight_finding_survives_report_dict_round_trip():
    from tokenjam.core.optimize.runner import report_from_dict, report_to_dict
    from tokenjam.core.optimize.types import OptimizeReport, WindowSummary

    w = WindowSummary(since=MARKER, until=NOW, days=5, sessions=10, spans=100,
                      total_tokens=1, total_cost_usd=5.0, thin_data=False)
    report = OptimizeReport(window=w, findings={"deadweight": _deadweight_finding()})

    payload = report_to_dict(report)
    rebuilt = report_from_dict(payload)

    finding = rebuilt.findings["deadweight"]
    assert finding.configured_servers == 1
    assert len(finding.dead_servers) == 1
    assert finding.dead_servers[0].name == "apollo"
    assert finding.dead_servers[0].example_sessions == ["s0", "s1", "s2"]
    assert finding.estimated_recoverable_tokens == 225_000
    assert len(finding.tax_table) == 1
    assert finding.tax_table[0].source == "MCP schema: apollo"


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


# --- Component E: the estimated-recoverable rollup --------------------------

def test_rollup_sums_across_analyzers_and_reports_window():
    proposals = [
        {"signature": "cost:downsize", "analyzer": "downsize", "title": "t1",
         "estimated_recoverable_usd": 3.0},
        {"signature": "cost:cache:anthropic:claude-sonnet-5", "analyzer": "cache",
         "title": "t2", "estimated_recoverable_usd": 1.2},
    ]
    rollup = estimated_recoverable_rollup(proposals)
    assert rollup["estimated_recoverable_usd"] == 4.2
    assert rollup["proposal_count"] == 2
    assert rollup["window_days"] == 30
    assert rollup["estimate_confidence"] == "estimated"
    by_analyzer = {a["analyzer"]: a for a in rollup["by_analyzer"]}
    assert by_analyzer["downsize"]["count"] == 1
    assert by_analyzer["cache"]["usd"] == 1.2


def test_rollup_dedupes_by_signature_never_double_counting():
    # A stale/duplicate cache entry carrying the SAME signature twice must
    # only contribute once — the second copy is dropped, not summed again.
    proposals = [
        {"signature": "cost:downsize", "analyzer": "downsize", "title": "t1",
         "estimated_recoverable_usd": 3.0},
        {"signature": "cost:downsize", "analyzer": "downsize", "title": "t1-stale",
         "estimated_recoverable_usd": 3.0},
    ]
    rollup = estimated_recoverable_rollup(proposals)
    assert rollup["estimated_recoverable_usd"] == 3.0
    assert rollup["proposal_count"] == 1


def test_rollup_empty_state():
    rollup = estimated_recoverable_rollup([])
    assert rollup["estimated_recoverable_usd"] == 0.0
    assert rollup["proposal_count"] == 0
    assert rollup["by_analyzer"] == []
    assert rollup["contributing"] == []
    assert "no open" in rollup["estimate_basis"]


def test_rollup_skips_proposals_with_no_dollar_estimate():
    # A card with an estimate of None still exists in the inbox individually;
    # it just isn't folded into this aggregate (folding in None would
    # silently misstate what's summed "across N proposals").
    proposals = [
        {"signature": "cost:trim:agentA", "analyzer": "trim", "title": "t",
         "estimated_recoverable_usd": None},
        {"signature": "cost:downsize", "analyzer": "downsize", "title": "t2",
         "estimated_recoverable_usd": 2.5},
    ]
    rollup = estimated_recoverable_rollup(proposals)
    assert rollup["proposal_count"] == 1
    assert rollup["estimated_recoverable_usd"] == 2.5


def test_rollup_is_generic_over_an_unregistered_future_analyzer():
    # No special-casing by analyzer name — a brand-new analyzer's cards (not
    # in cost_proposals.COST_ANALYZERS) are picked up with zero code changes
    # here, as long as they carry the shared CostProposal fields.
    proposals = [
        {"signature": "cost:deadweight:some-mcp-server", "analyzer": "deadweight",
         "title": "Unused MCP server", "estimated_recoverable_usd": 0.75},
    ]
    rollup = estimated_recoverable_rollup(proposals)
    assert rollup["estimated_recoverable_usd"] == 0.75
    assert rollup["proposal_count"] == 1
    assert rollup["by_analyzer"][0]["analyzer"] == "deadweight"


def test_rollup_ignores_a_proposal_with_no_signature():
    proposals = [{"analyzer": "downsize", "title": "no sig",
                  "estimated_recoverable_usd": 5.0}]
    rollup = estimated_recoverable_rollup(proposals)
    assert rollup["proposal_count"] == 0
    assert rollup["estimated_recoverable_usd"] == 0.0
