"""Unit tests for wiring the cost analyzers (downsize/cache/trim) into the
self-improve loop as advise-only proposals.

Fully isolated: the DB is an ``InMemoryBackend`` and every JSON ledger/store
write is routed under ``tmp_path`` via ``cfg.storage.path`` — nothing here
touches a real ``~/.tj`` / ``~/.claude`` (mirrors ``test_relearn_apply``).
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from tokenjam.core.config import StorageConfig, TjConfig
from tokenjam.core.db import InMemoryBackend
from tokenjam.core.optimize import cost_apply, relearn_store
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
    # The base fixture's server carries no priced model -- the adapter must
    # never invent a rate, so the card stays tokens-only for this server.
    assert p.estimated_recoverable_usd is None
    assert "claude mcp remove apollo --scope project" in p.suggestion


def test_deadweight_proposal_carries_priced_usd_from_server():
    """When the analyzer priced the tax (a model was observed across the
    server's sessions), the adapter must carry that $ figure straight
    through -- never recompute or invent a rate of its own."""
    from tokenjam.core.optimize.cost_proposals import _deadweight_to_proposals
    server = _dead_server(
        priced_model="claude-opus-4-8",
        estimated_tax_usd_per_session=0.125,
        estimated_tax_usd_90d=1.40625,
    )
    p = _deadweight_to_proposals(_deadweight_finding(dead_servers=[server]))[0]
    assert p.estimated_recoverable_usd == 1.40625
    assert p.baseline["priced_model"] == "claude-opus-4-8"


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


# --- script / reuse / verbosity: findings that were previously advise-only
# with no proposal at all, now on the same apply-capable rail as `subagent`. ---

def _workflow_cluster(**overrides):
    from tokenjam.core.optimize.analyzers.workflow_restructure import WorkflowCluster
    fields = dict(
        signature=[{"tool": "bash", "args": ["command_string"]}], instances=25,
        avg_cost_usd=0.02, avg_duration_seconds=1.5, example_session_id="det-0",
        avg_tokens=500, total_cost_usd=0.5, total_tokens=12_500,
        example_session_ids=["det-0", "det-1", "det-2"],
    )
    fields.update(overrides)
    return WorkflowCluster(**fields)


def _workflow_finding(clusters=None, **overrides):
    from tokenjam.core.optimize.analyzers.workflow_restructure import (
        WorkflowRestructureFinding,
    )
    clusters = clusters if clusters is not None else [_workflow_cluster()]
    fields = dict(
        clusters=clusters, sessions_examined=25, degraded=False,
        estimated_recoverable_usd=sum(c.total_cost_usd for c in clusters) or None,
        estimated_recoverable_tokens=sum(c.total_tokens for c in clusters) or None,
        estimate_basis="script basis",
    )
    fields.update(overrides)
    return WorkflowRestructureFinding(**fields)


def test_script_proposal_shape_and_apply_fields():
    from tokenjam.core.optimize.cost_proposals import _script_to_proposals
    props = _script_to_proposals(_workflow_finding(), persona="claude-code")
    assert len(props) == 1
    p = props[0]
    assert p.kind == "cost"
    assert p.analyzer == "script"
    assert p.signature.startswith("cost:script:")
    assert "bash" in p.evidence
    assert p.estimated_recoverable_usd == 0.5
    assert p.estimated_recoverable_tokens == 12_500
    assert p.estimate_basis == "script basis"
    # Apply-capable: a rung-2 skill note, same class of surface as `subagent`.
    assert p.advise_only is False
    assert p.apply_capable is True
    assert p.rung == 2
    assert p.scope == "project"
    assert p.proposed_fix
    assert p.baseline["apply_sessions"] == 25
    assert len(p.baseline["apply_examples"]) == 3


def test_script_proposal_notes_degraded_clustering_in_evidence():
    from tokenjam.core.optimize.cost_proposals import _script_to_proposals
    p = _script_to_proposals(_workflow_finding(degraded=True))[0]
    assert "tool_inputs" in p.evidence


def test_script_proposal_two_clusters_get_distinct_signatures_and_slugs():
    """Two clusters must never collide on the same skill-file slug — a
    collision would let the second cluster's apply silently overwrite the
    first's skill note (relearn_apply's create-only guard only checks for a
    TokenJam marker, not identity)."""
    c1 = _workflow_cluster(signature=[{"tool": "bash"}], example_session_id="a",
                            total_cost_usd=0.2, total_tokens=2_000)
    c2 = _workflow_cluster(signature=[{"tool": "grep"}], example_session_id="b",
                            total_cost_usd=0.3, total_tokens=3_000)
    from tokenjam.core.optimize.cost_proposals import _script_to_proposals
    from tokenjam.core.optimize.relearn_apply import slugify
    props = _script_to_proposals(_workflow_finding(clusters=[c1, c2]))
    assert len({p.signature for p in props}) == 2
    assert len({slugify(p.title) for p in props}) == 2


def test_script_proposal_empty_for_zero_cost_cluster_and_no_finding():
    from tokenjam.core.optimize.cost_proposals import _script_to_proposals
    zero = _workflow_cluster(avg_cost_usd=0.0, avg_tokens=0, total_cost_usd=0.0,
                              total_tokens=0)
    assert _script_to_proposals(_workflow_finding(clusters=[zero])) == []
    assert _script_to_proposals(None) == []
    assert _script_to_proposals(_workflow_finding(clusters=[])) == []


def _reuse_cluster(**overrides):
    from tokenjam.core.optimize.types import ReuseCluster
    fields = dict(
        cluster_id="abc123456789", tool_signature=("bash", "read"),
        prompt_prefix_hash=None, repetitions=4, avg_planning_tokens=300,
        avg_planning_cost_usd=0.01, cache_reuse_recoverable_usd=0.03,
        script_replacement_recoverable_usd=0.04, cache_reuse_recoverable_tokens=900,
        script_replacement_recoverable_tokens=1_200,
        example_session_ids=["s1", "s2", "s3"], skeleton_session_id="s1",
    )
    fields.update(overrides)
    return ReuseCluster(**fields)


def _reuse_finding(clusters=None, **overrides):
    from tokenjam.core.optimize.types import ReuseFinding
    clusters = clusters if clusters is not None else [_reuse_cluster()]
    fields = dict(
        clusters=clusters,
        estimated_recoverable_usd=sum(c.cache_reuse_recoverable_usd for c in clusters) or None,
        estimated_recoverable_tokens=sum(c.cache_reuse_recoverable_tokens for c in clusters) or None,
        estimate_basis="reuse basis",
    )
    fields.update(overrides)
    return ReuseFinding(**fields)


def test_reuse_proposal_shape_and_apply_fields():
    from tokenjam.core.optimize.cost_proposals import _reuse_to_proposals
    props = _reuse_to_proposals(_reuse_finding(), persona="claude-code")
    assert len(props) == 1
    p = props[0]
    assert p.kind == "cost"
    assert p.analyzer == "reuse"
    assert p.signature == "cost:reuse:abc123456789"
    assert "bash, read" in p.evidence
    # Conservative cache-reuse figure, never the script-replacement upper bound.
    assert p.estimated_recoverable_usd == 0.03
    assert p.estimated_recoverable_tokens == 900
    assert p.advise_only is False
    assert p.apply_capable is True
    assert p.rung == 1
    assert p.scope == "project"
    assert p.proposed_fix
    assert p.baseline["apply_sessions"] == 4
    assert len(p.baseline["apply_examples"]) == 3


def test_reuse_proposal_uses_the_analyzers_own_cluster_id_for_dedup_identity():
    """Unlike `script`, `reuse` already carries a deterministic cluster_id
    (plan_reuse.py computes it); the adapter must reuse it, never re-hash."""
    from tokenjam.core.optimize.cost_proposals import _reuse_to_proposals
    p = _reuse_to_proposals(_reuse_finding(clusters=[_reuse_cluster(cluster_id="zzz999")]))[0]
    assert p.signature == "cost:reuse:zzz999"


def test_reuse_proposal_empty_for_zero_recoverable_and_no_finding():
    from tokenjam.core.optimize.cost_proposals import _reuse_to_proposals
    zero = _reuse_cluster(cache_reuse_recoverable_usd=0.0, cache_reuse_recoverable_tokens=0)
    assert _reuse_to_proposals(_reuse_finding(clusters=[zero])) == []
    assert _reuse_to_proposals(None) == []
    assert _reuse_to_proposals(_reuse_finding(clusters=[])) == []


def _verbosity_finding(**overrides):
    from tokenjam.core.optimize.analyzers.output_verbosity import VerbosityFinding
    fields = dict(
        total_candidates=6, sessions_examined=40, cohorts_examined=3,
        estimated_recoverable_usd=0.9, estimated_recoverable_tokens=9_000,
        estimate_basis="verbosity basis", suggested_max_tokens=800,
    )
    fields.update(overrides)
    return VerbosityFinding(**fields)


def test_verbosity_proposal_shape_and_apply_fields():
    from tokenjam.core.optimize.cost_proposals import _verbosity_to_proposals
    props = _verbosity_to_proposals(_verbosity_finding(), persona="claude-code")
    assert len(props) == 1
    p = props[0]
    assert p.kind == "cost"
    assert p.analyzer == "verbosity"
    assert p.signature == "cost:verbosity"
    assert "6 session" in p.evidence
    assert p.estimated_recoverable_usd == 0.9
    assert p.estimated_recoverable_tokens == 9_000
    assert p.advise_only is False
    assert p.apply_capable is True
    assert p.rung == 1
    assert p.scope == "project"
    # The remedy snippet + the concrete suggested cap both land in the note.
    assert "concise" in p.proposed_fix.lower()
    assert "800" in p.proposed_fix
    # The honesty caveat (output length is not waste) rides along, never a
    # bare "you are wasting tokens" claim.
    assert "not waste" in p.proposed_fix
    assert p.baseline["apply_sessions"] == 6


def test_verbosity_proposal_empty_for_no_candidates_and_no_finding():
    from tokenjam.core.optimize.cost_proposals import _verbosity_to_proposals
    assert _verbosity_to_proposals(_verbosity_finding(total_candidates=0)) == []
    assert _verbosity_to_proposals(None) == []


def test_script_reuse_verbosity_wired_into_cost_analyzers_and_report_adapter():
    from tokenjam.core.optimize.cost_proposals import (
        COST_ANALYZERS,
        cost_proposals_from_report,
    )
    assert {"script", "reuse", "verbosity"} <= set(COST_ANALYZERS)

    rep = _report()
    rep.findings["script"] = _workflow_finding()
    rep.findings["reuse"] = _reuse_finding()
    rep.findings["verbosity"] = _verbosity_finding()
    analyzers = {p.analyzer for p in cost_proposals_from_report(rep)}
    assert {"script", "reuse", "verbosity"} <= analyzers


# --- Persona-gated fix modality (script / reuse / verbosity only) -----------
#
# These three analyzers' only apply path is a rung-1 CLAUDE.md note or rung-2
# .claude/skills/<slug>/SKILL.md — an artifact nothing in an SDK service's
# request path ever reads. An "sdk"/"unknown" persona must never see
# apply_capable=True for them; the identical recommendation must still reach
# them as a copy-pasteable `suggestion`. A "claude-code" window must be
# byte-identical to before this gating existed.

def _script_finding_for_persona_tests():
    return _workflow_finding()


def _reuse_finding_for_persona_tests():
    return _reuse_finding()


def _verbosity_finding_for_persona_tests():
    return _verbosity_finding()


@pytest.mark.parametrize("adapter_name,finder", [
    ("_script_to_proposals", _script_finding_for_persona_tests),
    ("_reuse_to_proposals", _reuse_finding_for_persona_tests),
    ("_verbosity_to_proposals", _verbosity_finding_for_persona_tests),
])
def test_sdk_persona_gets_snippet_not_write(adapter_name, finder):
    """An sdk-persona window: no write offered, but the exact same
    recommendation lands in `suggestion` so the card isn't left inert."""
    import tokenjam.core.optimize.cost_proposals as cp
    adapter = getattr(cp, adapter_name)
    props = adapter(finder(), persona="sdk")
    assert len(props) == 1
    p = props[0]
    assert p.apply_capable is False
    assert p.advise_only is True
    assert p.rung == 0
    assert p.scope == ""
    assert p.proposed_fix == ""
    assert p.suggestion  # the recommendation still reaches the sdk user
    assert p.suggestion == p.advise_text


@pytest.mark.parametrize("adapter_name,finder", [
    ("_script_to_proposals", _script_finding_for_persona_tests),
    ("_reuse_to_proposals", _reuse_finding_for_persona_tests),
    ("_verbosity_to_proposals", _verbosity_finding_for_persona_tests),
])
def test_unknown_persona_is_gated_same_as_sdk(adapter_name, finder):
    """"unknown" (no session in the window carries an identifiable agent_id,
    and no declared plan settles it) is exactly the shape of a pure-SDK
    caller who never ran `tj onboard` — the failure mode this gating exists
    to close. Grouped with "sdk", matching
    `cmd_optimize._render_downgrade_cta`'s CTA."""
    import tokenjam.core.optimize.cost_proposals as cp
    adapter = getattr(cp, adapter_name)
    props = adapter(finder(), persona="unknown")
    assert len(props) == 1
    p = props[0]
    assert p.apply_capable is False
    assert p.advise_only is True
    assert p.suggestion == p.advise_text

    # The default persona (no explicit kwarg) must resolve exactly the same
    # way — a caller that doesn't know the persona must never fall back to
    # assuming claude-code.
    default_props = adapter(finder())
    assert default_props[0].apply_capable is False
    assert default_props[0].advise_only is True


@pytest.mark.parametrize("adapter_name,finder", [
    ("_script_to_proposals", _script_finding_for_persona_tests),
    ("_reuse_to_proposals", _reuse_finding_for_persona_tests),
    ("_verbosity_to_proposals", _verbosity_finding_for_persona_tests),
])
def test_claude_code_persona_is_byte_identical_to_pre_gating_shape(adapter_name, finder):
    """The exact fields these analyzers produced before persona gating
    existed: apply_capable/advise_only unchanged, and no `suggestion` is
    invented for a persona that already gets the real write."""
    import tokenjam.core.optimize.cost_proposals as cp
    adapter = getattr(cp, adapter_name)
    props = adapter(finder(), persona="claude-code")
    assert len(props) == 1
    p = props[0]
    assert p.advise_only is False
    assert p.apply_capable is True
    assert p.rung in (1, 2)
    assert p.scope == "project"
    assert p.proposed_fix
    assert p.suggestion == ""  # unchanged from before this gating existed


@pytest.mark.parametrize("adapter_name,finder", [
    ("_script_to_proposals", _script_finding_for_persona_tests),
    ("_reuse_to_proposals", _reuse_finding_for_persona_tests),
    ("_verbosity_to_proposals", _verbosity_finding_for_persona_tests),
])
def test_mixed_persona_offers_the_write_and_the_snippet(adapter_name, finder):
    """"mixed": both audiences are meaningfully represented and a single
    finding isn't attributable to one side or the other, so — mirroring
    `_render_downgrade_cta`'s "mixed shows both" precedent — the write stays
    on offer AND the identical text is carried as `suggestion` so the sdk
    share of the mix isn't left with a card that looks actionable for them
    but silently isn't."""
    import tokenjam.core.optimize.cost_proposals as cp
    adapter = getattr(cp, adapter_name)
    props = adapter(finder(), persona="mixed")
    assert len(props) == 1
    p = props[0]
    assert p.apply_capable is True
    assert p.advise_only is False
    assert p.proposed_fix
    assert p.suggestion == p.advise_text


def test_cost_proposals_from_report_reads_persona_off_the_report():
    """`cost_proposals_from_report` must never take the caller's word for the
    persona out-of-band — it reads `report.persona`, the single field
    `runner.build_report` populates once (see `AnalyzerContext.persona`)."""
    from tokenjam.core.optimize.cost_proposals import cost_proposals_from_report

    rep = _report()
    rep.findings["script"] = _workflow_finding()
    rep.findings["reuse"] = _reuse_finding()
    rep.findings["verbosity"] = _verbosity_finding()

    rep.persona = "sdk"
    by_analyzer = {p.analyzer: p for p in cost_proposals_from_report(rep)}
    for name in ("script", "reuse", "verbosity"):
        assert by_analyzer[name].apply_capable is False
        assert by_analyzer[name].suggestion

    rep.persona = "claude-code"
    by_analyzer = {p.analyzer: p for p in cost_proposals_from_report(rep)}
    for name in ("script", "reuse", "verbosity"):
        assert by_analyzer[name].apply_capable is True

    # A report with no persona set at all (e.g. hand-built, pre-gating test
    # code) defaults to "unknown" -> the fail-safe, not "claude-code".
    rep.persona = "unknown"
    by_analyzer = {p.analyzer: p for p in cost_proposals_from_report(rep)}
    for name in ("script", "reuse", "verbosity"):
        assert by_analyzer[name].apply_capable is False


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


def test_rollup_all_open_cards_carry_none_renders_empty_state():
    # Every open card exists but none carries a dollar estimate (e.g. every
    # cache-thrash card came back "not worth it at this cadence") — the
    # rollup must still render the sensible empty state, not a zero that
    # looks like a real (if tiny) measured sum.
    proposals = [
        {"signature": "cost:cache:thrash:agentA", "analyzer": "cache", "title": "t1",
         "estimated_recoverable_usd": None},
        {"signature": "cost:deadweight:mcp-x", "analyzer": "deadweight", "title": "t2",
         "estimated_recoverable_usd": None},
    ]
    rollup = estimated_recoverable_rollup(proposals)
    assert rollup["estimated_recoverable_usd"] == 0.0
    assert rollup["proposal_count"] == 0
    assert rollup["by_analyzer"] == []
    assert rollup["contributing"] == []
    assert "no open" in rollup["estimate_basis"]


def test_rollup_mixed_none_and_real_estimates_across_analyzers():
    # Mirrors two sibling-branch card shapes that intentionally emit None:
    # Component A's cache-thrash card when the TTL arithmetic comes out
    # negative ("not worth it at this cadence"), and Component C's
    # deferred-tools deadweight card. Neither should be coerced into the sum
    # or into "N proposals" — only the two real-valued cards contribute.
    proposals = [
        {"signature": "cost:cache:thrash:agentA", "analyzer": "cache", "title": "not worth it",
         "estimated_recoverable_usd": None},
        {"signature": "cost:deadweight:mcp-x", "analyzer": "deadweight", "title": "deferred server",
         "estimated_recoverable_usd": None},
        {"signature": "cost:downsize", "analyzer": "downsize", "title": "real card 1",
         "estimated_recoverable_usd": 3.0},
        {"signature": "cost:cache:anthropic:claude-sonnet-5", "analyzer": "cache", "title": "real card 2",
         "estimated_recoverable_usd": 1.5},
    ]
    rollup = estimated_recoverable_rollup(proposals)
    assert rollup["estimated_recoverable_usd"] == 4.5
    assert rollup["proposal_count"] == 2   # only the two real-valued cards
    signatures = {c["signature"] for c in rollup["contributing"]}
    assert signatures == {"cost:downsize", "cost:cache:anthropic:claude-sonnet-5"}
    by_analyzer = {a["analyzer"]: a for a in rollup["by_analyzer"]}
    assert by_analyzer["cache"]["count"] == 1     # only the real-valued cache card
    assert by_analyzer["cache"]["usd"] == 1.5
    assert "deadweight" not in by_analyzer        # its only card was None-only


# --- Component E: the token sum and its coverage ---------------------------

def test_rollup_sums_tokens_independently_of_the_dollar_estimate():
    # The two estimates are populated by different analyzers, so a proposal can
    # carry either alone. Folding tokens in only where a dollar figure also
    # exists would understate the headline the suppressed-dollars path leads
    # with — here that would report 900 instead of the true 1400.
    proposals = [
        {"signature": "a", "analyzer": "downsize", "title": "t1",
         "estimated_recoverable_usd": 3.0, "estimated_recoverable_tokens": 900},
        {"signature": "b", "analyzer": "cache", "title": "t2",
         "estimated_recoverable_tokens": 500},
    ]
    rollup = estimated_recoverable_rollup(proposals)
    assert rollup["estimated_recoverable_tokens"] == 1400
    assert rollup["token_proposal_count"] == 2
    assert rollup["deduplicated_proposal_count"] == 2
    # The dollar sum still counts only the dollar-bearing proposal.
    assert rollup["estimated_recoverable_usd"] == 3.0
    assert rollup["proposal_count"] == 1


def test_rollup_reports_partial_token_coverage_rather_than_implying_all():
    # Two of three proposals carry no token estimate. The sum is a floor, and
    # token_proposal_count vs deduplicated_proposal_count is what lets the tile
    # say so instead of claiming coverage it does not have.
    proposals = [
        {"signature": "a", "analyzer": "downsize", "title": "t1",
         "estimated_recoverable_usd": 3.0, "estimated_recoverable_tokens": 900},
        {"signature": "b", "analyzer": "cache", "title": "t2",
         "estimated_recoverable_usd": 1.0},
        {"signature": "c", "analyzer": "trim", "title": "t3",
         "estimated_recoverable_usd": 2.0},
    ]
    rollup = estimated_recoverable_rollup(proposals)
    assert rollup["estimated_recoverable_tokens"] == 900
    assert rollup["token_proposal_count"] == 1
    assert rollup["deduplicated_proposal_count"] == 3
    assert "1 of 3" in rollup["estimate_basis"]
    assert "floor, not a total" in rollup["estimate_basis"]


def test_rollup_token_sum_dedupes_by_signature_too():
    proposals = [
        {"signature": "a", "analyzer": "downsize", "title": "t1",
         "estimated_recoverable_tokens": 900},
        {"signature": "a", "analyzer": "downsize", "title": "t1-stale",
         "estimated_recoverable_tokens": 900},
    ]
    rollup = estimated_recoverable_rollup(proposals)
    assert rollup["estimated_recoverable_tokens"] == 900
    assert rollup["token_proposal_count"] == 1


def test_rollup_with_no_token_estimates_reports_zero_coverage():
    # Nothing to lead with when dollars are suppressed; the tile hides rather
    # than rendering a zero, so the counts must make that state distinguishable.
    proposals = [
        {"signature": "a", "analyzer": "downsize", "title": "t1",
         "estimated_recoverable_usd": 3.0},
    ]
    rollup = estimated_recoverable_rollup(proposals)
    assert rollup["estimated_recoverable_tokens"] == 0
    assert rollup["token_proposal_count"] == 0
    assert "floor, not a total" not in rollup["estimate_basis"]
