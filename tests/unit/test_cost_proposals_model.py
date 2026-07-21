"""Cards for the two model-routing write paths, and the batch placement card.

Covers what the user actually reads: the arithmetic on the card, whether a
direct apply is offered or the one-paste artifact takes over, and the house
rules every runtime string has to hold to.
"""
from __future__ import annotations

import subprocess
from datetime import datetime, timedelta, timezone

import pytest

from tokenjam.core.config import AgentConfig, StorageConfig, TjConfig
from tokenjam.core.optimize import cost_proposals as cp
from tokenjam.core.optimize.analyzers.batch_placement import (
    BatchCandidate,
    BatchPlacementFinding,
)
from tokenjam.core.optimize.analyzers.downsize_agents import build_agent_price_rows
from tokenjam.core.optimize.analyzers.subagent_rightsizing import (
    SubagentRightsizingFinding,
    SubagentRow,
)
from tokenjam.core.optimize.model_apply import (
    APPLY_KIND_AGENT_MODEL,
    APPLY_KIND_MODEL_SWAP,
)
from tokenjam.core.optimize.types import DowngradeFinding, OptimizeReport, WindowSummary

NOW = datetime(2026, 7, 21, 12, 0, tzinfo=timezone.utc)

AGENT_FILE = """---
name: explore
model: claude-opus-4-8
---

Body.
"""


@pytest.fixture
def db_backend():
    from tokenjam.core.db import InMemoryBackend

    backend = InMemoryBackend()
    yield backend
    backend.close()


def _price_rows(agent_id="svc-a"):
    return build_agent_price_rows([{
        "session_id": "s1", "agent_id": agent_id, "provider": "anthropic",
        "model": "claude-opus-4-8", "alt_model": "claude-haiku-4-5",
        "input_tokens": 100_000, "output_tokens": 20_000,
        "cache_tokens": 500_000, "cache_write_tokens": 40_000,
    }], 30.0)


def _downsize_finding(agent_id="svc-a"):
    return DowngradeFinding(
        candidate_sessions=4, total_sessions=10, actual_cost_usd=5.0,
        alternative_cost_usd=2.0, monthly_savings_usd=3.0, percent_of_sessions=40.0,
        examples=[], suggestions={"claude-opus-4-8": "claude-haiku-4-5"},
        estimated_recoverable_usd=3.0, percent_of_tokens=35.0,
        estimate_basis="downsize basis", per_agent=_price_rows(agent_id),
    )


def _report(**findings):
    window = WindowSummary(
        since=NOW - timedelta(days=30), until=NOW, days=30, sessions=10,
        spans=100, total_tokens=1, total_cost_usd=10.0, thin_data=False,
    )
    return OptimizeReport(
        window=window, downgrade=findings.pop("downgrade", None), findings=findings,
    )


def _cfg(tmp_path, agents=None):
    return TjConfig(
        version="1",
        storage=StorageConfig(path=str(tmp_path / "t.duckdb")),
        agents=agents or {},
    )


def _git_repo(tmp_path, name="repo"):
    repo = tmp_path / name
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "t@example.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "T"], cwd=repo, check=True)
    return repo


def _commit_all(repo):
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=repo, check=True)


# --------------------------------------------------------------------------- #
# B1: per-agent arithmetic replaces the window-wide card
# --------------------------------------------------------------------------- #

def test_per_agent_cards_replace_the_aggregate_card(tmp_path):
    props = cp.cost_proposals_from_report(
        _report(downgrade=_downsize_finding()), config=_cfg(tmp_path),
    )
    downsize = [p for p in props if p.analyzer == "downsize"]
    assert [p.signature for p in downsize] == ["cost:downsize:svc-a"]
    row = _price_rows()[0]
    assert downsize[0].estimated_recoverable_usd == row.delta_usd
    assert downsize[0].estimated_recoverable_tokens == row.total_tokens
    # Both sides of the comparison are printed, not just the difference.
    assert "claude-opus-4-8" in downsize[0].evidence
    assert "claude-haiku-4-5" in downsize[0].evidence
    assert "cache read" in downsize[0].evidence and "cache write" in downsize[0].evidence
    assert downsize[0].estimate_basis


def test_finding_without_price_rows_keeps_the_aggregate_card(tmp_path):
    finding = _downsize_finding()
    finding.per_agent = []
    props = cp.cost_proposals_from_report(_report(downgrade=finding), config=_cfg(tmp_path))
    assert [p.signature for p in props if p.analyzer == "downsize"] == ["cost:downsize"]


# --------------------------------------------------------------------------- #
# B1b: the direct apply is offered only when every precondition holds
# --------------------------------------------------------------------------- #

def test_registered_clean_repo_offers_the_model_swap(tmp_path):
    repo = _git_repo(tmp_path)
    (repo / "agent.py").write_text('M = "claude-opus-4-8"\n', encoding="utf-8")
    _commit_all(repo)
    cfg = _cfg(tmp_path, {"svc-a": AgentConfig(source_path=str(repo))})

    card = [
        p for p in cp.cost_proposals_from_report(_report(downgrade=_downsize_finding()), config=cfg)
        if p.analyzer == "downsize"
    ][0]

    assert card.apply_capable is True
    assert card.apply_kind == APPLY_KIND_MODEL_SWAP
    assert card.target_path == str(repo / "agent.py")
    assert card.current_model == "claude-opus-4-8"
    assert card.proposed_model == "claude-haiku-4-5"
    assert card.apply_blocked_reason == ""
    # The redeploy caveat is not optional: nothing is measurable until the agent
    # actually runs on the new model.
    assert "redeploy" in card.advise_text


def test_unregistered_agent_falls_back_to_the_one_paste_fix(tmp_path):
    card = [
        p for p in cp.cost_proposals_from_report(
            _report(downgrade=_downsize_finding()), config=_cfg(tmp_path),
        )
        if p.analyzer == "downsize"
    ][0]
    assert card.apply_capable is False
    assert card.advise_only is True
    assert "no local source path is registered" in card.apply_blocked_reason
    assert card.one_paste_fix
    assert "claude-haiku-4-5" in card.one_paste_fix


def test_dirty_repo_falls_back_and_says_why(tmp_path):
    repo = _git_repo(tmp_path)
    (repo / "agent.py").write_text('M = "claude-opus-4-8"\n', encoding="utf-8")
    _commit_all(repo)
    (repo / "agent.py").write_text('M = "claude-opus-4-8"  # wip\n', encoding="utf-8")
    cfg = _cfg(tmp_path, {"svc-a": AgentConfig(source_path=str(repo))})

    card = [
        p for p in cp.cost_proposals_from_report(_report(downgrade=_downsize_finding()), config=cfg)
        if p.analyzer == "downsize"
    ][0]

    assert card.apply_capable is False
    assert "uncommitted changes" in card.apply_blocked_reason


# --------------------------------------------------------------------------- #
# B2: the subagent card routes to the agent file when there is one
# --------------------------------------------------------------------------- #

def _subagent_finding(sub_agent_id="explore"):
    row = SubagentRow(
        session_id="sess-1", sub_agent_id=sub_agent_id, model="claude-opus-4-8",
        llm_calls=3, tool_calls=1, input_tokens=80_000, output_tokens=500,
        cache_tokens=10_000, cache_write_tokens=2_000, cost_usd=1.2,
        provider="anthropic", flags=["over_powered"],
    )
    return SubagentRightsizingFinding(
        sessions_with_subagents=1, total_subagents=1, subagent_cost_usd=1.2,
        subagent_tokens=92_500, window_cost_usd=2.0, percent_of_cost=0.6,
        flagged_cost_usd=1.2, rows=[row], flagged=[row],
        estimated_recoverable_usd=0.9, estimated_recoverable_tokens=92_500,
    )


def test_named_subagent_with_a_definition_file_gets_the_model_apply(tmp_path, monkeypatch):
    repo = tmp_path / "workspace"
    agent_file = repo / ".claude" / "agents" / "explore.md"
    agent_file.parent.mkdir(parents=True)
    agent_file.write_text(AGENT_FILE, encoding="utf-8")
    monkeypatch.setattr(cp, "_session_cwds", lambda ids, config: {"sess-1": str(repo)})

    card = [
        p for p in cp.cost_proposals_from_report(
            _report(subagent=_subagent_finding()), config=_cfg(tmp_path),
        )
        if p.analyzer == "subagent"
    ][0]

    assert card.apply_kind == APPLY_KIND_AGENT_MODEL
    assert card.apply_capable is True
    assert card.advise_only is False
    assert card.agent_name == "explore"
    assert card.target_path == str(agent_file)
    assert card.proposed_model == "claude-haiku-4-5"
    assert card.scope == "project"
    assert card.signature == "cost:subagent:explore"


def test_inline_subagent_falls_back_to_the_guidance_block(tmp_path, monkeypatch):
    # A uuid sub_agent_id names no definition file: the rubric note stays the fix.
    monkeypatch.setattr(cp, "_session_cwds", lambda ids, config: {"sess-1": str(tmp_path)})
    card = [
        p for p in cp.cost_proposals_from_report(
            _report(subagent=_subagent_finding("3f2a9c11-77bd-4f0e-9b6a-2c1d8e5f0a44")),
            config=_cfg(tmp_path),
        )
        if p.analyzer == "subagent"
    ][0]
    assert card.apply_kind == ""
    assert card.signature == "cost:subagent"
    assert card.rung == 1
    assert card.proposed_fix


def test_missing_agent_file_falls_back_to_the_guidance_block(tmp_path, monkeypatch):
    # Name-shaped, but no file on disk for it.
    monkeypatch.setattr(cp, "_session_cwds", lambda ids, config: {"sess-1": str(tmp_path)})
    card = [
        p for p in cp.cost_proposals_from_report(
            _report(subagent=_subagent_finding()), config=_cfg(tmp_path),
        )
        if p.analyzer == "subagent"
    ][0]
    assert card.apply_kind == ""
    assert card.signature == "cost:subagent"


# --------------------------------------------------------------------------- #
# D1: the batch placement card
# --------------------------------------------------------------------------- #

def _placement_finding():
    return BatchPlacementFinding(
        candidates=[BatchCandidate(
            agent_id="nightly", sessions=6, first_start=NOW.isoformat(),
            last_start=NOW.isoformat(), median_gap_seconds=21_600.0, gap_cv=0.01,
            cost_usd=6.0, tokens=15_000, estimated_batch_saving_usd=3.0,
        )],
        window_cost_usd=12.0, candidate_cost_usd=6.0, percent_of_window_cost=50.0,
        estimated_recoverable_usd=3.0, estimated_recoverable_tokens=15_000,
    )


def test_placement_card_states_the_discount_and_the_friction(tmp_path):
    card = [
        p for p in cp.cost_proposals_from_report(
            _report(placement=_placement_finding()), config=_cfg(tmp_path),
        )
        if p.analyzer == "placement"
    ][0]
    assert card.signature == "cost:placement:batch"
    assert card.advise_only is True
    assert card.apply_capable is False
    assert card.estimated_recoverable_usd == 3.0
    assert "50%" in card.advise_text
    assert "architectural change" in card.advise_text
    assert "no human turn" in card.evidence
    assert card.estimate_basis


def test_no_placement_finding_means_no_placement_card(tmp_path):
    props = cp.cost_proposals_from_report(_report(), config=_cfg(tmp_path))
    assert [p for p in props if p.analyzer == "placement"] == []


# --------------------------------------------------------------------------- #
# House rules on every string these cards can print
# --------------------------------------------------------------------------- #

def _all_cards(tmp_path):
    return cp.cost_proposals_from_report(
        _report(
            downgrade=_downsize_finding(),
            subagent=_subagent_finding(),
            placement=_placement_finding(),
        ),
        config=_cfg(tmp_path),
    )


@pytest.mark.parametrize("field", [
    "title", "evidence", "advise_text", "suggestion", "one_paste_fix",
    "estimate_basis", "apply_blocked_reason", "caveat",
])
def test_card_copy_has_no_em_dash_and_never_says_quota(tmp_path, field):
    for card in _all_cards(tmp_path):
        text = getattr(card, field) or ""
        assert "—" not in text, f"em dash in {card.signature}.{field}"
        assert "quota" not in text.lower(), f"'quota' in {card.signature}.{field}"


def test_cards_carry_the_fields_the_rollup_sums(tmp_path):
    # The rollup reads signature, analyzer, title and estimated_recoverable_usd
    # generically, with no analyzer allowlist, so each card must fill all four
    # and no two may share a signature.
    cards = _all_cards(tmp_path)
    signatures = [c.signature for c in cards]
    assert len(signatures) == len(set(signatures))
    for card in cards:
        assert card.signature and card.analyzer and card.title
        assert card.estimated_recoverable_usd is not None
        assert card.estimated_recoverable_usd > 0


def test_an_agent_the_swap_would_not_save_on_gets_no_card(tmp_path):
    finding = _downsize_finding()
    finding.per_agent[0].delta_usd = -0.5
    props = cp.cost_proposals_from_report(_report(downgrade=finding), config=_cfg(tmp_path))
    # No per-agent card claiming a negative recovery; the window-wide card,
    # whose own estimate is finding-level, takes over.
    assert [p.signature for p in props if p.analyzer == "downsize"] == ["cost:downsize"]


def test_every_dollar_figure_is_tagged_and_has_a_construction_footnote(tmp_path):
    for card in _all_cards(tmp_path):
        if card.estimated_recoverable_usd is None:
            continue
        assert card.estimate_confidence in ("estimated", "measured")
        assert card.estimate_basis, f"{card.signature} prints a figure with no footnote"
        assert card.correlational is True
