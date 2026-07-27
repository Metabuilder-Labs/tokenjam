"""Cards for the two model-routing write paths, and the batch placement card.

Covers what the user actually reads: the arithmetic on the card, whether a
direct apply is offered or the one-paste artifact takes over, and the house
rules every runtime string has to hold to.
"""
from __future__ import annotations

import subprocess
from datetime import datetime, timedelta, timezone

from tokenjam.utils.time_parse import utcnow

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
        "started_at": utcnow() - timedelta(days=1),
    }], 30.0)


def _downsize_finding(agent_id="svc-a"):
    return DowngradeFinding(
        candidate_sessions=4, total_sessions=10, actual_cost_usd=5.0,
        alternative_cost_usd=2.0, monthly_savings_usd=3.0, percent_of_sessions=40.0,
        examples=[], suggestions={"claude-opus-4-8": "claude-haiku-4-5"},
        past_overspend_usd=3.0, percent_of_tokens=35.0,
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
    assert [p.signature for p in downsize] == [
        "cost:downsize:svc-a:anthropic:claude-opus-4-8:claude-haiku-4-5"
    ]
    row = _price_rows()[0]
    assert downsize[0].past_overspend_usd == row.delta_usd
    assert downsize[0].past_overspend_tokens == row.total_tokens
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


def test_unregistered_agent_asks_for_the_path_instead_of_giving_up(tmp_path):
    """An unregistered agent used to be the weakest row the inbox had: a measured,
    deterministic model swap reduced to a copy box and a "Mark applied" that only
    recorded the user doing it by hand. Eleven live rows read that way, all for the
    same reason — nobody had told tokenjam where those agents' source lives, and it
    will not go looking (`config.AgentConfig.source_path` is opt-in, never
    inferred). That is a QUESTION, so the row asks it.

    `apply_kind` stays UNSET while the path is missing. That is not an oversight:
    with no registered path there is no deterministic edit yet, so the row must not
    route to the apply endpoint that assumes one.
    """
    card = [
        p for p in cp.cost_proposals_from_report(
            _report(downgrade=_downsize_finding()), config=_cfg(tmp_path),
        )
        if p.analyzer == "downsize"
    ][0]
    assert card.needs_source_path is True
    assert card.apply_capable is True
    assert card.advise_only is False
    assert card.apply_kind == ""
    assert card.target_path == ""
    assert card.source_path == ""
    # The models the swap would substitute travel with it, so the row can name
    # them before any path is known.
    assert card.current_model == "claude-opus-4-8"
    assert card.proposed_model == "claude-haiku-4-5"
    # Nothing is BLOCKED, so nothing claims to be.
    assert card.apply_blocked_reason == ""
    # The copyable fix is still there for a reader who would rather do it by hand.
    assert card.one_paste_fix
    assert "claude-haiku-4-5" in card.one_paste_fix
    # And the row asks, rather than announcing a target it does not have.
    assert "point it at" in card.advise_text
    assert "redeploy" in card.advise_text


def test_an_apply_capable_swap_carries_its_quality_caveat_outside_the_prose(tmp_path):
    """Critical Rule 14 under a one-click write. The token-cost delta IS measured;
    quality equivalence is never claimed, and a button makes that distinction
    easier to lose. So the sentence rides its own field, which the card renders
    OUTSIDE the collapsed description — a caveat behind a "Read more" would not
    have counted as visible.

    One constant feeds both the field and the prose, so the sentence beside the
    button and the sentence in the paragraph cannot drift into two different
    strengths of claim.
    """
    unregistered = [
        p for p in cp.cost_proposals_from_report(
            _report(downgrade=_downsize_finding()), config=_cfg(tmp_path),
        )
        if p.analyzer == "downsize"
    ][0]

    repo = _git_repo(tmp_path)
    (repo / "agent.py").write_text('M = "claude-opus-4-8"\n', encoding="utf-8")
    _commit_all(repo)
    registered = [
        p for p in cp.cost_proposals_from_report(
            _report(downgrade=_downsize_finding()),
            config=_cfg(tmp_path, {"svc-a": AgentConfig(source_path=str(repo))}),
        )
        if p.analyzer == "downsize"
    ][0]

    for card in (unregistered, registered):
        assert card.apply_caveat == cp.MODEL_SWAP_QUALITY_CAVEAT
        assert "NOT measured here" in card.apply_caveat
        # Same words in the prose, from the same constant.
        assert cp.MODEL_SWAP_QUALITY_CAVEAT in card.advise_text


def test_a_gate_the_reader_cannot_answer_stays_advise_only(tmp_path):
    """The middle outcome must not swallow the third one. A registered path whose
    repo fails a LATER gate names something no question can fix, so that row keeps
    its one-paste artifact and says why — it does not get an Apply button that
    would fail on the click.
    """
    plain = tmp_path / "not-a-repo"
    plain.mkdir()
    (plain / "agent.py").write_text('M = "claude-opus-4-8"\n', encoding="utf-8")
    card = [
        p for p in cp.cost_proposals_from_report(
            _report(downgrade=_downsize_finding()),
            config=_cfg(tmp_path, {"svc-a": AgentConfig(source_path=str(plain))}),
        )
        if p.analyzer == "downsize"
    ][0]
    assert card.needs_source_path is False
    assert card.apply_capable is False
    assert card.apply_caveat == ""
    assert "not a git repository" in card.apply_blocked_reason


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

def _subagent_finding(sub_agent_type="explore",
                      sub_agent_id="aexplore-7e1dd2a1642d7c29"):
    # `sub_agent_id` is Claude Code's PER-DISPATCH id — `a` + an optional
    # caller-chosen label + a hex suffix. It names no file, and defaults here to
    # a realistic one so no test can accidentally rely on it resolving.
    # `sub_agent_type` is the stable identity that addresses
    # `.claude/agents/<type>.md`.
    row = SubagentRow(
        session_id="sess-1", sub_agent_id=sub_agent_id,
        sub_agent_type=sub_agent_type, model="claude-opus-4-8",
        llm_calls=3, tool_calls=1, input_tokens=80_000, output_tokens=500,
        cache_tokens=10_000, cache_write_tokens=2_000, cost_usd=1.2,
        provider="anthropic", flags=["over_powered"],
    )
    return SubagentRightsizingFinding(
        sessions_with_subagents=1, total_subagents=1, subagent_cost_usd=1.2,
        subagent_tokens=92_500, window_cost_usd=2.0, percent_of_cost=0.6,
        flagged_cost_usd=1.2, rows=[row], flagged=[row],
        # Scaled past the $5 write floor (`write_budget.MIN_NET_WRITE_USD`) so
        # these tests can exercise the OFFERED write path at all; the $/token
        # rate is held constant so the pair still divides back into a real
        # price band (CLAUDE.md rule 28).
        past_overspend_usd=9.0, past_overspend_tokens=925_000,
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


def test_a_dispatch_id_never_resolves_to_an_agent_file(tmp_path, monkeypatch):
    """The lookup keys on the stable TYPE, never on the per-dispatch id.

    This test used to assert the opposite by accident: `sub_agent_id` was the
    lookup key and a plain-slug shape check was supposed to keep dispatch ids
    out — but a Claude Code dispatch id IS slug-shaped (99.6% of 3,659 real ones
    matched), so every lookup went hunting for a `.claude/agents/a<hex>.md`.
    Here the file that WOULD satisfy the old behaviour exists on disk, named for
    the dispatch id; resolving it would be the defect, so the card must fall
    back to the guidance block.
    """
    repo = tmp_path / "workspace"
    decoy = repo / ".claude" / "agents" / "aexplore-7e1dd2a1642d7c29.md"
    decoy.parent.mkdir(parents=True)
    decoy.write_text(AGENT_FILE, encoding="utf-8")
    monkeypatch.setattr(cp, "_session_cwds", lambda ids, config: {"sess-1": str(repo)})
    card = [
        p for p in cp.cost_proposals_from_report(
            _report(subagent=_subagent_finding(sub_agent_type="")),
            config=_cfg(tmp_path),
        )
        if p.analyzer == "subagent"
    ][0]
    assert card.apply_kind == ""
    assert card.signature == "cost:subagent"
    assert card.target_path == ""
    assert card.rung == 1
    assert card.proposed_fix


def test_a_builtin_dispatch_type_falls_back_to_the_guidance_block(tmp_path, monkeypatch):
    # `Explore` is a built-in dispatch type with no definition file on disk:
    # the rubric note stays the fix. Its capital letter fails the slug shape,
    # which is the correct outcome for exactly that reason.
    monkeypatch.setattr(cp, "_session_cwds", lambda ids, config: {"sess-1": str(tmp_path)})
    card = [
        p for p in cp.cost_proposals_from_report(
            _report(subagent=_subagent_finding("Explore")),
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
        past_overspend_usd=3.0, past_overspend_tokens=15_000,
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
    assert card.past_overspend_usd == 3.0
    assert "50%" in card.advise_text
    assert "architectural change" in card.advise_text
    assert "no human turn" in card.evidence
    assert card.estimate_basis


def test_no_placement_finding_means_no_placement_card(tmp_path):
    props = cp.cost_proposals_from_report(_report(), config=_cfg(tmp_path))
    assert [p for p in props if p.analyzer == "placement"] == []


def test_placement_card_suppresses_dollars_on_subscription(tmp_path):
    """N42: the Batch API discount is an api-billed lever. Matches the CLI's
    `_render_placement`, which already suppresses this dollar figure on
    non-api plans (CLAUDE.md anti-pattern #22)."""
    card = [
        p for p in cp.cost_proposals_from_report(
            _report(placement=_placement_finding()), config=_cfg(tmp_path),
            pricing_mode="subscription",
        )
        if p.analyzer == "placement"
    ][0]
    assert card.past_overspend_usd is None
    assert "no dollar figure is shown for this plan" in card.advise_text
    assert "$3.00" not in card.advise_text
    # Token-level evidence (the workload shape) is still shown either way.
    assert "no human turn" in card.evidence


def test_placement_card_suppresses_dollars_on_local(tmp_path):
    card = [
        p for p in cp.cost_proposals_from_report(
            _report(placement=_placement_finding()), config=_cfg(tmp_path),
            pricing_mode="local",
        )
        if p.analyzer == "placement"
    ][0]
    assert card.past_overspend_usd is None


def test_recompute_cost_proposals_resolves_pricing_mode_from_sessions(tmp_path):
    """N42 end to end: `recompute_cost_proposals` reads the window's dominant
    plan tier off the `sessions` table and threads it through, so a
    subscription install's Review inbox suppresses the same placement dollar
    figure the CLI does — it isn't only reachable by passing the kwarg by
    hand."""
    from tokenjam.core.db import InMemoryBackend
    from tokenjam.utils.time_parse import utcnow
    from tests.factories import make_llm_span, make_session

    db = InMemoryBackend()
    try:
        base = utcnow() - timedelta(days=10)
        for i in range(6):
            start = base + timedelta(hours=6 * i)
            session_id = f"nightly-{i}"
            db.insert_span(make_llm_span(
                agent_id="nightly", model="claude-sonnet-4-6", provider="anthropic",
                input_tokens=2_000, output_tokens=500, cache_tokens=100,
                cache_write_tokens=50, cost_usd=1.0,
                session_id=session_id, start_time=start,
            ))
            db.upsert_session(make_session(
                agent_id="nightly", session_id=session_id, plan_tier="pro",
                started_at=start, ended_at=start + timedelta(minutes=1),
            ))

        proposals = cp.recompute_cost_proposals(db, _cfg(tmp_path), window_days=30)
        placement = [p for p in proposals if p.analyzer == "placement"]
        assert placement, "expected a placement card from the cadence-regular sessions"
        assert placement[0].past_overspend_usd is None
        assert "no dollar figure is shown for this plan" in placement[0].advise_text
    finally:
        db.close()


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
    # The rollup reads signature, analyzer, title and past_overspend_usd
    # generically, with no analyzer allowlist, so each card must fill all four
    # and no two may share a signature.
    cards = _all_cards(tmp_path)
    signatures = [c.signature for c in cards]
    assert len(signatures) == len(set(signatures))
    for card in cards:
        assert card.signature and card.analyzer and card.title
        assert card.past_overspend_usd is not None
        assert card.past_overspend_usd > 0


def test_an_agent_the_swap_would_not_save_on_gets_no_card(tmp_path):
    finding = _downsize_finding()
    finding.per_agent[0].delta_usd = -0.5
    props = cp.cost_proposals_from_report(_report(downgrade=finding), config=_cfg(tmp_path))
    # No per-agent card claiming a negative recovery; the window-wide card,
    # whose own estimate is finding-level, takes over.
    assert [p.signature for p in props if p.analyzer == "downsize"] == ["cost:downsize"]


def test_every_dollar_figure_is_tagged_and_has_a_construction_footnote(tmp_path):
    for card in _all_cards(tmp_path):
        if card.past_overspend_usd is None:
            continue
        assert card.estimate_confidence in ("estimated", "measured")
        assert card.estimate_basis, f"{card.signature} prints a figure with no footnote"
        assert card.correlational is True


# --------------------------------------------------------------------------- #
# B3: the two identity-resolution guards
# --------------------------------------------------------------------------- #

def test_a_dispatch_id_fails_the_agent_definition_name_check():
    """Real Claude Code dispatch ids, in both shapes seen on a live corpus.

    They are `a` + an optional caller-chosen instance label + a 16-17 hex
    suffix, so they satisfy the plain-slug shape that used to be the only gate —
    which is why 3,645 of 3,659 (99.6%) passed it and the lookup resolved
    nothing at all.
    """
    for dispatch_id in (
        "af8b26e872b7184a7",
        "aw-ratehistory-7e1dd2a1642d7c29",
        "aworker-428-63df1d3c53338de1",
        "apr543-e855e916eeb36f9e",
    ):
        assert cp._AGENT_NAME_RE.match(dispatch_id), (
            "precondition: the slug shape alone does NOT exclude a dispatch id"
        )
        assert not cp._names_agent_definition(dispatch_id)


def test_a_real_agent_type_passes_the_name_check():
    for name in ("code-reviewer", "general-purpose", "explore", "tdd-guide"):
        assert cp._names_agent_definition(name)
    # No name, and built-in types that carry no definition file.
    for name in ("", "Explore", "Plan"):
        assert not cp._names_agent_definition(name)


def test_the_scope_session_cap_counts_distinct_sessions_not_rows(tmp_path, monkeypatch):
    """`_session_cwds` is called with one entry per flagged ROW, and a session
    that fanned out to many subagents contributes that id many times. Slicing
    the raw list spent the whole cap on a handful of sessions."""
    seen: dict[str, list] = {}

    def _capture(pairs, root):
        seen["pairs"] = list(pairs)
        return {}

    monkeypatch.setattr(
        "tokenjam.core.optimize.analyzers.relearn._repo_cwd_map_for", _capture
    )
    # 40 sessions, each repeated 30 times — 1200 rows, far past the cap.
    session_ids = [f"sess-{i}" for i in range(40) for _ in range(30)]
    cp._session_cwds(session_ids, _cfg(tmp_path))

    got = [sid for sid, _ in seen["pairs"]]
    assert len(got) == cp._MAX_SCOPE_SESSIONS
    assert len(set(got)) == cp._MAX_SCOPE_SESSIONS, "every slot is a distinct session"
    assert got == [f"sess-{i}" for i in range(cp._MAX_SCOPE_SESSIONS)]
