"""Where a permanent rule gets written, and what that does to the arithmetic.

Three things are pinned here and each one is a defect if it breaks:

1. **A recorded cwd that no longer exists is COUNTED, never dropped.** Both
   analyzers that read live filesystem state return a plausible number rather
   than an error when they cannot see what they are measuring (Critical Rule
   31), so the only defence is a coverage statement — and it has to say the
   gap is what was NOT ANALYSED (Critical Rule 30), never what was
   unavoidable.
2. **Placement changes the OFFER decision.** A rule whose standing cost is
   charged against every session in the window can read net-negative and be
   suppressed, while the same rule charged against only the projects that
   exhibited the behaviour clears the bar. That flip is the whole point of the
   feature, so it is asserted directly rather than inferred.
3. **Tokens and dollars are split by the SAME weights**, so every
   destination's implied per-token rate equals the finding's own (Critical
   Rule 28).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from tokenjam.core.rulewrite.kinds import DELIVERY_CLAUDE_MD_RULE

from tokenjam.core.optimize import rule_placement as rp
from tokenjam.core.optimize import write_budget as wb
from tokenjam.core.optimize.projection import build_projection_basis


def _repo(tmp_path: Path, name: str) -> Path:
    root = tmp_path / name
    (root / ".git").mkdir(parents=True)
    (root / "CLAUDE.md").write_text("# existing\n", encoding="utf-8")
    return root


# --- 1. derivation ----------------------------------------------------------#

def test_sessions_are_grouped_into_the_claude_md_their_repo_actually_loads(tmp_path):
    alpha, beta = _repo(tmp_path, "alpha"), _repo(tmp_path, "beta")
    shares = [
        rp.SessionShare("s1", weight=600), rp.SessionShare("s2", weight=200),
        rp.SessionShare("s3", weight=150), rp.SessionShare("s4", weight=50),
    ]
    cwds = {
        "s1": str(alpha), "s2": str(alpha), "s3": str(beta), "s4": str(beta),
    }
    plan = rp.build_placement_plan(
        shares, cwds, total_tokens=1000, total_usd=10.0, within=tmp_path,
    )

    assert [d.path for d in plan.destinations] == [
        str(alpha / "CLAUDE.md"), str(beta / "CLAUDE.md"),
    ]
    assert [d.sessions for d in plan.destinations] == [2, 2]
    # Ranked by attributed tokens, and the split follows the weights: alpha
    # carries 800 of the 1000, beta 200.
    assert plan.destinations[0].tokens == 800
    assert plan.destinations[1].tokens == 200
    assert plan.unresolved_sessions == 0
    assert plan.coverage_note == ""
    assert plan.attribution_basis == "weighted"


def test_the_dollar_split_preserves_the_findings_own_per_token_rate(tmp_path):
    """Critical Rule 28, applied to a split rather than to a single figure.

    A destination whose tokens and dollars came from different weightings would
    publish an implied rate the finding never charged at — the same defect one
    level down from two analyzers claiming the same spans.
    """
    alpha, beta = _repo(tmp_path, "alpha"), _repo(tmp_path, "beta")
    plan = rp.build_placement_plan(
        [
            rp.SessionShare("s1", weight=700), rp.SessionShare("s2", weight=100),
            rp.SessionShare("s3", weight=150), rp.SessionShare("s4", weight=50),
        ],
        {"s1": str(alpha), "s2": str(alpha), "s3": str(beta), "s4": str(beta)},
        total_tokens=1_000_000, total_usd=3.0, within=tmp_path,
    )
    finding_rate = 3.0 / 1_000_000
    for destination in plan.destinations:
        assert destination.usd is not None
        assert destination.usd / destination.tokens == pytest.approx(
            finding_rate, rel=1e-6,
        )


def test_an_unpriced_finding_leaves_every_destination_unpriced(tmp_path):
    """Both fields degrade together (Critical Rule 28 corollary a). A 0.0 here
    would state "this project's share is worth nothing" for "not measured"."""
    alpha = _repo(tmp_path, "alpha")
    plan = rp.build_placement_plan(
        [rp.SessionShare("s1", weight=1), rp.SessionShare("s2", weight=1)],
        {"s1": str(alpha), "s2": str(alpha)},
        total_tokens=500, total_usd=None, within=tmp_path,
    )
    assert plan.destinations[0].usd is None
    assert plan.unresolved_usd is None


# --- 2. the coverage path (Critical Rules 30 + 31) --------------------------#

def test_a_vanished_cwd_is_counted_and_reported_not_silently_dropped(tmp_path):
    alpha = _repo(tmp_path, "alpha")
    plan = rp.build_placement_plan(
        [
            rp.SessionShare("s1", weight=400), rp.SessionShare("s2", weight=400),
            rp.SessionShare("gone1", weight=100), rp.SessionShare("gone2", weight=100),
        ],
        {
            "s1": str(alpha), "s2": str(alpha),
            "gone1": str(tmp_path / "deleted-repo"),
            "gone2": str(tmp_path / "also-deleted"),
        },
        total_tokens=1000, total_usd=10.0, within=tmp_path,
    )

    assert plan.unresolved_sessions == 2
    assert plan.vanished_cwds == 2
    # The share is REPORTED, never redistributed onto the surviving
    # destination — redistributing would quietly inflate alpha's claim.
    assert plan.unresolved_tokens == 200
    assert plan.unresolved_usd == pytest.approx(2.0)
    assert plan.destinations[0].tokens == 800

    note = plan.coverage_note
    assert "no longer on disk" in note
    # The sentence that is the whole point: the gap is what was NOT ANALYSED.
    assert "not analysed" in note
    assert "unavoidable" not in note.lower().replace("nowhere better", "")


def test_a_session_with_no_recorded_cwd_is_counted_separately_from_a_vanished_one(
    tmp_path,
):
    alpha = _repo(tmp_path, "alpha")
    plan = rp.build_placement_plan(
        [
            rp.SessionShare("s1"), rp.SessionShare("s2"),
            rp.SessionShare("nocwd1"), rp.SessionShare("nocwd2"),
        ],
        {"s1": str(alpha), "s2": str(alpha)},
        total_tokens=400, within=tmp_path,
    )
    assert plan.unresolved_sessions == 2
    assert plan.vanished_cwds == 0
    assert "recorded no working directory" in plan.coverage_note
    # No weights supplied: the split falls back to session count and says so,
    # rather than pretending to a precision it does not have.
    assert plan.attribution_basis == "by session count"
    assert plan.unresolved_tokens == 200


def test_a_home_directory_cwd_is_refused_as_a_destination(tmp_path, monkeypatch):
    """The boundary gate `repo_roots.is_safe_scan_root` already refuses `$HOME`
    and bare top-level paths. Placement must inherit that refusal rather than
    re-deciding it: a derivation that can be pointed at `$HOME` by a stray
    recorded cwd will eventually write a permanent rule into one."""
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    alpha = _repo(tmp_path, "alpha")
    plan = rp.build_placement_plan(
        [
            rp.SessionShare("s1"), rp.SessionShare("s2"),
            rp.SessionShare("h1"), rp.SessionShare("h2"),
        ],
        {"s1": str(alpha), "s2": str(alpha), "h1": str(tmp_path), "h2": str(tmp_path)},
        total_tokens=400, within=tmp_path,
    )
    assert [d.root for d in plan.destinations] == [str(alpha)]
    assert plan.unresolved_sessions == 2
    assert plan.refused_cwds == 1


def test_a_barely_touched_project_does_not_earn_a_permanent_block(tmp_path):
    """A repo the window touched once is folded into the unplaceable pool: a
    permanent rule written there spends a whole file's budget on a population
    too small to have shown anything."""
    alpha, thin = _repo(tmp_path, "alpha"), _repo(tmp_path, "thin")
    plan = rp.build_placement_plan(
        [
            rp.SessionShare("s1", weight=400), rp.SessionShare("s2", weight=400),
            rp.SessionShare("s3", weight=200),
        ],
        {"s1": str(alpha), "s2": str(alpha), "s3": str(thin)},
        total_tokens=1000, within=tmp_path,
    )
    assert [d.root for d in plan.destinations] == [str(alpha)]
    assert plan.unresolved_sessions == 1


# --- 3. the choice ----------------------------------------------------------#

def test_project_placement_wins_when_the_waste_is_concentrated(tmp_path):
    alpha = _repo(tmp_path, "alpha")
    plan = rp.build_placement_plan(
        [rp.SessionShare(f"s{i}", weight=10) for i in range(4)],
        {f"s{i}": str(alpha) for i in range(4)},
        total_tokens=1000, within=tmp_path,
    )
    choice = rp.choose_placement(
        plan, standing_tokens_per_session=50, total_sessions=200,
    )
    assert choice.scope == "project"
    assert choice.exposure_sessions == 4
    assert choice.standing_tokens == 50 * 4
    assert choice.alternative_standing_tokens == 50 * 200
    # One file grew, so the budget is charged for one block even though the
    # standing cost fell 50x.
    assert choice.footprint_tokens == 50


def test_the_user_global_file_wins_when_the_waste_really_is_everywhere(tmp_path):
    """The root file stays a legitimate destination. Decided by the arithmetic
    — when the finding's sessions ARE the window, one shared write is no more
    expensive to keep and is one file instead of many."""
    repos = [_repo(tmp_path, f"r{i}") for i in range(3)]
    shares, cwds = [], {}
    for i, repo in enumerate(repos):
        for j in range(2):
            sid = f"s{i}-{j}"
            shares.append(rp.SessionShare(sid, weight=10))
            cwds[sid] = str(repo)
    plan = rp.build_placement_plan(shares, cwds, total_tokens=600, within=tmp_path)
    choice = rp.choose_placement(
        plan, standing_tokens_per_session=40, total_sessions=6,
    )
    assert choice.scope == "user-global"
    assert choice.footprint_tokens == 40          # one file, not three
    assert choice.destinations[0].scope == "user-global"


def test_no_resolved_destination_falls_back_to_the_user_global_file(tmp_path):
    plan = rp.build_placement_plan(
        [rp.SessionShare("s1")], {}, total_tokens=100, within=tmp_path,
    )
    choice = rp.choose_placement(
        plan, standing_tokens_per_session=30, total_sessions=90,
    )
    assert choice.scope == "user-global"
    assert choice.standing_tokens == 30 * 90


# --- 4. the flip: placement is an INPUT to whether a write is offered -------#

def _candidate(**kw):
    base = dict(
        key="k", family="fam", delivery=DELIVERY_CLAUDE_MD_RULE,
        # ~500 chars of real rule text: past the placeholder floor, and big
        # enough that its standing cost matters.
        artifact_text="Route context-heavy sub-tasks to a subagent. " * 12,
        gross_tokens=40_000, gross_usd=12.0,
    )
    base.update(kw)
    return wb.WriteCandidate(**base)


def test_placement_flips_a_write_the_global_destination_would_have_suppressed():
    """The core claim of this change, asserted end to end on the budget.

    Charged against every session in the window the rule costs more to keep
    than it recovers and is suppressed as net-negative. Charged against only
    the sessions in the projects that actually exhibited the behaviour, the
    same rule, with the same text and the same gross, clears the bar and is
    offered. Nothing about the finding changed — only where the rule lands.
    """
    basis = build_projection_basis(30.0, 20, 400)
    budget = wb.build_write_budget(lane_budget_tokens=500, lane_max_writes=3)

    global_only = wb.allocate_writes([_candidate()], budget, basis)["k"]
    assert global_only.net_negative is True
    assert global_only.offered is False
    assert global_only.reason == wb.REASON_NET_NEGATIVE
    # A suppressed write claims nothing on any basis.
    assert global_only.claimed_tokens == 0

    placed = wb.allocate_writes(
        [_candidate(exposure_sessions=12, destinations=("/repo/a/CLAUDE.md",))],
        budget, basis,
    )["k"]
    assert placed.net_negative is False
    assert placed.offered is True
    assert placed.exposure_sessions == 12
    assert placed.claimed_tokens > 0
    # And the claim is still NET, never the gross — a proposal may never claim
    # a saving larger than its net value.
    assert placed.claimed_tokens == 40_000 - placed.standing_tokens


def test_a_multi_destination_write_is_charged_for_every_file_it_grows():
    """The two costs are distinct: standing follows the sessions, footprint
    follows the files. Collapsing them would make a three-project split look
    free to the budget."""
    basis = build_projection_basis(30.0, 20, 400)
    budget = wb.build_write_budget(lane_budget_tokens=10_000, lane_max_writes=3)
    decision = wb.allocate_writes(
        [_candidate(
            exposure_sessions=30,
            destinations=("/a/CLAUDE.md", "/b/CLAUDE.md", "/c/CLAUDE.md"),
        )],
        budget, basis,
    )["k"]
    assert decision.footprint_tokens == decision.standing_tokens_per_session * 3
    assert decision.standing_tokens == decision.standing_tokens_per_session * 30
    assert decision.destinations == (
        "/a/CLAUDE.md", "/b/CLAUDE.md", "/c/CLAUDE.md",
    )


class _Cand:
    """A summarize candidate, as the write budget reads it (duck-typed)."""

    def __init__(self, path: str, total_chars: int) -> None:
        self.path = path
        self.total_chars = total_chars
        self.always_resident_chars = total_chars


class _Summarize:
    def __init__(self, *candidates: _Cand) -> None:
        self.candidates = list(candidates)


def test_a_small_project_file_gets_a_small_growth_allowance_of_its_own(tmp_path):
    """One coordinated budget PER DESTINATION, sized against THAT file.

    While there was one destination, the aggregate agent-file footprint was the
    right denominator for "may this window grow the files at all". It stops
    being the right one the moment a rule can land in a small project
    ``CLAUDE.md`` while a large root file makes the corpus-wide figure look
    roomy: 10% of the aggregate can exceed the small file entirely, so two
    rules converge on it and add more than it contained.
    """
    small = tmp_path / "small" / "CLAUDE.md"
    large = tmp_path / "large" / "CLAUDE.md"
    for path in (small, large):
        path.parent.mkdir(parents=True)
        path.write_text("x", encoding="utf-8")
    finding = _Summarize(
        _Cand(str(small), total_chars=4_000),        # ~1k tokens -> below the floor
        _Cand(str(large), total_chars=160_000),      # ~40k tokens -> a real allowance
    )
    by_path = wb.measured_agent_file_tokens_by_path(finding)
    assert by_path[str(small)] < by_path[str(large)]

    budget = wb.build_write_budget(
        lane_budget_tokens=100_000, lane_max_writes=9,
        existing_agent_file_tokens=wb.measured_agent_file_tokens(finding),
        existing_by_path=by_path,
    )
    # The small file's allowance is its own floor, not a share of the corpus.
    assert budget.destination_budget(str(small)) == wb.MIN_WRITE_BUDGET_TOKENS
    assert budget.destination_budget(str(large)) > wb.MIN_WRITE_BUDGET_TOKENS
    # An unmeasured destination falls back to the lane budget — an unmeasured
    # file is not evidence of a full one.
    assert budget.destination_budget("/never/scanned/CLAUDE.md") == budget.budget_tokens


def test_two_rules_cannot_both_land_in_one_file_past_its_own_allowance(tmp_path):
    small = tmp_path / "small" / "CLAUDE.md"
    other = tmp_path / "other" / "CLAUDE.md"
    for path in (small, other):
        path.parent.mkdir(parents=True)
        path.write_text("x", encoding="utf-8")
    # A rule block big enough that two of them exceed the floor allowance.
    text = "Route context-heavy sub-tasks to a subagent. " * 26
    per_session = wb.standing_tokens_per_session(1, text)
    assert per_session <= wb.MIN_WRITE_BUDGET_TOKENS < per_session * 2

    # A large third file lifts the AGGREGATE (and so the lane budget) without
    # lifting either small file's own allowance — which is precisely the
    # divergence a per-file budget exists to catch.
    finding = _Summarize(
        _Cand(str(small), total_chars=4_000),
        _Cand(str(other), total_chars=4_000),
        _Cand(str(tmp_path / "root" / "CLAUDE.md"), total_chars=160_000),
    )
    budget = wb.build_write_budget(
        lane_budget_tokens=100_000, lane_max_writes=9,
        existing_agent_file_tokens=wb.measured_agent_file_tokens(finding),
        existing_by_path=wb.measured_agent_file_tokens_by_path(finding),
    )
    decisions = wb.allocate_writes(
        [
            _candidate(
                key="first", family="f1", artifact_text=text,
                gross_tokens=900_000, gross_usd=300.0,
                exposure_sessions=10, destinations=(str(small),),
            ),
            _candidate(
                key="second", family="f2", artifact_text=text,
                gross_tokens=800_000, gross_usd=250.0,
                exposure_sessions=10, destinations=(str(small),),
            ),
            # Same size, into a file nothing else touched — it must NOT be
            # blocked by the first file being full.
            _candidate(
                key="third", family="f3", artifact_text=text,
                gross_tokens=700_000, gross_usd=200.0,
                exposure_sessions=10, destinations=(str(other),),
            ),
        ],
        budget, build_projection_basis(30.0, 20, 400),
    )
    assert decisions["first"].offered is True
    assert decisions["second"].offered is False
    assert decisions["second"].reason == wb.REASON_BUDGET_FULL
    assert decisions["third"].offered is True
    # A deferred write is not a suppressed one: the saving is real and the
    # snippet stays copyable, so its claim survives.
    assert decisions["second"].claim_suppressed is False
    assert decisions["second"].claimed_tokens > 0
