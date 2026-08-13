"""Where a permanent rule gets written, and what that does to the arithmetic.

Two things are pinned here and each one is a defect if it breaks:

1. **A recorded cwd that no longer exists is COUNTED, never dropped.** Both
   analyzers that read live filesystem state return a plausible number rather
   than an error when they cannot see what they are measuring (Critical Rule
   31), so the only defence is a coverage statement — and it has to say the
   gap is what was NOT ANALYSED (Critical Rule 30), never what was
   unavoidable.
2. **Tokens and dollars are split by the SAME weights**, so every
   destination's implied per-token rate equals the finding's own (Critical
   Rule 28).

Placement no longer feeds an offer/suppress decision — there is no budget left
to flip. A rule confined to fewer sessions is simply cheaper to KEEP; every
`apply_capable` rule is offered regardless of which destination wins.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from tokenjam.core.optimize import rule_placement as rp


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


