"""Net-of-standing-cost accounting, the write budget, and the shared basis.

Covers `core/optimize/projection.py` and `core/optimize/write_budget.py`, plus
the two lanes that consume them (relearn clusters, cost proposals). The
invariant under test throughout: no proposal may claim a saving larger than its
net-of-standing-cost value, and the number of permanent rules offered per
window is bounded, ranked, and never a generic placeholder.
"""
from __future__ import annotations

import pytest

from tokenjam.core.rulewrite.kinds import (
    DELIVERY_CLAUDE_MD_RULE,
    DELIVERY_EXECUTING_HOOK,
    DELIVERY_INJECTING_HOOK,
    DELIVERY_SKILL,
    MAX_NUDGES_PER_SESSION,
)

from tokenjam.core.optimize import write_budget as wb
from tokenjam.core.optimize.projection import (
    MAX_PROJECTION_RATIO,
    build_projection_basis,
)

# A block long enough to clear the quality floor, with a known size.
_REAL_FIX = (
    "Always resolve the absolute path before a Read, and prefer the repo root "
    "over a relative path inherited from a previous tool call."
)


def _basis(sessions=100, active_days=10, window_days=30.0):
    return build_projection_basis(window_days, active_days, sessions)


# --- The shared projection basis ---------------------------------------------

def test_projection_uses_active_day_pace_within_the_cap():
    # Arrange: 200 sessions over 12 active days in a 30-day window.
    # Act
    basis = build_projection_basis(30.0, 12, 200)
    # Assert: r = 30/12 = 2.5, and the projected session count follows it.
    assert basis.projected is True
    assert basis.ratio == pytest.approx(2.5)
    assert basis.projected_sessions == pytest.approx(500.0)


def test_projection_ratio_is_capped_at_three():
    basis = build_projection_basis(30.0, 8, 40)
    assert basis.ratio == MAX_PROJECTION_RATIO   # 30/8 = 3.75, capped


def test_projection_is_suppressed_on_thin_data():
    """Below any guardrail the ratio is exactly 1.0, never an invented pace."""
    for window_days, active_days, sessions in [
        (7.0, 20, 200),      # window too short
        (30.0, 3, 200),      # too few active days
        (30.0, 10, 5),       # too few sessions
    ]:
        basis = build_projection_basis(window_days, active_days, sessions)
        assert basis.projected is False
        assert basis.ratio == 1.0
        assert "not projected" in basis.disclosure


def test_projection_may_normalise_down_over_a_long_window():
    """A 90-day window with 45 active days is 30/45 < 1: normalising down to a
    30-day figure, which the 1.0 floor must not block."""
    basis = build_projection_basis(90.0, 45, 300)
    assert basis.ratio == pytest.approx(30.0 / 45.0)


def test_projection_never_divides_by_a_missing_active_day_count():
    basis = build_projection_basis(0.0, 0, 0)
    assert basis.ratio == 1.0
    assert basis.projected_sessions == 0.0


def test_disclosure_names_every_input_it_used():
    basis = build_projection_basis(30.0, 12, 240)
    assert "240 sessions" in basis.disclosure
    assert "12 active days" in basis.disclosure
    assert "30 days" in basis.disclosure


# --- Standing cost by delivery mechanism ---------------------------------------

def test_a_claude_md_rule_charges_the_whole_block():
    text = "x" * 400
    assert wb.standing_tokens_per_session(DELIVERY_CLAUDE_MD_RULE, text) == 100


def test_a_skill_charges_only_the_always_loaded_frontmatter():
    """A skill's body loads on invoke; only its description is always sent."""
    long_skill = "x" * 8_000
    assert wb.standing_tokens_per_session(
        DELIVERY_SKILL, long_skill,
    ) == wb.tokens_from_chars(wb.SKILL_ALWAYS_LOADED_CHARS)
    # A skill shorter than the cap costs only what it is.
    assert wb.standing_tokens_per_session(DELIVERY_SKILL, "x" * 40) == 10


def test_only_the_executing_hook_has_no_standing_cost():
    """An EXECUTING hook is run by the harness, never sent as prompt text: a
    real zero, and the only one earned here.

    This assertion is the inverse of the one it replaces. That one said "a hook
    is executed, never sent as prompt text" of EVERY hook, which made the suite
    require the defect: an injecting hook exists precisely to put text in front
    of the model, and pricing it at zero understated the cost of most of the
    hook families that actually ship. Same guard, correct state."""
    assert wb.standing_tokens_per_session(DELIVERY_EXECUTING_HOOK, "x" * 4_000) == 0


def test_an_injecting_hook_is_charged_for_the_text_it_injects():
    """Never zero, and charged per nudge: a hook that fires twice injects
    twice. Stated as a floor rather than an exact figure — the injected block
    then rides in the conversation for every later turn."""
    injected = "x" * 400
    assert wb.standing_tokens_per_session(DELIVERY_INJECTING_HOOK, injected) == (
        100 * MAX_NUDGES_PER_SESSION
    )
    # And strictly more than the same words carried as a CLAUDE.md rule, which
    # is the whole reason it cannot inherit the executing hook's zero.
    assert wb.standing_tokens_per_session(
        DELIVERY_INJECTING_HOOK, injected,
    ) > wb.standing_tokens_per_session(DELIVERY_CLAUDE_MD_RULE, injected)


# --- The quality floor ---------------------------------------------------------

def test_generic_placeholder_fixes_are_rejected():
    assert wb.is_placeholder_fix("Review examples — no known fix template matched.")
    assert wb.is_placeholder_fix("TODO")
    assert wb.is_placeholder_fix("")
    assert wb.is_placeholder_fix("Fix it.")            # under the length floor


def test_placeholder_is_caught_inside_a_rendered_artifact_block():
    """What the budget pass actually inspects on the live relearn path is the
    RENDERED artifact, not the raw fix: the marker comment and heading come
    first, so the placeholder sentence lands mid-string. The unanchored
    "no known fix template matched" pattern is the only thing that catches it
    — anchoring it for symmetry with the other two switches the quality floor
    off for every cluster that has no fix template."""
    from tokenjam.core.optimize.relearn_apply import artifact_for_delivery

    rendered = artifact_for_delivery(
        {
            "title": "Some recurring failure",
            "proposed_fix": "Review examples — no known fix template matched.",
            "repos": ["r"], "occurrences": 5, "sessions": 3, "examples": [],
        },
        "relearn:x", 1, "some-recurring-failure",
    )
    assert not rendered.lstrip().lower().startswith("review examples"), (
        "precondition: the placeholder must sit mid-block, not at the start"
    )
    assert wb.is_placeholder_fix(rendered)


def test_an_honesty_caveat_is_not_mistaken_for_a_placeholder():
    """Every real fix in the tree says "review ... before applying". None of
    those may trip the anchored placeholder patterns."""
    assert not wb.is_placeholder_fix(_REAL_FIX)
    assert not wb.is_placeholder_fix(
        "Add a PostToolUseFailure hook for Bash. Review the example sessions "
        "before applying it.",
    )


# --- Netting and suppression ---------------------------------------------------

def _candidate(
    key, family="fam", delivery=DELIVERY_CLAUDE_MD_RULE, text=_REAL_FIX,
    tokens=1_000_000, usd=None,
):
    return wb.WriteCandidate(
        key=key, family=family, delivery=delivery, artifact_text=text,
        gross_tokens=tokens, gross_usd=usd,
    )


def test_saving_is_reported_net_of_the_rules_standing_cost():
    # Arrange: a 34-token rule re-sent across 100 observed sessions.
    basis = _basis(sessions=100)
    standing = wb.standing_tokens_per_session(1, _REAL_FIX) * 100
    # Act
    decision = wb.allocate_writes(
        [_candidate("a", tokens=1_000_000, usd=10.0)],
        wb.build_write_budget(lane_budget_tokens=1_000, lane_max_writes=5), basis,
    )["a"]
    # Assert
    assert decision.offered is True
    assert decision.standing_tokens == standing
    assert decision.net_tokens == 1_000_000 - standing
    assert decision.claimed_tokens == decision.net_tokens
    assert decision.claimed_tokens < 1_000_000        # the whole point
    assert decision.claimed_usd == pytest.approx(10.0 - standing * (10.0 / 1_000_000))


def test_a_net_negative_rule_is_suppressed_and_claims_nothing():
    """The ticket's worked example: a rule that costs more to keep than the
    failure it prevents. It must not be offered and must not claim a saving."""
    basis = _basis(sessions=100)
    decision = wb.allocate_writes(
        [_candidate("a", tokens=1_500, usd=0.02)],       # < 34 tok/session x 100
        wb.build_write_budget(lane_budget_tokens=1_000, lane_max_writes=5), basis,
    )["a"]
    assert decision.net_negative is True
    assert decision.offered is False
    assert decision.claimed_tokens == 0
    assert decision.claimed_usd == 0.0
    assert decision.payback_ratio is not None and decision.payback_ratio < 1.0
    assert decision.reason == wb.REASON_NET_NEGATIVE


def test_a_placeholder_never_becomes_a_permanent_rule():
    decision = wb.allocate_writes(
        [_candidate("a", text="Review examples — no known fix template matched.")],
        wb.build_write_budget(lane_budget_tokens=1_000, lane_max_writes=5), _basis(),
    )["a"]
    assert decision.offered is False
    assert decision.claimed_tokens == 0
    assert decision.reason == wb.REASON_PLACEHOLDER


def test_same_family_clusters_collapse_onto_one_block():
    """Three clusters of one family used to mean three identical appended
    blocks. Now the largest carries the write and the rest say so, and only
    ONE block's standing cost is charged."""
    basis = _basis(sessions=100)
    decisions = wb.allocate_writes(
        [
            _candidate("small", tokens=200_000),
            _candidate("big", tokens=900_000),
            _candidate("mid", tokens=500_000),
        ],
        wb.build_write_budget(lane_budget_tokens=1_000, lane_max_writes=5), basis,
    )
    assert decisions["big"].offered is True
    assert [decisions[k].offered for k in ("small", "mid")] == [False, False]
    assert decisions["small"].reason == wb.REASON_FAMILY_MERGED
    # Siblings write nothing, so they stand nothing and keep their own figure.
    assert decisions["small"].standing_tokens == 0
    assert decisions["small"].claimed_tokens == 200_000
    # Exactly one block is charged across the whole family.
    assert sum(d.standing_tokens_per_session for d in decisions.values()) == (
        wb.standing_tokens_per_session(1, _REAL_FIX)
    )


def test_unpriceable_family_still_offers_only_one_block():
    """The one-block-per-family invariant must hold on the UNPRICEABLE branch
    too. When the representative carries no token figure there is nothing to
    net, so every member takes the pass-through verdict — but only the
    representative's block is ever written, so a sibling handed `offered=True`
    would put the identical artifact text on offer once per cluster and
    bypass the budget counter entirely."""
    basis = _basis(sessions=100)
    decisions = wb.allocate_writes(
        [
            _candidate("rep", tokens=0),
            _candidate("sib1", tokens=0),
            _candidate("sib2", tokens=0),
        ],
        wb.build_write_budget(lane_budget_tokens=1_000, lane_max_writes=5), basis,
    )
    assert [d.offered for d in decisions.values()].count(True) == 1
    assert decisions["rep"].offered is True
    assert decisions["rep"].basis == wb.BASIS_NOT_PRICEABLE
    for key in ("sib1", "sib2"):
        assert decisions[key].offered is False, key
        assert decisions[key].reason == wb.REASON_FAMILY_MERGED, key
        # Unchanged from the pass-through verdict: no netting was invented.
        assert decisions[key].basis == wb.BASIS_NOT_PRICEABLE, key


def test_writes_are_ranked_by_net_value_and_bounded():
    basis = _basis(sessions=100)
    decisions = wb.allocate_writes(
        [
            _candidate("low", family="f1", tokens=300_000),
            _candidate("high", family="f2", tokens=900_000),
            _candidate("mid", family="f3", tokens=600_000),
        ],
        wb.build_write_budget(lane_budget_tokens=10_000, lane_max_writes=2), basis,
    )
    assert decisions["high"].offered is True
    assert decisions["mid"].offered is True
    assert decisions["low"].offered is False
    # Deferred, not deleted: the saving is real and the snippet still copyable.
    assert decisions["low"].reason == wb.REASON_BUDGET_FULL
    assert decisions["low"].claimed_tokens == decisions["low"].net_tokens > 0


def test_the_token_budget_binds_independently_of_the_write_count():
    basis = _basis(sessions=100)
    per_rule = wb.standing_tokens_per_session(1, _REAL_FIX)
    decisions = wb.allocate_writes(
        [
            _candidate("a", family="f1", tokens=900_000),
            _candidate("b", family="f2", tokens=800_000),
        ],
        # Room for exactly one rule's per-session cost.
        wb.build_write_budget(lane_budget_tokens=per_rule, lane_max_writes=99), basis,
    )
    assert [decisions["a"].offered, decisions["b"].offered] == [True, False]


# --- The summarize cross-reference ---------------------------------------------

class _Candidate:
    def __init__(self, path, total_chars):
        self.path = path
        self.total_chars = total_chars


class _SummarizeFinding:
    def __init__(self, *candidates):
        self.candidates = list(candidates)


def test_only_the_always_loaded_share_of_a_catalog_file_is_measured():
    """A skill body loads on invoke, so a 160 KB skill library is NOT always-on
    context. Measuring it as such would slam the budget shut for everyone."""
    finding = _SummarizeFinding(
        _Candidate("/proj/CLAUDE.md", 40_000),
        _Candidate("/home/.claude/skills/ship/SKILL.md", 160_000),
        _Candidate("/home/.claude/commands/plan.md", 4_000),
    )
    assert wb.measured_agent_file_tokens(finding) == wb.tokens_from_chars(
        40_000 + wb.SKILL_ALWAYS_LOADED_CHARS * 2,
    )


def test_the_budget_is_a_share_of_what_is_already_there():
    """Relative, not absolute: one 107-token rule on a 25k-token CLAUDE.md is
    fine, forty-one of them are not, and a small file still gets a floor."""
    big = wb.build_write_budget(
        lane_budget_tokens=1_000, lane_max_writes=5,
        existing_agent_file_tokens=25_000,
    )
    assert big.budget_tokens == 1_000       # 10% of 25k, clamped by the lane cap
    assert big.ceiling_reached is False

    mid = wb.build_write_budget(
        lane_budget_tokens=1_000, lane_max_writes=5,
        existing_agent_file_tokens=6_000,
    )
    assert mid.budget_tokens == 600         # 10% of 6k, under the lane cap

    tiny = wb.build_write_budget(
        lane_budget_tokens=1_000, lane_max_writes=5,
        existing_agent_file_tokens=200,
    )
    assert tiny.budget_tokens == wb.MIN_WRITE_BUDGET_TOKENS   # the floor holds


def test_no_new_rules_are_offered_once_the_agent_files_are_pathological():
    """The uncoordinated loop, closed: past the absolute ceiling the product
    can no longer offer new permanent rules for a file the same report says to
    compress."""
    existing = wb.measured_agent_file_tokens(
        _SummarizeFinding(_Candidate("/proj/CLAUDE.md", 400_000)),
    )
    assert existing >= wb.AGENT_FILE_STANDING_CEILING_TOKENS
    budget = wb.build_write_budget(
        lane_budget_tokens=1_000, lane_max_writes=5,
        existing_agent_file_tokens=existing,
    )
    assert budget.ceiling_reached is True
    assert budget.budget_tokens == 0 and budget.max_writes == 0
    decision = wb.allocate_writes([_candidate("a")], budget, _basis())["a"]
    assert decision.offered is False
    assert decision.reason == wb.REASON_CEILING_REACHED


def test_an_unmeasured_footprint_leaves_the_lane_cap_intact():
    """None means "the analyzer did not run", never "the file is empty"."""
    assert wb.measured_agent_file_tokens(None) is None
    assert wb.measured_agent_file_tokens(_SummarizeFinding()) is None
    budget = wb.build_write_budget(lane_budget_tokens=750, lane_max_writes=4)
    assert (budget.budget_tokens, budget.max_writes) == (750, 4)
    assert budget.ceiling_reached is False


# --- Lane integration: relearn --------------------------------------------------

def _episodes(prefix, n, day=1):
    from tokenjam.core.optimize.analyzers.relearn import FailureEpisode

    return [
        FailureEpisode(
            f"{prefix}-s{i}", "repo", f"2026-06-{day + (i % 20):02d}T10:00:00Z",
            "Bash", "cmd", "no such file or directory", "act", False, 0,
        )
        for i in range(n)
    ]


def _raw(signature, family_key, title, failures):
    from tokenjam.core.optimize.analyzers.relearn import _RawCluster

    return _RawCluster(
        signature=signature, family_key=family_key, title=title, failures=failures,
    )


def test_relearn_never_offers_a_placeholder_fix_as_a_permanent_rule():
    """A family-unmatched cluster falls back to "Review examples", which used
    to become a permanent CLAUDE.md block like any other.

    `persona="claude-code"` is the precondition for a write to be offered AT
    ALL (the persona gate in `build_proposals`); these tests are about what the
    write BUDGET does once that gate has passed, so they opt in explicitly
    rather than relying on the conservative "unknown" default."""
    from tokenjam.core.optimize.analyzers.relearn import build_proposals

    proposals, _ = build_proposals(
        [_raw("Bash:weird thing", None, "Bash: weird thing", _episodes("w", 6))],
        repo_cwd_map={"repo": "/tmp/repo"}, projection=_basis(sessions=6, active_days=6),
        persona="claude-code",
    )
    assert len(proposals) == 1
    p = proposals[0]
    assert p.write_offered is False
    assert p.suggested_target == ""
    assert p.advise_only is True
    assert not hasattr(p, "estimated_recoverable_tokens")
    assert not hasattr(p, "gross_recoverable_tokens")
    assert p.write_blocked_reason == wb.REASON_PLACEHOLDER
    # The observation is still inspectable and unaffected — only the
    # (now-retired) forward claim was ever zero.
    assert p.past_overspend_tokens > 0


def test_relearn_nets_by_mechanism_not_by_whether_it_is_a_hook():
    """Three families, three mechanisms, three different standing costs.

    `sleep_chain` is an EXECUTING hook: it blocks a command and injects
    nothing, so its zero is earned. `cwd_confusion` is an INJECTING hook: it
    hands the model ``additionalContext``, so it is prompt text and is charged.
    `read_too_large` is a CLAUDE.md rule and is netted down as it always was.

    **This inverts the assertion it replaces.** That one paired `cwd_confusion`
    with "no standing prompt cost, net == gross" — enforcing the exact defect,
    since the two hooks have opposite cost behaviour and only one of them is
    free. The guard is kept and now defends the correct state.

    (An earlier version of this test used `edit_before_read`, which stopped
    being a valid vehicle when that family became advisory-only: an advisory
    family is never offered and so cannot demonstrate netting at all, yet the
    assertions still passed. A fixture that is a real domain object can quietly
    stop testing what it claims to.)"""
    from tokenjam.core.optimize.analyzers.relearn import build_proposals

    proposals, _ = build_proposals(
        [
            _raw("sleep_chain", "sleep_chain", "sleep chain", _episodes("s", 40)),
            _raw("cwd_confusion", "cwd_confusion", "cwd confusion", _episodes("c", 40)),
            _raw("read_too_large", "read_too_large", "read too large",
                 _episodes("e", 40)),
        ],
        repo_cwd_map={"repo": "/tmp/repo"}, persona="claude-code",
        projection=_basis(sessions=80, active_days=10, window_days=30.0),
    )
    by_family = {p.family_key: p for p in proposals}
    guard = by_family["sleep_chain"]
    nudge = by_family["cwd_confusion"]
    note = by_family["read_too_large"]

    assert guard.delivery == DELIVERY_EXECUTING_HOOK
    assert guard.standing_cost_tokens == 0
    assert guard.payback_ratio is None or guard.payback_ratio >= 1.0
    assert not hasattr(guard, "estimated_recoverable_tokens")

    # The bug this whole change exists to close: same "it is a hook", opposite
    # answer, because the mechanism is what was asked.
    assert nudge.delivery == DELIVERY_INJECTING_HOOK
    assert nudge.standing_cost_tokens > 0
    assert nudge.standing_cost_basis

    assert note.delivery == DELIVERY_CLAUDE_MD_RULE
    assert note.standing_cost_tokens > 0
    assert note.standing_cost_basis
    assert not hasattr(note, "estimated_recoverable_tokens")


def test_relearn_bounds_how_many_permanent_rules_a_run_offers():
    """41 live clusters used to mean up to 41 separately appended blocks."""
    from tokenjam.core.optimize.analyzers.relearn import build_proposals

    clusters = [
        _raw(f"edit_before_read:{i}", f"fam{i}", f"cluster {i}", _episodes(f"c{i}", 40))
        for i in range(12)
    ]
    proposals, _ = build_proposals(
        clusters, repo_cwd_map={"repo": "/tmp/repo"}, persona="claude-code",
        projection=_basis(sessions=40, active_days=10),
    )
    offered = [p for p in proposals if p.write_offered]
    assert len(proposals) == 12
    assert len(offered) <= wb.RELEARN_MAX_OFFERED_WRITES
    # Ranked: nothing suppressed outranks anything offered, on the pre-net
    # observation the budget itself ranks by (no forward field is rendered
    # any more to check this against).
    suppressed = [p for p in proposals if not p.write_offered and not p.net_negative]
    if offered and suppressed:
        assert min(p.past_overspend_tokens for p in offered) >= max(
            p.past_overspend_tokens for p in suppressed
        )


def test_relearn_carries_no_monthly_forward_field():
    """The 30-day-basis forward claim (`estimated_monthly_*` /
    `gross_monthly_*`) is retired entirely — a relearn cluster's headline is
    `past_overspend_*`, the same past-tense figure every other analyzer's
    card shows. `gross_recoverable_tokens` is also gone: it held the SAME
    value as `past_overspend_tokens`, so the one observation now doubles as
    the netting input `write_budget.WriteCandidate.gross_tokens` needs to
    decide whether a permanent write is worth OFFERING
    (`standing_cost_*`, `payback_ratio` survive on that decision)."""
    from tokenjam.core.optimize.analyzers.relearn import build_proposals

    proposals, _ = build_proposals(
        [_raw("edit_before_read", "edit_before_read", "edit before read",
              _episodes("e", 40))],
        repo_cwd_map={"repo": "/tmp/repo"}, persona="claude-code",
        projection=_basis(sessions=80, active_days=10),
    )
    p = proposals[0]
    for retired in (
        "estimated_monthly_tokens", "estimated_monthly_usd",
        "gross_monthly_tokens", "gross_monthly_usd",
        "estimated_recoverable_tokens", "estimated_recoverable_usd",
        "gross_recoverable_tokens",
    ):
        assert not hasattr(p, retired)
    assert p.past_overspend_tokens > 0


# --- Lane integration: cost proposals -------------------------------------------

def _cost_report(sessions=100, active_days=10, days=30.0, summarize=None, persona="claude-code"):
    from tokenjam.core.optimize.types import OptimizeReport, WindowSummary
    from tokenjam.utils.time_parse import utcnow

    now = utcnow()
    window = WindowSummary(
        since=now, until=now, days=days, sessions=sessions, active_days=active_days,
        spans=sessions * 10, total_tokens=10_000_000, total_cost_usd=100.0,
        thin_data=False,
    )
    findings = {"summarize": summarize} if summarize is not None else {}
    return OptimizeReport(window=window, persona=persona, findings=findings)


def _reuse_finding(tokens, usd):
    from tokenjam.core.optimize.types import ReuseCluster, ReuseFinding

    cluster = ReuseCluster(
        cluster_id="abc123456789", tool_signature=("bash", "read"),
        prompt_prefix_hash=None, repetitions=4, avg_planning_tokens=300,
        avg_planning_cost_usd=0.01, cache_reuse_recoverable_usd=usd,
        script_replacement_recoverable_usd=usd * 2,
        cache_reuse_recoverable_tokens=tokens,
        script_replacement_recoverable_tokens=tokens * 2,
        example_session_ids=["s1"], skeleton_session_id="s1",
    )
    return ReuseFinding(
        clusters=[cluster], past_overspend_usd=usd,
        past_overspend_tokens=tokens, estimate_basis="reuse basis",
    )


def test_a_cost_write_reports_net_of_its_own_standing_cost():
    from tokenjam.core.optimize.cost_proposals import cost_proposals_from_report

    # "mixed" persona: reuse still offers its workspace write (claude-code
    # gates reuse out entirely — see test_persona_analyzer_gate.py), so
    # this is the write-netting mechanics, exercised on a persona the gate
    # leaves untouched.
    report = _cost_report(persona="mixed")
    report.findings["reuse"] = _reuse_finding(2_000_000, 60.0)
    p = next(p for p in cost_proposals_from_report(report) if p.analyzer == "reuse")

    assert p.write_offered is True
    assert p.apply_capable is True
    assert p.gross_recoverable_tokens == 2_000_000
    assert p.standing_cost_tokens_per_session > 0
    assert p.past_overspend_tokens < p.gross_recoverable_tokens
    assert p.past_overspend_usd < p.gross_recoverable_usd
    assert p.standing_cost_basis


def test_write_budget_suppresses_net_negative_cost_write():
    """A rule whose keep costs more than it recovers degrades to advise-only
    with the text still copyable, exactly the way the persona gate degrades
    one, and claims nothing."""
    from tokenjam.core.optimize.cost_proposals import cost_proposals_from_report

    report = _cost_report(sessions=100, persona="mixed")
    report.findings["reuse"] = _reuse_finding(900, 0.03)
    p = next(p for p in cost_proposals_from_report(report) if p.analyzer == "reuse")

    assert p.net_negative is True
    assert p.write_offered is False
    assert p.apply_capable is False and p.advise_only is True
    assert p.proposed_fix == "" and p.suggestion       # the fix is still copyable
    assert p.past_overspend_tokens == 0
    assert p.past_overspend_usd == 0.0
    assert p.gross_recoverable_tokens == 900           # inspectable, not hidden


def test_cost_writes_stop_when_the_agent_files_are_pathologically_large():
    from tokenjam.core.optimize.analyzers.summarize import (
        SummarizeCandidate,
        SummarizeFinding,
    )
    from tokenjam.core.optimize.cost_proposals import cost_proposals_from_report

    summarize = SummarizeFinding(candidates=[
        SummarizeCandidate(path="CLAUDE.md", kind="prompt", scope="project",
                           est_tokens_saved=5_000, total_chars=400_000),
    ])
    report = _cost_report(summarize=summarize, persona="mixed")
    report.findings["reuse"] = _reuse_finding(2_000_000, 60.0)
    p = next(p for p in cost_proposals_from_report(report) if p.analyzer == "reuse")

    assert p.write_offered is False
    assert p.write_blocked_reason == wb.REASON_CEILING_REACHED
    # Deferred, not denied: the saving is real, so the net claim stands.
    assert p.past_overspend_tokens > 0


def test_a_non_write_cost_card_is_left_completely_untouched():
    """`downsize` writes no standing prompt text, so it must pass through the
    budget pass with its figures exactly as the adapter built them."""
    from tokenjam.core.optimize.cost_proposals import cost_proposals_from_report
    from tokenjam.core.optimize.types import DowngradeFinding

    report = _cost_report()
    report.downgrade = DowngradeFinding(
        candidate_sessions=4, total_sessions=10, actual_cost_usd=5.0,
        alternative_cost_usd=2.0, monthly_savings_usd=3.0, percent_of_sessions=40.0,
        examples=[], suggestions={"claude-opus-4-8": "claude-sonnet-5"},
        past_overspend_usd=3.0, past_overspend_tokens=90_000,
        percent_of_tokens=35.0, estimate_basis="downsize basis",
    )
    p = next(p for p in cost_proposals_from_report(report) if p.analyzer == "downsize")
    assert p.past_overspend_usd == 3.0
    assert p.past_overspend_tokens == 90_000
    assert p.standing_cost_tokens == 0
    assert p.gross_recoverable_tokens is None       # never entered the pass


# --- The value floor (MIN_NET_WRITE_USD) --------------------------------------
# Netting alone only asks "does this break even". These pin the stronger
# question the floor asks: "is this worth one of five permanent blocks in a
# file the user re-sends forever".

def test_a_rule_returning_less_than_the_floor_is_not_offered():
    # Arrange: comfortably net-POSITIVE in tokens, but worth only $2.
    decision = wb.allocate_writes(
        [_candidate("a", tokens=1_000_000, usd=2.0)],
        wb.build_write_budget(lane_budget_tokens=1_000, lane_max_writes=5),
        _basis(sessions=100),
    )["a"]

    assert decision.net_negative is False          # it DID clear its own cost
    assert decision.offered is False               # ...and still isn't worth a block
    assert decision.reason == wb.REASON_BELOW_VALUE_FLOOR


def test_the_floor_defers_the_write_but_never_suppresses_the_claim():
    """The floor is a decision about OUR remedy, not about the user's money.

    Repo CLAUDE.md rule 32: a gate on action-availability may never make an
    incurred cost read as zero. So a below-floor family keeps its net claim and
    its copyable snippet, exactly like a budget-deferred one — the only thing
    it loses is the permanent write.
    """
    decision = wb.allocate_writes(
        [_candidate("a", tokens=1_000_000, usd=2.0)],
        wb.build_write_budget(lane_budget_tokens=1_000, lane_max_writes=5),
        _basis(sessions=100),
    )["a"]

    assert decision.claim_suppressed is False
    assert decision.claimed_tokens > 0
    assert decision.claimed_usd is not None and decision.claimed_usd > 0


def test_a_rule_at_or_above_the_floor_is_offered():
    decision = wb.allocate_writes(
        [_candidate("a", tokens=1_000_000, usd=wb.MIN_NET_WRITE_USD * 3)],
        wb.build_write_budget(lane_budget_tokens=1_000, lane_max_writes=5),
        _basis(sessions=100),
    )["a"]
    assert decision.offered is True
    assert decision.reason == ""


def test_an_unpriced_family_is_never_held_to_a_dollar_floor():
    """No dollar figure means no comparison, and inventing a rate to make one
    would kill real fixes for the sin of being unquantified — the same reason
    `BASIS_NOT_PRICEABLE` exists."""
    decision = wb.allocate_writes(
        [_candidate("a", tokens=1_000_000, usd=None)],
        wb.build_write_budget(lane_budget_tokens=1_000, lane_max_writes=5),
        _basis(sessions=100),
    )["a"]
    assert decision.offered is True
    assert decision.reason == ""


def test_a_below_floor_family_does_not_consume_a_write_slot():
    """The floor exists to protect the scarce slots, so it has to be applied
    BEFORE the budget — otherwise a $2 family could crowd out a $50 one."""
    candidates = [
        _candidate("small", family="f1", tokens=1_000_000, usd=1.0),
        _candidate("big", family="f2", tokens=1_000_000, usd=50.0),
    ]
    decisions = wb.allocate_writes(
        candidates,
        wb.build_write_budget(lane_budget_tokens=1_000, lane_max_writes=1),
        _basis(sessions=100),
    )
    assert decisions["small"].offered is False
    assert decisions["small"].reason == wb.REASON_BELOW_VALUE_FLOOR
    assert decisions["big"].offered is True        # the slot went to the big one


def test_the_floor_has_a_short_label_so_it_never_renders_generic():
    assert wb.short_reason(wb.REASON_BELOW_VALUE_FLOOR) != "no permanent fix offered"
    assert "5" in wb.short_reason(wb.REASON_BELOW_VALUE_FLOOR)


def test_the_floor_applies_to_the_combined_family_value_not_one_members_share():
    """Regression guard: several clusters sharing one write family collapse
    onto ONE block (`_decide_family`), so the floor must compare against the
    family's COMBINED net, not just the representative member's own share.
    Two members at $3 each are individually below MIN_NET_WRITE_USD, but
    their combined $6 clears it -- the shared rule must still be offered."""
    candidates = [
        _candidate("a", family="combofam", tokens=1_000_000, usd=3.0),
        _candidate("b", family="combofam", tokens=1_000_000, usd=3.0),
    ]
    decisions = wb.allocate_writes(
        candidates,
        wb.build_write_budget(lane_budget_tokens=1_000, lane_max_writes=5),
        _basis(sessions=100),
    )
    rep_decision = next(d for d in decisions.values() if d.reason != wb.REASON_FAMILY_MERGED)
    assert rep_decision.offered is True
    assert rep_decision.net_usd is not None and rep_decision.net_usd > wb.MIN_NET_WRITE_USD
