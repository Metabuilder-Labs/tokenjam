"""The past-overspend contract: the number the product leads with.

Every user-facing figure on the Dashboard hero and the Review inbox is
PAST TENSE and window-OBSERVED — priced at the rates it actually billed at,
over a window that has already happened. These tests pin the four properties
that make that figure safe to show, because each of them failing turns the
feature net-negative rather than merely wrong:

  0. the figure a reader will call waste/overspend is the AVOIDABLE amount,
     never the full observed cost. Waste is only what could have been avoided;
     unavoidable spend is cost. Leading with the cost figure implicitly
     asserted that everything it exceeded the avoidable figure by had been
     shown to be unavoidable — a claim nothing in the analyzer supports, since
     the two are computed over DIFFERENT POPULATIONS;
  1. neither figure is ever summed into a recoverable total, and the two are
     never summed with each other (the avoidable figure is a subset of the
     cost figure, so adding them double-counts);
  2. neither is multiplied by the central 30-day pacing ratio — both sides of
     the pair read the same raw analysed window, so their difference is
     attributable to avoidability alone and never to a time-basis artifact;
  3. the surfaces read them off the payload rather than deriving them in JS
     (two derivations of one number drift the moment either side is edited).
"""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from tokenjam.core.rulewrite.kinds import DELIVERY_CLAUDE_MD_RULE

from tokenjam.core.optimize.cost_proposals import (
    CostProposal,
    _with_past_overspend,
    backfill_legacy_past_overspend_fields,
    cost_proposals_from_report,
    past_overspend_rollup,
)
from tokenjam.core.optimize.analyzers.relearn import RelearnFinding
from tokenjam.core.optimize.types import (
    DowngradeFinding,
    OptimizeReport,
    WindowSummary,
)

#: The per-analyzer dollar/token field names retired by the field collapse.
#: `estimated_recoverable_*` was `past_overspend_*` under a forward-framed
#: name; `estimated_monthly_*` was that number times a pace ratio. Neither may
#: reappear on a proposal, a payload, or a rollup — see the field contract in
#: the repo CLAUDE.md.
_RETIRED_DOLLAR_FIELDS = frozenset({
    "estimated_recoverable_usd", "estimated_recoverable_tokens",
    "estimated_monthly_usd", "estimated_monthly_tokens",
    # The total-observed-cost pair: `cost_of_waste_*` was the analyzer-side
    # input, `observed_cost_*` the field it was published on. Deleted by founder
    # decision — two analyzers of twelve emitted it, one rendered it, and the
    # rollup total it fed covered 2 of the headline's 13 proposals while its
    # disclosure called the headline a subset of it.
    "cost_of_waste_usd", "cost_of_waste_tokens", "cost_of_waste_basis",
    "observed_cost_usd", "observed_cost_tokens", "observed_cost_basis",
})

NOW = datetime(2026, 7, 20, 12, 0, tzinfo=timezone.utc)

UI = Path(__file__).resolve().parents[2] / "tokenjam" / "ui" / "index.html"


def _proposal(**kw) -> CostProposal:
    base = dict(
        kind="cost", analyzer="downsize", signature="cost:downsize",
        title="t", target_key={}, evidence="e", baseline={}, advise_text="a",
    )
    base.update(kw)
    return CostProposal(**base)


def _resend_finding():
    from tokenjam.core.optimize.analyzers.context_resend import ResendFinding

    return ResendFinding(
        sessions_examined=40, repeat_share=0.93, repeat_tokens=10_000,
        past_overspend_tokens=1_400_000_000,
        past_overspend_usd=703.78,
        estimate_basis="resend basis",
        fix_compaction="Run /compact.",
        # The real analyzer always emits this wherever the avoidable figure was
        # computed over a subset (pinned by test_context_resend.py); stated here
        # because this fixture builds the finding directly rather than running
        # the analyzer.
        coverage_note=(
            "COVERAGE. 40 session(s) in this window carried repeat volume; the "
            "avoidable figure was computed over 12 of them."
        ),
    )


def _relearn_finding():
    """A relearn finding shaped like a real gated run: some clusters have no
    fix template, some are net-negative to codify — both still cost real
    money, which lands on this finding's ``past_overspend_*``.

    It produces NO ``CostProposal`` any more: relearn's one aggregate card
    carried only the retired total-observed-cost field, so the card went with
    the field (see
    ``test_relearn_archive_and_cost.test_relearn_produces_no_cost_proposal_and_keeps_its_claim_on_its_clusters``).
    The fixture stays because a report carrying a relearn finding must still
    adapt cleanly and contribute nothing, which is the case a report built with
    only resend would not exercise.
    """
    from tokenjam.core.optimize.analyzers.relearn import RelearnCluster
    from tokenjam.core.optimize.write_budget import REASON_NET_NEGATIVE, REASON_PLACEHOLDER

    clusters = [
        RelearnCluster(
            signature="no-fix", family_key=None, title="No fix template",
            sessions=3, occurrences=5, repos=["demo"], delivery=DELIVERY_CLAUDE_MD_RULE, scope="project",
            proposed_fix="", write_offered=False,
            write_blocked_reason=REASON_PLACEHOLDER,
        ),
        RelearnCluster(
            signature="net-neg", family_key=None, title="Net-negative rule",
            sessions=4, occurrences=6, repos=["demo"], delivery=DELIVERY_CLAUDE_MD_RULE, scope="project",
            proposed_fix="Add a rule.", write_offered=False,
            write_blocked_reason=REASON_NET_NEGATIVE,
        ),
    ]
    return RelearnFinding(
        clusters=clusters, past_overspend_usd=46.30,
        past_overspend_tokens=100_000,
        past_overspend_basis="occurrences x measured re-read tail, observed",
    )


# --- 0. the waste-labelled figure is the AVOIDABLE one --------------------- #

def test_the_waste_labelled_figure_never_exceeds_the_avoidable_figure():
    """THE invariant this whole surface rests on.

    `past_overspend_usd` is what every waste/overspend-worded surface renders.
    It may never exceed the avoidable figure, because a figure a reader calls
    "wasted" is a claim that it could have been avoided. Before this was
    enforced, resend put its full $6,972 cost of re-sending context here beside
    a $398 avoidable figure — implicitly asserting 94.3% of it was unavoidable,
    when in truth that 94.3% was analysed on another card, filtered out below
    the context floor, or outside the tail definition, and was never shown to
    be unavoidable at all.

    Asserted across a whole report rather than one hand-built proposal, so a
    future analyzer that routes a cost figure into the headline slot fails here
    rather than shipping.
    """
    window = WindowSummary(
        since=NOW - timedelta(days=30), until=NOW, days=30, sessions=200,
        spans=1_000, total_tokens=1, total_cost_usd=50.0, thin_data=False,
        active_days=28,
    )
    dg = DowngradeFinding(
        candidate_sessions=4, total_sessions=10, actual_cost_usd=5.0,
        alternative_cost_usd=2.0, monthly_savings_usd=3.0, percent_of_sessions=40.0,
        examples=[], suggestions={"claude-opus-4-8": "claude-sonnet-5"},
        past_overspend_usd=3.0, past_overspend_tokens=1_000,
        percent_of_tokens=35.0, estimate_basis="downsize basis",
    )
    report = OptimizeReport(window=window, downgrade=dg)
    report.findings["resend"] = _resend_finding()
    report.findings["relearn"] = _relearn_finding()

    props = cost_proposals_from_report(report)
    assert props, "fixture produced no proposals — the guard would be vacuous"
    for p in props:
        # The invariant is now STRUCTURAL rather than a comparison: there is
        # exactly one avoidable field, so the waste-labelled figure cannot be
        # anything other than the avoidable one. What still has to be checked
        # is that no SECOND avoidable quantity has crept back onto the card.
        fields = set(asdict(p))
        assert not (fields & _RETIRED_DOLLAR_FIELDS), (
            f"{p.analyzer} carries a second per-analyzer dollar field "
            f"({fields & _RETIRED_DOLLAR_FIELDS}) beside past_overspend_usd"
        )
    block = past_overspend_rollup(props)
    # Every dollar figure the block publishes is the one canonical total.
    dollar_keys = {k for k, v in block.items()
                   if k.endswith("_usd") and isinstance(v, (int, float))}
    assert dollar_keys == {"past_overspend_usd"}


def test_a_rollup_block_publishes_one_dollar_total_over_one_population():
    """THE invariant the purge exposed: **any two figures published together in
    one rollup block must cover the same set of proposals.**

    The block used to publish two dollar totals. ``past_overspend_usd`` summed
    every proposal carrying the canonical figure; ``observed_cost_usd`` summed
    the 2 of 12 analyzers that also emitted a total-cost figure. They were
    shipped adjacent, under a ``cost_disclosure`` reading "the avoidable figure
    is a subset of it" — and on live data that was false: the headline summed 13
    proposals while the cost total covered 2, and roughly $5,754 of the $6,163
    headline came from proposals with no observed cost at all, so most of the
    headline lay OUTSIDE the figure it was described as part of.

    Stated as a rule rather than as a bug: a second total is summed over its own
    population, the reader computes the ratio anyway, and the ratio of two
    figures over two populations means nothing. So the guard is that there is
    exactly ONE dollar total, and that every other published quantity is counted
    over the proposals that total covers.

    Written to catch a REGRESSION, not just today's shape: it builds proposals
    where a divergent second population would be plainly visible (one proposal
    carrying a large figure that the canonical total does not include) and
    asserts no key reports it.
    """
    props = [
        _with_past_overspend(_proposal(
            analyzer="resend", signature="cost:resend", past_overspend_usd=398.41,
            coverage_note="COVERAGE. ...",
        )),
        _with_past_overspend(_proposal(signature="cost:downsize",
                                       past_overspend_usd=40.0)),
        # The shape that made the old claim false: a big contributor to the
        # headline that a second, differently-populated total would have missed.
        _with_past_overspend(_proposal(
            analyzer="summarize", signature="cost:summarize",
            past_overspend_usd=4_811.33,
        )),
    ]
    block = past_overspend_rollup(props)

    # ONE dollar total, and it sums exactly the proposals `proposal_count` names.
    dollar_keys = {k for k, v in block.items()
                   if k.endswith("_usd") and isinstance(v, (int, float))}
    assert dollar_keys == {"past_overspend_usd"}
    assert block["past_overspend_usd"] == pytest.approx(398.41 + 40.0 + 4_811.33)
    assert block["proposal_count"] == 3

    # The per-analyzer breakdown covers that same set and carries no second
    # dollar key of its own, so a renderer cannot reconstruct one from it.
    by_analyzer = {a["analyzer"]: a for a in block["by_analyzer"]}
    assert set(by_analyzer) == {"resend", "downsize", "summarize"}
    assert sum(a["usd"] for a in by_analyzer.values()) == pytest.approx(
        block["past_overspend_usd"]
    )
    for entry in by_analyzer.values():
        # Exactly these keys. A second dollar key here is how a renderer would
        # rebuild the removed total per-analyzer even with the top-level one gone.
        assert set(entry) == {"analyzer", "count", "usd", "tokens"}

    # And the disclosure that existed only to explain the removed figure is gone
    # rather than orphaned. An orphaned disclosure is worse than none: it
    # describes a relationship between figures the reader can no longer see.
    assert "cost_disclosure" not in block
    assert not [k for k in block if "observed_cost" in k]
    assert block["disclosure"]      # the surviving one, about the figure shown


def _report():
    """A minimal report carrying downsize/cache/trim findings — mirrors
    test_cost_proposals.py's own ``_report()`` (kept local rather than
    imported so this file's fixtures stay self-contained)."""
    from tokenjam.core.optimize.analyzers.cache_efficacy import (
        CacheEfficacyFinding,
        CacheEfficacyRow,
    )
    from tokenjam.core.optimize.analyzers.prompt_bloat import BloatPrompt, PromptBloatFinding

    dg = DowngradeFinding(
        candidate_sessions=4, total_sessions=10, actual_cost_usd=5.0,
        alternative_cost_usd=2.0, monthly_savings_usd=3.0, percent_of_sessions=40.0,
        examples=[], suggestions={"claude-opus-4-8": "claude-sonnet-5"},
        past_overspend_usd=3.0, percent_of_tokens=35.0,
        estimate_basis="downsize basis",
    )
    cache = CacheEfficacyFinding(
        flagged=[CacheEfficacyRow("anthropic", "claude-sonnet-5", 100_000, 5_000,
                                  0.05, "full", True)],
        past_overspend_usd=1.2, estimate_basis="cache basis",
    )
    trim = PromptBloatFinding(
        enabled=True,
        per_prompt=[BloatPrompt(agent_id="svc-a", sample_chars="x", prompt_chars=8000,
                                significant_chars=3000, bloat_chars=5000,
                                estimated_token_reduction=1250)],
        past_overspend_usd=0.8, estimate_basis="trim basis",
    )
    window = WindowSummary(since=NOW - timedelta(days=5), until=NOW, days=5, sessions=10,
                           spans=100, total_tokens=1, total_cost_usd=5.0, thin_data=False)
    return OptimizeReport(window=window, downgrade=dg, findings={"cache": cache, "trim": trim})


# --- 1. never summed into a recoverable total ------------------------------ #

def test_a_retired_cost_field_cannot_be_set_on_a_proposal_at_all():
    # The strongest form of "never summed": the field a caller would sum does not
    # exist, so an adapter that tried to attach a total-cost figure raises rather
    # than quietly shipping an untracked number a later rollup might pick up.
    for retired in sorted(_RETIRED_DOLLAR_FIELDS):
        with pytest.raises(TypeError):
            _proposal(**{retired: 7_038.85})

    prop = _with_past_overspend(_proposal(
        analyzer="resend", signature="cost:resend",
        past_overspend_usd=703.78, past_overspend_tokens=1_400_000_000,
    ))
    rollup = past_overspend_rollup([prop])
    assert rollup["past_overspend_usd"] == 703.78
    assert rollup["past_overspend_tokens"] == 1_400_000_000


def test_there_is_exactly_one_rollup_and_one_per_analyzer_dollar_field():
    """The collapse itself, pinned.

    Three near-identical per-analyzer dollar quantities used to coexist
    (`past_overspend_usd`, `estimated_recoverable_usd`, and
    `estimated_monthly_usd` = that number x a pace ratio). A session comparing
    the first two reported them "identical for 6 of 7 analyzers" while the
    Review inbox rendered the third, 7.14% larger. One field now, one rollup.
    """
    import tokenjam.core.optimize.cost_proposals as cp

    assert not hasattr(cp, "estimated_recoverable_rollup")
    assert not hasattr(cp, "compute_projection_ratio")
    assert not hasattr(cp, "_with_rollup_projection")
    assert not (set(CostProposal.__dataclass_fields__) & _RETIRED_DOLLAR_FIELDS)

    priced = _proposal(signature="a", past_overspend_usd=100.0, past_overspend_tokens=10)
    unpriced = _proposal(signature="b")
    block = past_overspend_rollup([priced, unpriced])
    assert block["past_overspend_usd"] == 100.0
    assert block["proposal_count"] == 1              # only the priced one counts
    assert block["deduplicated_proposal_count"] == 2  # both still render


def test_a_stale_cache_cannot_resurrect_a_retired_cost_figure():
    # A cache written before the purge still carries the retired keys, and it also
    # carries `past_overspend_basis` — which is the early-return condition of the
    # read-time migration. So the strip has to happen BEFORE that return, or a
    # warm daemon keeps rendering a deleted figure for a whole recompute interval.
    warm = backfill_legacy_past_overspend_fields({
        "analyzer": "resend", "signature": "cost:resend",
        "past_overspend_usd": 703.78, "past_overspend_basis": "already stamped",
        "observed_cost_usd": 7_038.85, "observed_cost_basis": "observed",
        "cost_of_waste_usd": 7_038.85,
    })
    assert warm["past_overspend_usd"] == 703.78
    assert warm["past_overspend_basis"] == "already stamped"
    assert not (set(warm) & _RETIRED_DOLLAR_FIELDS)


# --- 2. never paced ---------------------------------------------------------#

def test_no_pacing_ratio_is_applied_to_a_past_overspend_figure():
    # Pacing is gone from the cost pipeline entirely: multiplying an
    # observation by a forward ratio turns the one figure that needs no trust
    # into the one that needs the most. A window whose guardrails WOULD have
    # produced a live 3.0x ratio (10 active days, 200 sessions over 30 days)
    # must still yield the raw window figure.
    dg = DowngradeFinding(
        candidate_sessions=4, total_sessions=10, actual_cost_usd=5.0,
        alternative_cost_usd=2.0, monthly_savings_usd=3.0, percent_of_sessions=40.0,
        examples=[], suggestions={"claude-opus-4-8": "claude-sonnet-5"},
        past_overspend_usd=3.0, past_overspend_tokens=1_000,
        percent_of_tokens=35.0, estimate_basis="downsize basis",
    )
    window = WindowSummary(
        since=NOW - timedelta(days=30), until=NOW, days=30, sessions=200, spans=1_000,
        total_tokens=1, total_cost_usd=50.0, thin_data=False, active_days=10,
    )
    props = cost_proposals_from_report(OptimizeReport(window=window, downgrade=dg))
    prop = next(p for p in props if p.analyzer == "downsize")

    # The figure is the raw window observation — 3.0, not 3.0 x 3.0.
    assert prop.past_overspend_usd == pytest.approx(3.0)
    assert prop.past_overspend_tokens == 1_000
    assert not (set(asdict(prop)) & _RETIRED_DOLLAR_FIELDS)

    # And the rollup can't pace it either: it takes no pace to project from.
    block = past_overspend_rollup(props)
    assert block["past_overspend_usd"] == pytest.approx(3.0)
    assert "projection_ratio" not in block
    assert "projected_usd_30d" not in block
    with pytest.raises(TypeError):
        past_overspend_rollup(props, active_days=10, n_sessions=200)  # type: ignore[call-arg]


def test_past_overspend_reads_the_netted_figure_not_the_gross_one():
    # A write-bearing card is netted against what its rule costs to KEEP. The
    # observed figure must follow the netting, not the pre-net gross, or the
    # card would state an overspend larger than the product's own arithmetic
    # says it is.
    prop = _with_past_overspend(_proposal(
        signature="cost:reuse", analyzer="reuse",
        past_overspend_usd=4.0, gross_recoverable_usd=9.0,
    ))
    assert prop.past_overspend_usd == 4.0


# --- the single-number vs paired-number rule ------------------------------- #

def test_every_card_renders_exactly_one_number():
    # There is no paired-number shape left. `resend` was the one analyzer that had
    # a second, larger figure to show; with it deleted, every card is the
    # single-number shape and the stamper has one basis string to attach.
    single = _with_past_overspend(_proposal(past_overspend_usd=12.0,
                                            past_overspend_tokens=99,
                                            estimate_basis="downsize basis"))
    assert single.past_overspend_usd == 12.0
    assert "downsize basis" in single.past_overspend_basis

    resend = _with_past_overspend(_proposal(
        analyzer="resend", past_overspend_usd=703.78,
        estimate_basis="resend basis",
    ))
    assert resend.past_overspend_usd == 703.78
    assert "resend basis" in resend.past_overspend_basis
    row = asdict(resend)
    # `gross_recoverable_usd` (the netting disclosure's pre-net figure) is not set
    # here, so the canonical field is the only one carrying a number at all.
    assert {f for f in row if f.endswith("_usd") and row[f] is not None} == {
        "past_overspend_usd"
    }


def test_resend_adapter_carries_one_figure_and_promises_no_second_one():
    from tokenjam.core.optimize.cost_proposals import _resend_to_proposals

    prop = _with_past_overspend(
        _resend_to_proposals(_resend_finding(), persona="claude-code")[0]
    )
    assert prop.past_overspend_usd == 703.78
    assert prop.past_overspend_tokens == 1_400_000_000
    # The evidence line states the observation without recovery vocabulary, and no
    # longer promises a cost figure "reported below": there is nothing below to
    # report it, so the sentence would have pointed at an empty slot.
    assert "recoverable" not in prop.evidence
    assert "as cost, not waste" not in prop.evidence
    assert "reported below" not in prop.evidence
    assert "inherently re-sends" not in prop.evidence
    # The coverage question that sentence gestured at is still answered, in words.
    assert "COVERAGE" in prop.coverage_note


def test_legacy_cached_proposal_migrates_the_old_field_names_on_read():
    # A cache written before the collapse carries `estimated_recoverable_*`
    # (the same quantity under its old name) and `estimated_monthly_*` (that
    # number paced). The first migrates; the second is DROPPED rather than
    # promoted — reviving a paced figure as the past-tense headline is the
    # exact mistake this collapse exists to prevent.
    single = backfill_legacy_past_overspend_fields(
        {"analyzer": "cache", "estimated_recoverable_usd": 4.5,
         "estimated_recoverable_tokens": 700, "estimated_monthly_usd": 4.82,
         "estimated_monthly_tokens": 750, "estimate_basis": "cache basis"}
    )
    assert single["past_overspend_usd"] == 4.5
    assert single["past_overspend_tokens"] == 700
    assert "cache basis" in single["past_overspend_basis"]
    assert not (set(single) & _RETIRED_DOLLAR_FIELDS)

    # A legacy entry carrying the retired cost keys migrates its canonical figure
    # and DROPS them: there is no field left to migrate them onto.
    with_cost = backfill_legacy_past_overspend_fields(
        {"analyzer": "resend", "estimated_recoverable_usd": 703.78,
         "cost_of_waste_usd": 7_038.85, "cost_of_waste_basis": "observed"}
    )
    assert with_cost["past_overspend_usd"] == 703.78
    assert not (set(with_cost) & _RETIRED_DOLLAR_FIELDS)

    # A legacy entry carrying ONLY the paced figure renders no dollar figure —
    # the honest degradation, until the next recompute.
    monthly_only = backfill_legacy_past_overspend_fields(
        {"analyzer": "trim", "estimated_monthly_usd": 12.0}
    )
    assert monthly_only["past_overspend_usd"] is None
    assert not (set(monthly_only) & _RETIRED_DOLLAR_FIELDS)

    # Never overwrites a current entry.
    current = {"past_overspend_usd": 1.0, "past_overspend_basis": "already stamped"}
    assert backfill_legacy_past_overspend_fields(current)["past_overspend_usd"] == 1.0


# --- summarize is a real peer card, not a link-only disclosure ------------- #
#
# #326 tried to soften "summarize's dollars never reach the headline" with an
# `excluded` footnote ("$X more, not summed above -> review it"). That got
# revisited and rejected: the Review inbox is the complete index of
# everything actionable, so
# summarize gets a normal card like every other analyzer — its window figure
# reaches `past_overspend_rollup` exactly the way downsize/cache/trim/etc.'s
# do. Only its AFFORDANCE differs (links to the curate/diff surface instead
# of an inline apply), not its presence in the headline.

def _summarize_finding(**overrides):
    from tokenjam.core.optimize.analyzers.summarize import (
        SummarizeCandidate,
        SummarizeFinding,
    )
    candidates = overrides.pop("candidates", None) or [
        SummarizeCandidate(
            path="/repo/CLAUDE.md", kind="prompt", scope="project",
            est_tokens_saved=4_000, total_chars=32_000, reduction_pct=40,
            sessions_loading=120, est_usd_saved=3_969.74,
            est_tokens_saved_window=6_986_110_266,
        ),
    ]
    fields = dict(
        candidates=candidates, files=len(candidates),
        past_overspend_usd=3_969.74,
        past_overspend_tokens=6_986_110_266,
        estimate_basis="summarize basis", avg_reduction_pct=40,
        sessions_examined=120,
    )
    fields.update(overrides)
    return SummarizeFinding(**fields)


def test_summarize_is_a_cost_analyzer_producing_a_real_peer_card():
    from tokenjam.core.optimize.cost_proposals import COST_ANALYZERS, _summarize_to_proposals

    assert "summarize" in COST_ANALYZERS

    props = _summarize_to_proposals(_summarize_finding())
    assert len(props) == 1
    p = props[0]
    assert p.kind == "cost"
    assert p.analyzer == "summarize"
    assert p.signature == "cost:summarize"
    assert p.past_overspend_usd == 3_969.74
    assert p.past_overspend_tokens == 6_986_110_266
    # Copy names the review flow, not a claimable "Apply" — the exact
    # phrasing behind that decision: "Review N oversized files, $X reads
    # correctly; a bare Apply button would misrepresent the flow".
    assert p.title == "Review 1 oversized file, $3,969.74"
    assert p.advise_only is True
    assert p.apply_capable is False
    assert p.apply_kind == ""
    assert p.delivery == ""


def test_summarize_card_empty_with_no_candidates_or_no_priced_evidence():
    from tokenjam.core.optimize.cost_proposals import _summarize_to_proposals

    assert _summarize_to_proposals(None) == []
    from tokenjam.core.optimize.analyzers.summarize import SummarizeFinding

    assert _summarize_to_proposals(SummarizeFinding()) == []          # dead window
    assert _summarize_to_proposals(_summarize_finding(
        past_overspend_usd=None, past_overspend_tokens=None,
    )) == []   # candidates found, but no session observed loading them


def test_summarize_card_reaches_the_same_headline_every_other_analyzer_does():
    rep = _report()
    rep.findings["summarize"] = _summarize_finding()
    props = cost_proposals_from_report(rep)
    summarize_prop = next(p for p in props if p.analyzer == "summarize")
    assert summarize_prop.past_overspend_usd == pytest.approx(3_969.74)

    block = past_overspend_rollup(props)
    by_analyzer = {a["analyzer"]: a for a in block["by_analyzer"]}
    assert "summarize" in by_analyzer
    assert by_analyzer["summarize"]["usd"] == pytest.approx(3_969.74)
    assert block["past_overspend_usd"] >= 3_969.74


def test_past_overspend_headline_accounts_for_every_cost_analyzer():
    """Regression guard: the headline total must account for every analyzer
    that produces a figure, whether by inclusion or by
    explicit disclosure. Builds a report where (nearly) every COST_ANALYZERS
    member produces a priced finding, and asserts every one of them shows up
    in the past-overspend headline's `by_analyzer` breakdown — the exact
    regression this guards against is a future analyzer silently going
    invisible from the headline the way `summarize` did between #326 and
    #328. `cache-recommend` is the one COST_ANALYZERS member without a ready
    fixture here; it is covered by its own dedicated tests elsewhere
    (test_cache_root_cause_proposals.py) and by the generic, per-analyzer
    contract this test pins for everyone else.
    """
    from tokenjam.core.optimize.analyzers.deadweight import ContextTaxRow, DeadweightFinding, ServerDeadweight
    from tokenjam.core.optimize.analyzers.output_verbosity import VerbosityFinding
    from tokenjam.core.optimize.analyzers.subagent_rightsizing import (
        SubagentRightsizingFinding,
        SubagentRow,
    )
    from tokenjam.core.optimize.analyzers.workflow_restructure import (
        WorkflowCluster,
        WorkflowRestructureFinding,
    )
    from tokenjam.core.optimize.types import ReuseCluster, ReuseFinding

    dg = DowngradeFinding(
        candidate_sessions=4, total_sessions=10, actual_cost_usd=5.0,
        alternative_cost_usd=2.0, monthly_savings_usd=3.0, percent_of_sessions=40.0,
        examples=[], suggestions={"claude-opus-4-8": "claude-sonnet-5"},
        past_overspend_usd=3.0, percent_of_tokens=35.0,
        estimate_basis="downsize basis",
    )
    from tokenjam.core.optimize.analyzers.cache_efficacy import (
        CacheEfficacyFinding,
        CacheEfficacyRow,
    )
    cache = CacheEfficacyFinding(
        flagged=[CacheEfficacyRow("anthropic", "claude-sonnet-5", 100_000, 5_000,
                                  0.05, "full", True)],
        past_overspend_usd=1.2, estimate_basis="cache basis",
    )
    from tokenjam.core.optimize.analyzers.prompt_bloat import BloatPrompt, PromptBloatFinding
    trim = PromptBloatFinding(
        enabled=True,
        per_prompt=[BloatPrompt(agent_id="svc-a", sample_chars="x", prompt_chars=8000,
                                significant_chars=3000, bloat_chars=5000,
                                estimated_token_reduction=1250)],
        past_overspend_usd=0.8, estimate_basis="trim basis",
    )
    subagent = SubagentRightsizingFinding(
        flagged=[SubagentRow(session_id="s1", sub_agent_id="sa0", model="claude-opus-4-8",
                             llm_calls=2, tool_calls=1, input_tokens=60000, output_tokens=500,
                             cache_tokens=0, cache_write_tokens=0, cost_usd=1.2,
                             provider="anthropic", flags=["over_powered"])],
        percent_of_cost=0.66, flagged_cost_usd=1.2, subagent_cost_usd=1.5,
        past_overspend_usd=0.4, past_overspend_tokens=60500,
    )
    dead_server = ServerDeadweight(
        name="apollo", scope="project", source="/repo/.mcp.json",
        sessions_present=10, invocations=0, deferred_sessions=0, dead=True,
        estimated_tax_tokens_per_session=25_000, estimated_tax_tokens_window=225_000,
        tax_construction="25,000 tok/session, cited estimate.",
        fix="Remove or project-scope apollo.", example_sessions=["s0"],
    )
    deadweight = DeadweightFinding(
        sessions_scanned=10, configured_servers=1,
        servers=[dead_server], dead_servers=[dead_server],
        tax_table=[ContextTaxRow(source="MCP schema: apollo", sessions=10,
                                 avg_tokens_per_session=25_000, total_tokens_window=250_000)],
        past_overspend_tokens=225_000,
        estimate_basis="sum of each dead server's schema-injection tax observed over this window",
    )
    script_cluster = WorkflowCluster(
        signature=[{"tool": "bash", "args": ["command_string"]}], instances=25,
        avg_cost_usd=0.02, avg_duration_seconds=1.5, example_session_id="det-0",
        avg_tokens=500, total_cost_usd=0.5, total_tokens=12_500,
        example_session_ids=["det-0", "det-1", "det-2"],
    )
    script = WorkflowRestructureFinding(
        clusters=[script_cluster], sessions_examined=25, degraded=False,
        past_overspend_usd=0.5, past_overspend_tokens=12_500,
        estimate_basis="script basis",
    )
    reuse_cluster = ReuseCluster(
        cluster_id="abc123456789", tool_signature=("bash", "read"),
        prompt_prefix_hash=None, repetitions=4, avg_planning_tokens=300,
        avg_planning_cost_usd=0.01, cache_reuse_recoverable_usd=0.03,
        script_replacement_recoverable_usd=0.04, cache_reuse_recoverable_tokens=900,
        script_replacement_recoverable_tokens=1_200,
        example_session_ids=["s1", "s2", "s3"], skeleton_session_id="s1",
    )
    reuse = ReuseFinding(
        clusters=[reuse_cluster], past_overspend_usd=0.03,
        past_overspend_tokens=900, estimate_basis="reuse basis",
    )
    verbosity = VerbosityFinding(
        total_candidates=6, sessions_examined=40, cohorts_examined=3,
        past_overspend_usd=0.9, past_overspend_tokens=9_000,
        estimate_basis="verbosity basis", suggested_max_tokens=800,
    )
    resend = _resend_finding()
    summarize = _summarize_finding()

    window = WindowSummary(
        since=NOW - timedelta(days=5), until=NOW, days=5, sessions=10, spans=100,
        total_tokens=1, total_cost_usd=5.0, thin_data=False,
    )
    rep = OptimizeReport(
        window=window, downgrade=dg,
        findings={
            "cache": cache, "trim": trim, "subagent": subagent,
            "deadweight": deadweight, "script": script, "reuse": reuse,
            "verbosity": verbosity, "resend": resend, "summarize": summarize,
        },
    )
    props = cost_proposals_from_report(rep)
    block = past_overspend_rollup(props)
    by_analyzer = {a["analyzer"] for a in block["by_analyzer"]}

    priced_analyzers = {
        p.analyzer for p in props if p.past_overspend_usd is not None
    }
    assert priced_analyzers  # sanity: the fixture actually produced priced cards
    assert priced_analyzers <= by_analyzer, (
        f"analyzer(s) produced a priced card but never reached the headline: "
        f"{priced_analyzers - by_analyzer}"
    )
    assert "summarize" in by_analyzer
    assert "downsize" in by_analyzer


# --- 3. the UI reads the payload, it does not recompute -------------------- #
# No JS runner in CI (see CLAUDE.md -> Testing the UI), so the guard is a
# static grep over the single-file SPA, same as every other Lens regression.

@pytest.fixture(scope="module")
def ui() -> str:
    return UI.read_text()


def test_ui_renders_the_observed_figure_at_all(ui):
    # The gap this closes: the backend computed the figure, priced it per
    # token class, wrote an honesty basis for it, shipped it on the payload,
    # and handed it to a dashboard that referenced it zero times.
    assert ui.count("past_overspend_usd") > 0
    assert "PastOverspendTile" in ui


def test_ui_never_derives_a_past_overspend_figure_client_side(ui):
    # Single-compute-path: if the UI needs a number, the endpoint provides it.
    # No pricing, no pacing, no window arithmetic in JS.
    for forbidden in (
        "past_overspend_usd *", "past_overspend_tokens *",
        "* past_overspend", "past_overspend_usd /", "cost_of_waste",
    ):
        assert forbidden not in ui, f"UI derives its own figure: {forbidden}"


def test_the_observed_figure_renders_from_the_server_block_only(ui):
    # The Dashboard hero band was removed and the inbox band became a compact
    # tile, so this figure now has exactly ONE
    # render site. The guarantee that survives is the one that mattered: it is
    # read from the server's `past_overspend` block, never reduced client-side,
    # so what renders cannot drift from what the endpoint computed.
    assert ui.count("<${PastOverspendTile}") == 1
    assert "PastOverspendBand" not in ui, "the removed band must not linger"
    assert "setCostPastOverspend(r.past_overspend || null)" in ui
    # One render site now has exactly one reader. The Dashboard used to keep its
    # own `heroPast` copy of this read for a band it no longer renders, so the
    # page paid for a request per mount and displayed nothing from it; it is gone.
    # This assertion is the same guarantee stated the other way round: nothing may
    # read that block except the surface that renders it.
    assert "setHeroPast" not in ui
    dash = ui[ui.index("function DashboardView"):ui.index("// Two lenses, one router")]
    # The FETCH, not the string: a comment in that view still names the endpoint,
    # deliberately, to say where the figure must come from if it is re-added.
    assert "api('/relearn/cost-proposals')" not in dash
    # If a Dashboard summary of this figure is ever re-added, it must read the
    # server's own `past_overspend` block rather than reduce over rendered cards.
    # That rule now lives in a comment at the old render site, so keep it findable.
    assert "past_overspend` block, the same one" in ui


def test_ui_labels_are_past_tense_and_carry_no_recovery_vocabulary(ui):
    band = ui[ui.index("function PastOverspendTile"):]
    band = band[:band.index("\n}")]
    # The headline is money ALREADY SPENT, and the tile's own label says so in the
    # past tense. The wording is the founder's ("You Overspent", replacing "What
    # you could have avoided"); what this pins is the TENSE, not the phrasing —
    # the tile carries its own tense in a screenshot, so a forward-looking or
    # conditional label here would make the figure read as a projection.
    assert "You Overspent" in band
    for forward in ("you could save", "you will save", "projected", "per month"):
        assert forward not in band.lower(), forward
    assert "recoverable" not in band
    assert "could save" not in ui
    # No ratio framing ("recovering $X of a $Y problem") anywhere.
    assert "recovering $" not in ui

    # INVERTED, and this half used to assert the DEFECT. The tile rendered
    # "... 13 causes of $7,653.24 total cost — that is cost, not waste", and this
    # test required both the "was avoidable" wording and that whole clause to be
    # present. Measured on the live block, the clause was false as rendered:
    # `past_overspend_usd` sums 13 proposals while `observed_cost_usd` covers 2
    # (resend + relearn), so it attached a two-proposal denominator to "13 causes".
    # It is also not the part-of-a-whole relationship `cost_disclosure` claimed:
    # summarize alone contributes ~4,811 of avoidable from a proposal with NO
    # observed cost, so most of the avoidable total lies outside the 7,653 entirely.
    #
    # Rule 30's disclosure is not being dropped, it is being made unnecessary: with
    # no companion total on the tile there is no adjacency to misread. The
    # PER-ROW disclosure is a different quantity, true of a single proposal, and is
    # asserted intact below.
    assert "was avoidable" not in band
    assert "total cost" not in band
    assert "cost, not waste" not in band.lower()
    # Scoped to the RETURNED markup: the comment above the render names the removed
    # field to explain why it went, and a whole-function check matches that
    # explanation instead of the render.
    markup = band[band.index("return html`"):]
    assert "observed_cost_usd" not in markup
    # And no orphaned disclosure describing a figure the tile no longer shows.
    assert "cost_disclosure" not in markup
    # The per-row cost sentence is gone with the field it reported. Asserted as an
    # absence, not re-pointed: the helper that built it no longer exists, and a
    # test looking it up would fail on the lookup rather than on the claim.
    assert "function observedCostSentence" not in ui
    # Matched on the RENDERED string, not the phrase: the comment where the helper
    # used to live quotes the sentence while explaining why it went, and a bare
    # phrase check matches the explanation instead of a render.
    assert "'In total, this behaviour cost '" not in ui


def test_no_ui_surface_reads_a_retired_cost_field(ui):
    # The total-observed-cost fields are deleted from the contract, so the payload
    # never carries them. A branch still reading one would be dead on a fresh
    # recompute and, worse, alive on a stale cache — which is exactly how a
    # deleted figure comes back on screen. Comments naming the fields to explain
    # the removal are fine and are what the `${...}` scoping here allows for.
    for marker in ("observed_cost_usd", "observed_cost_tokens",
                   "observed_cost_basis", "cost_of_waste_usd"):
        for template in ("${prop." + marker, "${item." + marker,
                         "item." + marker + " !=", "prop." + marker + " !="):
            assert template not in ui, f"the UI still reads {marker}"


def test_the_card_states_what_its_figure_does_not_cover(ui):
    # The avoidable figure is computed over a filtered subset, so the card has to
    # say which sessions it covered. This used to be the second half of a pair
    # (the note explained the gap to a total-cost figure beside it); the total is
    # deleted and the note is not, because the filtering it describes still
    # happens and is invisible without it.
    card = ui[ui.index("function CostProposalCard"):]
    card = card[:card.index("\n// The headline band")]
    assert "${prop.coverage_note}" in card
    assert "<summary>What this figure does and does not cover</summary>" in card
    assert "observedCostSentence" not in card


def _all_indices(haystack: str, needle: str):
    start = 0
    while True:
        i = haystack.find(needle, start)
        if i < 0:
            return
        yield i
        start = i + 1


def test_the_basis_is_reachable_from_the_card_not_only_on_hover(ui):
    # "Do NOT read this as a saving" only protects a reader who can reach it —
    # a hover title is unreachable on touch, so the card carries an expandable
    # block as well.
    card = ui[ui.index("function CostProposalCard"):]
    card = card[:card.index("\n// The headline band")] if "\n// The headline band" in card else card
    assert 'title=${prop.past_overspend_basis' in card
    assert "How this number was derived" in card
    assert "${prop.past_overspend_basis}" in card


def test_the_observed_figure_is_visually_separated_from_recoverable_tiles(ui):
    # Not the same colour treatment: every "what you could get back" surface is
    # accent-blue (.rec-amount) or success-green; this one renders in plain body
    # text. Asserted over the rule that actually renders it, `.po-amount`. The
    # companion `.po-observed-tag` rule that used to sit in this range is gone
    # with the chip it styled, so the range now covers `.po-amount` alone — which
    # is the figure, and the figure is what must not be coloured like a claim.
    css = ui[ui.index(".po-amount {"):ui.index(".po-basis {")]
    assert "var(--accent)" not in css
    assert "var(--success)" not in css
    assert "color: var(--text);" in css
