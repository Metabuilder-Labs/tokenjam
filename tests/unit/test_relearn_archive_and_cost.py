"""relearn's horizon, its ungated past-tense cost, and its rollup presence.

Four defects, one module:

1. relearn read Claude Code's on-disk transcripts, which Claude Code rotates
   (``cleanupPeriodDays``), so the one analyzer whose premise is long-horizon
   recurrence could never accumulate history. The archive lane recovers the
   sessions tokenjam retained telemetry for after the transcript was rotated.
2. A cluster with no fix template in our library reported ``$0`` — an
   action-availability gate zeroing an observed cost.
3. A cluster whose rule is uneconomic to keep reported ``$0`` for the same
   wrong reason: "is codifying this worth it?" is not "did this cost anything?"
4. A future fix's standing cost was netted out of a PAST figure.

Plus the rollup: relearn produced ``RelearnCluster``s, never ``CostProposal``s,
so no aggregate surface could see it at all.

All spans/sessions go through ``tests/factories`` (Critical Rule 8); the backend
is in-memory and no real ``~/.tj`` or ``~/.claude`` is touched.

NOTE ON THE FIXTURE FAILURE MESSAGE. These tests used to seed an "has not been
read yet" episode, which clusters as ``edit_before_read``. That family is
ADVISORY-ONLY — its own fix text says the harness already blocks the mistake
and no rule is needed — so it never carries an apply path and cannot
demonstrate any apply-offer behaviour at all. The seed is now a
``read_too_large`` episode, a non-advisory CLAUDE.md-rule family. Every
assertion below is unchanged; only the family standing in for them (Critical
Rule 23 in spirit: the vehicle became invalid, the property being pinned did
not).
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from tokenjam.core.rulewrite.kinds import DELIVERY_CLAUDE_MD_RULE

from tokenjam.core.db import InMemoryBackend
from tokenjam.core.optimize.analyzers.relearn import (
    GROUNDED_TOKENS_PER_OCCURRENCE,
    MIN_RECURRING_SESSIONS,
    FailureEpisode,
    RelearnCluster,
    RelearnFinding,
    _corpus_window_days,
    analyze_relearns,
    build_proposals,
    compute_relearn_finding,
)
from tests.factories import make_llm_span, make_session, make_tool_span

BASE = datetime(2026, 5, 10, tzinfo=timezone.utc)
CODING_AGENT = "claude-code-demo"

#: Long enough that the transcript lane could never have reached it: Claude Code
#: rotates at 30 days by default, so an episode this old exists ONLY in the
#: archive.
BEYOND_TRANSCRIPT_RETENTION_DAYS = 75


@pytest.fixture
def db():
    backend = InMemoryBackend()
    yield backend
    backend.close()


def _episode(
    session_id: str, error: str, *, ts: str | None = None, tool: str = "Bash",
) -> FailureEpisode:
    return FailureEpisode(
        session_id=session_id, repo="demo", ts=ts, tool_name=tool, label="",
        error_text=error, kind="act", is_retry=False, depth=0,
    )


def _priced_session(
    db, session_id: str, *, agent_id: str = CODING_AGENT, at=BASE,
    input_tokens: int = 2_000,
):
    """A session row plus one priced LLM span, so a rate profile can be blended.

    ``input_tokens`` is a knob for sizing the priced figures a test asserts on.
    """
    db.upsert_session(make_session(
        agent_id=agent_id, session_id=session_id, started_at=at, ended_at=at,
    ))
    db.insert_span(make_llm_span(
        agent_id=agent_id, session_id=session_id, model="claude-sonnet-4-5",
        input_tokens=input_tokens, output_tokens=200, start_time=at,
    ))


# --------------------------------------------------------------------------- #
# 1. The horizon is tokenjam's archive, not Claude Code's rotation
# --------------------------------------------------------------------------- #

def test_episode_older_than_transcript_retention_still_counts(db, tmp_path):
    """An episode whose transcript Claude Code already rotated away must still
    reach a cluster. This is the whole premise of the analyzer and it used to be
    structurally impossible: with no ``.jsonl`` on disk the session was invisible
    however long tokenjam had been retaining its telemetry."""
    old = BASE - timedelta(days=BEYOND_TRANSCRIPT_RETENTION_DAYS)
    for i in range(MIN_RECURRING_SESSIONS):
        session_id = f"archived-{i}"
        _priced_session(db, session_id, at=old)
        span = make_tool_span(
            agent_id=CODING_AGENT, tool_name="Bash", status="error",
            session_id=session_id, start_time=old + timedelta(minutes=i),
        )
        span.status_message = "(eval):cd:1: no such file or directory: orchestrator"
        db.insert_span(span)

    # tmp_path is an EMPTY projects root: not one of these sessions has a
    # transcript, exactly as it would be after Claude Code's rotation ran.
    finding = compute_relearn_finding(
        db.conn, projects_root=tmp_path, distill_enabled=False,
    )

    assert finding.transcript_sessions_scanned == 0
    assert finding.archived_sessions_scanned == MIN_RECURRING_SESSIONS
    assert finding.clusters, "an archived-only recurrence produced no cluster"
    assert finding.clusters[0].occurrences == MIN_RECURRING_SESSIONS
    assert "rotated" in finding.corpus_basis


def test_archive_lane_never_double_counts_a_session_with_a_transcript(db, tmp_path):
    """A session that still HAS a transcript belongs to the transcript lane
    alone. The two lanes are disjoint by session id, so a failure is extracted
    once, never twice."""
    from tokenjam.core.optimize.relearn_otel import extract_archived_coding_failures

    _priced_session(db, "still-on-disk")
    span = make_tool_span(
        agent_id=CODING_AGENT, tool_name="Bash", status="error",
        session_id="still-on-disk", start_time=BASE,
    )
    span.status_message = "no such file or directory: orchestrator"
    db.insert_span(span)

    # The caller hands the archive lane only the transcript-LESS ids.
    assert extract_archived_coding_failures(db.conn, set()) == []
    assert extract_archived_coding_failures(db.conn, {"still-on-disk"})


def test_sentinel_timestamp_does_not_stretch_the_observed_window():
    """A ``1970-01-01`` sentinel must not become the corpus MIN. Widening the
    horizon to the DB brings sentinel-stamped rows into range, and one of them
    reaching the span derivation would report a ~56-year window and crush every
    monthly figure to nearly zero."""
    real = [
        _episode("s1", "boom", ts="2026-06-01T00:00:00Z"),
        _episode("s2", "boom", ts="2026-06-11T00:00:00Z"),
    ]
    assert _corpus_window_days(real) == pytest.approx(10.0)

    with_sentinel = [*real, _episode("s3", "boom", ts="1970-01-01T00:00:00Z")]
    assert _corpus_window_days(with_sentinel) == pytest.approx(10.0)


# --------------------------------------------------------------------------- #
# 2 + 3. No gate may zero an observed cost
# --------------------------------------------------------------------------- #

def _no_fix_template_clusters(db):
    """Three sessions sharing an error no known family matches, so the fix falls
    back to the generic "Review examples" placeholder."""
    failures = []
    for i in range(MIN_RECURRING_SESSIONS):
        session_id = f"s{i}"
        _priced_session(db, session_id)
        failures.append(_episode(
            session_id, "Unclassifiable widget explosion in the frobnicator",
            ts=(BASE + timedelta(days=i)).isoformat(),
        ))
    from tokenjam.core.optimize.analyzers.relearn import cluster_failures

    return list(cluster_failures(failures).values())


def test_no_fix_template_cluster_still_carries_its_observed_cost(db):
    """Absence of a fix template is a gap in OUR library, not a property of the
    waste. The cluster claims nothing (correct — there is no fix to claim) and
    still reports what it cost (previously: $0 on every basis)."""
    proposals, _ = build_proposals(
        _no_fix_template_clusters(db), conn=db.conn, window_days=30.0,
        persona="claude-code",
    )

    assert len(proposals) == 1
    p = proposals[0]
    assert "no known fix template" in p.proposed_fix.lower() or "review examples" in p.proposed_fix.lower()

    # No forward claim exists at all any more — the one canonical dollar
    # field is `past_overspend_*`, no carve-out.
    assert not hasattr(p, "estimated_recoverable_tokens")
    assert not hasattr(p, "estimated_monthly_tokens")
    # A fix that is nothing but the generic "no fix template matched" text is
    # still offered like any other apply_capable cluster: there is no
    # budget/quality gate withdrawing the offer any more.
    assert not p.advise_only

    # The OBSERVATION is not zero.
    assert p.past_overspend_tokens >= MIN_RECURRING_SESSIONS * GROUNDED_TOKENS_PER_OCCURRENCE
    assert p.past_overspend_usd is not None and p.past_overspend_usd > 0
    assert p.past_overspend_basis


def test_recurrence_gate_residue_is_counted_not_silently_dropped(db):
    """A one-off failure is not a relearn, so it gets no cluster and no claim —
    but it is not free either, and the finding says so instead of dropping it."""
    _priced_session(db, "solo")
    finding = analyze_relearns(
        [], conn=db.conn, distill_enabled=False,
        extra_failures=[_episode("solo", "a one-off boom", ts=BASE.isoformat())],
    )

    assert finding.clusters == []
    assert finding.below_threshold_clusters == 1
    assert finding.below_threshold_occurrences == 1
    # Priced on the SAME head basis the kept clusters use — the measured cost
    # of the recovery turn, not the error text's size. Asserted as the
    # invariant rather than a literal, because the head is corpus-derived now:
    # pinning it to the constant is what let the residue silently keep the old
    # ~15-20x-low basis when the head term moved.
    from tokenjam.core.optimize.analyzers.relearn import _measured_turn_tokens
    from tokenjam.core.optimize.rate_profile import blended_rate_profile

    expected = _measured_turn_tokens(
        db.conn, {"solo"}, blended_rate_profile(db.conn, session_ids={"solo"}),
    )
    assert expected is not None and expected >= GROUNDED_TOKENS_PER_OCCURRENCE
    assert finding.below_threshold_past_overspend_tokens == expected


# --------------------------------------------------------------------------- #
# 4. relearn reaches the rollup
# --------------------------------------------------------------------------- #

class _Window:
    days = 30.0
    active_days = 30
    sessions = 100


class _Report:
    persona = "claude-code"
    downgrade = None
    window = _Window()

    def __init__(self, finding):
        self.findings = {"relearn": finding}


def test_relearn_produces_no_cost_proposal_and_keeps_its_claim_on_its_clusters(db):
    """relearn contributes NO ``CostProposal``, and that is the whole design.

    It used to emit exactly one aggregate card whose only figure was
    ``observed_cost_usd`` — the retired total-observed-cost field —
    with ``past_overspend_usd`` deliberately ``None``, because relearn's re-read
    tail is the same re-sent context ``resend`` already claims in full and two
    analyzers claiming one span is CLAUDE.md rule 27. With that field deleted the
    card had no number left to lead with, so the card is deleted too.

    ``relearn`` stays in ``COST_ANALYZERS``: membership is what keeps the analyzer
    running inside the inbox recompute, which relearn's own per-cluster rows
    depend on. What must not come back is a proposal.
    """
    from tokenjam.core.optimize.cost_proposals import (
        COST_ANALYZERS,
        cost_proposals_from_report,
        past_overspend_rollup,
    )

    assert "relearn" in COST_ANALYZERS

    failures = []
    for i in range(MIN_RECURRING_SESSIONS):
        session_id = f"r{i}"
        _priced_session(db, session_id)
        failures.append(_episode(
            session_id,
            "File content (52000 tokens) exceeds maximum allowed tokens (25000).",
            ts=(BASE + timedelta(days=i)).isoformat(), tool="Read",
        ))
    finding = analyze_relearns(
        [], conn=db.conn, distill_enabled=False, extra_failures=failures,
        persona="claude-code",
    )
    # The MEASUREMENT is untouched: relearn still prices what the recurrences
    # already cost. Only the card that republished it as a second aggregate total
    # is gone.
    assert finding.clusters and finding.past_overspend_usd

    proposals = cost_proposals_from_report(_Report(finding))
    assert [p for p in proposals if p.analyzer == "relearn"] == []

    rollup = past_overspend_rollup(proposals)
    assert "relearn" not in {a["analyzer"] for a in rollup["by_analyzer"]}
    # No retired key survives on the block under any name.
    assert not [k for k in rollup if "observed_cost" in k or k == "cost_disclosure"]

    # And the claim is not lost: every cluster still carries its own figure, on
    # the canonical field, which is what the Review inbox rows render.
    assert all(c.past_overspend_usd is not None or c.past_overspend_tokens
               for c in finding.clusters)


def test_advise_snippet_offered_separates_a_real_fix_from_an_offered_write():
    """The Review inbox routes every row onto a two-valued mechanism axis: tj
    applies it, or tj hands over the exact change. A cluster whose write is
    offered (``not advise_only``) must not ALSO advertise "copy this" and
    print its own fix twice; an advise-only cluster with a real fix text must.
    """
    from tokenjam.core.optimize.relearn_proposals import (
        advise_snippet_offered,
        stamp_proposal_ids,
    )

    # A write WAS offered (`advise_only` False/absent), so there is nothing to
    # hand over: a cluster's `proposed_fix` is the same content the write
    # would write.
    assert advise_snippet_offered({
        "proposed_fix": "PreToolUse hook: block a busy-wait sleep chain.",
        "advise_only": False,
    }) is False
    # Advise-only (no workspace, an advisory family, or a persona with no
    # write surface): the recommendation is still real and the user can apply
    # it themselves.
    assert advise_snippet_offered({
        "proposed_fix": "Use absolute paths in parallel Bash calls.",
        "advise_only": True,
    }) is True
    # No fix text at all is never a snippet, whatever `advise_only` says.
    assert advise_snippet_offered({"proposed_fix": "", "advise_only": True}) is False
    assert advise_snippet_offered({"advise_only": False}) is False

    # Stamped on read, like proposal_id / advise_only_reason, so a cache
    # written before the field existed classifies correctly on the first read
    # with no recompute — and without mutating the caller's dict.
    legacy = {"clusters": [
        {"signature": "a", "proposed_fix": "do the thing", "advise_only": True},
        {"signature": "b", "proposed_fix": "already offered", "advise_only": False},
    ]}
    out = stamp_proposal_ids(legacy)["clusters"]
    assert out[0]["advise_snippet_offered"] is True
    assert out[1]["advise_snippet_offered"] is False
    assert "advise_snippet_offered" not in legacy["clusters"][0]


def test_relearn_finding_round_trips_its_observed_dollar_figure():
    """`report_to_dict` / `report_from_dict` must preserve relearn's observed
    figure. relearn's reader was the one in `runner.py` restoring the token
    field but not its USD twin, so a rehydrated report kept ~16.7M tokens and
    silently lost the $46 that went with them -- and the dollar field is what
    every leading surface renders.
    """
    from tokenjam.core.optimize.runner import report_from_dict, report_to_dict

    finding = RelearnFinding(
        clusters=[], sessions_scanned=3,
        past_overspend_tokens=1_234_567,
        past_overspend_usd=46.62,
        past_overspend_basis="observed over the scanned corpus",
    )
    # Round-trip through the report envelope the cache actually stores.
    from tokenjam.core.optimize.runner import OptimizeReport, WindowSummary

    window = WindowSummary(
        since=BASE - timedelta(days=30), until=BASE, days=30, sessions=3,
        spans=10, total_tokens=1, total_cost_usd=1.0, thin_data=False,
        active_days=28,
    )
    report = OptimizeReport(window=window, downgrade=None)
    report.findings["relearn"] = finding
    back = report_from_dict(report_to_dict(report)).findings["relearn"]

    assert back.past_overspend_tokens == 1_234_567
    assert back.past_overspend_usd == 46.62
    assert back.past_overspend_basis == "observed over the scanned corpus"
    # A payload written before the fields existed rehydrates to the declared
    # defaults, never to None on a non-optional int.
    legacy = report_from_dict({"findings": {"relearn": {"sessions_scanned": 1}}})
    assert legacy.findings["relearn"].past_overspend_tokens == 0


# --------------------------------------------------------------------------- #
# The head term is MEASURED, and the claim is disjoint from `resend`.
#
# relearn priced a pothole at a flat 1,500 tokens -- an unmeasured guess for
# "one extra assistant turn's overhead". Measured on a real corpus, one turn
# costs ~140k tokens (836 fresh + 138k cache-read + 795 out), because a coding
# turn re-sends the whole context. The constant was ~15-20x low, which is why
# relearn read as noise: 70% of the inbox cards carrying 0.8% of the money.
#
# The fix splits one conflated constant into the two quantities it was standing
# in for: the forced RETRY TURN (measured, and relearn's own claim) and the
# error TEXT re-read on later calls (~1,500 tokens really is right for a block
# of error text, and it is `resend`'s money, so it is observed but not claimed).
# --------------------------------------------------------------------------- #


def test_head_term_is_measured_from_the_corpus_not_a_constant(db):
    """The recovery turn is priced from what a call actually cost in the
    contributing sessions, so a corpus of expensive calls yields a bigger
    figure than one of cheap calls off the identical failure count."""
    from tokenjam.core.optimize.analyzers.relearn import (
        _measured_turn_tokens,
        cluster_failures,
    )
    from tokenjam.core.optimize.rate_profile import blended_rate_profile

    failures = []
    for i in range(MIN_RECURRING_SESSIONS):
        session_id = f"m{i}"
        _priced_session(db, session_id)
        failures.append(_episode(
            session_id,
            "File content (52000 tokens) exceeds maximum allowed tokens (25000).",
            ts=(BASE + timedelta(days=i)).isoformat(), tool="Read",
        ))
    sessions = {f.session_id for f in failures}
    profile = blended_rate_profile(db.conn, session_ids=sessions)
    measured = _measured_turn_tokens(db.conn, sessions, profile)

    assert measured is not None, "a priced corpus must yield a measured turn"
    # Never BELOW the error-text constant: a forced retry carries the error
    # text, so it cannot cost less than the text alone.
    assert measured >= GROUNDED_TOKENS_PER_OCCURRENCE
    # And it must actually be derived, not the constant echoed back: these
    # sessions carry 2,000-token calls, well above the 1,500 floor.
    assert measured > GROUNDED_TOKENS_PER_OCCURRENCE

    clusters = list(cluster_failures(failures).values())
    proposals, _ = build_proposals(
        clusters, conn=db.conn, window_days=30.0, persona="claude-code",
        repo_cwd_map={"demo": "/tmp/demo"},
    )
    p = proposals[0]
    # The head alone already exceeds what the OLD flat model charged for head
    # plus tail combined.
    assert p.past_overspend_tokens > p.occurrences * GROUNDED_TOKENS_PER_OCCURRENCE


def test_relearn_claims_the_retry_turn_and_never_the_reread_tail(db):
    """The disjointness rule between the two analyzers, asserted.

    `resend` prices redundant context inside calls that had to happen.
    `relearn` prices a call that should not have happened at all. The error
    text's re-read tail belongs to the first, so it appears in relearn's
    OBSERVED figure and never in its CLAIM -- otherwise the same tokens are
    priced on two cards (CLAUDE.md rule 27).
    """
    from tokenjam.core.optimize.analyzers.relearn import cluster_failures

    # 5 sessions of ~100k-token turns hitting the same pothole 4 times each,
    # which is an ordinary shape on a real coding corpus.
    failures = []
    for i in range(MIN_RECURRING_SESSIONS + 2):
        session_id = f"d{i}"
        _priced_session(db, session_id, input_tokens=100_000)
        for occurrence in range(4):
            failures.append(_episode(
                session_id,
                "File content (52000 tokens) exceeds maximum allowed tokens (25000).",
                ts=(BASE + timedelta(days=i, minutes=occurrence)).isoformat(),
                tool="Read",
            ))
    clusters = list(cluster_failures(failures).values())
    proposals, _ = build_proposals(
        clusters, conn=db.conn, window_days=30.0, persona="claude-code",
        repo_cwd_map={"demo": "/tmp/demo"},
    )
    p = proposals[0]
    assert not p.advise_only, "this family has a real fix; expected an apply offer"

    # No forward claim field exists any more — a relearn cluster shows its
    # PAST figure only. The disjointness rule this test used to assert on
    # `estimated_recoverable_tokens` now lives entirely inside the observed
    # figure: the re-read tail is a COMPONENT of `past_overspend_tokens` (not
    # an addend), broken out separately as `past_reread_tokens`, and never
    # exceeds the total it is a component of.
    assert not hasattr(p, "estimated_recoverable_tokens")
    assert p.past_reread_tokens <= p.past_overspend_tokens
    # The one basis string says the overlap is disclosed, never claimed twice.
    from tokenjam.core.optimize.analyzers.relearn import PAST_OVERSPEND_BASIS

    assert "must never be added together" in PAST_OVERSPEND_BASIS


def test_apply_offered_regardless_of_write_size_or_session_scale(db):
    """Pins the new contract: an apply_capable relearn cluster is offered no
    matter how large its write would have been, or how many sessions would
    have re-sent it. There is no standing-cost ceiling, growth-share cap, or
    write-count limit left to withhold the offer -- those were the write
    budget's job, and the write budget is gone.
    """
    from tokenjam.core.optimize.analyzers.relearn import cluster_failures

    # A pathologically large recurrence: far more sessions/occurrences than any
    # of the old AGENT_FILE_STANDING_CEILING_TOKENS / MAX_OFFERED_WRITES /
    # WRITE_BUDGET_TOKENS constants would have tolerated for one cluster.
    failures = []
    for i in range(200):
        session_id = f"huge{i}"
        _priced_session(db, session_id, input_tokens=500_000)
        failures.append(_episode(
            session_id,
            "File content (52000 tokens) exceeds maximum allowed tokens (25000).",
            ts=(BASE + timedelta(days=i % 30)).isoformat(), tool="Read",
        ))
    clusters = list(cluster_failures(failures).values())
    proposals, _ = build_proposals(
        clusters, conn=db.conn, window_days=30.0, persona="claude-code",
        repo_cwd_map={"demo": "/tmp/demo"},
    )
    p = proposals[0]
    # This cluster's implied standing cost (were it still computed) and its
    # exposure would both have been enormous -- and it is still offered.
    assert p.sessions >= 200
    assert not p.advise_only
    assert p.suggested_target


def test_dry_run_by_default_still_holds_for_relearn_apply(db, tmp_path):
    """The human write guard is untouched by this purge: relearn's apply path
    writes nothing to disk unless the caller explicitly passes ``go=True``."""
    from tokenjam.core.optimize import relearn_apply
    from tokenjam.core.config import TjConfig

    target = tmp_path / "CLAUDE.md"
    cluster = {
        "signature": "sig:dry-run-check", "title": "dry run check",
        "proposed_fix": "Use absolute paths in parallel Bash calls.",
        "delivery": DELIVERY_CLAUDE_MD_RULE, "sessions": 3, "repos": ["demo"],
    }
    config = TjConfig(version="1")
    relearn_apply.apply_relearn_fix(
        config, cluster, target_path=str(target), scope="project", go=False,
    )
    assert not target.exists(), "a dry run (go=False) must not write anything"
