"""Unit tests for the recoverable-total presentation contract in
`api/routes/cost.py` (A1, analyzer-audit #482).

`/cost/components` used to expose a flat `sum()` of every analyzer's
`past_overspend_usd` with nothing signaling that the analyzers price
waste from overlapping angles over the same sessions. These tests pin the
fix: individual analyzer estimates are NEVER touched (the operator does not
want the headline magnitude deflated), but the response now says, in the
data itself, that the sum is a ceiling rather than an achievable total, and
carries a standalone "largest single line" figure that IS honest on its own.
"""
from __future__ import annotations

from dataclasses import dataclass

from datetime import datetime, timedelta, timezone

from tokenjam.api.routes.cost import (
    _collect_recoverable,
    _recoverable_basis_note,
    _recoverable_overlap_note,
    _recoverable_window_bounds,
)


@dataclass
class _FakeFinding:
    past_overspend_usd: float | None
    past_overspend_tokens: int | None
    estimate_basis: str = ""
    caveat: str = ""


@dataclass
class _FakeReport:
    downgrade: object | None
    findings: dict


def test_collect_recoverable_sorted_biggest_first():
    """Order must be deterministic and USD-descending so 'largest opportunity
    + N more' can render directly off list order, with no client-side sort."""
    report = _FakeReport(
        downgrade=None,
        findings={
            "cache": _FakeFinding(past_overspend_usd=5.0, past_overspend_tokens=100),
            "trim": _FakeFinding(past_overspend_usd=50.0, past_overspend_tokens=200),
            "reuse": _FakeFinding(past_overspend_usd=20.0, past_overspend_tokens=50),
        },
    )
    out = _collect_recoverable(report)
    assert [r["analyzer"] for r in out] == ["trim", "reuse", "cache"]
    assert [r["past_overspend_usd"] for r in out] == [50.0, 20.0, 5.0]


def test_collect_recoverable_ties_broken_by_tokens():
    report = _FakeReport(
        downgrade=None,
        findings={
            "cache": _FakeFinding(past_overspend_usd=10.0, past_overspend_tokens=100),
            "trim": _FakeFinding(past_overspend_usd=10.0, past_overspend_tokens=500),
        },
    )
    out = _collect_recoverable(report)
    assert [r["analyzer"] for r in out] == ["trim", "cache"]


def test_overlap_note_empty_for_zero_or_one_finding():
    # Zero findings: nothing to disclose, no false "these overlap" claim.
    assert _recoverable_overlap_note([]) == ""
    # A single finding cannot double-count anything by construction.
    assert _recoverable_overlap_note([{"past_overspend_usd": 10.0}]) == ""


def test_overlap_note_present_for_two_or_more_findings():
    note = _recoverable_overlap_note([
        {"past_overspend_usd": 10.0},
        {"past_overspend_usd": 5.0},
    ])
    assert note != ""
    assert "2" in note  # names how many estimates it's disclaiming
    # House style: never claim the analyzers' figures were changed here.
    assert "reduce" not in note.lower()
    # No em dashes in user-facing copy (house rule).
    assert "—" not in note


def test_overlap_note_scales_the_count_with_the_list():
    note = _recoverable_overlap_note([
        {"past_overspend_usd": 1.0},
        {"past_overspend_usd": 1.0},
        {"past_overspend_usd": 1.0},
    ])
    assert "3" in note


# --------------------------------------------------------------------------- #
# The spend bar's denominator: which window the ceiling was measured over.
#
# `total_recoverable_usd` is read out of the stored analyzer report, so it
# answers the SCAN's window, not the caller's. Drawing it as a shaded share of
# `total_cost_usd` (which does follow `since`) published two windows as one
# ratio. `_recoverable_window_bounds` is what lets the route compute a
# denominator over the ceiling's own window instead. The end-to-end coupling is
# pinned in tests/integration/test_cost_components_recoverable_population.py;
# these cover the bounds resolution itself, including the cases where it must
# REFUSE to answer.
# --------------------------------------------------------------------------- #
_NOW = datetime(2026, 8, 6, 21, 8, 53, tzinfo=timezone.utc)


def test_bounds_prefer_the_cycle_record_s_own_resolved_window():
    since, until = _recoverable_window_bounds({
        "scan_since": (_NOW - timedelta(days=30)).isoformat(),
        "scan_until": _NOW.isoformat(),
        "computed_at": (_NOW + timedelta(hours=4)).isoformat(),
        "window_days": 30,
    })
    # `computed_at` is when the pass FINISHED, which is later than the window it
    # observed. The sealed bounds win so the denominator is not silently shifted
    # by however long the scan took.
    assert until == _NOW
    assert since == _NOW - timedelta(days=30)


def test_bounds_fall_back_to_computed_at_minus_window_days():
    """A pre-record artifact still carries `computed_at` and `window_days`, and
    a denominator derived from those is far better than none."""
    since, until = _recoverable_window_bounds({
        "scan_since": None, "scan_until": None,
        "computed_at": _NOW.isoformat(), "window_days": 7,
    })
    assert until == _NOW
    assert since == _NOW - timedelta(days=7)


def test_bounds_are_never_invented():
    """THE BOUNDS ARE NOT GUESSED. A denominator over a window nobody recorded
    is the defect this helper exists to prevent, wearing a different hat: it
    would look exactly as authoritative as a correct one. `(None, None)` makes
    the route publish no share at all, which the UI renders as unknown."""
    assert _recoverable_window_bounds({}) == (None, None)
    # An anchor with no length, and a length with no anchor: neither is enough.
    assert _recoverable_window_bounds({"computed_at": _NOW.isoformat()}) == (None, None)
    assert _recoverable_window_bounds({"window_days": 30}) == (None, None)
    # A window_days of zero is not a window.
    assert _recoverable_window_bounds(
        {"computed_at": _NOW.isoformat(), "window_days": 0},
    ) == (None, None)
    # Unparseable values are treated as absent, never as "now".
    assert _recoverable_window_bounds(
        {"scan_since": "not-a-date", "scan_until": "not-a-date"},
    ) == (None, None)


def test_bounds_accept_epochs_and_naive_datetimes_as_utc():
    since, until = _recoverable_window_bounds({
        "scan_since": (_NOW - timedelta(days=30)).timestamp(),
        "scan_until": _NOW.replace(tzinfo=None),
    })
    assert until == _NOW
    assert since == _NOW - timedelta(days=30)


def test_basis_note_states_the_window_and_that_the_picker_does_not_move_it():
    note = _recoverable_basis_note(_NOW - timedelta(days=30), _NOW, 30)
    # The window, as a length and as dates a reader can check.
    assert "30 days" in note
    assert "2026-07-07" in note and "2026-08-06" in note
    # That the range picker above does not move this bar. Still true, still the
    # sentence this note was originally written for.
    assert "does not follow the range selected above" in note
    # House rule: no em dashes in user-facing copy.
    assert "—" not in note


# --------------------------------------------------------------------------- #
# The persona sentence, INVERTED
# --------------------------------------------------------------------------- #
# This assertion used to read `assert "not which traffic is measured" in note`,
# and it was defending the defect rather than guarding against it. The note
# claimed the picker selects a lever set only and that both figures cover every
# agent. Both stopped being true when the daemon began storing a separately
# scoped pass per persona: on a real corpus the same bar's denominator reads
# 13733.55 under claude-code and 1.37 under sdk. The test kept passing because
# it pinned the string, not the claim. So it is inverted here rather than
# deleted: the old wording is pinned ABSENT, and the note is checked against the
# behaviour it describes.
def test_the_note_no_longer_claims_the_picker_leaves_the_traffic_alone():
    for persona in (None, "claude-code", "sdk"):
        note = _recoverable_basis_note(_NOW - timedelta(days=30), _NOW, 30, persona, "claude-code")
        assert "not which traffic is measured" not in note, (
            f"persona={persona!r}: the picker scopes the spend too"
        )
        assert "changes which analyzers contribute to the ceiling, not" not in note


def test_a_selected_persona_is_named_and_each_figure_gets_its_own_claim():
    for persona, label in (("claude-code", "Claude Code"), ("sdk", "SDK workflow")):
        note = _recoverable_basis_note(_NOW - timedelta(days=30), _NOW, 30, persona)
        assert f"The spend covers {label} traffic" in note
        # It must NOT claim to cover every agent while a persona narrows the spend.
        assert "every agent" not in note, f"{persona} note still claims corpus scope"
        # Nor may it claim the ceiling is scoped to that persona; see
        # test_no_sentence_describes_both_figures_at_once for why.
        assert "not every basis is scoped to this persona" in note
        assert "—" not in note


def test_no_sentence_describes_both_figures_at_once():
    """THE INVARIANT. The two figures do not share a population.

    The spend is exactly persona-scoped; the ceiling is a mix, because
    `relearn` and `deadweight` measure on an unbounded corpus scan and return
    the same figure scoped or not. So "Both figures cover every agent" is false
    under a selected persona, and "both cover this persona's traffic" is false
    for that corpus-wide share. Any single coverage claim about the pair is
    wrong in one direction or the other, which is why each figure states its
    own. Pinned as a shape, not as a phrase, so the next rewrite cannot
    reintroduce it under different words.
    """
    for persona in (None, "claude-code", "sdk"):
        for lever in (None, "claude-code", "sdk", "mixed"):
            note = _recoverable_basis_note(
                _NOW - timedelta(days=30), _NOW, 30, persona, lever,
            )
            assert "Both figures" not in note, (
                f"persona={persona!r} lever={lever!r}: one claim about both "
                "figures cannot be true, they do not share a population"
            )
            # The spend always gets its own sentence, and it is the only half
            # that may carry a flat coverage claim.
            assert note.startswith("The spend covers "), note[:60]


def test_the_unscoped_note_does_not_imply_a_clean_corpus_total():
    """No persona selected is a HYBRID, and the note has to say so.

    `_collect_recoverable` falls back to the corpus's own dominant persona, so
    the spend covers every agent while the ceiling counts only the analyzers
    that persona can act on. Describing that as one corpus-wide answer is the
    same over-claim the old wording made, one layer down.
    """
    note = _recoverable_basis_note(_NOW - timedelta(days=30), _NOW, 30, None, "claude-code")
    assert "The spend covers every agent" in note
    assert "The ceiling does not" in note
    assert "Claude Code" in note
    assert "—" not in note


def test_an_unclassified_corpus_gets_the_plain_note_not_a_false_lever_claim():
    """`mixed` / `unknown` disable nothing, so every analyzer really does count."""
    for lever in (None, "", "mixed", "unknown"):
        note = _recoverable_basis_note(_NOW - timedelta(days=30), _NOW, 30, None, lever)
        assert "Every analyzer contributes" in note
        assert "The ceiling does not" not in note


def test_the_ceiling_really_does_mix_scoped_and_corpus_wide_components():
    """The measurement the note's wording is built on, encoded as a fixture.

    Measured per-analyzer on the operator's corpus over one 30-day window:
    `relearn` and `deadweight` come back byte-identical from the claude-code
    pass and the unscoped pass, because their basis is an unbounded corpus scan;
    `resend`, `subagent` and `summarize` move with the persona; `downsize` is
    exactly additive across the two. So the ceiling is neither wholly scoped
    nor wholly corpus-wide, and no single coverage sentence can cover it.

    This drives `_collect_recoverable` over two reports shaped like those two
    passes and asserts the mix survives, so if the corpus-wide analyzers ever
    become window-scoped the note's hedge can be tightened deliberately rather
    than by accident.
    """
    corpus_wide = {
        "relearn": _FakeFinding(past_overspend_usd=250.735256, past_overspend_tokens=107529906),
        "deadweight": _FakeFinding(past_overspend_usd=2.76376, past_overspend_tokens=552752),
    }
    scoped = _FakeReport(downgrade=None, findings={
        **{k: _FakeFinding(v.past_overspend_usd, v.past_overspend_tokens) for k, v in corpus_wide.items()},
        "resend": _FakeFinding(past_overspend_usd=429.120533, past_overspend_tokens=828619941),
    })
    unscoped = _FakeReport(downgrade=None, findings={
        **{k: _FakeFinding(v.past_overspend_usd, v.past_overspend_tokens) for k, v in corpus_wide.items()},
        "resend": _FakeFinding(past_overspend_usd=430.123459, past_overspend_tokens=830539733),
    })

    by_name = lambda rep, p: {  # noqa: E731
        r["analyzer"]: r["past_overspend_usd"] for r in _collect_recoverable(rep, persona=p)
    }
    a, b = by_name(scoped, "claude-code"), by_name(unscoped, None)

    # The corpus-wide half does not move with the persona. This is the fact that
    # makes "both figures cover this persona's traffic" false.
    assert a["relearn"] == b["relearn"]
    assert a["deadweight"] == b["deadweight"]
    # The scoped half does. This is the fact that makes "both figures cover
    # every agent" false.
    assert a["resend"] != b["resend"]

    # And the note hedges exactly that, rather than claiming either extreme.
    note = _recoverable_basis_note(_NOW - timedelta(days=30), _NOW, 30, "claude-code")
    assert "not every basis is scoped to this persona" in note


def test_the_note_never_claims_the_ceilings_population_is_persona_scoped():
    """The analyzer SET is scoped; a contributor's own basis need not be.

    `relearn` scans unbounded on-disk history by design and lands the same
    figure in a scoped total as in an unscoped one, so a note asserting the
    ceiling covers only this persona's traffic would be quietly false for it.
    """
    for persona in ("claude-code", "sdk", None):
        note = _recoverable_basis_note(_NOW - timedelta(days=30), _NOW, 30, persona, "claude-code")
        assert "each measured on its own basis" in note, (
            "the ceiling's contributors are not all bounded by this window, and "
            "the note has to leave room for that"
        )
        assert "ceiling is measured over" not in note


def test_basis_note_admits_an_unknown_window_rather_than_implying_one():
    note = _recoverable_basis_note(None, None, None)
    assert "did not record the window" in note
    assert "cannot be shown as a share of spend" in note
    # It must not name a window it does not have.
    assert "days," not in note
    assert "—" not in note


# --------------------------------------------------------------------------- #
# The note checked against the BEHAVIOUR, not against itself
# --------------------------------------------------------------------------- #
# A string assertion is what let the old wording survive the change that
# falsified it. So this drives the real denominator query over a two-persona
# corpus and asserts the note's claim and the query agree. Same shape as the
# PERSONA_HIDDEN_VIEWS mirror: pin the declaration against the enforcement.
def test_the_picker_really_moves_the_denominator_the_note_describes(tmp_path):
    import pytest

    from tests.factories import make_llm_span, make_session
    from tokenjam.api.routes.cost import _component_costs
    from tokenjam.core.config import StorageConfig
    from tokenjam.core.db import DuckDBBackend

    now = datetime.now(timezone.utc)
    db = DuckDBBackend(StorageConfig(path=str(tmp_path / "note.duckdb")))
    try:
        # `claude-code` is the coding-agent prefix the persona scope buckets on;
        # anything else falls in the sdk complement.
        for agent, n, cost in (("claude-code", 6, 0.50), ("svc-pipeline", 4, 0.25)):
            for i in range(n):
                sid = f"{agent}-{i}"
                db.upsert_session(make_session(
                    agent_id=agent, session_id=sid, total_cost_usd=cost,
                    started_at=now - timedelta(days=1),
                ))
                db.insert_span(make_llm_span(
                    agent_id=agent, session_id=sid, cost_usd=cost,
                    start_time=now - timedelta(days=1),
                ))

        since, until = now - timedelta(days=7), now
        total = lambda p: sum(  # noqa: E731
            v["cost_usd"] for v in _component_costs(db.conn, None, since, until, p).values()
        )
        whole, cc, sdk = total(None), total("claude-code"), total("sdk")
    finally:
        db.close()

    # THE CLAIM: the picker scopes the spend, not just the analyzer set. If this
    # ever stops holding, the note goes back to being a lie and this fails.
    assert cc != whole, "the persona picker must actually narrow the denominator"
    assert sdk != whole
    assert cc > 0 and sdk > 0
    # And the two personas partition the corpus exactly, which is what lets the
    # note say each figure covers "that persona's own traffic" without a gap.
    assert cc + sdk == pytest.approx(whole)

    # The note for a scoped read names that persona and drops the corpus claim.
    scoped = _recoverable_basis_note(since, until, 7, "claude-code")
    assert "Claude Code traffic" in scoped and "every agent" not in scoped
    # The unscoped read keeps the corpus claim for the SPEND only.
    unscoped = _recoverable_basis_note(since, until, 7, None, "claude-code")
    assert "The spend covers every agent" in unscoped
