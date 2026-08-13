"""The Review inbox headline covers EVERY row in the inbox, exactly once.

The inbox is one list fed by two producers (cost proposals and relearn
clusters), and its headline was summed over the cost feed alone. Everything
derived from the list was therefore false in a way no reader could see: the
collapsed tail summed rows of both kinds, and the below-floor note claimed the
hidden items were "still counted in the total above" when most of that money had
never entered the total.

WHAT THESE TESTS PIN, AND WHY IT IS NOT EITHER RETIRED MECHANISM. Two earlier
attempts to put relearn's dollars into a shared aggregate were retired and are
still guarded (``test_cost_proposals.
test_the_rollup_has_no_per_analyzer_side_channel`` and ``test_relearn.
test_retired_forward_fields_stay_gone``). Neither guard is weakened here:

  * the first removed a relearn-only PARAMETER that computed relearn's figure
    inside ``past_overspend_rollup`` on a second time basis and published it in
    its own key. Nothing here adds a parameter to that function or a key to its
    result. A relearn cluster arrives as an ordinary row on the one canonical
    field and earns an ordinary ``by_analyzer`` entry;
  * the second removed a RESCALE. The contribution is the detector's own
    bounded bucket: a date filter over the same occurrences at the same price,
    capped at the unbounded figure, minus a component of itself. Every test
    below that touches direction pins it downward.

Read ``core/optimize/inbox_contribution.py``'s module docstring first: it owns
the design (why net of re-read, why an exactly matching window or nothing, why
the unbounded fields are untouchable).
"""
from __future__ import annotations

import pytest

from tokenjam.core.optimize.cost_proposals import past_overspend_rollup
from tokenjam.core.optimize.inbox_contribution import (
    contribution_window_label,
    exact_window_label,
    relearn_contribution,
    relearn_contribution_rows,
    relearn_excluded_entry,
    stamp_cost_contribution,
    stamp_relearn_contributions,
    unrepresented_relearn,
)

WINDOW = "30d"


def _bucket(
    label: str = WINDOW, *, usd: float | None, reread_usd: float | None,
    tokens: int = 0, reread_tokens: int = 0, occurrences: int = 3,
) -> dict[str, object]:
    return {
        "label": label, "window_days": 30.0,
        "window_start": "2026-06-27T12:00:00+00:00",
        "window_end": "2026-07-27T12:00:00+00:00",
        "occurrences": occurrences, "sessions": 2, "detour_turns": 4.0,
        "undated_occurrences": 0, "tail_calls_median": 3, "tail_multiplier": 1.4,
        "past_overspend_tokens": tokens, "past_overspend_usd": usd,
        "past_reread_tokens": reread_tokens, "past_reread_usd": reread_usd,
        "capped_at_unbounded": False, "basis": "windowed",
    }


def _cluster(
    signature: str, *, unbounded_usd: float | None = 40.0,
    unbounded_tokens: int = 400_000, windows: dict[str, object] | None = None,
) -> dict[str, object]:
    return {
        "signature": signature, "title": f"title {signature}",
        "past_overspend_usd": unbounded_usd,
        "past_overspend_tokens": unbounded_tokens,
        "past_reread_usd": 3.0, "past_reread_tokens": 30_000,
        "past_overspend_windows": windows,
    }


def _finding(clusters: list[dict[str, object]], *, labels=(WINDOW,)) -> dict[str, object]:
    return {
        "clusters": clusters,
        "past_overspend_usd": sum(
            c["past_overspend_usd"] or 0.0 for c in clusters),  # type: ignore[misc]
        "past_overspend_windows": {label: {"label": label} for label in labels},
    }


# --- the invariant: headline == sum of every row's contribution ------------- #

def test_the_headline_equals_the_sum_of_every_inbox_rows_contribution():
    """The one thing the founder asked for. Every row in the list, including the
    ones the noise floor hides and the ones folded into the collapsed tail,
    contributes to the headline, and the headline is nothing but their sum."""
    cost = [
        {"signature": "cost:summarize", "analyzer": "summarize", "title": "big",
         "past_overspend_usd": 100.0, "past_overspend_tokens": 1_000},
        # Below the inbox's own display floor, and still in the total.
        {"signature": "cost:downsize:tiny", "analyzer": "downsize", "title": "tiny",
         "past_overspend_usd": 0.75, "past_overspend_tokens": 10},
    ]
    clusters = [
        _cluster("relearn:a", windows={WINDOW: _bucket(usd=30.0, reread_usd=5.0,
                                                       tokens=300, reread_tokens=50)}),
        # Also below the floor, also still in the total.
        _cluster("relearn:b", windows={WINDOW: _bucket(usd=2.0, reread_usd=0.5,
                                                      tokens=20, reread_tokens=5)}),
    ]
    finding = _finding(clusters)

    rows = [stamp_cost_contribution(p, window=WINDOW) for p in cost]
    stamped = stamp_relearn_contributions(finding, label=WINDOW)
    rollup = past_overspend_rollup(
        cost + relearn_contribution_rows(finding, label=WINDOW), window_days=30,
    )

    every_row = rows + list(stamped["clusters"])          # type: ignore[index]
    assert len(every_row) == 4
    assert rollup["past_overspend_usd"] == pytest.approx(
        sum(r["inbox_contribution_usd"] for r in every_row))
    assert rollup["past_overspend_tokens"] == sum(
        r["inbox_contribution_tokens"] for r in every_row)
    # 100 + 0.75 + (30 - 5) + (2 - 0.5)
    assert rollup["past_overspend_usd"] == pytest.approx(127.25)
    # Attributable, not smuggled in: relearn's share is a named contributor.
    assert {e["analyzer"] for e in rollup["by_analyzer"]} == {
        "summarize", "downsize", "relearn"}
    relearn_entry = next(e for e in rollup["by_analyzer"] if e["analyzer"] == "relearn")
    assert relearn_entry["usd"] == pytest.approx(26.5)
    assert relearn_entry["count"] == 2


def test_a_below_floor_relearn_row_is_in_the_total():
    """The sentence that was false for most of the money it described: "N
    smaller items are hidden, $X combined, still counted in the total above"."""
    finding = _finding([
        _cluster("relearn:small", unbounded_usd=9.0,
                 windows={WINDOW: _bucket(usd=1.20, reread_usd=0.20)}),
    ])
    rollup = past_overspend_rollup(
        relearn_contribution_rows(finding, label=WINDOW), window_days=30)
    stamped = stamp_relearn_contributions(finding, label=WINDOW)
    hidden = stamped["clusters"][0]                        # type: ignore[index]

    assert hidden["inbox_contribution_usd"] == pytest.approx(1.0)
    assert hidden["inbox_contribution_usd"] < 5            # the inbox's floor
    assert rollup["past_overspend_usd"] == pytest.approx(1.0)


# --- nothing is counted twice ---------------------------------------------- #

def test_the_reread_share_is_netted_out_so_the_resend_proposal_is_not_doubled():
    """Relearn's figure CONTAINS a re-read component, and the context re-send
    proposal prices re-sent context in full. Adding relearn gross would bill the
    same tokens in two rows of one total."""
    finding = _finding([
        _cluster("relearn:a", windows={WINDOW: _bucket(
            usd=30.0, reread_usd=8.0, tokens=3_000, reread_tokens=800)}),
    ])
    resend = {"signature": "cost:resend", "analyzer": "resend", "title": "trim",
              "past_overspend_usd": 412.44, "past_overspend_tokens": 50_000}

    rows = relearn_contribution_rows(finding, label=WINDOW)
    assert rows[0]["past_overspend_usd"] == pytest.approx(22.0)
    assert rows[0]["past_overspend_tokens"] == 2_200

    rollup = past_overspend_rollup([resend, *rows], window_days=30)
    gross = past_overspend_rollup([
        resend,
        {"signature": "x", "analyzer": "relearn", "past_overspend_usd": 30.0,
         "past_overspend_tokens": 3_000},
    ], window_days=30)
    # The overlap is exactly the re-read share, and it is counted once, inside
    # the resend proposal.
    assert gross["past_overspend_usd"] - rollup["past_overspend_usd"] == pytest.approx(8.0)
    assert rollup["past_overspend_usd"] == pytest.approx(434.44)


def test_an_already_applied_cluster_is_out_of_the_total_like_an_applied_proposal():
    finding = _finding([
        _cluster("relearn:open", windows={WINDOW: _bucket(usd=10.0, reread_usd=1.0)}),
        _cluster("relearn:done", windows={WINDOW: _bucket(usd=99.0, reread_usd=1.0)}),
    ])
    rows = relearn_contribution_rows(
        finding, label=WINDOW, applied_signatures={"relearn:done"})
    assert [r["past_overspend_usd"] for r in rows] == [pytest.approx(9.0)]


def test_a_contribution_row_signature_cannot_collide_with_a_cost_proposal():
    """The rollup deduplicates by signature, so a collision would silently drop
    one of the two rows and lose real money."""
    finding = _finding([
        _cluster("cost:downsize:driver-role",
                 windows={WINDOW: _bucket(usd=10.0, reread_usd=0.0)}),
    ])
    cost = {"signature": "cost:downsize:driver-role", "analyzer": "downsize",
            "past_overspend_usd": 522.43, "past_overspend_tokens": 1}
    rollup = past_overspend_rollup(
        [cost, *relearn_contribution_rows(finding, label=WINDOW)], window_days=30)
    assert rollup["past_overspend_usd"] == pytest.approx(532.43)
    assert rollup["deduplicated_proposal_count"] == 2


# --- one basis, or an honest exclusion ------------------------------------- #

def test_only_an_exactly_matching_window_contributes():
    """The headline publishes a window label. A bucket computed for another span
    cannot enter a total labelled with this one, and there is no nearest-match
    fallback to blur that."""
    assert exact_window_label(30, ["1h", "24h", "7d", "30d", "90d"]) == "30d"
    assert exact_window_label(14, ["1h", "24h", "7d", "30d", "90d"]) is None
    assert exact_window_label(1, ["24h"]) == "24h"         # same span, other spelling
    assert exact_window_label(30, []) is None
    assert exact_window_label(None, ["30d"]) is None
    # A malformed cache key is skipped, never raised through the aggregate.
    assert exact_window_label(30, ["not-a-window", "30d"]) == "30d"


def test_a_window_the_finding_never_computed_yields_no_contribution():
    finding = _finding([_cluster("relearn:a", windows={"90d": _bucket("90d", usd=9.0,
                                                                     reread_usd=1.0)})],
                       labels=("90d",))
    assert contribution_window_label(finding, 30) is None
    assert relearn_contribution_rows(finding, label=None) == []
    stamped = stamp_relearn_contributions(finding, label=None)
    assert stamped["clusters"][0]["inbox_contribution_usd"] is None  # type: ignore[index]


def test_unknown_money_is_disclosed_as_excluded_never_dropped_or_zeroed():
    """Absent is not zero. A cluster with no bounded figure for the headline's
    window is stated through the rollup's ``excluded`` channel, on its own
    basis, and summed into no total."""
    finding = _finding([
        _cluster("relearn:no-windows", unbounded_usd=26.24, windows=None),
        _cluster("relearn:priced", unbounded_usd=40.0,
                 windows={WINDOW: _bucket(usd=10.0, reread_usd=1.0)}),
    ])
    unrepresented = unrepresented_relearn(finding, label=WINDOW)
    assert unrepresented["clusters"] == 1
    assert unrepresented["past_overspend_usd"] == pytest.approx(26.24)

    excluded = relearn_excluded_entry(unrepresented, reason="no bounded figures")
    rollup = past_overspend_rollup(
        relearn_contribution_rows(finding, label=WINDOW),
        window_days=30, excluded=excluded,
    )
    assert rollup["past_overspend_usd"] == pytest.approx(9.0)   # NOT 35.24
    entry = rollup["excluded"]["relearn"]
    assert entry["past_overspend_usd"] == pytest.approx(26.24)
    assert entry["clusters"] == 1
    # The renderer must not fall back to the previous occupant's name, and must
    # not claim this figure is on the headline's window: it is excluded because
    # it is not.
    assert entry["label"] == "recurring mistakes"
    assert "not this window" in entry["note"]


def test_nothing_unrepresented_means_no_excluded_entry_at_all():
    finding = _finding([
        _cluster("relearn:a", windows={WINDOW: _bucket(usd=10.0, reread_usd=1.0)}),
    ])
    unrepresented = unrepresented_relearn(finding, label=WINDOW)
    assert unrepresented["clusters"] == 0
    assert relearn_excluded_entry(unrepresented, reason="whatever") == {}


def test_an_unpriced_bucket_is_unknown_not_a_free_cluster():
    """A bucket the detector could not price cannot be netted against the resend
    proposal, so it contributes nothing and is disclosed instead."""
    finding = _finding([
        _cluster("relearn:unpriced",
                 windows={WINDOW: _bucket(usd=None, reread_usd=None, tokens=900)}),
    ])
    assert relearn_contribution(finding["clusters"][0], label=WINDOW) is None  # type: ignore[index]
    assert relearn_contribution_rows(finding, label=WINDOW) == []
    assert unrepresented_relearn(finding, label=WINDOW)["clusters"] == 1


def test_a_cluster_quiet_in_the_window_contributes_a_known_zero():
    """Different from unknown: the detector priced this window and it holds no
    occurrence of this cluster. Zero is the measurement, not a placeholder."""
    finding = _finding([
        _cluster("relearn:quiet", unbounded_usd=12.0, windows={
            WINDOW: _bucket(usd=0.0, reread_usd=0.0, occurrences=0)}),
    ])
    contribution = relearn_contribution(finding["clusters"][0], label=WINDOW)  # type: ignore[index]
    assert contribution is not None and contribution["usd"] == 0.0
    assert unrepresented_relearn(finding, label=WINDOW)["clusters"] == 0


# --- the figures the write budget depends on stay untouched ----------------- #

def test_the_unbounded_fields_are_byte_identical_after_stamping():
    """``past_overspend_*`` is the write budget's pre-net gross. Shrinking it in
    place would silently flip clusters between "worth a permanent rule" and
    net-negative, so the contribution is a NEW field beside it."""
    finding = _finding([
        _cluster("relearn:a", unbounded_usd=40.0, unbounded_tokens=400_000,
                 windows={WINDOW: _bucket(usd=10.0, reread_usd=1.0)}),
    ])
    before = {k: v for k, v in finding["clusters"][0].items()}  # type: ignore[union-attr]
    stamped = stamp_relearn_contributions(finding, label=WINDOW)
    after = stamped["clusters"][0]                          # type: ignore[index]

    for key, value in before.items():
        assert after[key] == value, key
    assert set(after) - set(before) == {
        "inbox_contribution_usd", "inbox_contribution_tokens",
        "inbox_contribution_window", "inbox_contribution_basis",
    }
    # And the input object itself was not mutated.
    assert "inbox_contribution_usd" not in finding["clusters"][0]  # type: ignore[operator]


def test_the_contribution_can_only_be_smaller_than_the_unbounded_figure():
    """A filter minus a component of itself. There is no input that makes this
    a rescale, which is what the second retired mechanism was."""
    finding = _finding([
        _cluster("relearn:a", unbounded_usd=40.0,
                 windows={WINDOW: _bucket(usd=39.0, reread_usd=0.5)}),
    ])
    stamped = stamp_relearn_contributions(finding, label=WINDOW)
    row = stamped["clusters"][0]                            # type: ignore[index]
    assert row["inbox_contribution_usd"] <= row["past_overspend_usd"]


def test_a_net_figure_never_goes_negative():
    """A capped subset bucket could in principle report a re-read share above
    its own total. Money already spent is never negative."""
    finding = _finding([
        _cluster("relearn:a", windows={WINDOW: _bucket(
            usd=1.0, reread_usd=4.0, tokens=100, reread_tokens=900)}),
    ])
    row = relearn_contribution(finding["clusters"][0], label=WINDOW)  # type: ignore[index]
    assert row is not None
    assert row["usd"] == 0.0
    assert row["tokens"] == 0


# --- one field for the floor, the tail and the headline --------------------- #

def test_both_feeds_carry_the_same_contribution_field_under_one_window_label():
    """The UI tests the floor and sums the tail off ONE field, so a row's kind
    can never decide which quantity it is judged on."""
    cost = stamp_cost_contribution(
        {"signature": "cost:a", "analyzer": "downsize", "past_overspend_usd": 3.5,
         "past_overspend_tokens": 7}, window=WINDOW)
    finding = _finding([
        _cluster("relearn:a", windows={WINDOW: _bucket(usd=3.0, reread_usd=1.0)}),
    ])
    relearn_row = stamp_relearn_contributions(
        finding, label=WINDOW)["clusters"][0]               # type: ignore[index]

    for row in (cost, relearn_row):
        assert set(row) >= {
            "inbox_contribution_usd", "inbox_contribution_tokens",
            "inbox_contribution_window", "inbox_contribution_basis",
        }
        assert row["inbox_contribution_window"] == WINDOW
    assert cost["inbox_contribution_usd"] == pytest.approx(3.5)
    assert relearn_row["inbox_contribution_usd"] == pytest.approx(2.0)


def test_an_unpriced_cost_proposal_stays_unpriced_never_zero():
    """The floor may only hide a KNOWN small figure; an unpriced row is not
    cheap, and no combined figure may include it."""
    stamped = stamp_cost_contribution(
        {"signature": "cost:a", "analyzer": "trim", "past_overspend_usd": None,
         "past_overspend_tokens": None}, window=WINDOW)
    assert stamped["inbox_contribution_usd"] is None
    assert stamped["inbox_contribution_tokens"] is None


# --- the tile and the inbox cover the same POPULATION, not just the window -- #

def test_the_tile_figure_excludes_applied_clusters_exactly_like_the_inbox():
    """One analyzer, two surfaces, and the population is half the derivation.

    The window was unified first, which fixed the loud half and hid the rest:
    the tile still read the finding-level bucket, which covers EVERY retained
    cluster, while the inbox summed the OPEN ones — so the two disagreed by
    exactly what the user had already fixed, and applying a fix moved the
    inbox while the Dashboard sat still, still claiming the recovered money.

    Pinned as an equality against the inbox's own derivation rather than
    against a literal, so the two cannot drift apart again by one side being
    retuned.
    """
    from tokenjam.core.optimize.inbox_contribution import (
        window_scoped_finding_figure,
    )

    open_a = _cluster("open-a", windows={WINDOW: _bucket(usd=30.0, reread_usd=5.0)})
    open_b = _cluster("open-b", windows={WINDOW: _bucket(usd=20.0, reread_usd=2.0)})
    done = _cluster("done", windows={WINDOW: _bucket(usd=100.0, reread_usd=10.0)})
    finding = {
        "clusters": [open_a, open_b, done],
        # What the detector precomputes at finding level: every cluster.
        "past_overspend_windows": {
            WINDOW: _bucket(usd=150.0, reread_usd=17.0),
        },
    }
    applied = ["done"]

    inbox = sum(
        (row.get("past_overspend_usd") or 0.0)
        for row in relearn_contribution_rows(
            finding, label=WINDOW, applied_signatures=applied,
        )
    )
    tile = window_scoped_finding_figure(
        finding, days=30, applied_signatures=applied,
    )

    assert inbox == pytest.approx(43.0)      # (30-5) + (20-2)
    assert tile is not None
    assert tile["usd"] == pytest.approx(inbox), (
        "the Dashboard tile and the Review inbox published different "
        "populations of the same analyzer over the same window"
    )
    # And the applied cluster's money is genuinely gone from it, not merely
    # equal by coincidence of the fixture.
    whole = window_scoped_finding_figure(finding, days=30)
    assert whole is not None and whole["usd"] == pytest.approx(133.0)
