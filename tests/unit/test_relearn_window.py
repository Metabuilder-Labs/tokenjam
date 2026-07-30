"""Bounded trailing-window views of relearn's observed cost, and the robust
"how many days of data do we actually have" helper.

WHAT THIS MECHANISM IS, AND WHAT IT DELIBERATELY IS NOT. Two earlier attempts
to put relearn's dollars on a 30-day basis were built and retired, and both
retirements are still guarded by their own tests
(``test_cost_proposals.test_the_rollup_has_no_per_analyzer_side_channel`` and
``test_relearn.test_retired_forward_fields_stay_gone``). Neither is weakened
here, because this is a different operation:

  * the first retirement removed a SECOND TIME BASIS smuggled into a shared
    aggregate through a relearn-only parameter. Nothing in THIS module touches
    ``past_overspend_rollup`` or introduces a second aggregate: the windowed
    figures live on relearn's own cluster and finding and are summed only over
    each other (``sum_windowed``). A separate module, ``core/optimize/
    inbox_contribution.py``, later reads a bounded bucket and hands it to the one
    shared aggregate as an ordinary row on the canonical field, on the window
    that aggregate is labelled with; see its own tests for why that is the
    sanctioned path and not the retired parameter. Producing the bounded figure
    and spending it are deliberately different modules.
  * the second retirement removed a RESCALE: a corpus observation multiplied by
    ``30 / window_days`` and published as a month's money that was never
    observed. This is a FILTER over the same observed occurrences, so it can
    only ever be smaller than the unbounded figure, and the tests below pin
    that direction.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import pytest

from tokenjam.core.rulewrite.kinds import DELIVERY_INJECTING_HOOK

from tokenjam.core.data_span import DataSpan, available_data_span, data_span_from_days
from tokenjam.core.optimize.analyzers.relearn import (
    GROUNDED_TOKENS_PER_OCCURRENCE,
    FailureEpisode,
    _RawCluster,
    _windowed_observations,
    build_proposals,
)
from tokenjam.core.optimize.rate_profile import RateProfile
from tokenjam.core.optimize.relearn_window import (
    RELEARN_WINDOW_LABELS,
    resolve_window_label,
    window_days,
)

ANCHOR = datetime(2026, 7, 27, 12, 0, tzinfo=timezone.utc)


def _iso(days_ago: float) -> str:
    return (ANCHOR - timedelta(days=days_ago)).isoformat().replace("+00:00", "Z")


def _failure(session: str, days_ago: float, *, detour: float | None = 2.0) -> FailureEpisode:
    return FailureEpisode(
        session_id=session, repo="repo-a", ts=_iso(days_ago),
        tool_name="Bash", label="cd x",
        error_text="(eval):cd:1: no such file or directory: x",
        kind="act", is_retry=False, depth=0, detour_turns=detour,
    )


def _cluster(failures: list[FailureEpisode]) -> _RawCluster:
    return _RawCluster(
        signature="cwd_confusion", family_key="cwd_confusion",
        title="cwd / relative-path confusion", failures=failures,
    )


# --- window label vocabulary ------------------------------------------------ #

def test_window_labels_are_the_since_vocabulary_the_other_routes_use():
    # The precomputed set matches the window selector's own option list, so a
    # `since` the UI can produce always has an exactly-matching bucket.
    assert RELEARN_WINDOW_LABELS == ("1h", "24h", "7d", "30d", "90d")
    assert window_days("24h") == pytest.approx(1.0)
    assert window_days("7d") == pytest.approx(7.0)


def test_an_exact_label_resolves_to_itself_and_an_odd_one_to_the_nearest():
    assert resolve_window_label("30d", RELEARN_WINDOW_LABELS) == "30d"
    # 45d sits between 30d and 90d; nearest by duration is 30d.
    assert resolve_window_label("45d", RELEARN_WINDOW_LABELS) == "30d"
    # A tie resolves to the LONGER window: under-reporting an observed cost is
    # the one direction this module is not allowed to bias toward.
    assert resolve_window_label("2d", ("24h", "3d")) == "3d"


def test_a_malformed_since_raises_rather_than_guessing():
    with pytest.raises(ValueError):
        window_days("banana")
    with pytest.raises(ValueError):
        resolve_window_label("banana", RELEARN_WINDOW_LABELS)


# --- the bounded observation ------------------------------------------------ #

def test_bounded_figure_never_exceeds_the_unbounded_one():
    # Nine occurrences spread over 60 days. Every window's figure is a subset
    # sum of the same observation, so it can only shrink.
    failures = [_failure(f"s{i}", days_ago) for i, days_ago in enumerate(
        [0.1, 0.5, 2, 5, 9, 20, 35, 50, 60],
    )]
    proposals, _ = build_proposals(
        [_cluster(failures)], min_sessions=3,
        window_labels=RELEARN_WINDOW_LABELS, window_anchor=ANCHOR,
    )
    cluster = proposals[0]
    assert cluster.past_overspend_windows is not None
    for label in RELEARN_WINDOW_LABELS:
        bucket = cluster.past_overspend_windows[label]
        assert bucket.past_overspend_tokens <= cluster.past_overspend_tokens
        assert bucket.occurrences <= cluster.occurrences


def test_occurrences_and_detour_turns_are_re_derived_not_reused():
    # Six occurrences, three of them inside the last 7 days, with DIFFERENT
    # detour costs so a reused (whole-cluster) figure would be visible.
    failures = (
        [_failure(f"recent{i}", days_ago, detour=1.0) for i, days_ago in enumerate([1, 2, 3])]
        + [_failure(f"old{i}", days_ago, detour=5.0) for i, days_ago in enumerate([40, 50, 60])]
    )
    proposals, _ = build_proposals(
        [_cluster(failures)], min_sessions=3,
        window_labels=RELEARN_WINDOW_LABELS, window_anchor=ANCHOR,
    )
    cluster = proposals[0]
    week = cluster.past_overspend_windows["7d"]
    assert cluster.occurrences == 6
    assert week.occurrences == 3                 # re-counted, not inherited
    assert week.sessions == 3
    assert week.detour_turns == pytest.approx(3.0)   # 3 x 1.0, not 6 x avg
    # The head term follows the re-derived detour turns, not the cluster's.
    assert week.past_overspend_tokens == round(3.0 * GROUNDED_TOKENS_PER_OCCURRENCE)


def test_the_tail_multiplier_is_re_derived_over_the_filtered_failures():
    # Two occurrences: the recent one sat in context for 20 more calls, the old
    # one for 2. The whole-cluster median tail is therefore 11 while the
    # last-24h median is 20 — a reused multiplier could not produce 20.
    profile = RateProfile(input_rate_per_token=3e-6, cache_read_ratio=0.1, basis="test")
    recent = _failure("recent", 0.2)
    old = _failure("old", 40)
    base = ANCHOR - timedelta(days=41)
    timelines = {
        # 2 calls after the old failure, then a hard compaction stops the tail.
        "old": [(base + timedelta(minutes=i), 100_000) for i in range(3)]
               + [(base + timedelta(minutes=3), 1_000)],
        "recent": [(ANCHOR - timedelta(hours=4) + timedelta(minutes=i), 100_000)
                   for i in range(21)],
    }
    # Timeline entries must straddle each failure's own timestamp.
    timelines["old"] = [(ANCHOR - timedelta(days=40, minutes=1), 100_000)] + [
        (ANCHOR - timedelta(days=40) + timedelta(minutes=i + 1), 100_000) for i in range(2)
    ] + [(ANCHOR - timedelta(days=40) + timedelta(minutes=5), 1_000)]
    timelines["recent"] = [(ANCHOR - timedelta(days=0.2) - timedelta(minutes=1), 100_000)] + [
        (ANCHOR - timedelta(days=0.2) + timedelta(minutes=i + 1), 100_000) for i in range(20)
    ]

    buckets = _windowed_observations(
        [recent, old], labels=("24h", "90d"), anchor=ANCHOR,
        turn_tokens=10_000, profile=profile, timelines=timelines,
        unbounded_tokens=10**9, unbounded_head_tokens=0,
    )
    assert buckets is not None
    assert buckets["24h"].tail_calls_median == 20
    assert buckets["90d"].tail_calls_median in (2, 11)   # median of [20, 2]
    assert buckets["24h"].tail_multiplier > buckets["90d"].tail_multiplier


def test_a_subset_median_that_inflates_the_multiplier_is_capped_not_published():
    # The clamp: a filtered sample's median tail can legitimately exceed the
    # whole cluster's, but the money already spent cannot. The bounded figure is
    # capped at the unbounded one and says so, rather than publishing a subset
    # that costs more than the whole.
    profile = RateProfile(input_rate_per_token=3e-6, cache_read_ratio=0.1, basis="test")
    recent = _failure("recent", 0.2)
    timelines = {
        "recent": [(ANCHOR - timedelta(days=0.2) - timedelta(minutes=1), 100_000)] + [
            (ANCHOR - timedelta(days=0.2) + timedelta(minutes=i + 1), 100_000)
            for i in range(50)
        ],
    }
    buckets = _windowed_observations(
        [recent], labels=("24h",), anchor=ANCHOR,
        turn_tokens=1_000, profile=profile, timelines=timelines,
        unbounded_tokens=2_000, unbounded_head_tokens=1_000,
    )
    assert buckets is not None
    bucket = buckets["24h"]
    assert bucket.past_overspend_tokens <= 2_000
    assert bucket.capped_at_unbounded is True


def test_undated_occurrences_are_disclosed_not_silently_dropped():
    # An occurrence with no parseable timestamp cannot be placed in a window.
    # It is neither counted in nor quietly discarded: the bucket says how many.
    dated = [_failure(f"s{i}", days_ago) for i, days_ago in enumerate([1, 2, 3])]
    undated = FailureEpisode(
        session_id="nots", repo="repo-a", ts=None, tool_name="Bash", label="",
        error_text="no such file or directory", kind="act", is_retry=False,
        depth=0, detour_turns=2.0,
    )
    proposals, _ = build_proposals(
        [_cluster(dated + [undated])], min_sessions=3,
        window_labels=("30d",), window_anchor=ANCHOR,
    )
    bucket = proposals[0].past_overspend_windows["30d"]
    assert bucket.occurrences == 3
    assert bucket.undated_occurrences == 1


def test_a_cluster_with_no_timestamps_at_all_is_unknown_never_zero():
    failures = [
        FailureEpisode(
            session_id=f"s{i}", repo="repo-a", ts=None, tool_name="Bash", label="",
            error_text="no such file or directory", kind="act", is_retry=False,
            depth=0, detour_turns=2.0,
        )
        for i in range(3)
    ]
    proposals, _ = build_proposals(
        [_cluster(failures)], min_sessions=3,
        window_labels=("30d",), window_anchor=ANCHOR,
    )
    # No window can be asserted over occurrences that carry no dates. `None`
    # (unknown) rather than a bucket reading 0 (a claim of "cost nothing").
    assert proposals[0].past_overspend_windows is None


def test_no_window_labels_leaves_the_unbounded_behaviour_exactly_as_it_was():
    failures = [_failure(f"s{i}", days_ago) for i, days_ago in enumerate([1, 2, 3])]
    proposals, _ = build_proposals([_cluster(failures)], min_sessions=3)
    assert proposals[0].past_overspend_windows is None
    assert proposals[0].past_overspend_tokens > 0


def test_the_unbounded_fields_are_identical_with_and_without_windowing():
    # The write budget nets against `past_overspend_tokens` as its pre-net
    # gross, so windowing must not move it by a single token.
    failures = [_failure(f"s{i}", days_ago) for i, days_ago in enumerate([1, 9, 40])]
    plain, _ = build_proposals([_cluster(failures)], min_sessions=3)
    windowed, _ = build_proposals(
        [_cluster(failures)], min_sessions=3,
        window_labels=RELEARN_WINDOW_LABELS, window_anchor=ANCHOR,
    )
    for field_name in (
        "past_overspend_tokens", "past_overspend_usd", "past_reread_tokens",
        "past_reread_usd", "past_overspend_basis", "tail_multiplier",
        "standing_cost_tokens", "payback_ratio", "net_negative",
        "write_offered", "write_blocked_reason",
    ):
        assert getattr(plain[0], field_name) == getattr(windowed[0], field_name)


# --- the finding-level total shares the per-cluster basis ------------------- #

def test_the_windowed_total_is_the_sum_of_the_per_cluster_windowed_figures():
    # The floor note and the headline must be able to read ONE quantity: a
    # windowed total that is not the sum of the rows' own windowed figures is
    # the mixed-population defect all over again.
    from tokenjam.core.optimize.analyzers.relearn import analyze_relearns

    finding = analyze_relearns(
        [], extra_failures=[
            _failure(f"a{i}", days_ago) for i, days_ago in enumerate([1, 2, 3])
        ] + [
            FailureEpisode(
                session_id=f"b{i}", repo="repo-b", ts=_iso(days_ago),
                tool_name="Read", label="", error_text="File has not been read yet",
                kind="act", is_retry=False, depth=0, detour_turns=1.0,
            )
            for i, days_ago in enumerate([1, 40, 50])
        ],
        distill_enabled=False, min_sessions=3,
        window_labels=("7d", "90d"), window_anchor=ANCHOR,
    )
    assert finding.past_overspend_windows is not None
    for label in ("7d", "90d"):
        total = finding.past_overspend_windows[label]
        rows = [
            c.past_overspend_windows[label] for c in finding.clusters
            if c.past_overspend_windows is not None
        ]
        assert total.past_overspend_tokens == sum(r.past_overspend_tokens for r in rows)
        assert total.occurrences == sum(r.occurrences for r in rows)
        assert total.clusters == len(rows)
        assert total.past_overspend_tokens <= finding.past_overspend_tokens


# --- cache compatibility ---------------------------------------------------- #

def test_a_cache_written_before_windowing_loads_and_reads_unknown():
    """An older cache carries no windowed keys at all. It must load, and the
    absent figure must read as UNKNOWN (None), never as a zero-dollar claim."""
    from dataclasses import asdict

    from tokenjam.core.optimize.analyzers.relearn import RelearnCluster, RelearnFinding

    old_cluster = {
        "signature": "cwd_confusion", "family_key": "cwd_confusion",
        "title": "cwd confusion", "sessions": 3, "occurrences": 9,
        "repos": ["repo-a"], "delivery": DELIVERY_INJECTING_HOOK,
        "scope": "project",
        "proposed_fix": "hook", "past_overspend_tokens": 12345,
        "past_overspend_usd": 1.25,
    }
    revived = RelearnCluster(**old_cluster)
    assert revived.past_overspend_windows is None
    finding = RelearnFinding(clusters=[revived], past_overspend_tokens=12345)
    assert finding.past_overspend_windows is None
    # Round-trips through the cache's own serializer without inventing a zero.
    assert asdict(finding)["past_overspend_windows"] is None


# --- days of data available ------------------------------------------------- #

def test_the_daemon_to_cli_round_trip_keeps_the_windowed_figures():
    """``report_to_dict`` / ``report_from_dict`` is the daemon-to-CLI boundary.

    Nested dataclasses cross it as plain dicts, so they have to be revived
    explicitly. A field this module knows how to carry and silently drops is
    indistinguishable downstream from one that was never computed, which is the
    exact bug the neighbouring comment in ``runner._relearn`` records about
    relearn's dollar figure.
    """
    from tokenjam.core.optimize.analyzers.relearn import analyze_relearns
    # `_finding_constructor_for` (a hand-written per-field builder) was
    # replaced by a name -> dataclass table plus generic hydration; those
    # hand-written builders were exactly what dropped fields silently.
    from tokenjam.core.optimize.runner import _finding_class_for, hydrate_dataclass
    from dataclasses import asdict

    finding = analyze_relearns(
        [], extra_failures=[_failure(f"s{i}", d) for i, d in enumerate([1, 2, 3])],
        distill_enabled=False, min_sessions=3,
        window_labels=("7d",), window_anchor=ANCHOR,
    )
    revived = hydrate_dataclass(_finding_class_for("relearn"), asdict(finding))
    assert revived.past_overspend_windows is not None
    assert revived.past_overspend_windows["7d"].past_overspend_tokens == \
        finding.past_overspend_windows["7d"].past_overspend_tokens
    assert revived.clusters[0].past_overspend_windows["7d"].occurrences == 3
    # And the revived bucket is the same TYPE, not a bare dict.
    assert revived.clusters[0].past_overspend_windows["7d"].label == "7d"


def test_an_older_payload_without_the_windowed_keys_revives_as_unknown():
    from tokenjam.core.optimize.relearn_window import (
        observations_from_dict,
        totals_from_dict,
    )

    for absent in (None, {}, "nonsense", []):
        assert observations_from_dict(absent) is None
        assert totals_from_dict(absent) is None
    # A bucket missing fields is UNKNOWN, never zero-filled: a defaulted 0 would
    # publish "this window cost nothing" off an absent key.
    assert observations_from_dict({"30d": {"label": "30d"}}) is None


def test_an_ancient_outlier_row_does_not_stretch_the_available_span():
    """The naive measure is wrong here and this is the row that proves it.

    ``max(ts) - min(ts)`` over this corpus reports ~2,400 days because one
    fixture row is dated 2020-01-01. The real, usable history is the most
    recent contiguous block of days that actually carry data.
    """
    days = [date(2020, 1, 1)] + [
        date(2026, 5, 22) + timedelta(days=i) for i in range(67)
    ]
    span = data_span_from_days(days)
    naive = (max(days) - min(days)).days + 1
    assert naive > 2_000
    assert span.available_days == 67
    assert span.newest == "2026-07-27"
    assert span.oldest_in_block == "2026-05-22"
    assert span.ignored_days_before_block == 1
    # The second, independently robust measure: an outlier can add at most one.
    assert span.days_with_data == 68


def test_a_gap_inside_the_tolerance_does_not_split_the_block():
    days = [date(2026, 7, 1), date(2026, 7, 4), date(2026, 7, 5), date(2026, 7, 10)]
    span = data_span_from_days(days, max_gap_days=7)
    assert span.available_days == 10          # 2026-07-01 .. 2026-07-10
    assert span.ignored_days_before_block == 0


def test_a_gap_past_the_tolerance_ends_the_block():
    days = [date(2026, 1, 1), date(2026, 7, 1), date(2026, 7, 2)]
    span = data_span_from_days(days, max_gap_days=7)
    assert span.available_days == 2
    assert span.ignored_days_before_block == 1


def test_implausible_and_future_days_are_discarded_before_measuring():
    today = datetime.now(tz=timezone.utc).date()
    days = [date(1970, 1, 1), today + timedelta(days=400), today]
    span = data_span_from_days(days)
    assert span.available_days == 1
    assert span.newest == today.isoformat()


@pytest.mark.parametrize("db_timezone", ["UTC", "Asia/Kolkata", "America/Los_Angeles"])
def test_the_day_union_reads_utc_days_whatever_the_databases_timezone_is(db_timezone):
    """The read side must group by the SAME day boundary the filter uses.

    ``_plausible_days`` drops any day later than the UTC date, so a bare
    ``CAST(ts AS DATE)`` — which DuckDB resolves through the session timezone
    before truncating (Critical Rule 1) — silently deletes TODAY from the span
    for every hour the local date runs ahead of the UTC one. On a ``+05:30``
    machine that is the last five and a half hours of every day: the served
    ``available_days`` drops by one at 18:30 local and comes back at midnight,
    with no change behind it. Late-in-the-UTC-day rows are the ones that
    expose it, so the fixture pins them there rather than at ``utcnow()``.
    """
    import duckdb

    conn = duckdb.connect()
    conn.execute(f"SET TimeZone='{db_timezone}'")
    conn.execute("CREATE TABLE spans (start_time TIMESTAMPTZ)")
    conn.execute("CREATE TABLE sessions (started_at TIMESTAMPTZ)")

    today = datetime.now(tz=timezone.utc).date()
    late = datetime(today.year, today.month, today.day, 23, 30, tzinfo=timezone.utc)
    conn.execute("INSERT INTO spans VALUES (?)", [late - timedelta(days=1)])
    conn.execute("INSERT INTO sessions VALUES (?)", [late])

    span = available_data_span(conn)
    assert span.newest == today.isoformat()
    assert span.days_with_data == 2
    assert span.available_days == 2


def test_no_data_at_all_is_unknown_not_a_zero_day_span():
    span = data_span_from_days([])
    assert isinstance(span, DataSpan)
    assert span.available_days is None
    assert span.days_with_data == 0
    assert span.newest is None
    assert "no dated" in span.basis
