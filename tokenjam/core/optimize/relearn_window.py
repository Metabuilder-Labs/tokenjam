"""Bounded trailing-window views of relearn's observed cost.

WHY THE BOUND IS COMPUTED IN THE ANALYZER AND NOT AT THE API. Relearn's
per-occurrence dates exist only deep in the pipeline: ``_RawCluster.failures``
holds a ``FailureEpisode`` per occurrence, each with its own ``ts``. The
dataclass that is CACHED and served (``RelearnCluster``) keeps a scalar
``occurrences`` and the dollar figures; the only surviving dates ride on
``examples``, which is capped at three and sorted newest-first. A route reading
that cache therefore has a biased, incomplete sample and cannot honestly
recompute anything over a window. So the analyzer computes the bounded figures
for a small fixed vocabulary of windows while it still holds the dates, and the
route selects among them.

WHAT A BOUNDED FIGURE IS. A FILTER over the same observed occurrences, priced
exactly as they are already priced. It is not a rescale, not a projection and
not a pace: nothing is multiplied up to fill the window. Two earlier
mechanisms did rescale relearn's dollars to 30 days and were retired for it;
their retirement tests still stand untouched, because a subset sum is the
opposite operation. The direction is pinned in code as well as in tests: a
window's figure is capped at the unbounded one.

WHAT IS RE-DERIVED, AND WHAT IS DELIBERATELY NOT. Bounding is not a filter over
one sum, because the unbounded figure is a product of several terms that each
depend on the occurrence set. Re-derived over the filtered failures:
occurrences, distinct sessions, recovery-arc turns (``detour_turns``) and the
measured re-read tail. Deliberately reused from the FULL cluster: the blended
rate profile and the measured per-turn token cost. Three reasons, in order of
weight:

  1. it keeps the bounded figure a subset SUM of the unbounded one. Recomputing
     the price per subset would make the two figures price identical
     occurrences differently, so a 24-hour figure could legitimately exceed a
     90-day one, which is incoherent for an observation;
  2. the per-turn cost is a MEDIAN over the cluster's sessions. A median over a
     handful of filtered sessions is noise, and noise in a price is a lie in a
     dollar figure;
  3. it costs no extra database round trips. The rate profile, the per-turn
     cost and the prompt timelines are all computed once per cluster for the
     unbounded figure; every window reuses them, so the whole vocabulary is
     free and the expensive full-corpus job does not get five times slower.

The one place re-derivation can still misbehave is the tail multiplier: a
median over a small filtered sample can exceed the whole cluster's median (the
window happens to contain the one occurrence that sat in context longest). That
is sampling noise in a multiplier, not extra money spent, so the resulting
total is capped at the unbounded figure and the bucket says ``capped_at_
unbounded`` rather than quietly publishing a subset that costs more than the
whole.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Sequence

from tokenjam.utils.time_parse import parse_since, utcnow

#: The window vocabulary the analyzer precomputes, matching the web UI's own
#: window selector options so a `since` the UI can produce always has an
#: exactly-matching bucket. A `since` from anywhere else resolves to the
#: nearest of these and the response says which was applied.
RELEARN_WINDOW_LABELS: tuple[str, ...] = ("1h", "24h", "7d", "30d", "90d")


def window_labels_including(label: str | None) -> tuple[str, ...]:
    """The fixed vocabulary plus the resolved analysis span, deduplicated.

    The Review inbox publishes ONE window label across both feeds, and the cost
    side's window is the resolved analysis span (`core/analysis_span
    .window_days_for`) rather than a member of the fixed vocabulary above. A
    span that had no precomputed bucket here could never contribute a relearn
    row to that headline — the exact-match lookup would find nothing and every
    cluster would fall into the `excluded` channel — so the span joins the
    vocabulary and the two sides meet.

    Both sides derive the label from the same seam, but not in the same pass:
    the detector runs on its own schedule, so a span that moves between runs
    leaves the cache one recompute behind. That degrades honestly through the
    existing `excluded` channel rather than mislabelling, and self-heals.
    """
    if not label or label in RELEARN_WINDOW_LABELS:
        return RELEARN_WINDOW_LABELS
    return RELEARN_WINDOW_LABELS + (label,)

_LABEL_RE = re.compile(r"^(\d+)([mhd])$")
_UNIT_DAYS = {"m": 1.0 / 1440.0, "h": 1.0 / 24.0, "d": 1.0}

WINDOWED_BASIS = (
    "the SAME observed occurrences the unbounded past_overspend figure covers, "
    "filtered to this trailing window and re-derived over that subset: its own "
    "occurrence count, its own distinct sessions, its own measured recovery-arc "
    "turns and its own measured re-read tail. A filter, never a rescale: "
    "nothing is paced, projected or extrapolated to fill the window, so this "
    "figure can only be smaller than the unbounded one and is capped at it. "
    "The per-turn price and the contributing-session set are the full "
    "cluster's, not the subset's, so both figures price identical occurrences "
    "identically. Occurrences whose timestamp could not be parsed cannot be "
    "placed in any window; they are reported as undated_occurrences rather "
    "than counted in or silently discarded. The window runs backward from "
    "window_end, which is when the detector last ran, not when this was read"
)

WINDOW_TOTAL_BASIS = (
    "the sum of every cluster's own figure for this same window, and of "
    "nothing else. Clusters whose occurrences carry no parseable timestamps "
    "have no windowed figure at all and are counted in clusters_unknown "
    "instead of contributing a zero"
)


def window_days(label: str) -> float:
    """A precomputed window label as a span in days. Raises ``ValueError`` on
    anything that is not a bare relative span (``90d``, ``24h``, ``30m``).

    Strict on purpose: these labels are the cache's own keys, so a typo must
    fail at the producer rather than silently become a different window.
    """
    match = _LABEL_RE.match((label or "").strip())
    if not match:
        raise ValueError(
            f"not a window label: {label!r}. Expected a bare relative span "
            f"such as 24h, 7d or 90d."
        )
    amount = int(match.group(1))
    if amount <= 0:
        raise ValueError(f"not a window label: {label!r} (amount must be > 0)")
    return amount * _UNIT_DAYS[match.group(2)]


def since_span_days(value: str) -> float:
    """How many days back a caller's ``since`` reaches.

    Accepts everything ``parse_since`` accepts (relative spans, a date, an ISO
    datetime) so this endpoint's ``since`` means exactly what it means on
    ``/cost``, ``/alerts``, ``/traces`` and ``/optimize``. Raises ``ValueError``
    on a malformed value, which the routes translate to a 400.
    """
    try:
        return window_days(value)
    except ValueError:
        pass
    start = parse_since(value)          # raises ValueError on a malformed value
    span = (utcnow() - start).total_seconds() / 86400.0
    return max(span, 0.0)


def resolve_window_label(value: str, available: Sequence[str]) -> str:
    """The precomputed window closest in duration to ``value``.

    A caller may ask for any span; only ``available`` ones exist in the cache.
    Rather than refusing, the nearest is applied and the response reports which
    one, so a reader is never shown a figure under a window label it was not
    computed for. Ties resolve to the LONGER window: under-reporting an already
    incurred cost is the one direction this module must not bias toward.
    """
    if not available:
        raise ValueError("no precomputed windows are available")
    wanted = since_span_days(value)
    return min(available, key=lambda label: (abs(window_days(label) - wanted), -window_days(label)))


@dataclass
class RelearnWindowedObservation:
    """One cluster's observed cost, bounded to one trailing window.

    Parallel to the cluster's unbounded ``past_overspend_*`` fields, never a
    replacement for them: those feed the write budget's netting as the pre-net
    gross, and shrinking them would silently flip clusters between "worth a
    permanent rule" and net-negative.
    """
    label:                 str
    window_days:           float
    window_start:          str
    window_end:            str
    occurrences:           int
    sessions:              int
    detour_turns:          float
    #: Occurrences in this cluster that carry no parseable timestamp and so sit
    #: in no window. Disclosed rather than dropped.
    undated_occurrences:   int
    tail_calls_median:     int
    tail_multiplier:       float
    past_overspend_tokens: int
    past_overspend_usd:    float | None
    past_reread_tokens:    int
    past_reread_usd:       float | None
    #: True when the subset's own re-read median would have produced a figure
    #: larger than the unbounded one and it was capped. See this module's
    #: docstring: sampling noise in a multiplier, not money.
    capped_at_unbounded:   bool
    basis:                 str


@dataclass
class RelearnWindowTotal:
    """Every cluster's figure for one window, summed. Nothing else enters it."""
    label:                 str
    window_days:           float
    window_start:          str
    window_end:            str
    clusters:              int
    #: Clusters that have no windowed figure at all (no parseable timestamps).
    #: Counted, never summed in as zero.
    clusters_unknown:      int
    occurrences:           int
    undated_occurrences:   int
    past_overspend_tokens: int
    past_overspend_usd:    float | None
    past_reread_tokens:    int
    past_reread_usd:       float | None
    basis:                 str


def sum_windowed(
    per_cluster: Sequence[dict[str, RelearnWindowedObservation] | None],
    label: str,
    *,
    anchor_start: str,
    anchor_end: str,
) -> RelearnWindowTotal:
    """Total one window across clusters.

    The total is the sum of exactly the per-cluster figures a reader can see on
    the rows, which is what lets a floor note ("N smaller items are hidden, $X
    combined, still counted in the total above") stay true: the hidden sum and
    the total are the same quantity summed over the same population.
    """
    buckets = [
        windows[label] for windows in per_cluster
        if windows is not None and label in windows
    ]
    priced = [b.past_overspend_usd for b in buckets if b.past_overspend_usd is not None]
    reread_priced = [b.past_reread_usd for b in buckets if b.past_reread_usd is not None]
    return RelearnWindowTotal(
        label=label,
        window_days=window_days(label),
        window_start=anchor_start,
        window_end=anchor_end,
        clusters=len(buckets),
        clusters_unknown=sum(1 for w in per_cluster if w is None or label not in w),
        occurrences=sum(b.occurrences for b in buckets),
        undated_occurrences=sum(b.undated_occurrences for b in buckets),
        past_overspend_tokens=sum(b.past_overspend_tokens for b in buckets),
        past_overspend_usd=round(sum(priced), 6) if priced else None,
        past_reread_tokens=sum(b.past_reread_tokens for b in buckets),
        past_reread_usd=round(sum(reread_priced), 6) if reread_priced else None,
        basis=WINDOW_TOTAL_BASIS,
    )


def _revive(cls: type, raw: Any) -> Any | None:
    """One serialized bucket back into ``cls``, or ``None`` if it cannot be.

    A bucket missing any field it needs is UNKNOWN, not a zero-filled one, so it
    is dropped rather than defaulted: a figure defaulted to 0 would publish "this
    window cost nothing" on the strength of an absent key. Unknown extra keys are
    ignored, so a payload from a NEWER producer still loads here.
    """
    from dataclasses import fields

    if not isinstance(raw, dict):
        return None
    names = {f.name for f in fields(cls)}
    if not names.issubset(raw.keys()):
        return None
    try:
        return cls(**{k: raw[k] for k in names})
    except (TypeError, ValueError):
        return None


def observations_from_dict(
    raw: Any,
) -> dict[str, RelearnWindowedObservation] | None:
    """A cluster's serialized windowed figures back into dataclasses.

    ``None`` for anything absent, empty or unreadable, which is what a cache or
    an HTTP payload written before these fields existed produces. Absent means
    UNKNOWN here and every reader must treat it that way.
    """
    if not isinstance(raw, dict) or not raw:
        return None
    out = {
        label: revived for label, value in raw.items()
        if (revived := _revive(RelearnWindowedObservation, value)) is not None
    }
    return out or None


def totals_from_dict(raw: Any) -> dict[str, RelearnWindowTotal] | None:
    """A finding's serialized windowed totals back into dataclasses. Same
    absent-is-unknown rule as ``observations_from_dict``."""
    if not isinstance(raw, dict) or not raw:
        return None
    out = {
        label: revived for label, value in raw.items()
        if (revived := _revive(RelearnWindowTotal, value)) is not None
    }
    return out or None


def window_report(
    *,
    since: str | None,
    applied: dict[str, Any] | None,
    available: Sequence[str],
    unavailable_reason: str | None = None,
    clusters_in_window: int | None = None,
    clusters_omitted: int | None = None,
) -> dict[str, Any]:
    """The block a route publishes so a reader can see WHICH window produced
    the figures beside it, and which population each figure covers.

    ``since`` is what the caller asked for; ``applied`` is the windowed TOTAL
    that actually existed (already serialized, since it comes off the JSON
    cache). They differ whenever a caller asks for a span the detector did not
    precompute, and a surface that renders the requested label over figures
    computed for another one is the mixed-basis defect in miniature.
    """
    return {
        "since_requested": since,
        "applied": applied.get("label") if applied is not None else None,
        "window_days": applied.get("window_days") if applied is not None else None,
        "window_start": applied.get("window_start") if applied is not None else None,
        "window_end": applied.get("window_end") if applied is not None else None,
        "available": list(available),
        "clusters_in_window": clusters_in_window,
        "clusters_omitted_outside_window": clusters_omitted,
        "clusters_unknown": (
            applied.get("clusters_unknown") if applied is not None else None
        ),
        "unavailable_reason": unavailable_reason,
        "population_note": (
            "the rows in this response cover the applied window only; the "
            "finding's own past_overspend_* totals still cover every cluster, "
            "including the ones omitted here. past_overspend_windowed is the "
            "figure that covers exactly these rows"
            if applied is not None else None
        ),
    }
