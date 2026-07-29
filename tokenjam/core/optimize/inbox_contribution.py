"""What each Review inbox row contributed to the ONE headline total.

THE REQUIREMENT THIS MODULE EXISTS FOR. The Review inbox is a single list that
merges two feeds: cost proposals (``core/optimize/cost_proposals.py``) and
relearn clusters (``core/optimize/analyzers/relearn.py``). Its headline total
was summed over the cost feed alone, so the money on relearn's rows sat outside
it. That made every derived sentence false in a way a reader could not see: the
collapsed tail summed rows of BOTH kinds, and the below-floor note said the
hidden items were "still counted in the total above" when most of that money
never entered the total. The rule is now one line long: every inbox row's money
appears in the headline, exactly once, on one basis.

WHY THIS IS NOT EITHER RETIRED MECHANISM. Two earlier attempts to put relearn's
dollars into a shared aggregate were built and deliberately retired, and both
retirements are still guarded by tests that must keep passing
(``test_cost_proposals.test_the_rollup_has_no_per_analyzer_side_channel`` and
``test_relearn.test_retired_forward_fields_stay_gone``):

  * the first was a SIDE CHANNEL: a relearn-only ``relearn_clusters=``
    parameter on ``past_overspend_rollup`` which computed relearn's figure
    INSIDE the rollup, on relearn's own 30-day basis, and published it in a
    separate ``projected_usd_30d`` key no other analyzer's figure shared. Two
    time bases and two keys in one aggregate.
  * the second was a RESCALE: a corpus observation multiplied by
    ``30 / window_days`` and published as a month of money nobody ever spent.

This module is neither. It adds NO parameter to the rollup (that retirement's
``TypeError`` guard still holds literally), computes nothing inside it, and
introduces no second aggregate, no second key and no forward figure. A relearn
cluster arrives at the rollup the way every other analyzer's finding already
does: as an ordinary row carrying the one canonical field
(``past_overspend_usd``/``past_overspend_tokens``), earning an ordinary
``by_analyzer`` entry, summed by the same code path as everything else. The
retired docstring's own "or not at all" clause is what changes; the shape it
forbade does not come back.

THE FIGURE IS A FILTER, NEVER A RESCALE. The contribution is the DETECTOR's own
bounded trailing-window bucket for the cluster (``core/optimize/
relearn_window.py``), which is the same observed occurrences at the same price,
date-filtered and capped at the unbounded figure. Nothing here multiplies,
paces or projects anything, and nothing here recomputes a dollar figure: this
module only SELECTS a precomputed bucket, subtracts a component of it, and
labels the result.

NET OF THE RE-READ COMPONENT, BECAUSE resend ALREADY PRICES IT. Relearn's
``past_overspend_usd`` contains a ``past_reread_*`` COMPONENT (not an addend):
what the failure text cost on every later call that still carried it. That is
re-sent context, which is exactly the quantity ``analyzers/context_resend.py``
prices in full, and relearn's own basis string says the two must never be added
together. The resend proposal is already in the headline, so relearn's
contribution is its bounded figure MINUS its bounded re-read share. The union is
then complete with nothing doubled: the head term arrives here, the re-read term
arrives inside resend's proposal.

AN EXACTLY MATCHING WINDOW, OR NO CONTRIBUTION AT ALL. The headline publishes a
``window_days`` label, and this codebase treats a mixed basis as a first-order
bug, so a contribution may only be added under a bucket whose span EQUALS that
label. There is no nearest-match fallback here (unlike a reader's own ``since``
on ``/relearn/proposals``, where the applied label is reported beside the
figures): the headline names one window for every row it covers, so a bucket
computed for another span cannot enter it. When no exact bucket exists (a cache
written before windowing, or occurrences with no parseable timestamp), the
contribution is UNKNOWN. Unknown is never zero: the money is disclosed through
the rollup's ``excluded`` channel instead, which states waste the headline did
not sum in rather than dropping it silently.

THE UNBOUNDED FIELDS ARE NOT TOUCHED. ``past_overspend_usd``/``_tokens`` on the
cluster stay exactly as the detector wrote them. They are the write budget's
pre-net gross (``core/optimize/write_budget.py``), and shrinking them in place
would silently flip clusters between "worth a permanent rule" and net-negative.
The contribution is a new, separately named field beside them.

ONE FIGURE FOR THE FLOOR, THE TAIL AND THE HEADLINE. Every row gets
``inbox_contribution_usd`` stamped on it, cost and relearn alike, so the noise
floor, the collapsed tail's combined figure and the headline are the same
quantity summed over the same population by construction rather than by
coincidence. A row whose contribution is ``None`` is UNPRICED, not cheap: the
floor may not hide it and no combined figure may include it.
"""
from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

from tokenjam.core.optimize import cost_proposals as _cost_proposals_mod
from tokenjam.core.optimize.relearn_window import window_days

#: The analyzer name a relearn contribution row carries into the rollup's
#: ``by_analyzer`` breakdown, so its share of the total is attributable rather
#: than an unexplained delta.
RELEARN_ANALYZER = "relearn"

#: Contribution rows are internal to the aggregate (never served), and the
#: rollup deduplicates by signature, so they are namespaced: a relearn cluster
#: signature can never collide with a cost proposal's and silently displace it.
_ROW_SIGNATURE_PREFIX = "relearn-window:"

COST_CONTRIBUTION_BASIS = (
    "this proposal's own past_overspend_usd, which is already the avoidable "
    "figure observed over the window the headline names. Nothing is adjusted "
    "here: a cost proposal's figure and its contribution to the headline are "
    "the same number"
)

RELEARN_CONTRIBUTION_BASIS = (
    "this cluster's observed cost filtered to the headline's own trailing "
    "window (the detector's bounded bucket: the same occurrences at the same "
    "price, never a rescale), minus its measured re-read share. The re-read "
    "share is subtracted because it is re-sent context, which the context "
    "re-send proposal already prices in full; counting it in both places would "
    "bill the same tokens twice. The cluster's unbounded past_overspend_usd is "
    "unchanged and still reported beside this"
)

RELEARN_CONTRIBUTION_UNKNOWN_BASIS = (
    "unknown, not zero. This cluster has no bounded figure for the window the "
    "headline names, so its money cannot be added on the headline's basis. It "
    "is disclosed as excluded from the total rather than counted as nothing"
)

NO_BOUNDED_WINDOW_REASON = (
    "relearn's clusters carry no bounded figure for this window, so their "
    "money cannot be added to a total labelled with it. Either this cached "
    "result predates bounded window figures or none of the occurrences carry a "
    "parseable timestamp. Refresh the relearn proposals to fold this in"
)

_EXCLUDED_BASIS = (
    "observed over relearn's full history, not over the headline's window: it "
    "is stated here precisely because it could not be put on the headline's "
    "basis, and it is summed into no total on this block"
)


def headline_window_days(cached: Any) -> int:
    """The window the Review inbox headline is LABELLED with.

    Read from the cost block's own ``cost_window_days`` (the window the
    stored cost figures were observed over), which is where the resolved
    analysis span lands — so the headline follows the span the user chose with
    no derivation of its own. The constant is reached only when a cache predates
    the key or carries a zero, and labels a figure nobody can now attribute to a
    window. EVERY surface
    that builds the headline resolves it through this one function -- the
    web ``/relearn/cost-proposals`` and ``/relearn/proposals`` routes and the
    CLI's ``tj relearn cost-proposals`` alike -- so a row can never publish a
    contribution a headline built elsewhere never counted. ``cached`` is
    whatever ``relearn_store.read_cache``/``read_cost_proposals`` returned
    (both carry the same ``cost_window_days`` key, one raw and one a
    projection of it), so one helper serves every caller.
    """
    raw = cached.get("cost_window_days") if isinstance(cached, dict) else None
    try:
        days = int(raw or 0)
    except (TypeError, ValueError):
        days = 0
    return days or _cost_proposals_mod.FALLBACK_COST_WINDOW_DAYS


def exact_window_label(
    days: float | int | None, available: Sequence[str],
) -> str | None:
    """The available window label whose span EXACTLY equals ``days``.

    ``None`` when there is no exact match, which is the honest answer rather
    than the nearest one: see this module's docstring on why the headline takes
    no nearest-match fallback. A malformed label in ``available`` is skipped
    instead of raising, so one bad cache key cannot sink the whole aggregate.
    """
    if not days or not available:
        return None
    for label in available:
        try:
            if window_days(label) == float(days):
                return label
        except (TypeError, ValueError):
            continue
    return None


def contribution_window_label(finding: Any, days: float | int | None) -> str | None:
    """The bucket on ``finding`` that matches the headline's window, if any."""
    if not isinstance(finding, Mapping):
        return None
    available = list((finding.get("past_overspend_windows") or {}))
    return exact_window_label(days, available)


def _bucket(cluster: Mapping[str, Any], label: str) -> Mapping[str, Any] | None:
    windows = cluster.get("past_overspend_windows")
    if not isinstance(windows, Mapping):
        return None
    bucket = windows.get(label)
    return bucket if isinstance(bucket, Mapping) else None


def _net_of_reread(bucket: Mapping[str, Any]) -> dict[str, Any] | None:
    """One bounded bucket's figure MINUS its measured re-read share, or ``None``.

    The netting rule, in exactly one place, because two surfaces publish it: the
    Review inbox row's contribution to the headline, and the Dashboard's relearn
    waste tile (via :func:`window_scoped_finding_figure`). Both sit in a set
    whose other members include the context re-send proposal, which prices
    re-sent context in full — so both have to subtract the same component or
    whichever one forgot bills the same tokens twice.

    ``None`` means UNKNOWN. An unpriced bucket cannot be netted, and treating an
    unknown re-read share as zero would publish a figure that double-counts
    against resend. Unknown propagates rather than degrading to a number.
    """
    gross_usd = bucket.get("past_overspend_usd")
    reread_usd = bucket.get("past_reread_usd")
    if gross_usd is None or reread_usd is None:
        return None
    gross_tokens = int(bucket.get("past_overspend_tokens") or 0)
    reread_tokens = int(bucket.get("past_reread_tokens") or 0)
    return {
        "usd": round(max(float(gross_usd) - float(reread_usd), 0.0), 6),
        "tokens": max(gross_tokens - reread_tokens, 0),
        "gross_usd": float(gross_usd),
        "reread_usd": float(reread_usd),
    }


def window_scoped_finding_figure(
    finding: Any, *, days: float | int | None,
    applied_signatures: Iterable[str] = (),
) -> dict[str, Any] | None:
    """Relearn's FINDING-level past overspend, on the window ``days`` names.

    THE DEFECT THIS EXISTS FOR. ``RelearnFinding.past_overspend_usd`` is
    deliberately UNBOUNDED — relearn's whole signal is recurrence across
    history, so ``run(ctx)`` does not forward the report's ``since`` and the
    field covers everything the detector retained. That is correct for the field
    and wrong for the Dashboard's recoverable-waste row, which rendered it as a
    peer beside five window-scoped tiles with nothing saying it was on another
    footing. Measured on a real corpus the tile read $386.64 (all history) while
    the Review inbox published $260.21 for the same analyzer over the same 30
    days, and the correctly-bounded figure was already sitting unused on the
    same payload.

    So this SELECTS the finding's own precomputed bucket for that window and
    nets it exactly the way an inbox row is netted. Nothing is rescaled, paced
    or projected — same filter, same price, same subtraction. ``None`` when no
    bucket EXACTLY matches (see this module's docstring on why there is no
    nearest-match fallback): a surface must then say it has no figure on this
    window's basis, never fall back to the unbounded one.

    **SAME POPULATION AS THE INBOX, NOT JUST THE SAME WINDOW.** The
    finding-level bucket covers EVERY cluster the detector retained, including
    ones the user has already applied a fix for; the Review inbox sums the
    OPEN clusters, because a headline answers what is still outstanding. Same
    window, same netting, different population — so the two disagreed by
    exactly the applied clusters' worth, and the Dashboard went on claiming
    money the user had already recovered (apply a fix, watch the tile not
    move). ``applied_signatures`` closes that: the figure is summed from the
    open clusters through ``relearn_contribution_rows`` — the inbox's OWN
    derivation, not a second one that agrees today — so the two surfaces are
    equal by construction. The bucket still decides whether this window has a
    basis at all, and still supplies the window labels.

    Passing no signatures keeps the whole-population figure, which is correct
    for a caller that genuinely wants every cluster (nothing applied is the
    normal case, and then the two are identical anyway).

    Immutable: reads the finding, returns a new dict, never writes to it.
    """
    if not isinstance(finding, Mapping):
        return None
    totals = finding.get("past_overspend_windows")
    if not isinstance(totals, Mapping):
        return None
    label = exact_window_label(days, list(totals))
    if not label:
        return None
    bucket = totals.get(label)
    if not isinstance(bucket, Mapping):
        return None
    netted = _net_of_reread(bucket)
    if netted is None:
        return None
    applied = {str(s) for s in applied_signatures}
    if applied:
        rows = relearn_contribution_rows(
            finding, label=label, applied_signatures=applied,
        )
        netted = {
            **netted,
            "usd": sum((r.get("past_overspend_usd") or 0.0) for r in rows),
            "tokens": sum((r.get("past_overspend_tokens") or 0) for r in rows),
        }
    return {
        **netted,
        "window": label,
        "window_days": bucket.get("window_days"),
        "basis": RELEARN_CONTRIBUTION_BASIS,
    }


def relearn_contribution(
    cluster: Mapping[str, Any], *, label: str | None,
) -> dict[str, Any] | None:
    """One relearn cluster's contribution to the headline, or ``None``.

    ``None`` means UNKNOWN and every caller must treat it that way: no bounded
    bucket for this window, or a bucket the detector could not price. A cluster
    whose bucket holds no occurrences inside the window DOES contribute, as a
    known ``0.0``: it genuinely spent nothing in the window, which is a
    different fact from "we could not tell".

    Immutable: reads the cluster, returns a new dict, never writes to it.
    """
    if not label:
        return None
    bucket = _bucket(cluster, label)
    if bucket is None:
        return None
    netted = _net_of_reread(bucket)
    if netted is None:
        return None
    return {
        **netted,
        "window": label,
        "window_days": bucket.get("window_days"),
        "basis": RELEARN_CONTRIBUTION_BASIS,
    }


def stamp_relearn_contribution(
    cluster: Mapping[str, Any], *, label: str | None,
) -> dict[str, Any]:
    """A copy of ``cluster`` carrying its ``inbox_contribution_*`` fields.

    Always PRESENT, even when unknown: a renderer that indexes the field must
    not blow up on an older row, and an explicit ``None`` reads as "not
    measured on this basis" rather than "worth nothing".
    """
    contribution = relearn_contribution(cluster, label=label)
    if contribution is None:
        return {
            **cluster,
            "inbox_contribution_usd": None,
            "inbox_contribution_tokens": None,
            "inbox_contribution_window": label,
            "inbox_contribution_basis": RELEARN_CONTRIBUTION_UNKNOWN_BASIS,
        }
    return {
        **cluster,
        "inbox_contribution_usd": contribution["usd"],
        "inbox_contribution_tokens": contribution["tokens"],
        "inbox_contribution_window": contribution["window"],
        "inbox_contribution_basis": contribution["basis"],
    }


def stamp_relearn_contributions(finding: Any, *, label: str | None) -> Any:
    """A copy of ``finding`` with every cluster's contribution stamped on it.

    The stamped figure is the HEADLINE's window, which is deliberately NOT the
    window a caller's own ``since`` selected: the floor, the tail sum and the
    headline have to read one quantity, and the headline names one window for
    every row. A reader's selected window still travels separately, on each
    row's ``window`` bucket.
    """
    if not isinstance(finding, Mapping):
        return finding
    clusters = finding.get("clusters")
    if not isinstance(clusters, list):
        return finding
    return {
        **finding,
        "clusters": [
            stamp_relearn_contribution(c, label=label) if isinstance(c, Mapping) else c
            for c in clusters
        ],
    }


def stamp_cost_contribution(
    proposal: Mapping[str, Any], *, window: str | None = None,
) -> dict[str, Any]:
    """A copy of a cost proposal carrying the same ``inbox_contribution_*``
    fields, so the UI reads ONE field across both feeds and cannot pick the
    wrong one per row kind. A cost proposal's figure is already the avoidable
    observation over the headline's window, so the contribution is that figure
    unchanged.

    ``window`` is the headline's own window label, carried on the row for the
    same reason relearn's is: a reader (or a test) can see that both feeds
    published their contribution under ONE label rather than trusting it.
    """
    usd = proposal.get("past_overspend_usd")
    tokens = proposal.get("past_overspend_tokens")
    return {
        **proposal,
        "inbox_contribution_usd": None if usd is None else float(usd),
        "inbox_contribution_tokens": None if tokens is None else int(tokens),
        "inbox_contribution_window": window,
        "inbox_contribution_basis": COST_CONTRIBUTION_BASIS,
    }


def relearn_contribution_rows(
    finding: Any,
    *,
    label: str | None,
    applied_signatures: Iterable[str] = (),
) -> list[dict[str, Any]]:
    """Relearn's clusters as ordinary rollup rows on the canonical field.

    One row per OPEN cluster that has a known contribution. An already-applied
    cluster is left out for the same reason an applied cost proposal is: the
    headline is what is still outstanding. A cluster with an unknown
    contribution produces no row at all rather than a zero one; the caller
    discloses it through ``excluded`` (see ``relearn_excluded_entry``).

    These rows never leave the aggregate. They are the rollup's input, not
    payload: the row a reader sees is the cluster itself, stamped by
    ``stamp_relearn_contributions`` with the same figure.
    """
    if not isinstance(finding, Mapping):
        return []
    clusters = finding.get("clusters")
    if not isinstance(clusters, list):
        return []
    applied = {str(s) for s in applied_signatures}
    rows: list[dict[str, Any]] = []
    for cluster in clusters:
        if not isinstance(cluster, Mapping):
            continue
        signature = str(cluster.get("signature") or "")
        if not signature or signature in applied:
            continue
        contribution = relearn_contribution(cluster, label=label)
        if contribution is None:
            continue
        rows.append({
            "signature": f"{_ROW_SIGNATURE_PREFIX}{signature}",
            "analyzer": RELEARN_ANALYZER,
            "title": str(cluster.get("title") or ""),
            "past_overspend_usd": contribution["usd"],
            "past_overspend_tokens": contribution["tokens"],
            "past_overspend_basis": contribution["basis"],
        })
    return rows


def unrepresented_relearn(
    finding: Any,
    *,
    label: str | None,
    applied_signatures: Iterable[str] = (),
) -> dict[str, Any]:
    """The open clusters whose money could NOT be put on the headline's basis.

    Returns ``{"clusters": int, "past_overspend_usd": float | None,
    "past_overspend_tokens": int}`` over exactly those clusters, on their own
    unbounded basis (the only basis they have). ``past_overspend_usd`` is
    ``None`` when not one of them carries a priced figure: absent, not zero.
    """
    clusters = (finding or {}).get("clusters") if isinstance(finding, Mapping) else None
    if not isinstance(clusters, list):
        return {"clusters": 0, "past_overspend_usd": None, "past_overspend_tokens": 0}
    applied = {str(s) for s in applied_signatures}
    count = 0
    usd = 0.0
    priced = False
    tokens = 0
    for cluster in clusters:
        if not isinstance(cluster, Mapping):
            continue
        signature = str(cluster.get("signature") or "")
        if not signature or signature in applied:
            continue
        if relearn_contribution(cluster, label=label) is not None:
            continue
        count += 1
        tokens += int(cluster.get("past_overspend_tokens") or 0)
        cluster_usd = cluster.get("past_overspend_usd")
        if cluster_usd is not None:
            usd += float(cluster_usd)
            priced = True
    return {
        "clusters": count,
        "past_overspend_usd": round(usd, 6) if priced else None,
        "past_overspend_tokens": tokens,
    }


def relearn_excluded_entry(
    unrepresented: Mapping[str, Any], *, reason: str,
) -> dict[str, dict[str, Any]]:
    """The rollup ``excluded`` entry for relearn money the headline omitted.

    ``excluded`` is the existing channel for waste a caller deliberately did
    NOT sum in: carried on the block unchanged so a surface can state it
    instead of silently dropping a real figure. Empty (no entry at all) when
    every open cluster contributed, which is the normal case.

    ``label`` and ``note`` are for the renderer: the existing excluded note was
    written for one occupant and defaults its wording to that analyzer's name and
    its "over the same window, on its own review surface" framing, both of which
    are wrong for this entry. The figure here is NOT on the headline's window
    (that is precisely why it is excluded) and the rows it covers are in THIS
    surface, not another one. ``note`` carries the sentence a renderer should
    use instead of assuming either.
    """
    if not unrepresented.get("clusters"):
        return {}
    count = int(unrepresented.get("clusters") or 0)
    return {
        RELEARN_ANALYZER: {
            "past_overspend_usd": unrepresented.get("past_overspend_usd"),
            "past_overspend_tokens": unrepresented.get("past_overspend_tokens"),
            "clusters": count,
            "label": "recurring mistakes",
            "note": (
                f"{count} recurring-mistake row(s) below could not be put on this "
                f"total's window, so their money is stated here and summed into "
                f"nothing. It is their full-history figure, not this window's."
            ),
            "reason": reason,
            "basis": _EXCLUDED_BASIS,
            "href": "#/review",
        },
    }
