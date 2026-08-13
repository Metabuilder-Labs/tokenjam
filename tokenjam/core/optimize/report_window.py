"""The ONE trailing window every past-overspend surface publishes.

THE DEFECT THIS EXISTS FOR. Two surfaces published the same metric
(``past_overspend_usd``) under two independently-derived windows, and neither
said so where the figure was read. The Dashboard's recoverable-waste tiles came
off the stored analyzer report, whose window was ``[optimize]
scan_window_days`` — a fixed config int. The Review inbox's headline came off
the cost-proposal store, whose window was the resolved analysis span
(``core/analysis_span.window_days_for``: the span chosen at onboarding, bounded
by how much history the store holds). On a real corpus those were 30 and 69,
so the six tiles summed to roughly three and a half thousand dollars while the
inbox headline — read minutes later, on the same corpus, under the same
label — said nearly five. Nothing was miscomputed. The two surfaces were
answering different questions while looking like they answered one.

THE RESOLUTION IS ONE SEAM, NOT TWO CORRECT DERIVATIONS. Both callers now
resolve through :func:`report_window_days`, so the two windows cannot drift
apart again by construction rather than by two docstrings agreeing to be
careful. The window is:

  ``min(scan_window_days, the chosen analysis span, the history actually held)``

Each bound answers a different question and the honest answer is the smallest:

  * ``[optimize] scan_window_days`` is the PRODUCT choice — how far back the
    waste figures look. It is the knob, and its default is what a user is
    shown unless they say otherwise. The Review inbox does not expose a window
    picker at all, so this is the only thing that decides its window;
  * the chosen analysis span (``storage.analysis_span``) bounds it because
    retention is derived from that span — a figure may not claim a window whose
    data the retention job is entitled to have deleted;
  * the measured history bounds it because a 30-day claim over a store that has
    been running a week is answerable for a week.

WHY THE SPAN NO LONGER WIDENS THE WINDOW. It used to, on the cost side alone:
``analysis_span = "90d"`` over a 69-day store produced a 69-day inbox headline
the reader could neither see the provenance of nor change. The span's job is to
bound what may be claimed, not to decide what IS claimed — the analyzers' own
look-back is ``scan_window_days``. A user who wants the waste figures to cover
more history raises that knob; raising the span alone now widens only what is
kept and what may be asked for.

NOT A WINDOW PICKER. The Dashboard's own ``since`` selector re-scopes the spend
chart and the KPI row, which are live queries. It does not re-scope the waste
tiles and must not be read as doing so: those come from a stored analyzer pass
that ran once, at this window, in the background (no route runs an analyzer —
see ``api/routes/optimize.py``). Every surface built on this window states it
beside the figure for exactly that reason.
"""
from __future__ import annotations

from typing import Any

from tokenjam.core.analysis_span import analysis_span_days, days_of_history

#: LAST RESORT ONLY — the look-back to use when the config carries no usable
#: ``scan_window_days``. It is not "what a normal run uses"
#: (:func:`report_window_days` is that); it exists because a window has to be a
#: number of days to subtract from now, and this is the smallest honest claim.
FALLBACK_WINDOW_DAYS = 30


def configured_scan_window_days(config: Any) -> int:
    """``[optimize] scan_window_days``, or the fallback when it is unusable.

    Unusable means absent, unparseable or non-positive. A zero here would make
    every window-scoped figure cover nothing at all and read as "no waste", so
    it resolves to the fallback rather than being honoured.
    """
    optimize = getattr(config, "optimize", None)
    try:
        days = int(getattr(optimize, "scan_window_days", FALLBACK_WINDOW_DAYS))
    except (TypeError, ValueError):
        days = 0
    return days if days > 0 else FALLBACK_WINDOW_DAYS


def report_window_days(config: Any, conn: Any) -> int:
    """The trailing window, in days, that past-overspend figures are observed
    over on EVERY surface that publishes them.

    ``conn`` may be ``None`` (a proxy backend, or a caller with no direct
    connection): an unmeasurable history cannot narrow anything, so the choice
    stands unbounded by it — the same distinction ``core/data_span`` makes, and
    for the same reason. Never returns less than 1: a zero-day window is not a
    smaller claim, it is an empty one.
    """
    days = configured_scan_window_days(config)
    try:
        span = analysis_span_days(getattr(config, "storage", None))
    except ValueError:
        # A span nobody can parse must not sink the scan that is about to run.
        # It cannot narrow the window either, so the configured look-back
        # stands; the malformed value is reported where it can be acted on
        # (`tj doctor`, onboarding), not by silently shrinking a figure here.
        span = None
    if span is not None:
        days = min(days, int(span))
    held = days_of_history(conn)
    if held is not None:
        days = min(days, int(held))
    return max(1, days)


def report_window_label(config: Any, conn: Any) -> str:
    """:func:`report_window_days` as the label surfaces publish it.

    Relearn precomputes its bounded figures against a vocabulary of labels
    while it still holds the per-occurrence dates, and the Review inbox matches
    its headline window against that vocabulary EXACTLY (no nearest-match — see
    ``core/optimize/inbox_contribution``). So the label this returns has to be
    in relearn's vocabulary or every cluster falls out of the headline into the
    excluded channel. Both sides therefore resolve it here.
    """
    return f"{report_window_days(config, conn)}d"
