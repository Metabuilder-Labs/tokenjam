"""The single 30-day projection basis every surface shares.

A window-scoped observation ("this wasted $119 over the last 30 days") and a
per-session standing cost ("this CLAUDE.md rule costs 180 tokens on every
future session") are only comparable once both sit on the SAME horizon. This
module owns that horizon and is the only place in the tree allowed to define
it. Two consumers share it and neither may re-derive it:

* the recoverable-savings rollup, which rescales each analyzer's
  window-scoped observation onto the 30-day display basis;
* :mod:`tokenjam.core.optimize.write_budget`, which needs "how many sessions
  will re-send this artifact in a month" before it can price a permanent
  CLAUDE.md rule.

**The basis.** Project to a 30-day month at the user's OWN pace, measured from
their own data::

    D_active = distinct days in the window with at least one session
    r        = clamp(30 / D_active, 1.0, 3.0)
    display  = observed * r                      # labelled "per 30 days"

Stretching the *time axis* to an arbitrary horizon (a 90-day multiplier over a
30-day window) is un-provable and the first thing a skeptical user attacks.
Deriving a heavier *session pace* from the user's own active days is not: every
input is a number they can count themselves.

**Guardrails.** Thin data cannot support a projection at all, so ``r`` is
forced to 1.0 and ``projected`` is False unless ALL of::

    window_days >= 14   AND   active_days >= 8   AND   sessions >= 20

``r`` is capped at 3.0 always. The 1.0 floor applies only when the window is at
most 30 days long: over a LONGER window ``r = 30 / D_active`` may legitimately
fall below 1.0, which is normalising down to a 30-day figure, not inflating.

**Forbidden**, because it crosses from defensible into un-provable: reverse-
fitting a factor to reproduce a previously-shown number; applying ``r`` to a
one-time saving; a peak-N-day or p90-daily base.
"""
from __future__ import annotations

from dataclasses import dataclass

#: The horizon every projected figure is expressed on. A month is the unit
#: users already reason in (a billing cycle), so no other horizon is offered.
PROJECTION_HORIZON_DAYS = 30.0

#: Hard cap on the pace multiplier. A user cannot be assumed to sustain more
#: than 3x their observed window pace, however few active days they had.
MAX_PROJECTION_RATIO = 3.0
#: Floor, applied ONLY to windows of at most the horizon (see module docstring).
MIN_PROJECTION_RATIO = 1.0

#: Projection guardrails: below any of these the data cannot carry a
#: projection and the observed figure is shown as-is.
MIN_WINDOW_DAYS_FOR_PROJECTION = 14.0
MIN_ACTIVE_DAYS_FOR_PROJECTION = 8
MIN_SESSIONS_FOR_PROJECTION = 20


@dataclass(frozen=True)
class ProjectionBasis:
    """One window's projection inputs, its ratio, and the disclosure line.

    ``projected`` is False when the guardrails suppressed the projection; the
    ratio is then exactly 1.0 so a caller can always multiply unconditionally
    and still be honest. ``disclosure`` is the user-facing sentence naming
    every input the ratio came from, so the derivation is never a black box.
    """

    window_days: float
    active_days: int
    sessions: int
    ratio: float
    projected: bool
    disclosure: str

    @property
    def projected_sessions(self) -> float:
        """Sessions expected across the 30-day horizon at this pace.

        This is the multiplier a permanent artifact's per-session cost is
        charged against: a rule written today is re-sent on every one of
        these. Falls back to the observed session count when the guardrails
        suppressed the projection, which is the conservative direction (a
        smaller session count charges a permanent rule LESS, so a suppressed
        projection can never manufacture a net-negative verdict).
        """
        return self.sessions * self.ratio


def build_projection_basis(
    window_days: float, active_days: int, sessions: int,
) -> ProjectionBasis:
    """Resolve the 30-day pace ratio for one window. Never raises.

    Degenerate inputs (zero/negative days, no sessions) resolve to an
    unprojected basis with ``ratio == 1.0`` rather than a division error or an
    invented multiplier: a projection is never derived from missing data.
    """
    window_days = max(float(window_days or 0.0), 0.0)
    active_days = max(int(active_days or 0), 0)
    sessions = max(int(sessions or 0), 0)

    eligible = (
        window_days >= MIN_WINDOW_DAYS_FOR_PROJECTION
        and active_days >= MIN_ACTIVE_DAYS_FOR_PROJECTION
        and sessions >= MIN_SESSIONS_FOR_PROJECTION
    )
    if not eligible or active_days <= 0:
        return ProjectionBasis(
            window_days=window_days, active_days=active_days, sessions=sessions,
            ratio=1.0, projected=False,
            disclosure=_unprojected_disclosure(window_days),
        )

    ratio = PROJECTION_HORIZON_DAYS / active_days
    if window_days <= PROJECTION_HORIZON_DAYS:
        ratio = max(ratio, MIN_PROJECTION_RATIO)
    ratio = min(ratio, MAX_PROJECTION_RATIO)
    return ProjectionBasis(
        window_days=window_days, active_days=active_days, sessions=sessions,
        ratio=ratio, projected=True,
        disclosure=_projected_disclosure(window_days, active_days, sessions),
    )


def _projected_disclosure(window_days: float, active_days: int, sessions: int) -> str:
    per_day = sessions / active_days if active_days > 0 else 0.0
    return (
        f"Projected to a {PROJECTION_HORIZON_DAYS:.0f}-day month at your own pace: "
        f"{per_day:.1f} sessions per active coding day, measured from {sessions} "
        f"sessions across {active_days} active days in the last "
        f"{window_days:.0f} days."
    )


def _unprojected_disclosure(window_days: float) -> str:
    return (
        f"Observed in the last {window_days:.0f} days, not projected: a "
        f"{PROJECTION_HORIZON_DAYS:.0f}-day projection needs at least "
        f"{MIN_WINDOW_DAYS_FOR_PROJECTION:.0f} days, "
        f"{MIN_ACTIVE_DAYS_FOR_PROJECTION} active days and "
        f"{MIN_SESSIONS_FOR_PROJECTION} sessions of history."
    )
