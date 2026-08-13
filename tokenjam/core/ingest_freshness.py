"""Shared corpus-freshness math for `tj doctor` and `tj status`.

Both surfaces answer "how long since anything was ingested" — doctor's
pass/fail check and status's advisory nudge — against the same threshold,
derived from the daemon's own configured ingest cadence
(``[ingest] interval_minutes``) rather than each guessing its own constant,
so they can never silently disagree about what "stale" means.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

# "Stale" means the newest ingested session is older than this many multiples
# of the configured ingest interval — wide enough that a single missed cycle
# (daemon briefly restarting, a slow backfill) never fires, tight enough to
# catch a daemon that has been silently dead for a meaningful stretch. The
# floor guards a very short configured interval from producing a sub-hour
# threshold that would fire during a completely normal idle stretch.
STALENESS_INTERVAL_MULTIPLIER = 6
STALENESS_FLOOR_HOURS = 2.0


def staleness_threshold_hours(interval_minutes: int) -> float:
    """The corpus-staleness threshold, in hours, for a given ingest cadence."""
    return max(
        STALENESS_FLOOR_HOURS,
        interval_minutes * STALENESS_INTERVAL_MULTIPLIER / 60.0,
    )


@dataclass(frozen=True)
class CorpusFreshness:
    newest_session_at: datetime | None
    age_hours: float | None
    threshold_hours: float
    is_stale: bool


def corpus_freshness(conn, interval_minutes: int, *, now: datetime) -> CorpusFreshness:
    """Read the newest ``sessions.started_at`` and compare it to the
    cadence-derived threshold.

    ``now`` is injected (matches the rest of the codebase's ``utcnow()``
    seam) so callers/tests never depend on wall-clock time. Never raises on
    "no sessions yet" — that's a legitimate pre-ingest state, not staleness
    (mirrors ``_check_span_staleness``'s empty-DB handling in
    ``cmd_doctor.py``).
    """
    threshold = staleness_threshold_hours(interval_minutes)
    row = conn.execute("SELECT MAX(started_at) FROM sessions").fetchone()
    newest = row[0] if row else None
    if newest is None:
        return CorpusFreshness(None, None, threshold, False)
    if newest.tzinfo is None:
        newest = newest.replace(tzinfo=timezone.utc)
    age_hours = (now - newest).total_seconds() / 3600.0
    return CorpusFreshness(newest, age_hours, threshold, age_hours > threshold)
