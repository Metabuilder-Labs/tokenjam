"""How many days of data this store actually holds.

Every window-scoped surface eventually needs to know whether the window it is
offering has any data behind it: a 90-day figure over a store that has been
running for a week is not wrong so much as unanswerable, and a selector that
offers 90d there invites a reader to conclude "nothing happened" from "nothing
was kept".

THE NAIVE MEASURE IS WRONG HERE, AND IT IS WRONG BY A FACTOR OF THIRTY-SIX.
``max(ts) - min(ts)`` looks like the answer and is not. Ingest writes sentinel
timestamps for rows that carry no usable time (the same ``1970-01-01`` zero
epoch ``analyzers/relearn._parse_failure_ts`` already had to defend against),
and a single fixture or import row dated years before the store existed drags
the minimum back with it. Measured on a real local corpus whose usable history
is about two months, the naive span reads in the thousands of days because of
ONE row. A measure that any single row can move by that much is not a measure.

So two robust quantities are reported instead, and neither can be moved far by
one outlier:

  * ``available_days`` -- the most recent CONTIGUOUS block of days that carry
    data, walking backward from the newest and stopping at the first gap wider
    than ``MAX_GAP_DAYS``. This is the "how far back can I actually ask?"
    number. An ancient outlier sits behind a gap of years, so it is excluded by
    construction rather than by a hand-tuned threshold on the answer.
  * ``days_with_data`` -- how many distinct calendar days carry data at all.
    Independent of contiguity, and an outlier can add at most one to it.

Both are ``None``/0 rather than 0/0 when the store holds no dated rows: an
unknown span is not a zero-day span.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Any, Iterable

#: Anything stamped before this year is a SENTINEL, not an observation. Same
#: rule and same reason as ``analyzers/relearn.MIN_PLAUSIBLE_TS_YEAR``: ingest
#: writes a zero epoch when a row carries no usable timestamp.
MIN_PLAUSIBLE_YEAR = 2000

#: How wide a hole in the calendar the recent block tolerates before it is
#: treated as the edge of the data rather than a quiet week. A week of silence
#: is a holiday; a year of silence is a different corpus.
MAX_GAP_DAYS = 7


@dataclass(frozen=True)
class DataSpan:
    """What the store can actually answer questions about.

    ``available_days`` is ``None`` when nothing dated could be found, never 0:
    "we do not know how much history there is" and "there is none" are
    different answers and only one of them is safe to clamp a selector against.
    """
    available_days:            int | None
    days_with_data:            int
    newest:                    str | None
    oldest_in_block:           str | None
    ignored_days_before_block: int
    basis:                     str
    #: The OLDEST PLAUSIBLE day carrying data, ISO, ``None`` when nothing dated
    #: was found. Behind a gap, and so deliberately NOT ``oldest_in_block``.
    #:
    #: This is the far edge of the measure the module docstring calls wrong for
    #: "how far back can I ask", and it is the right one for a different
    #: question: "how far back must an analyzer look to see everything it is
    #: entitled to". Bounding a query by ``available_days`` would discard
    #: everything behind a gap — a fortnight away would silently delete the
    #: quarter before it from a past-tense figure — whereas widening a query
    #: past the data costs nothing, since rows that are not there return nothing
    #: either way. So the two are not interchangeable and neither replaces the
    #: other: clamp a SELECTOR against ``available_days``, size a LOOKBACK
    #: against this.
    #:
    #: Only safe to expose because ingest no longer ADMITS an epoch sentinel: a
    #: record with no observed time is rejected at the boundary rather than
    #: stored with a made-up one. A single 1970 row used to move this by
    #: decades, which is what the contiguous-block measure was introduced to
    #: survive; the plausible-year floor below remains the backstop for rows an
    #: older build already wrote (and `tj doctor --repair` removes those).
    oldest:                    str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "available_days": self.available_days,
            "days_with_data": self.days_with_data,
            "newest": self.newest,
            "oldest_in_block": self.oldest_in_block,
            "ignored_days_before_block": self.ignored_days_before_block,
            "basis": self.basis,
            "oldest": self.oldest,
        }


_UNKNOWN_BASIS = (
    "no dated rows were found, so the available span is unknown rather than "
    "zero. Nothing may be clamped against this"
)


def _plausible_days(days: Iterable[Any]) -> list[date]:
    """The input coerced to dates, with sentinels and future stamps dropped.

    A future date is as much a data error as a 1970 one and would make the
    newest end of the block a day nothing was observed on.
    """
    today = datetime.now(tz=timezone.utc).date()
    out: set[date] = set()
    for raw in days:
        day = _as_date(raw)
        if day is None:
            continue
        if day.year < MIN_PLAUSIBLE_YEAR or day > today:
            continue
        out.add(day)
    return sorted(out)


def _as_date(raw: Any) -> date | None:
    """Best-effort date out of a date, datetime or ISO-ish string. ``None`` on
    anything else, never an exception: this runs over whatever the store holds."""
    if isinstance(raw, datetime):
        return raw.date()
    if isinstance(raw, date):
        return raw
    if not raw:
        return None
    try:
        return datetime.fromisoformat(str(raw).replace("Z", "+00:00")).date()
    except (TypeError, ValueError):
        return None


def data_span_from_days(
    days: Iterable[Any], *, max_gap_days: int = MAX_GAP_DAYS,
) -> DataSpan:
    """The robust span over an explicit set of days-with-data.

    Pure, so the measure can be tested against the exact corpus shape that
    breaks the naive one (an ancient outlier plus a recent contiguous block)
    without needing a database.
    """
    plausible = _plausible_days(days)
    if not plausible:
        return DataSpan(
            available_days=None, days_with_data=0, newest=None,
            oldest_in_block=None, ignored_days_before_block=0,
            basis=_UNKNOWN_BASIS, oldest=None,
        )

    newest = plausible[-1]
    oldest_in_block = newest
    index = len(plausible) - 1
    while index > 0:
        gap = (plausible[index] - plausible[index - 1]).days
        if gap > max_gap_days:
            break
        index -= 1
        oldest_in_block = plausible[index]

    return DataSpan(
        available_days=(newest - oldest_in_block).days + 1,
        days_with_data=len(plausible),
        newest=newest.isoformat(),
        oldest_in_block=oldest_in_block.isoformat(),
        ignored_days_before_block=index,
        basis=(
            f"the most recent unbroken run of days carrying data "
            f"({oldest_in_block.isoformat()} to {newest.isoformat()}), stopping "
            f"at the first gap wider than {max_gap_days} days. Deliberately not "
            f"newest minus oldest: one sentinel or back-imported row would "
            f"stretch that by years, and {index} dated day(s) sit behind such a "
            f"gap here. days_with_data counts every distinct day instead, which "
            f"one outlier can move by at most one"
        ),
        oldest=plausible[0].isoformat(),
    )


def _distinct_days(conn: Any, sql: str) -> list[Any]:
    """One best-effort day query. A missing table or an unreadable column
    contributes nothing rather than failing the whole measurement."""
    try:
        return [row[0] for row in conn.execute(sql).fetchall()]
    except Exception:
        return []


def available_data_span(
    conn: Any | None, *, max_gap_days: int = MAX_GAP_DAYS,
) -> DataSpan:
    """The robust span over everything the store holds.

    Unions the days present in ``spans`` and in ``sessions``: a session with no
    spans and a span with no session row are both real evidence that a day
    carried data, and asking only one table would under-report a partial
    ingest. ``None``/unreadable degrades to the unknown span, never to zero.
    """
    if conn is None:
        return data_span_from_days([], max_gap_days=max_gap_days)
    # `AT TIME ZONE 'UTC'` before the cast is load-bearing, not decoration
    # (Critical Rule 1): DuckDB resolves a bare `CAST(TIMESTAMPTZ AS DATE)`
    # through the session timezone, so on a machine running ahead of UTC the
    # newest rows come back stamped with TOMORROW's date — and `_plausible_days`
    # then drops them as future. The whole of today would vanish from the span
    # for the hours the local date leads, and reappear at local midnight.
    days = _distinct_days(
        conn,
        "SELECT DISTINCT CAST(start_time AT TIME ZONE 'UTC' AS DATE) FROM spans "
        "WHERE start_time IS NOT NULL",
    ) + _distinct_days(
        conn,
        "SELECT DISTINCT CAST(started_at AT TIME ZONE 'UTC' AS DATE) FROM sessions "
        "WHERE started_at IS NOT NULL",
    )
    return data_span_from_days(days, max_gap_days=max_gap_days)
