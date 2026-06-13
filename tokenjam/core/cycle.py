"""
Billing-cycle bounds, shared by the budget-projection analyzer and the cost
API's run-rate caption (#138).

A "cycle" is a monthly window that begins on `cycle_start_day` of each month
(configured per-provider via `[budget.<provider>] cycle_start_day`). With
`cycle_start_day = 1` this is exactly the calendar month, which is the fallback
when no provider cycle is configured.
"""
from __future__ import annotations

from datetime import datetime


def cycle_bounds(now: datetime, start_day: int) -> tuple[datetime, datetime]:
    """
    Return (cycle_start, cycle_end) for the monthly cycle that contains `now`,
    given a cycle that begins on `start_day` of each month.

    `start_day` is clamped to 1..28 to avoid month-length edge cases (Feb).
    `cycle_end` is the *next* cycle's start (exclusive upper bound).
    """
    start_day = max(1, min(start_day, 28))
    if now.day >= start_day:
        cs = now.replace(day=start_day, hour=0, minute=0, second=0, microsecond=0)
    else:
        prev_month_year = now.year if now.month > 1 else now.year - 1
        prev_month = now.month - 1 if now.month > 1 else 12
        cs = datetime(prev_month_year, prev_month, start_day, tzinfo=now.tzinfo)
    next_month_year = cs.year + (1 if cs.month == 12 else 0)
    next_month = 1 if cs.month == 12 else cs.month + 1
    ce = datetime(next_month_year, next_month, start_day, tzinfo=cs.tzinfo)
    return cs, ce


def effective_cycle_start_day(config) -> int:
    """
    The `cycle_start_day` to use for the *global* run-rate caption (the Overview
    / Cost chart projects a single cross-provider figure, so it needs one cycle).

    Honors `[budget.<provider>] cycle_start_day` when the configured providers
    agree on a single non-default value; falls back to 1 (calendar month) when
    none are configured or they conflict — we don't guess between competing
    cycles. Per-provider projections (the budget-projection analyzer) always use
    each provider's own `cycle_start_day` and are unaffected by this resolution.
    """
    days = {b.cycle_start_day for b in config.budgets.values()}
    days.discard(1)  # default == calendar month; not a distinguishing signal
    if len(days) == 1:
        return next(iter(days))
    return 1
