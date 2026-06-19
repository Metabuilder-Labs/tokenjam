"""Billing-cycle bounds shared by budget-projection + cost run-rate (#138)."""
from __future__ import annotations

from datetime import datetime, timezone

from tokenjam.core.config import ProviderBudget, TjConfig
from tokenjam.core.cycle import cycle_bounds, effective_cycle_start_day


def _dt(y, m, d, h=0):
    return datetime(y, m, d, h, tzinfo=timezone.utc)


# --- cycle_bounds: default start_day=1 == calendar month ------------------- #
def test_cycle_bounds_calendar_month():
    cs, ce = cycle_bounds(_dt(2026, 6, 19, 12), start_day=1)
    assert cs == _dt(2026, 6, 1)
    assert ce == _dt(2026, 7, 1)


def test_cycle_bounds_calendar_month_december_wraps_year():
    cs, ce = cycle_bounds(_dt(2026, 12, 5), start_day=1)
    assert cs == _dt(2026, 12, 1)
    assert ce == _dt(2027, 1, 1)


# --- cycle_bounds: explicit cycle_start_day -------------------------------- #
def test_cycle_bounds_explicit_start_day_after():
    # now is on/after the start day -> cycle started this month.
    cs, ce = cycle_bounds(_dt(2026, 6, 19), start_day=15)
    assert cs == _dt(2026, 6, 15)
    assert ce == _dt(2026, 7, 15)


def test_cycle_bounds_explicit_start_day_before():
    # now is before the start day -> cycle started last month.
    cs, ce = cycle_bounds(_dt(2026, 6, 10), start_day=15)
    assert cs == _dt(2026, 5, 15)
    assert ce == _dt(2026, 6, 15)


def test_cycle_bounds_clamps_high_start_day():
    # start_day > 28 is clamped to 28 (avoids Feb month-length edge cases).
    # June 19 is before the 28th, so the cycle started May 28.
    cs, ce = cycle_bounds(_dt(2026, 6, 19), start_day=31)
    assert cs == _dt(2026, 5, 28)
    assert ce == _dt(2026, 6, 28)


# --- effective_cycle_start_day: config resolution -------------------------- #
def _cfg(**budgets) -> TjConfig:
    cfg = TjConfig(version="1")
    cfg.budgets = dict(budgets)
    return cfg


def test_effective_start_day_none_configured_falls_back_to_one():
    assert effective_cycle_start_day(_cfg()) == 1


def test_effective_start_day_default_only_is_calendar_month():
    assert effective_cycle_start_day(_cfg(anthropic=ProviderBudget())) == 1


def test_effective_start_day_single_explicit_is_honored():
    cfg = _cfg(anthropic=ProviderBudget(cycle_start_day=15))
    assert effective_cycle_start_day(cfg) == 15


def test_effective_start_day_agreeing_providers_honored():
    cfg = _cfg(
        anthropic=ProviderBudget(cycle_start_day=15),
        openai=ProviderBudget(cycle_start_day=15),
    )
    assert effective_cycle_start_day(cfg) == 15


def test_effective_start_day_conflicting_providers_fall_back():
    cfg = _cfg(
        anthropic=ProviderBudget(cycle_start_day=15),
        openai=ProviderBudget(cycle_start_day=20),
    )
    assert effective_cycle_start_day(cfg) == 1
