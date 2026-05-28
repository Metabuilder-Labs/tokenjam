# Task 1.3 — Period comparison

**Wave 1. Dispatch after Task 0 fully merges. Runs in parallel with Tasks 1.1 and 1.2.**

## Prerequisites

- Read `CLAUDE.md`, `docs/internal/tasks/decisions-locked.md`.

## Summary

Add `--compare <period>` flag to `tj cost` and `tj optimize` enabling week-over-week, month-over-month, and custom-period comparisons. Free-tier feature; no Pro gating.

## Scope

**Period parsing:**

- `last-week`, `last-month`, `last-7d`, `last-30d`
- Custom date ranges in `YYYY-MM-DD:YYYY-MM-DD` form, e.g. `2026-04-01:2026-04-30`
- Reuse existing time parsing from `tokenjam/utils/time_parse.py` where possible.

**Diff computation:**

- Spend delta (or token delta for subscription-tier sessions, per `pricing_mode`).
- Top cost-shifts (which models / agents drove the change).
- Model-mix-drift (did the user shift between Opus/Sonnet/Haiku?).

**Output rendering:**

- Sign indicators (`▲` for increases, `▼` for decreases) and percentage changes.
- Plan-tier-aware rendering: unknown-tier sessions render token deltas only, never dollar deltas. Subscription-tier sessions render token deltas (consistent with v1.1 spec).
- `--json` output extension with structured diff data.

## Files touched

- `tokenjam/core/cost.py` (period comparison logic)
- `tokenjam/cli/cmd_cost.py`, `tokenjam/cli/cmd_optimize.py` (add `--compare` flag)
- `tests/unit/test_cost.py` (period comparison tests using factory with various `plan_tier` values)
- `docs/cli/cost.md`, `docs/cli/optimize.md`
- `CHANGELOG.md`

## Coordination

- Adds a flag to `cmd_optimize.py` and `cmd_cost.py`. No other Wave 1 task touches `cmd_cost.py`. Task 1.1 modifies output rendering in `cmd_optimize.py` but in a separate code path from `--compare`; small rebase if conflicts.

## Done-when

- `tj cost --compare last-week` produces a diff report against the prior week.
- `tj optimize --compare last-month` shows month-over-month optimization findings drift.
- Custom date ranges like `tj cost --compare 2026-04-01:2026-04-30` work.
- Subscription-tier and unknown-tier sessions render token deltas; api-tier renders dollar deltas.
- `--json` output contains structured `current_period`, `previous_period`, and `delta` blocks.
- Tests exercise api / subscription / local / unknown rendering paths.
