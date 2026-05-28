# Task 1.1 — v1.1 honest output for subscription users

**Wave 1. Dispatch after Task 0 fully merges. Runs in parallel with Tasks 1.2 and 1.3.**

## Prerequisites

- Read `CLAUDE.md`, `docs/internal/tasks/decisions-locked.md`.
- Read `docs/internal/specs/v1.1-honest-output.md` — **this is the canonical spec.** Where this task file and the spec differ on rendering language, the spec wins.

## Summary

Output reframing for subscription users (implied API value rather than "spend"), plan-tier-aware rendering across the optimize command. The plumbing — plan-tier on `SessionRecord`, the onboard prompt, the `unknown`-tier handling — already lives in Task 0. This task adds the rendering logic that consumes it.

## Scope

`tj optimize` output rendering branches on `SessionRecord.plan_tier`:

- **`api` mode:** dollar figures as actual spend (current behavior preserved).
- **`subscription` modes** (`pro` / `max_5x` / `max_20x` / `plus` / `team` / `enterprise`): dollar figures relabeled "implied API value" with explicit framing in the header.
- **`local` mode:** zero dollar figures, all framing in tokens.

Token savings framed differently per plan:

- `api`: "would save $X/month"
- `subscription`: "would free X tokens (~Y% headroom against your plan's programmatic cap)" — but defer to the spec's exact wording where they differ.
- `local`: "would free X tokens (zero marginal cost; relevant for capacity planning)"

Other rules:

- The `!` caveat line on model-downgrade output is mandatory regardless of plan tier.
- `budget-projection` analyzer: suppress for subscription users by default. If explicitly configured, label clearly as "Self-imposed soft ceiling."
- JSON output: top-level `plan` and `pricing_mode` fields per spec §3.
- MCP tool `get_optimize_report` docstring updated per spec §4.

## Files touched

- `tokenjam/core/optimize/analyzers/model_downgrade.py` and `budget_projection.py` (read plan_tier via session JOIN; suppress dollar fields for non-api modes)
- `tokenjam/cli/cmd_optimize.py` (output rendering branch on plan_tier)
- `tokenjam/mcp/server.py` (update `get_optimize_report` docstring with plan-tier-aware return shape and trigger phrases)
- Tests covering each plan_tier rendering path (use `make_session(plan_tier=...)` per Task 0.6)
- `docs/optimize/overview.md` (document the plan-tier-aware framing)
- `CHANGELOG.md`

## Coordination

- This task does not touch `cmd_optimize.py`'s `--finding` choice list (registry-driven from Task 0).
- Multiple analyzer modules edited here; no other Wave 1 task touches them.

## Done-when

- A Max-20x user running `tj optimize` sees output framed as "implied API value" with no dollar-spend claim against their flat-rate plan.
- An API user sees output identical to pre-Task-0 behavior.
- A local-mode user sees no dollar figures at all.
- An unknown-tier session continues to display the Task-0 header note and suppresses dollar figures.
- The `!` caveat line on model-downgrade renders in all modes.
- JSON output carries `plan` and `pricing_mode` fields.
- Tests cover all four rendering paths (api / subscription / local / unknown).
