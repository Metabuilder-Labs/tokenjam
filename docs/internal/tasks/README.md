# TokenJam OSS sprint — task queue

Implementation tasks for the May 26 OSS feature push from `tokenjam-product-strategy.md` v3 §11.

## How to use these files

Each task file is self-contained. To dispatch a task to an agent:

1. The agent reads `CLAUDE.md` at repo root (critical rules).
2. The agent reads `decisions-locked.md` in this directory (shared decisions across all tasks).
3. The agent reads the specific task file (e.g. `task-1.2-langfuse-ingest.md`).
4. The agent reads `tokenjam-product-strategy.md` v3 if it needs broader context.

Each task file lists scope, files touched, done-when criteria, and any coordination notes specific to that task.

## Dispatch order

**Day 0 — sequential, blocks everything else:**
- [task-0-foundation.md](task-0-foundation.md)

**Wave 1 — three parallel agents, after Task 0 merges:**
- [task-1.1-honest-output.md](task-1.1-honest-output.md)
- [task-1.2-langfuse-ingest.md](task-1.2-langfuse-ingest.md)
- [task-1.3-period-comparison.md](task-1.3-period-comparison.md)

**Wave 2 — three parallel agents, after Wave 1 fully merges:**
- [task-2.1-trim-analyzer.md](task-2.1-trim-analyzer.md)
- [task-2.2-cache-analyzer.md](task-2.2-cache-analyzer.md)
- [task-2.3-script-analyzer.md](task-2.3-script-analyzer.md)

**Wave 3 — three parallel agents, after Wave 2 fully merges:**
- [task-3.1-helicone-ingest.md](task-3.1-helicone-ingest.md)
- [task-3.2-otlp-ingest.md](task-3.2-otlp-ingest.md)
- [task-3.3-config-export.md](task-3.3-config-export.md)

**Alongside Wave 3 (optional, free agent):**
- [task-4-policy-list.md](task-4-policy-list.md)

**Wave 4 — one agent, after Waves 1–3 fully merge:**
- [task-5-docs-sweep.md](task-5-docs-sweep.md)

## Realistic sprint length

10–12 days. Each wave fully merges before the next opens. Reviewer is the bottleneck (~60–90 min per analyzer PR × 12 PRs = 15–25 hours over the sprint).

If pace pressure forces compression, drop in this order (least painful first): Task 3.2 (OTLP), Task 4 (policy list), Task 5 (docs sweep — roll into next sprint).

## Quality bar (every task)

- Feature works as specified.
- Unit and integration tests pass.
- `ruff check` and `mypy` pass cleanly.
- Documentation updated (analyzer docs, CLI reference, CHANGELOG).
- No new hard dependencies added without justification. Optional extras are fine when properly gated.
- PR description includes screenshots or terminal output where applicable.
- Honesty constraints (strategy doc §4.5, §10) respected in all new user-facing output.
- Plan-tier-aware rendering applied consistently (`api` vs `subscription` vs `unknown` framing where dollar figures appear).
- Each analyzer self-registers in the registry; no analyzer task edits `cmd_optimize.py`.

Each task should be small enough to merge within 24 hours of PR open. If a task is growing past that, split it.
