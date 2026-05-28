# Task 4 — `tj policy list` read-only view

**Optional, alongside Wave 3 if a fourth agent is available. Skip if pace pressure forces compression.**

## Prerequisites

- Read `CLAUDE.md`, `docs/internal/tasks/decisions-locked.md`.

## Summary

A small read-only `tj policy list` command that surfaces existing alerts, drift, schema, and budget configuration under the unified `policy` framing, without migrating the underlying config structure.

**Why this exists.** Ships the framing of the future unified policy surface (which the strategy doc relies on) without committing to the full multi-week migration. Users get a glimpse; we get to learn what shapes well in practice before next sprint's full migration.

## Scope

- New: `tokenjam/cli/cmd_policy.py` with single subcommand `list`.
- Reads existing `[alerts]`, `[drift]`, `[schema]`, `[budget.*]` config sections from the loaded `TjConfig`.
- Presents them as a unified table with columns: policy primitive name, current setting, source config section.
- Read-only. No `add` / `edit` / `apply` / `remove` / `test` subcommands this sprint.
- The full migration to a unified `[policy]` config structure is next sprint's work; **Task 4 does not touch the underlying config schema.**
- `--json` output supported for machine readers.

## Example output

```
$ tj policy list

POLICY               SETTING                          SOURCE
sensitive-actions    block: email_send, file_delete   [alerts.sensitive_actions]
budget.anthropic     daily_usd=50, plan=max_20x       [budget.anthropic]
drift                z_threshold=3.0                  [drift]
schema               validate=true                    [schema]

Note: this is a read-only preview. The unified `tj policy add|edit|apply`
surface lands next sprint.
```

## Files touched

- New: `tokenjam/cli/cmd_policy.py`
- `tokenjam/cli/main.py` (register `cmd_policy`)
- New: `tests/unit/test_cmd_policy.py`
- `docs/policy/overview.md` (notes that this is a read-only preview; full surface lands next sprint)
- `CHANGELOG.md`

## Coordination

- Registers a new top-level command in `cli/main.py`. Tasks 2.1 and 3.3 also touch `main.py` (2.1 adds `cmd_report`, 3.3 may or may not). Trivial conflict.

## Done-when

- `tj policy list` produces a consolidated view of existing policy-adjacent configuration with consistent formatting.
- `tj policy list --json` returns structured data.
- `tj policy add|edit|apply` etc. do NOT exist this sprint (verify they fail with a "coming next sprint" message or are simply absent).
- `docs/policy/overview.md` clearly states this is a preview.
