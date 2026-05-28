# Policy (preview)

`tj policy` is the unified surface for the configuration that controls TokenJam's
runtime behavior: alerts, drift detection, output-schema validation, sensitive-action
blocks, and per-provider budgets.

**This sprint ships a read-only preview only.** `tj policy list` consolidates the
existing config sections under a single view so you can see everything that drives
runtime behavior in one place. The underlying TOML structure has not been migrated —
each row in the table points back to the config section it was read from.

The full surface (`add`, `edit`, `apply`, `remove`, `test`) lands next sprint.

## `tj policy list`

```
$ tj policy list

POLICY                              SETTING                                         SOURCE
alerts                              cooldown_seconds=60, include_captured_content=false, channels=1  [alerts]
alerts.channels[0]                  type=stdout, min_severity=info                  [[alerts.channels]]
budget.anthropic                    usd=200, plan=max_20x, cycle_start_day=1        [budget.anthropic]
agents.checkout-bot.budget          daily_usd=5                                     [agents.checkout-bot.budget]
agents.checkout-bot.sensitive_actions  block: email_send, file_delete               [agents.checkout-bot]

Note: this is a read-only preview. The unified `tj policy add|edit|apply` surface lands next sprint.
```

`--json` returns the same data as a structured payload:

```json
{
  "policies": [
    {"policy": "alerts", "setting": "cooldown_seconds=60, ...", "source": "[alerts]"},
    {"policy": "budget.anthropic", "setting": "usd=200, plan=max_20x, ...", "source": "[budget.anthropic]"}
  ],
  "note": "Note: this is a read-only preview. ..."
}
```

## What gets surfaced

| Policy primitive            | Source TOML section                       |
| --------------------------- | ----------------------------------------- |
| `alerts`                    | `[alerts]`                                |
| `alerts.channels[N]`        | `[[alerts.channels]]`                     |
| `defaults.budget`           | `[defaults.budget]`                       |
| `budget.<provider>`         | `[budget.<provider>]`                     |
| `agents.<id>.budget`        | `[agents.<id>.budget]`                    |
| `agents.<id>.drift`         | `[agents.<id>.drift]`                     |
| `agents.<id>.sensitive_actions` | `[agents.<id>]`                       |
| `agents.<id>.schema`        | `[agents.<id>]` (`output_schema = ...`)   |
| `capture`                   | `[capture]`                               |

Rows are only emitted when something is actually configured. A bare config without
any per-agent overrides still produces the default `alerts` rows (every config
ships with at least one channel) but nothing else.

## Why a preview?

The unified policy framing is what the next sprint's `tj policy add | edit | apply`
work will build on. Shipping the read-only view first lets users see the existing
surface in the new shape, surface mismatches early, and lets us learn what reads
well before committing to the full config migration.
