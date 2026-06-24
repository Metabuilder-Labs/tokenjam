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

## Enforcement-plane policies (`[[policies]]`, #220)

The proxy's policy engine (#220) loads **data-driven** policies from `[[policies]]`
and evaluates *eligible* requests through the proxy substrate (#219). A policy is
data, not code — it binds a registered `kind` (an evaluator) to a target with
kind-specific `params`:

```toml
[[policies]]
name = "demo-cap"
kind = "noop"            # the reference example kind (ships); budget_cap is #222
mode = "suggest"         # suggest only — enforce is gated off in the OSS rails
target_provider = "openai"
```

**Suggest mode only.** The engine *evaluates and records* what each policy WOULD
do, then the request is forwarded **unmodified** — nothing is enforced. The
enforce-mode path is scaffolded but gated off behind the certification gate (a
separate, private track).

**API-only.** The engine only ever sees `api`/usage-billed traffic. Subscription /
`unknown` traffic is observe-only at the gate (#219) and never reaches the engine —
enforced belt-and-suspenders by an api-only guard inside the engine itself.

**Unvalidated.** There is no certification engine in the open tree, so every policy
decision carries an explicit `unvalidated` label. `tj policy list` and
`tj policy decisions` surface it — a suggestion is never implied to be validated safe.

```
$ tj policy list
POLICY            SETTING                                                  SOURCE
policies.demo-cap kind=noop, mode=suggest, label=unvalidated, provider=openai  [[policies]][0]
...
Enforcement-plane policies ([[policies]]) run 'unvalidated' (suggest mode only — ...).

$ tj policy decisions        # persisted decisions + the estimated-recoverable meter
TIME       PROVIDER  PATH                  WOULD-DO  POLICY    LABEL
2026-…     openai    /v1/chat/completions  noop      demo-cap  unvalidated

Estimated recoverable: ~$0.0000 vs actual spend $12.3400 (3 decisions, label=unvalidated)
Estimated recoverable — suggest mode enforces nothing, so this is what these
policies WOULD have recovered if enforced, not realized savings. Unvalidated.
```

### Audit log + savings meter (#221)

Every recorded decision is persisted to an append-only DuckDB audit log
(`policy_decisions`), and each eligible POLICY-path decision also writes a
`savings_ledger` row. The audit log records **both** paths: `gate_decision` +
`passthrough_tos` distinguish "we *chose* not to act" (policy path, action
`noop`) from "we were *not permitted* to act" (subscription, TOS).

**The savings meter is estimated-recoverable, never realized.** Suggest mode
enforces nothing, so `tj policy decisions` shows what these policies *would have
recovered if enforced* — reconciled against actual spend from the same source
`tj cost` reads — and `realized` is always `False`. It never says "saved", and
the `unvalidated` label rides through to every persisted row.

`tj policy decisions` reads the **persisted** decisions + meter from the DB; if
a running `tj serve` holds the DB lock, it falls back to the proxy's recent
in-memory ring. The `add | edit | apply` lifecycle remains out of scope this
sprint.

## Why a preview?

The unified policy framing is what the next sprint's `tj policy add | edit | apply`
work will build on. Shipping the read-only view first lets users see the existing
surface in the new shape, surface mismatches early, and lets us learn what reads
well before committing to the full config migration.
