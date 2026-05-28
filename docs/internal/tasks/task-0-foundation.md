# Task 0 — Foundation

**Sequential. Blocks every parallel task. One agent. One PR. Merge before Wave 1.**

Estimated: 3 days for one experienced agent, plus review iteration.

## Prerequisites

- Read `CLAUDE.md` at repo root.
- Read `docs/internal/tasks/decisions-locked.md` in this directory.
- Read `tokenjam-product-strategy.md` v3 for context.

## Why this task exists

Multiple independent foundation changes that downstream analyzer tasks all need. Doing them in one focused PR avoids parallel agents making different assumptions about the same things.

## Scope

### 0.1 Data-model migration for plan tier and billing account

**Schema additions:**

- Add `billing_account` column to the `spans` table. **Provider-only**, no plan info: one of `anthropic`, `openai`, `google`, `bedrock`, `local.ollama`. No API-key fingerprint, no composite encoding.
- Add `plan_tier` column on `SessionRecord` as the canonical plan identifier. Values: `api`, `pro`, `max_5x`, `max_20x`, `plus`, `team`, `enterprise`, `local`, `unknown`. Extensible per provider.
- **No stored `pricing_mode` column.** Implement `pricing_mode` as a derived Python property on `SessionRecord` per decision 3 in `decisions-locked.md`.
- Spans do **not** carry plan_tier. They carry `billing_account` only. Analyzers JOIN through `SessionRecord` when they need plan info.

**Migration:**

- Write the DuckDB migration. New columns use `ALTER TABLE ADD COLUMN ... DEFAULT NULL`. **Do not backfill heuristics into existing rows** — the product has no external users and the dev's local DB will get sane defaults from regular use after the schema lands. New rows always have `billing_account` set by the writer; existing rows can stay NULL until a future reset.
- Update `ProviderBudget` config schema with a `plan` field. Default `api` only when set explicitly during fresh onboard; never auto-defaulted.

**Onboard prompt addition:**

- Add to `tokenjam/cli/cmd_onboard.py`: during `tj onboard --claude-code`, after Claude Code detection, prompt the user for their plan:
  ```
  How do you pay for Claude?
    1) API (per-token billing through console.anthropic.com)
    2) Pro plan ($20/mo subscription)
    3) Max 5x plan ($100/mo subscription)
    4) Max 20x plan ($200/mo subscription)

  [1]:
  ```
- Write the answer to `[budget.anthropic] plan = "<value>"` in config.
- **Do not auto-write `usd = 200`** for subscription plan users. If a Max-plan user wants a self-imposed soft ceiling, they configure it explicitly.
- Mirror the prompt for OpenAI when Codex is being onboarded (Plus / Team / Enterprise / API options).
- Add a `--reconfigure` flag to the existing `tj onboard` command: re-runs the prompts against an existing config without re-detecting the agent runtime.

**Propagation:**

- Update every place that writes spans: OTel receiver, OpenAI/Anthropic/Bedrock/Gemini patches, LangChain/LangGraph/CrewAI/AutoGen patches, Claude Code JSONL backfill, OpenClaw events. `grep` for `NormalizedSpan` constructors and `INSERT INTO spans` and audit each one. Each should set `billing_account` correctly based on which integration is emitting.
- Add OTel semconv attribute extensions for `billing_account` and `plan_tier` (the latter on session-level attributes, not span-level). Document the convention in `docs/architecture.md`.

### 0.2 Split `optimize.py` into a package with registry-driven analyzers

Refactor `tokenjam/core/optimize.py` into `tokenjam/core/optimize/` package:

```
tokenjam/core/optimize/
  __init__.py             # public API surface, re-exports run_optimize()
  runner.py               # orchestrates analyzer dispatch
  registry.py             # ANALYZER_REGISTRY dict; auto-populated at import
  types.py                # OptimizeFinding, OptimizeReport, ConfidenceLevel
  analyzers/
    __init__.py           # AUTO-DISCOVERY: walks dir, imports every .py
    model_downgrade.py    # self-registers via @register decorator
    budget_projection.py  # self-registers via @register decorator
```

- Each analyzer module ends with `@register("<name>")` decorator usage.
- **`analyzers/__init__.py` uses auto-discovery:**
  ```python
  import importlib, pkgutil
  for _, name, _ in pkgutil.iter_modules(__path__):
      importlib.import_module(f"{__name__}.{name}")
  ```
- **`cmd_optimize.py` reads valid choices from `REGISTRY.keys()` at click decoration time**: `@click.option("--finding", type=click.Choice(list(ANALYZER_REGISTRY.keys())))`. Analyzer tasks never edit the choice list.
- Move `analyze_model_downgrade()` and `project_budget()` into their new module homes, each with `@register` decoration.
- Update imports: `cmd_optimize.py`, `mcp/server.py` (`get_optimize_report`), all tests in `test_optimize.py`. Existing public function names stay importable via `tokenjam.core.optimize` re-exports.
- Add `tokenjam/core/optimize/README.md` explaining the package layout (TL;DR: drop a `.py` file under `analyzers/` with a `@register("name")` decorator).

### 0.3 Capture-content config key

- Add new config key: `capture.include_content` (boolean, default `false`).
- Distinct from existing `alerts.include_captured_content` (which controls dispatch payload, not capture).
- Gate enforcement at a single point: **`IngestPipeline.process()`** calls `strip_captured_content()` on the span before INSERT when `capture.include_content` is `false`. Receiver and JSONL backfill both go through this pipeline.
- **Strip the exact attribute keys listed in `decisions-locked.md` "Content stripping" section.** Both `tool_output` and `gen_ai.tool.output` are stripped intentionally; do not consolidate.
- Precedence: if `capture.include_content = false`, the `alerts.include_captured_content` flag is moot.
- Document privacy semantics in `docs/configuration.md` under a new "Content capture and privacy" section.

### 0.4 `tj onboard --reconfigure` plus plan-tier-unknown handling

- **No new `tj reconfigure` top-level command.** Add `--reconfigure` flag to existing `tj onboard`. Same prompts; no agent-runtime re-detection.
- `tj status`: when sessions exist with `plan_tier = unknown`, print a one-line note at end of output: `Note: N session(s) have unknown plan tier. Run 'tj onboard --reconfigure' to set it.` **Exit code unchanged.**
- `tj optimize`: when invoked against sessions with `plan_tier = unknown`, refuse to render dollar figures for those sessions. Token counts and structural findings only, with header note: `M of N sessions have unknown plan tier; dollar figures suppressed for those. Run 'tj onboard --reconfigure' to resolve.` Sessions with known plan tier render normally.

### 0.5 Embed the v1.1 honest-output spec into the repo

Create `docs/internal/specs/v1.1-honest-output.md` with the exact content below. This file is referenced by Task 1.1 and must exist when Task 1.1 dispatches. **Copy verbatim. The spec is canonical when this queue and the spec differ.**

````markdown
# `tj optimize` v1.1 — honest output for subscription users

## What's wrong today

Running `tj optimize` on a Claude Code user who's on Anthropic's Max 20x subscription plan produces output like:

```
Analyzing 17 sessions, 4.0M tokens, $2977.0660 spend (last 30d)…
No candidates flagged in this window. Either spend is small or all sessions already use a cost-effective model.
```

The user paid $100 (their subscription fee). They did not spend $2,977. The "$2,977 spend" figure is the **list-price equivalent of their token usage** — what the same usage would have cost at Anthropic API list prices. That number is meaningful, but presenting it as "spend" misleads.

This task fixes the output so it's honest for both API users and subscription users, without changing the underlying analyzers.

## The change in plain terms

For an **API user** (pays per-token), nothing changes.

For a **subscription user** (pays a flat fee with an allocation/cap), the output reframes:

- The "$2,977 spend" line becomes a value-vs-plan-cost framing: *"4.0M tokens this period (implied API value: $2,977, ~30× your $100 plan cost)"*. The big number is still shown, contextually, as evidence the user is getting their money's worth.
- The model-downgrade savings number stops showing dollar projections. Instead, savings express as a fraction of token usage: *"Switching these 5 sessions to Haiku would have used 0.4M fewer tokens (~10% of your monthly usage)."* The structural heuristic and caveat line stay identical.
- The budget-projection finding for subscription users either suppresses entirely or shifts to a clearly-labeled "self-imposed soft ceiling" framing if the user has explicitly set one.

## What you're being asked to build

### 1. Plan-tier as a config concept

(Already handled by Task 0.1's onboard prompt addition; this section is informational for the agent.)

### 2. Output reframing based on plan

`tj optimize` reads `SessionRecord.plan_tier` and changes its output:

**API mode (current behavior).** No changes.

**Subscription mode.** Three changes:

- Header line: replace *"Analyzing 17 sessions, 4.0M tokens, $2977.07 spend"* with:
  > *Analyzing 17 sessions, 4.0M tokens this cycle (Max 20x plan, $100/mo flat).*
  > *Implied API value: $2,977.07 — about 30× your plan cost.*

- Model-downgrade savings line: replace *"Projected savings if pattern holds: $1,632/mo"* with token-fraction framing:
  > *If this pattern held, you'd use ~10% fewer tokens this cycle — freeing capacity for harder tasks.*

- Budget projection: suppress for subscription users by default. If explicitly configured, label clearly as "Self-imposed soft ceiling" so it's never confused with a real billing limit.

**Local mode.** Zero dollar figures throughout. Token-only framing.

**Unknown mode (plan_tier = unknown).** Dollar figures suppressed for affected sessions with the header note from Task 0.4.

### 3. JSON output reflects the same change

- Top-level field: `plan: "max_20x"` (or whatever the SessionRecord plan_tier is)
- `pricing_mode: "subscription" | "api" | "local" | "unknown"` derived from plan_tier
- The `actual_cost_usd` field carries the computed-against-list-prices number for all users (useful data)
- For subscription users: surface a sibling `monthly_tokens_freed` field with the token-fraction equivalent; zero out or omit the `savings_usd` field

### 4. MCP tool description updated

The `get_optimize_report` MCP tool's docstring should surface for both audiences:
- API user asking "how can I save money on my Claude Code bill" — works as today
- Subscription user asking "am I using my Claude plan efficiently" or "am I getting my money's worth" — same tool surfaces

Update the natural-language trigger phrases in the docstring accordingly. The underlying implementation doesn't change.

## Done means

- A subscription user runs `tj optimize` and sees output that doesn't quote a dollar "spend" figure they didn't actually pay
- An API user runs `tj optimize` and sees output identical to today
- A local-mode user sees no dollar figures at all
- The model-downgrade caveat line is preserved in all modes
- The JSON output carries an explicit pricing-mode signal
- The MCP tool description surfaces for both "save money" and "use my plan efficiently" question phrasings
- No new analyzers, no new dependencies

## Honesty constraints

1. No quality-equivalence claims for downgrade recommendations
2. The `!` caveat line on the model-downgrade finding renders in all output modes
3. Numbers reconcile with `tj cost`
4. The output never presents list-price-computed costs as a user's bill if the user is on a subscription plan. The list-price number can be shown as a *value-comparison*, *implied-API-value*, or *internal computation reference*, but never as "spend," "cost," or "you paid."
````

### 0.6 Test factory updates

- Update `tests/factories.py`:
  - `make_session()` accepts a `plan_tier` parameter, **default `"api"`** (least-disruption — existing tests still see dollar figures).
  - `make_llm_span()` and similar accept a `billing_account` parameter, default `"anthropic"`.
- Existing tests should pass unchanged after this update. New tests in Wave 1 and Wave 2 can override these defaults to exercise subscription/local/unknown paths.

## Files touched

- DuckDB migration file (new, under existing migrations directory)
- `tokenjam/core/storage.py` or equivalent (schema additions)
- `tokenjam/core/session.py` or wherever `SessionRecord` lives (add `plan_tier` field; add `pricing_mode` derived property)
- `tokenjam/core/config.py` (new `[capture]` section, `plan` field on `ProviderBudget`)
- `tokenjam/core/optimize.py` deleted; replaced by `tokenjam/core/optimize/` package
- `tokenjam/core/optimize/registry.py`, `runner.py`, `types.py`, `analyzers/__init__.py` (auto-discovery), `analyzers/model_downgrade.py`, `analyzers/budget_projection.py`, `README.md`
- `tokenjam/cli/cmd_optimize.py` (registry-driven `--finding` choices, unknown-tier note)
- `tokenjam/cli/cmd_status.py` (unknown-tier note)
- `tokenjam/cli/cmd_onboard.py` (add plan-tier prompt; add `--reconfigure` flag)
- `tokenjam/mcp/server.py` (update `get_optimize_report` for new fields)
- All integration patch files (`anthropic.py`, `openai.py`, `bedrock.py`, `gemini.py`, `litellm.py`, langchain.py, etc.) — update span construction to set `billing_account`
- `tokenjam/core/ingest_pipeline.py` (gate content capture; strip the explicit attribute keys listed in `decisions-locked.md`)
- `tests/factories.py` (new optional parameters per 0.6)
- New: `docs/internal/specs/v1.1-honest-output.md` (verbatim content from 0.5)
- `docs/architecture.md`, `docs/configuration.md`
- `CHANGELOG.md`

## Done-when

- All existing tests pass after the refactor.
- Fresh `tj onboard --claude-code` flow prompts for plan tier, writes `[budget.anthropic] plan = "..."`, writes spans with `billing_account` populated and `SessionRecord.plan_tier` set.
- Fresh `tj optimize` on a new install produces output consistent with the v1.1 spec.
- `tj status` against a database with unknown-plan-tier sessions prints the note, exits 0.
- `tj optimize` against unknown-plan-tier sessions suppresses dollar figures with the header note.
- `tj onboard --reconfigure` runs end-to-end and updates an existing config.
- `capture.include_content` defaults to false and is gated in `IngestPipeline.process()` with all five named attribute keys stripped.
- `docs/internal/specs/v1.1-honest-output.md` exists in the repo with the spec content from 0.5 verbatim.
- `pricing_mode` is a derived property on `SessionRecord`, not a stored column; check by reading the schema.
- Adding a new file under `tokenjam/core/optimize/analyzers/` with `@register("foo")` makes `foo` appear in `tj optimize --finding` choices without any other edit.
- Test factories accept `plan_tier` and `billing_account` parameters with safe defaults.
