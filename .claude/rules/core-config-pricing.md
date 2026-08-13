---
description: Core config resolution and the pricing engine — precedence, env overrides, plan tiers, subscription vs API pricing modes.
paths:
  - "tokenjam/core/config.py"
  - "tokenjam/core/pricing.py"
  - "tokenjam/pricing/models.toml"
---

# `tokenjam/core/` — config and pricing

## Config

Config is TOML, discovered at: `tj.toml` -> `.tj/config.toml` -> `~/.config/tj/config.toml`. Override with `--config` or `TJ_CONFIG` env var. Full config hierarchy is in `config.py` (`TjConfig` dataclass).

Two distinct budget concepts coexist — do not conflate:
- **`[defaults.budget]` / `[agents.<id>.budget]`** (`daily_usd`, `session_usd`) — per-agent alert thresholds checked on every span by `AlertEngine`.
- **`[budget.<provider>]`** (`plan`, `usd`, `cycle_start_day`, `applies_to_services`) — per-provider budget config. `plan` is the user's declared plan tier (api / pro / max_5x / max_20x / plus / team / enterprise / local), prompted for by `tj onboard` and used by `IngestPipeline` to populate `SessionRecord.plan_tier` at session creation. `usd` is a periodic monthly ceiling used only by `tj optimize` budget-projection (read-only; no alerts fire from it). Onboard does NOT auto-write `usd = 200` — subscription users get only the `plan` field; API users are explicitly asked whether they want a self-imposed ceiling. The budget-projection analyzer scopes spend by `provider` column and (optionally) by `agent_id IN applies_to_services`.

**`[ingest]`** controls the daemon's continuous transcript catch-up (`auto_catch_up`, `interval_minutes`, `lookback_hours`, `startup_lookback_days`). It exists because Claude Code's OTLP exporter has no retry and no buffer, so the live path drops any session whose shell lacked the telemetry env vars or that ran while `tj serve` was down — and the on-disk transcript that could still rescue it is pruned by Claude Code on its own retention setting (`cleanupPeriodDays` — see Critical Rule 32's horizon note). `tj serve` runs a catch-up on startup (wider `startup_lookback_days` window, so downtime self-heals) and then every `interval_minutes`. `startup_lookback_days` is deliberately the wider of the two: the startup pass has to cover however long the daemon was off, the steady-state pass only one interval plus slack. See Critical Rule 33 before changing how a windowed pass behaves.

`tj onboard --claude-code` and `tj onboard --codex` always write to the **global** config (`~/.config/tj/config.toml`) regardless of cwd. This is intentional: each coding-agent integration reads one ingest secret from a single global location (`~/.claude/settings.json` or `~/.codex/config.toml`), and per-project configs would rotate that secret on every onboard, breaking auth for previously onboarded projects. Onboarded Claude Code project paths are tracked in `~/.config/tj/projects.json` for clean uninstall. Codex onboarding is fully project-agnostic — Codex hardcodes `service.name=codex_exec` in its binary, so there is one Codex agent ID for all projects.

## Pricing

Model pricing lives in `tokenjam/pricing/models.toml` (USD per million tokens) — the packaged file `pricing.py` loads via `PRICING_FILE = Path(__file__).parent.parent / "pricing" / "models.toml"`. There is no repo-root `pricing/` copy (it was moved into the package in v0.1.x so it ships in the wheel; editing a repo-root file would have no runtime effect). Structure: `[provider.model_name]` with `input_per_mtok`, `output_per_mtok`, and optional `cache_read_per_mtok`/`cache_write_per_mtok`. Unknown models fall back to the `DEFAULT_INPUT_PER_MTOK` / `DEFAULT_OUTPUT_PER_MTOK` constants in `pricing.py` with a logged warning (read the constants for the rates — never a copy of them here). The pricing table is LRU-cached at process startup — restart to pick up changes.

**The table has three axes: model, time, and variant.** A `[provider.model]` table carrying rate keys directly is one rate that has always applied — the original and still the common form, and adding the other two axes left every such row valid unchanged. The additions:

- **Time (rate history).** When a provider changes a price, add a dated `[[provider.model.rates]]` row with `valid_from` rather than editing the existing rate in place. `get_rates(..., at=<instant>)` picks the row whose window contains that instant, and `CostEngine` / the backfill adapters pass the span's own timestamp — so a span recorded before a change keeps pricing at the rate that actually billed it while a later one picks up the new rate. Editing in place instead makes past and future figures silently incomparable, with no record of which rate applied. Fields a dated row omits are inherited from the rate it supersedes.
- **Variant.** A variant is a different way of buying the *same* model id (fast mode, the Batch API, a longer cache TTL). Per-model rates go on a `[[provider.model.rates]]` row carrying `variant = "..."`; model-independent ones go under `[variants.<name>]`, where each field is an absolute rate or `{ multiplier = X }` / `{ multiplier = X, of = "<field>" }` (a multiple of a different standard field — how the 1-hour cache write is defined). `get_rates(..., variant=...)` resolves per-model rows first, then a `[variants]` definition, then falls back to the standard rate with a one-time warning. **A price multiplier belongs here, never as a constant in an analyzer file** — `batch_placement.batch_discount()` and `cache_efficacy.ONE_HOUR_TTL_VARIANT` both read the table.
- `load_pricing_rows()` is the full structure; `load_pricing_table()` remains the flat `{provider: {model: ModelRates}}` view (standard variant, in effect now) that `tj pricing list` reads.

The variant a live span was billed under comes off its own captured request params (`TjAttributes.REQUEST_SPEED`, read by `cost.rate_variant_for_span`) — deliberately *not* gated by a `[capture]` toggle, since it names which price applied rather than any content.

The packaged table is community-maintained: submit a PR editing `tokenjam/pricing/models.toml` when provider prices change. No code changes needed — the file is loaded at runtime.

**Local user overrides (no PR needed)** — users correct or add rates *for their own install* via override layers that `pricing.py` merges over the packaged table. Two sources, two key forms (see `docs/configuration.md` → "Pricing overrides" for the user-facing version):

- **Sources** (lowest priority first; later wins): the packaged `models.toml`, then a standalone file (`~/.config/tj/pricing.toml`, or `TJ_PRICING_FILE`), then a `[pricing]` section in the main config (`tj.toml`). The project-local config `[pricing]` wins over the global standalone file.
- **Key forms** — told apart deterministically by section name in `_split_pricing_raw()` (the reserved `models` section vs everything-else-is-a-provider; no value-shape guessing, no ordering dependency):
  - **Provider-keyed** (`[pricing.anthropic]` / `[anthropic]` whose values are model sub-tables) — merged per `(provider, model)` over the packaged table. This is the long-standing `[provider.model]` format.
  - **Model-keyed** (the reserved `[pricing.models]` section in tj.toml, or `[models]` in the standalone file) — keyed by **bare model name**, applied **regardless of inferred provider**. This is the attribution-proof path: it prices a span even when the provider resolved to `"unknown"` (the #194 open-weight class). `models` is a reserved key (`MODEL_SECTION_KEY`), never a provider, so the forms never collide.
- **`get_rates(provider, model, *, at=..., variant=...)` lookup order** (first match wins): model-keyed override → provider-keyed table (user `[provider.model]` over packaged) → `None` (→ `calculate_cost` applies the `pricing.py` default-rate constants and logs once). Each step tries an exact match, then strips a trailing `-YYYYMMDD` suffix.
- An override may itself carry dated `[[...rates]]` rows and its own `[variants]` / `[pricing.variants]` section; a provider/model in an override replaces that model's WHOLE row list, and variant definitions merge per name.
- The parsed layers are LRU-cached (`load_pricing_rows` + `load_model_pricing_row_overrides` + `load_rate_variants`); call `clear_pricing_cache()` or **restart the daemon** to pick up an edit.

The packaged table stays the zero-config default — the override is a *layer*, never a replacement; no user ever has to declare a rate to get started. Read-only inspection ships via `tj pricing list`.
