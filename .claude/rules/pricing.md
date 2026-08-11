---
paths:
  - "tokenjam/pricing/**"
  - "tokenjam/core/pricing.py"
  - "tokenjam/core/cost.py"
---

# Pricing and cost calculation

> Moved verbatim out of `CLAUDE.md` so it loads only when you touch these files.
> Cross-cutting rules stay in `CLAUDE.md`.

- **`tokenjam/core/pricing.py`**: `ModelRates` (frozen dataclass) + `RateRow` (a rate plus the `valid_from` and `variant` it applies under) + `VariantSpec`, `load_pricing_rows()` (LRU-cached; the full three-axis structure), `load_pricing_table()` (the flat standard-variant-now view), `get_rates(provider, model, *, at=..., variant=...)`. Falls back to default rates for unknown models. See the Pricing section for the time and variant axes.
- **`tokenjam/core/cost.py`**: `calculate_cost()` (pure function, rounds to 8dp) + `CostEngine` (post-ingest hook that updates `spans.cost_usd` and `sessions.total_cost_usd` via `db.conn` — see db.py note). Pricing loaded from `tokenjam/pricing/models.toml`. **Cache-read vs cache-write are separate fields** on `NormalizedSpan` (`cache_tokens` = read, `cache_write_tokens` = create); they bill at different rates and `calculate_cost` charges each at its own rate. The early-return no-op guard checks all four token counts (input/output/cache_read/cache_write) — see PR #90 and PR #92 for the cache-only-span and cache-write-on-live-path fixes.

## Pricing

Model pricing lives in `tokenjam/pricing/models.toml` (USD per million tokens) — the packaged file `core/pricing.py` loads via `PRICING_FILE = Path(__file__).parent.parent / "pricing" / "models.toml"`. There is no repo-root `pricing/` copy (it was moved into the package in v0.1.x so it ships in the wheel; editing a repo-root file would have no runtime effect). Structure: `[provider.model_name]` with `input_per_mtok`, `output_per_mtok`, and optional `cache_read_per_mtok`/`cache_write_per_mtok`. Unknown models fall back to the `DEFAULT_INPUT_PER_MTOK` / `DEFAULT_OUTPUT_PER_MTOK` constants in `core/pricing.py` with a logged warning (read the constants for the rates — never a copy of them here). The pricing table is LRU-cached at process startup — restart to pick up changes.

**The table has three axes: model, time, and variant.** A `[provider.model]` table carrying rate keys directly is one rate that has always applied — the original and still the common form, and adding the other two axes left every such row valid unchanged. The additions:

- **Time (rate history).** When a provider changes a price, add a dated `[[provider.model.rates]]` row with `valid_from` rather than editing the existing rate in place. `get_rates(..., at=<instant>)` picks the row whose window contains that instant, and `CostEngine` / the backfill adapters pass the span's own timestamp — so a span recorded before a change keeps pricing at the rate that actually billed it while a later one picks up the new rate. Editing in place instead makes past and future figures silently incomparable, with no record of which rate applied. Fields a dated row omits are inherited from the rate it supersedes.
- **Variant.** A variant is a different way of buying the *same* model id (fast mode, the Batch API, a longer cache TTL). Per-model rates go on a `[[provider.model.rates]]` row carrying `variant = "..."`; model-independent ones go under `[variants.<name>]`, where each field is an absolute rate or `{ multiplier = X }` / `{ multiplier = X, of = "<field>" }` (a multiple of a different standard field — how the 1-hour cache write is defined). `get_rates(..., variant=...)` resolves per-model rows first, then a `[variants]` definition, then falls back to the standard rate with a one-time warning. **A price multiplier belongs here, never as a constant in an analyzer file** — `batch_placement.batch_discount()` and `cache_efficacy.ONE_HOUR_TTL_VARIANT` both read the table.
- `load_pricing_rows()` is the full structure; `load_pricing_table()` remains the flat `{provider: {model: ModelRates}}` view (standard variant, in effect now) that `tj pricing list` reads.

The variant a live span was billed under comes off its own captured request params (`TjAttributes.REQUEST_SPEED`, read by `core/cost.rate_variant_for_span`) — deliberately *not* gated by a `[capture]` toggle, since it names which price applied rather than any content.

The packaged table is community-maintained: submit a PR editing `tokenjam/pricing/models.toml` when provider prices change. No code changes needed — the file is loaded at runtime.

**Local user overrides (no PR needed)** — users correct or add rates *for their own install* via override layers that `core/pricing.py` merges over the packaged table. Two sources, two key forms (see `docs/configuration.md` → "Pricing overrides" for the user-facing version):

- **Sources** (lowest priority first; later wins): the packaged `models.toml`, then a standalone file (`~/.config/tj/pricing.toml`, or `TJ_PRICING_FILE`), then a `[pricing]` section in the main config (`tj.toml`). The project-local config `[pricing]` wins over the global standalone file.
- **Key forms** — told apart deterministically by section name in `_split_pricing_raw()` (the reserved `models` section vs everything-else-is-a-provider; no value-shape guessing, no ordering dependency):
  - **Provider-keyed** (`[pricing.anthropic]` / `[anthropic]` whose values are model sub-tables) — merged per `(provider, model)` over the packaged table. This is the long-standing `[provider.model]` format.
  - **Model-keyed** (the reserved `[pricing.models]` section in tj.toml, or `[models]` in the standalone file) — keyed by **bare model name**, applied **regardless of inferred provider**. This is the attribution-proof path: it prices a span even when the provider resolved to `"unknown"` (the #194 open-weight class). `models` is a reserved key (`MODEL_SECTION_KEY`), never a provider, so the forms never collide.
- **`get_rates(provider, model, *, at=..., variant=...)` lookup order** (first match wins): model-keyed override → provider-keyed table (user `[provider.model]` over packaged) → `None` (→ `calculate_cost` applies the `core/pricing.py` default-rate constants and logs once). Each step tries an exact match, then strips a trailing `-YYYYMMDD` suffix.
- An override may itself carry dated `[[...rates]]` rows and its own `[variants]` / `[pricing.variants]` section; a provider/model in an override replaces that model's WHOLE row list, and variant definitions merge per name.
- The parsed layers are LRU-cached (`load_pricing_rows` + `load_model_pricing_row_overrides` + `load_rate_variants`); call `clear_pricing_cache()` or **restart the daemon** to pick up an edit.

The packaged table stays the zero-config default — the override is a *layer*, never a replacement; no user ever has to declare a rate to get started. Read-only inspection ships via `tj pricing list` (see CLI Commands; #282). The `set` half — a `tj pricing set` to edit overrides without hand-writing TOML — is not built yet.
