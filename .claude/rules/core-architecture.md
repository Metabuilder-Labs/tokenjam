---
description: The tokenjam/core/ domain layer — data flow, post-ingest hooks, session continuity, the top-level core modules, and StorageBackend parity.
paths:
  - "tokenjam/core/alerts.py"
  - "tokenjam/core/api_backend.py"
  - "tokenjam/core/backfill.py"
  - "tokenjam/core/config.py"
  - "tokenjam/core/cost.py"
  - "tokenjam/core/db.py"
  - "tokenjam/core/drift.py"
  - "tokenjam/core/framing.py"
  - "tokenjam/core/ingest.py"
  - "tokenjam/core/loop.py"
  - "tokenjam/core/models.py"
  - "tokenjam/core/pricing.py"
  - "tokenjam/core/recommendations.py"
  - "tokenjam/core/savings_log.py"
  - "tokenjam/core/schema_validator.py"
  - "tokenjam/core/export/**"
  - "tokenjam/core/ingest_adapters/**"
  - "tests/integration/test_storage_backend_parity.py"
---

# `tokenjam/core/` — domain layer

Pure domain logic. **Must never import from `tokenjam.cli` or `tokenjam.api`** — CLI and API import
from core, not the reverse.

The `optimize/`, `summarize/`, `rulewrite/` and `fixes/` sub-packages are not covered here; see
[`optimize-architecture.md`](optimize-architecture.md),
[`optimize-analyzers.md`](optimize-analyzers.md) and
[`optimize-cost-figures.md`](optimize-cost-figures.md).

## Data Flow

Spans enter from two paths, both converging at `IngestPipeline.process()`:

1. **In-process**: Python SDK `@watch()` + provider patches -> `TjSpanExporter` -> `IngestPipeline`
2. **HTTP**: TypeScript SDK (or any OTLP client) -> `POST /api/v1/spans` (auth required) -> `IngestPipeline`

Post-ingest hooks run after each span is written: `CostEngine.process_span()` (USD cost from token
counts — **always synchronous**, on the ingest thread, so budget/cost reads immediately after ingest
are accurate), then `AlertEngine.evaluate()` (per-span alert rules), then `SchemaValidator.validate()`
(tool outputs vs JSON Schema).

**Sync vs async hooks (`[alerts] async_hooks`, default `false`).** By default every hook runs inline
on the ingest thread. With `async_hooks = true`, only the *advisory* hooks (`AlertEngine` +
`SchemaValidator`) move onto one background worker thread (`TjHookWorker`) fed by a bounded
`queue.Queue` (`HOOK_QUEUE_MAXSIZE`, drop-**oldest** overflow — dropped spans are logged, never
silent); `CostEngine` stays synchronous regardless. Trades eventual consistency for alerts against
span-acknowledgment latency.

- **Concurrency:** in async mode the worker writes alerts/validations while the ingest thread writes spans/cost. DuckDB's **only** write-write conflict is same-row UPDATE overlap (`TransactionException: Conflict on update!`); disjoint-row/-table writes never conflict. `DuckDBBackend` serializes **writes** through a re-entrant `write_lock` (`self._write_lock`) held by every mutating method; the per-thread-cursor **read** path (#124) stays lock-free. **Do not remove the write lock** — its guard is `test_async_hooks_same_row_update_contention` in `tests/synthetic/test_async_hooks_concurrency.py` (many threads incrementing one shared session row; verified to raise `Conflict on update!` when the lock is stubbed with `nullcontext()`). The sibling disjoint-INSERT tests there don't catch a lock regression on their own. **Caveat:** direct `db.conn` writes from routes *outside* `DuckDBBackend`'s mutating methods (`alerts.py`'s `UPDATE alerts SET acknowledged`, `set_session_label`) do **not** take the lock — safe only while they hit rows/tables disjoint from the worker's; a direct write to a worker-touched table on a hot row must acquire `db.write_lock` explicitly.
- **Shutdown is lossless:** `IngestPipeline.flush()` blocks until the queue drains (guarded against hanging when the worker isn't alive); `close()` flushes, then stops the worker via a sentinel — the worker drains to empty before exiting rather than bailing at the top of the loop on the shutdown event. Every shutdown path (`tj serve` lifespan, `TjSpanExporter.shutdown`/`force_flush`) flushes before or while closing, so no queued alert is dropped.

## Session Continuity

A span whose `conversation_id` matches an existing session joins it, even across process restarts; a
new `conversation_id` starts a new session. A span lacking both `session_id` and `conversation_id`
joins the session of any sibling span on the same `trace_id`.

## Key Modules

- **`db.py`** — `StorageBackend` protocol + `DuckDBBackend` + `InMemoryBackend` (tests) + migration runner. Migrations are `(version, sql)` tuples in a `MIGRATIONS` list: never modify an existing one, only append. `StorageBackend` does not cover every query — `CostEngine` and `cmd_status` use `db.conn` directly for cost updates and active-session lookups; `_row_to_session()` converts raw DuckDB rows.

  **Schema self-heal (#382).** `run_migrations` keys purely on the version INTEGER, so a version recorded-applied under an older or renumbered definition never re-runs and its DDL silently never lands. A missing `ADD COLUMN` then makes every ingest writing that column hit a DuckDB Binder Error and be dropped (blank Status page); a missing `CREATE TABLE` (#382) makes any peripheral write to it raise. Guard: on every open, `run_migrations` also calls `ensure_expected_columns` (`ADD COLUMN IF NOT EXISTS` for `EXPECTED_ADDITIVE_COLUMNS`) and `ensure_expected_tables` (`CREATE TABLE IF NOT EXISTS` for `EXPECTED_TABLES`, table name → self-contained CREATE DDL), regardless of recorded versions. **Appending a migration that adds a column or a table the code reads/writes means adding it to `EXPECTED_ADDITIVE_COLUMNS` / `EXPECTED_TABLES` too.** Only additive/backward-compatible column defs belong there; table DDL must be idempotent/`IF NOT EXISTS`; indexes are out of scope (perf hints, not correctness). `tj doctor` surfaces any residual gap ("Schema integrity"), `tj doctor --repair` reconciles both.

- **`ingest.py`** — `IngestPipeline` (central hub), `SpanSanitizer` (rejects oversized/malformed spans), `strip_captured_content()`. Post-ingest hooks are optional and error-tolerant: failures are logged, never propagated. `flush()` / `close()` give the async-hook worker a lossless shutdown — see "Sync vs async hooks" above.

- **`pricing.py`** — `ModelRates` (frozen dataclass) + `RateRow` (a rate plus the `valid_from` and `variant` it applies under) + `VariantSpec`; `load_pricing_rows()` (LRU-cached, the full three-axis structure), `load_pricing_table()` (flat standard-variant-now view), `get_rates(provider, model, *, at=..., variant=...)`. Falls back to default rates for unknown models. Time and variant axes: [`core-config-pricing.md`](core-config-pricing.md).

- **`cost.py`** — `calculate_cost()` (pure, rounds to 8dp) + `CostEngine` (post-ingest hook updating `spans.cost_usd` and `sessions.total_cost_usd` via `db.conn`; see the `db.py` note). Pricing loads from `tokenjam/pricing/models.toml`. **Cache-read and cache-write are separate fields** on `NormalizedSpan` (`cache_tokens` = read, `cache_write_tokens` = create), billed at different rates, each charged at its own by `calculate_cost`. The early-return no-op guard checks all four token counts (input/output/cache_read/cache_write) — PR #90 and PR #92 fixed the cache-only-span and cache-write-on-live-path cases.

  **Rule: both cache fields belong on both sides of every token sum and every price comparison** — an omission that recurs in `core/optimize/`. **(a) In aggregates**, sum cache reads AND cache writes in the same expression; `COALESCE(SUM(input_tokens + output_tokens + cache_tokens + cache_write_tokens), 0)` (the form in `optimize/runner.py`) is canonical for per-cluster and per-session totals — grep `SUM.*cache_tokens` when adding one. **(b) In pricing comparisons**, an actual-vs-alternative-model savings figure must price cache identically on both sides: a hand-rolled alt-price helper omitting cache-write while the actual `cost_usd` bills it inflates every derived saving by the candidate's full cache-write cost, and evades that grep because the SQL is correct and the Python pricing is not — the omission lives in a per-row helper, not a `SUM`. **Keep exactly one pricer:** route every alternative-side computation through `calculate_cost`, which covers all four token classes.

- **`alerts.py`** — `AlertEngine` (the kinds it can fire are the `AlertType` enum in `core/models.py`; read the enum, never a count kept here), `CooldownTracker` (in-memory, per agent+type, resets on restart), `AlertDispatcher` routing to the channel types the `_build_channel` factory knows (stdout, file, ntfy, webhook, Discord, Telegram — that factory is the live set). `AlertEngine.fire()` is the entry point for `SchemaValidator` and `DriftDetector`. Suppressed alerts are still persisted, just not dispatched. Thresholds are module-level constants, not config — read them for values: **retry loop** fires when the same tool call repeats often enough inside a trailing span window (`_RETRY_LOOP_THRESHOLD` / `_RETRY_LOOP_WINDOW`); **failure rate** when the error share over a trailing span window crosses a ceiling, re-evaluated only every Nth error to bound cost (`_FAILURE_RATE_THRESHOLD` / `_FAILURE_RATE_WINDOW` / `_FAILURE_RATE_CHECK_INTERVAL`); **session duration** past a default wall-clock ceiling (`_SESSION_DURATION_DEFAULT`). Stdout and file channels always include full detail regardless of `include_captured_content`.

- **`drift.py`** — `DriftDetector`, Z-score based behavioral drift detection, fires at session end.

- **`loop.py`** — Close-the-loop primitive, pairing with the capture half (`tj trace`). Pure domain; storage helpers accept a backend or a raw conn like `db.set_session_label`. Three additive tables (migration 16): `run_annotations` (append-only human note plus optional verdict `good`/`bad`/`mixed`/`unknown` on a run/session — NOT the single-row `session_labels` renamed), `expectations` (a named case, optionally promoted `origin_session_id` FROM a bad run), `expectation_runs` (fix-history ledger, one row per rerun, `outcome` ∈ `pass`/`regress`/`unknown`). **Deliberately local-first and NOT an eval-runner:** pass/regress is a recorded *human verdict*, never an automated score (Critical Rule 14), and tokenjam does not push to Langfuse (that integration is inbound-only). Surfaces over this one storage: `api/routes/loop.py` (`/sessions/{id}/annotations`, `/expectations`, `/expectations/{id}` with history, `/expectations/{id}/runs`), the `tj loop` CLI group (`cmd_loop.py`, dual-path — api_mode when `tj serve` holds the lock, else direct DuckDB), and the Lens Session-Detail **"Loop" tab**. Product decision: `docs/internal/close-the-loop.md`.

- **`ingest_adapters/`** — adapters normalizing third-party trace exports (`langfuse.py`, `helicone.py`, `otlp.py`, `codex.py`) into `NormalizedSpan`, each reachable as a `tj backfill <name>` subcommand. Live-API adapters accept `--source-url` / `--source-file`; the on-disk `codex.py` accepts `--root`, mirroring `core/backfill.py`'s Claude Code path. All write deterministic span IDs derived from the source's identifiers, so re-runs are idempotent. `otlp.py` shares span-mapping logic with the live `POST /api/v1/spans` route via `tokenjam/otel/otlp_parsing.py`. `codex.py` parses Codex CLI rollout JSONL from `~/.codex/sessions/YYYY/MM/DD/rollout-*.jsonl` (provider/agent always `openai`/`codex_exec`, plan tier from `[budget.openai]`, cost recomputed from `pricing/models.toml`; one LLM span per `token_count` per-turn delta, tool spans from `response_item` `function_call`).

- **`export/`** — routing-config snippet generators for `tj optimize --export-config`. `claude_code.py` emits a JSONC fragment under a `tokenjam.routing_recommendations` namespace with honest-framing caveat comments baked in, writing to `~/.config/tokenjam/exports/`; it never touches `~/.claude/settings.json` or any other external config. There is deliberately **no `--apply` flag** — Claude Code does not honor TokenJam routing keys, so auto-writing would change nothing and erode trust.

- **`savings_log.py`** — a lock-free JSONL-sink utility; its remaining consumer is `core/recommendations.py`, which reuses `hooks_dir`. It outlived the output-cap hook (`tj hook cap-output`, `core/output_cap.py`, `cli/cmd_hook.py`, the `tj savings` CLI), cut because an A/B measured it NEGATIVE: whole-session cost went UP on Claude Code, since CC pre-truncates Bash output before the hook sees it, so the hook only added overhead. Do not reintroduce that shape. `tj onboard` / `tj uninstall` still strip an already-installed legacy cap-output hook entry from `~/.claude/settings.json` for prior-release users.

- **`recommendations.py`** — recommendation-outcome ledger: did a recommendation get *acted on*, and what did it recover? Records the two directly-observable actions (`summarize apply --go` → `record_summarize_apply`; `tj optimize --export-config` → `record_config_export`, stashing the downsize `suggestions` plus analysis window as the adoption baseline) and does **post-hoc downsize adoption detection** (`detect_downsize_adoption`): for each ripe export (at least `MIN_OBSERVATION_DAYS` old) it compares the recommended premium models' spend *rate* in the pre-window against the post-export observation window, marking adopted/ignored with a **measured** delta. `summarize_outcomes` aggregates into the two honest columns the Lens render and recommendations API surface — **measured-recovered stays strictly separate from estimated-recoverable** (Critical Rule 14; `$` summed only for `api` pricing-mode records). Idempotent via a deterministic `outcome_id`.

  **Storage is an append-only JSONL sink** (`hooks_dir(config)/recommendations.jsonl`, mirroring `core/savings_log.py`), deliberately NOT the `savings_ledger` table: the write paths need a lock-free sink (`summarize` is in `no_db_commands`; `optimize --export-config` runs against the read-only serve shim when the daemon holds the lock), and `savings_ledger` (#221) is the proxy would-have-saved meter whose `realized` invariant is always FALSE. Detection needs a direct DuckDB conn, so it runs from two triggers that both write the shared sink: server-side on `GET /api/v1/recommendations` (daemon owns the conn), and opportunistically from `tj optimize` when the daemon is down.

- **`backfill.py`** — parses Claude Code on-disk session JSONL into `NormalizedSpan`s, recomputing cost from `pricing/models.toml` because the on-disk format has no `cost_usd`. Tolerates the dated `claude-<family>-<ver>-YYYYMMDD` model-name suffixes Anthropic ships, via `core/pricing.py.get_rates()`, which strips the trailing 8-digit date suffix when no exact match exists. Idempotency relies on deterministic span IDs derived from `(session_id, message uuid)` / `(session_id, tool_use id)`. **Plan tier:** `ingest_claude_code(db, …, config=…)` resolves `plan_tier` from `config.budgets["anthropic"].plan` (Claude Code is always Anthropic) and stamps it on each `SessionRecord`, mirroring the live `IngestPipeline._resolve_plan_tier` so backfilled sessions aren't all `"unknown"` (#176) — pass `config` from callers (`cmd_backfill`, `tj onboard`). The Langfuse/Helicone/OTLP adapters create **no** `SessionRecord` (spans only), so there is no plan tier to propagate there.

- **`schema_validator.py`** — validates tool outputs against declared or genson-inferred JSON Schema. Fires only on `gen_ai.tool.call` spans carrying `gen_ai.tool.output` in attributes. Schema priority: 1) declared file from agent config `output_schema`, 2) inferred schema from `DriftBaseline.output_schema_inferred`. Caches schemas in-memory per agent.

- **`models.py`** — all domain dataclasses: `NormalizedSpan`, `SessionRecord`, `Alert`, `DriftBaseline`, filter types. `NormalizedSpan` carries `billing_account` (provider-only: `anthropic` / `openai` / `google` / `bedrock` / `local.ollama`). `SessionRecord` carries `plan_tier` (api / pro / max_5x / max_20x / plus / team / enterprise / local / unknown) plus a derived `pricing_mode` property (`local` / `subscription` / `api` / `unknown`). Spans inherit plan via the session FK — analyzers JOIN through `SessionRecord` for plan context. Full derivation rules: [`docs/architecture.md`](../../docs/architecture.md) → "OTel semconv extensions".

- **`config.py`** — `TjConfig` dataclass tree, TOML loading/writing, config file discovery. `ProviderBudget` carries an optional `plan` field (set by `tj onboard`'s plan-tier prompt) that `IngestPipeline._build_or_update_session` reads to populate `SessionRecord.plan_tier` at session creation. `CaptureConfig` has fine-grained content-capture toggles (`prompts` / `completions` / `tool_inputs` / `tool_outputs`), enforced by `strip_captured_content()` in `core/ingest.py` at the single ingest-pipeline gate. Resolution order and the pricing layers: [`core-config-pricing.md`](core-config-pricing.md).

- **`framing.py`** — **single source of truth for plan-tier-aware rendering** (issue #110). `compute_framing(config, window_summary, by_provider_breakdown) -> Framing` decides whether dollar figures are shown verbatim (`api`), suppressed for token-share framing (`subscription`), shown tokens-only (`local`), or shown with an "may overstate" qualifier (`unknown`). Also `render_dollar()` / `render_savings()` (UI-facing compact formatters) and the shared helpers `pricing_mode_for` / `dominant_plan` / `config_declared_plan` (with the global-config fallback) / `plan_tier_mix`.

  **Consumed by both the CLI (`cmd_optimize`, `cmd_tokenmaxx`, `cmd_cost` — both the `--compare` diff and the bare cost table, #175) and the REST API** (which emits `Framing.to_dict()` as the `framing` block); neither re-derives the rules. The bare `tj cost` table renders COST cells via `render_dollar()` (subscription → "% of cycle", local → "—", api/unknown → `format_cost`) with the qualifier surfaced above; under the daemon it reuses the `framing` block from `/api/v1/cost` via `ApiBackend.fetch_cost_framing`. This module *reads* plan-tier/pricing-mode; the canonical derivation lives on `SessionRecord.pricing_mode` + `SUBSCRIPTION_PLAN_TIERS` (semconv). **When adding a dollar-bearing surface, consume this — never re-implement the suppression rules.**

## StorageBackend parity (serve shim vs DB)

`tj` ships three `StorageBackend` implementations: `DuckDBBackend`, `InMemoryBackend` (tests), and
`ApiBackend` — the read-only HTTP shim the CLI uses when `tj serve` holds the DB write-lock. The shim
reconstructs objects from JSON, so it diverges from the DB backend *silently* (dropped
`cache_write_tokens` on daemon-fetched spans, cache columns zeroed, `get_daily_cost` returning a
cumulative total). That whole class of bug is guarded by
`tests/integration/test_storage_backend_parity.py`, which runs each faithfully-mirrored read method
against both the DuckDB backend and the `ApiBackend` shim — over a **real in-process uvicorn server**,
not the ASGI shortcut a sync `ApiBackend` can't use — and fails on any divergence.

**Rule: every `StorageBackend` protocol method must be classified in that file** — parity-covered
(`SHIM_PARITY_METHODS` plus a spec in `_parity_specs`), a documented `SHIM_KNOWN_GAPS`, deliberately
unimplemented (`SHIM_NOT_IMPLEMENTED`), or lifecycle. Adding a protocol method, or teaching the shim
a new one, fails CI (`test_every_protocol_method_is_classified` /
`test_unimplemented_methods_have_no_silent_shim`) until you classify it — which for a new read method
means adding a parity assertion. Don't skip the classification to make CI green; that reopens the
exact hole this file closes.
