# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`tj` (TokenJam) is a local-first, OTel-native **cost-optimization layer** for AI agents (with a full observability stack underneath). No cloud backend, no signup. It captures telemetry from agent runtimes, stores it in a local DuckDB database, and runs four named analyzers (`downsize` / `cache` / `script` / `trim`) that surface cost-saving candidates from real usage — plus a CLI, local REST API, web UI, and MCP server for querying. Install via `pipx install tokenjam` (recommended — sidesteps PEP 668 on Homebrew Python and Debian 12+/Ubuntu 24+) or `pip install tokenjam` in a venv. Run via `tj <subcommand>`. Requires Python >=3.10.

## Build & Development

```bash
# Install in dev mode
pip install -e ".[dev]"

# Linting and type checking
ruff check tokenjam/                  # line-length=100, target py310
mypy tokenjam/                        # partial config (not --strict; see [tool.mypy] in pyproject.toml)

# Tests (CI runs all except e2e)
pytest tests/unit/ tests/synthetic/ tests/agents/ tests/integration/

# Individual test layers
pytest tests/unit/               # Pure logic, no I/O, <1s
pytest tests/synthetic/          # Span injection via factories, zero cost
pytest tests/agents/             # Mock agent scenarios, full SDK path
pytest tests/integration/        # CLI + API integration

# Run a single test file or test function
pytest tests/unit/test_config.py
pytest tests/unit/test_config.py::test_function_name -v

# Real LLM tests (requires TJ_ANTHROPIC_API_KEY — auto-skipped otherwise)
pytest tests/e2e/

# TypeScript SDK (independent package)
cd sdk-ts && npm install && npm test
```


## Working with concurrent agents

When more than one agent is editing this repo in parallel, **each agent must operate in its own git worktree**. A single working directory shares one `HEAD`, so two `git commit` calls from different agents land on whichever branch was checked out last — leading to commits leaking into the wrong PR. We've hit this multiple times.

Spin up a per-task worktree before starting:
```bash
git worktree add ../tokenjam-<task> main
cd ../tokenjam-<task>
git checkout -b feat/<task>
```

When the PR merges and the branch is deleted, prune the worktree:
```bash
git worktree remove ../tokenjam-<task>
```

Symptom of a missed worktree: `git log` shows a commit on a branch you didn't intend (because another agent's `HEAD` was the checked-out one when your `git commit` ran). If you see this, do **not** force-push — rebase the stray commit off your branch first, and only force-push if you own every commit being rewritten.

`.tj/config.toml` is intentionally untracked (see PR #145 + Critical Rule 20) and gets mutated at runtime by `tj onboard` / `tj serve` regenerating the local `ingest_secret`. Don't `git add` it back. The CI test `tests/unit/test_no_tracked_dev_secrets.py` guards against this.


## PR and commit conventions (for any agent producing a PR)

These conventions apply to any agent — feature work, bug fixes, docs, content. Briefs may add task-specific structure but should not contradict these.

### Branch + PR titles

- **Branch names** are slash-separated, kebab-case, prefixed by type:
  - `fix/<issue-or-area>` — bug fixes (e.g. `fix/175-176-cost-framing-backfill-plan`)
  - `feat/<area>` — new features (e.g. `feat/reuse-analyzer-115`)
  - `docs/<area>` — documentation (e.g. `docs/readme-cleanup-v0.4.1`)
  - `chore/<area>` — refactors, renames, infra
  - `release/<X.Y.Z>` — release-cut PRs
- **PR titles** lead with the verb / type and reference issues by number when applicable:
  - `Fix #175, #176: tj cost framing + backfill plan_tier propagation (v0.4.2)` (bug fixes)
  - `[feature] Add Reuse analyzer (#115)` (features)
  - `docs: drop stale CHANGELOG.md + add maintainer contact` (docs)
  - `Bump version to 0.4.1` (release-cut PRs — keep these terse)
- Use **`Closes #N`** in the PR body (not just title) when fixing an issue, so GitHub auto-closes the issue on merge. Multiple `Closes` lines if you're closing several. Do not use the comma form `Closes #1, #2` — GitHub only catches the first; use separate lines.

### Commit messages

- **Subject line** (first line, ≤72 chars): one-line summary in active voice. Reference issues with `#N` when applicable.
- **Body** (after blank line): explain *why* the change is needed, not *what* it changes (the diff shows that). Use full sentences, paragraphs, bullet lists.
- **Trailers** (after another blank line, at the very end):
  - Always include: `Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>` (or the appropriate model identifier)
  - When fixing an externally-reported bug: also include `Co-Authored-By: <reporter-handle> <noreply@github.com>` (e.g. `ashwmu` for the external contributor's reports)
- Use **HEREDOC for multi-line messages** to preserve formatting: `git commit -m "$(cat <<'EOF' ... EOF)"`

### PR body structure

```markdown
[1-2 sentence framing of why this exists]

## Summary
- [bullet — what changed at a high level]
- [bullet — another high-level change]

## [Per-issue or per-feature section, repeated as needed]
[Detail per issue, including the symptom, root cause, fix]

## Tests / Verification
- [test files added or modified, what they cover]
- [any live verification: workflow run URL, screenshot, command output]

## What's NOT in this PR (if scope was deliberately limited)
- [out-of-scope item 1 — explain why deferred]
- [out-of-scope item 2]

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: <reporter-handle> <noreply@github.com>   # if applicable
```

The "What's NOT in this PR" section is load-bearing — it makes the reviewer's job 10x easier when the agent explicitly named what they decided to defer. Use it whenever scope is non-obvious.

### Self-review checklist before requesting review

Run through this before pushing the PR:

1. **Tests pass locally.** `pytest tests/unit/ tests/integration/` (or `tests/unit/<file>.py` if narrow).
2. **`ruff check tokenjam/` and `mypy tokenjam/` clean** for any files you touched.
3. **CI on the branch is green** for at least the test-ts job (Python jobs may still be running when you push).
4. **Acceptance criteria from the issue are met** — go through them one by one and verify.
5. **No accidental files in the diff** — `.tj/config.toml`, `.tj-test-data/`, screenshots that were just for debugging, etc.
6. **PR body explains the WHY** — symptom + root cause + fix, not just "fixes the bug."
7. **Honesty discipline preserved.** If the change touches any user-facing string ("recoverable," "estimated," "savings"), verify it matches existing analyzer caveat language. Never silently strengthen claims.

### Scope discipline

- **Do what the brief / issue says, no more.** If you notice an adjacent issue, file it as a separate issue rather than expanding the PR. Reviewers should never have to mentally separate "the fix" from "drive-by cleanup."
- **Exception:** when an adjacent change is functionally required to make the primary fix work (e.g., updating a caller of a function you changed). Note it explicitly in the PR body under "What's also in this PR."
- **When in doubt about scope, ask the master agent before expanding.** A 30-second clarification beats a 30-minute scope review.

### Worker vs master

- **Worker agents do not merge their own PRs.** Open the PR, request review, the master + Anil handle merge.
- **Worker agents do not file follow-up issues unprompted.** If you notice something during your work that's out of scope, mention it in the PR body and let the master decide whether to file.
- **Worker agents do not bump versions.** Release-cut PRs are a separate concern handled by the master / Anil.


## Architecture

### Data Flow

Spans enter from two paths, both converging at `IngestPipeline.process()`:
1. **In-process**: Python SDK `@watch()` + provider patches -> `TjSpanExporter` -> `IngestPipeline`
2. **HTTP**: TypeScript SDK (or any OTLP client) -> `POST /api/v1/spans` (auth required) -> `IngestPipeline`

Post-ingest hooks run synchronously after each span is written to DB:
1. `CostEngine.process_span()` — calculates USD cost from token counts
2. `AlertEngine.evaluate()` — checks all per-span alert rules
3. `SchemaValidator.validate()` — validates tool outputs against JSON Schema

### Package Dependency Rules

- `tokenjam/core/` is pure domain logic. **Must never import from `tokenjam.cli` or `tokenjam.api`**. CLI and API import from core, not the reverse.
- `tokenjam/otel/semconv.py` is pure constants with no internal imports.
- `sdk-ts/` is fully independent from Python — communicates only via HTTP.

### Key Modules

- **`tokenjam/core/db.py`**: `StorageBackend` protocol + `DuckDBBackend` + `InMemoryBackend` (for tests) + migration runner. Migrations are `(version, sql)` tuples in a `MIGRATIONS` list — never modify existing ones, only append. **Note:** `StorageBackend` doesn't cover every query. Some callers (e.g. `CostEngine`, `cmd_status`) access `db.conn` directly for queries not in the protocol (cost updates, active session lookups). Helper `_row_to_session()` is used to convert raw DuckDB rows.
- **`tokenjam/core/ingest.py`**: `IngestPipeline` (central hub), `SpanSanitizer` (rejects oversized/malformed spans), `strip_captured_content()`. Post-ingest hooks (cost, alerts, schema) are optional and error-tolerant — hook failures are logged, never propagated.
- **`tokenjam/core/pricing.py`**: `ModelRates` (frozen dataclass), `load_pricing_table()` (LRU-cached), `get_rates(provider, model)`. Falls back to default rates for unknown models.
- **`tokenjam/core/cost.py`**: `calculate_cost()` (pure function, rounds to 8dp) + `CostEngine` (post-ingest hook that updates `spans.cost_usd` and `sessions.total_cost_usd` via `db.conn` — see db.py note). Pricing loaded from `tokenjam/pricing/models.toml`. **Cache-read vs cache-write are separate fields** on `NormalizedSpan` (`cache_tokens` = read, `cache_write_tokens` = create); they bill at different rates and `calculate_cost` charges each at its own rate. The early-return no-op guard checks all four token counts (input/output/cache_read/cache_write) — see PR #90 and PR #92 for the cache-only-span and cache-write-on-live-path fixes.
- **`tokenjam/core/alerts.py`**: `AlertEngine` with 13 alert types, `CooldownTracker` (in-memory, per agent+type, resets on restart), `AlertDispatcher` routing to 6 channel types (stdout, file, ntfy, webhook, Discord, Telegram). `AlertEngine.fire()` is the external entry point for other modules (SchemaValidator, DriftDetector) to fire alerts. Suppressed alerts are still persisted to DB but not dispatched to channels. Hardcoded thresholds: retry loop fires at 4+ identical tool calls in last 6 spans; failure rate fires at >20% errors in last 20 spans (checked every 5th error); session duration default 3600s. Stdout and file channels always include full detail regardless of `include_captured_content` config.
- **`tokenjam/core/drift.py`**: `DriftDetector` — Z-score based behavioral drift detection, fires at session end.
- **`tokenjam/core/optimize/`**: Package powering `tj optimize` and the `get_optimize_report` MCP tool. Public API re-exported from `__init__.py`: `build_report()` (orchestrator), `report_to_dict()`, `ANALYZER_REGISTRY`, `ANALYZER_ORDER`, plus result dataclasses. Architecture: `registry.py` holds the `@register("name")` decorator and `ANALYZER_REGISTRY` dict; `runner.py` defines `ANALYZER_ORDER` and orchestrates execution; `types.py` holds `AnalyzerContext` + result dataclasses + `MODEL_DOWNGRADE_CAVEAT`. Individual analyzers live in `analyzers/`, each as a single file registering via `@register`. **Registry strings (the user-facing names) and file names are decoupled**:
  - `model_downgrade.py` → `@register("downsize")` — structural candidates (input < 5K tokens AND output < 500 tokens AND tool_calls ≤ 5; never claims quality equivalence, caveat baked into dataclass default)
  - `budget_projection.py` → `@register("budget-projection")` — per-provider cycle spend vs `[budget.<provider>]` ceiling; only fires when budget > 0
  - `cache_efficacy.py` → `@register("cache")` — current cache-read efficacy per (provider, model)
  - `cache_recommend.py` → `@register("cache-recommend")` — Anthropic-only structural prefix detection for `cache_control` placement
  - `workflow_restructure.py` → `@register("script")` — `(tool_name, arg_shape)` cluster detection for deterministic-script candidates
  - `plan_reuse.py` → `@register("reuse")` — repeated-planning cluster detection; a savings analyzer (carries `estimated_recoverable_usd`). Has a dedicated endpoint/report path (`GET /api/v1/reuse/clusters`, `tj report --reuse`) because its per-cluster planner text can be many KB — see the routes/report bullets below
  - `prompt_bloat.py` → `@register("trim")` — LLMLingua-2 token-significance classification (requires `tokenjam[bloat]` extra)
  Analyzers receive an `AnalyzerContext` and operate on `db.conn` directly. To add a new analyzer: drop a file under `analyzers/`, decorate with `@register("name")`, append to `ANALYZER_ORDER` if ordering matters — `cmd_optimize`'s positional `findings` Click choices auto-derive from the registry.
  **Recoverable-savings contract** (issues #111/#122): every *savings* analyzer's result dataclass carries `estimated_recoverable_usd` / `estimated_recoverable_tokens` / `estimate_basis` / `estimate_confidence` (`"heuristic"`). All four are on **one time basis — recoverable over the analyzed window** (`downsize` keeps a separate `monthly_savings_usd` for its CLI projection line, but `estimated_recoverable_usd` is the window figure so Overview tiles are comparable). `cache-recommend` and `budget-projection` deliberately carry **no** recoverable field (not savings analyzers); the Overview waste band is registry-driven off the presence of `estimated_recoverable_usd`, so a future analyzer (e.g. reuse) appears with no UI change. `report_to_dict`/`report_from_dict` round-trip these fields. Honesty discipline (Critical Rule 14) is mandatory — every estimate is "estimated recoverable", never "saves you".
- **`tokenjam/core/ingest_adapters/`**: Third-party trace-export adapters that normalize external payloads (`langfuse.py`, `helicone.py`, `otlp.py`) into `NormalizedSpan` for ingest. Each is reachable as a `tj backfill <name>` subcommand and accepts `--source-url` (live API) or `--source-file` (offline JSON dump). Adapters write deterministic span IDs derived from the source's identifiers so re-runs are idempotent. `otlp.py` shares span-mapping logic with the live `POST /api/v1/spans` route via `tokenjam/otel/otlp_parsing.py`.
- **`tokenjam/core/export/`**: Routing-config snippet generators for `tj optimize --export-config`. Currently `claude_code.py` emits a JSONC fragment under a `tokenjam.routing_recommendations` namespace with honest-framing caveat comments baked in. Writes to `~/.config/tokenjam/exports/`; never touches `~/.claude/settings.json` or other external configs (no `--apply` flag — Claude Code doesn't currently honor TokenJam routing keys, so auto-writing would change nothing and erode trust).
- **`tokenjam/core/backfill.py`**: Parses Claude Code on-disk session JSONL files into `NormalizedSpan`s. Cost is recomputed from `pricing/models.toml` because the on-disk format has no `cost_usd`. The parser tolerates the dated `claude-<family>-<ver>-YYYYMMDD` model-name suffixes Anthropic ships (handled by `core/pricing.py.get_rates()`, which strips the trailing 8-digit date suffix when no exact pricing match exists). Idempotency relies on deterministic span IDs derived from `(session_id, message uuid)` / `(session_id, tool_use id)`. **Plan tier:** `ingest_claude_code(db, …, config=…)` resolves `plan_tier` from `config.budgets["anthropic"].plan` (Claude Code is always Anthropic) and stamps it on each `SessionRecord` — mirroring the live `IngestPipeline._resolve_plan_tier` so backfilled sessions aren't all `"unknown"` (#176). Pass `config` from callers (`cmd_backfill`, `tj onboard`). The Langfuse/Helicone/OTLP adapters create **no** `SessionRecord` (spans only), so there's no plan tier to propagate there.
- **`tokenjam/core/schema_validator.py`**: Validates tool outputs against declared or genson-inferred JSON Schema. Only fires on `gen_ai.tool.call` spans with `gen_ai.tool.output` in attributes. Schema priority: 1) declared file from agent config `output_schema`, 2) inferred schema from `DriftBaseline.output_schema_inferred`. Caches schemas in-memory per agent.
- **`tokenjam/core/models.py`**: All domain dataclasses — `NormalizedSpan`, `SessionRecord`, `Alert`, `DriftBaseline`, filter types, etc. `NormalizedSpan` carries `billing_account` (provider-only: `anthropic` / `openai` / `google` / `bedrock` / `local.ollama`). `SessionRecord` carries `plan_tier` (api / pro / max_5x / max_20x / plus / team / enterprise / local / unknown) plus a derived `pricing_mode` property (`local` / `subscription` / `api` / `unknown`). Spans inherit plan via the session FK — analyzers JOIN through `SessionRecord` when they need plan context. See [`docs/architecture.md`](docs/architecture.md) → "OTel semconv extensions" for the full derivation rules.
- **`tokenjam/core/config.py`**: `TjConfig` dataclass tree, TOML loading/writing, config file discovery. `ProviderBudget` carries an optional `plan` field (set by `tj onboard`'s plan-tier prompt) that `IngestPipeline._build_or_update_session` reads to populate `SessionRecord.plan_tier` at session creation. `CaptureConfig` has four fine-grained content-capture toggles (`prompts` / `completions` / `tool_inputs` / `tool_outputs`); `strip_captured_content()` in `core/ingest.py` enforces them at the single ingest-pipeline gate.
- **`tokenjam/core/framing.py`**: **Single source of truth for plan-tier-aware rendering** (issue #110). `compute_framing(config, window_summary, by_provider_breakdown) -> Framing` decides whether dollar figures are shown verbatim (`api`), suppressed for token-share framing (`subscription`), shown as tokens-only (`local`), or shown with an "may overstate" qualifier (`unknown`). Plus `render_dollar()` / `render_savings()` (UI-facing compact formatters), and the shared helpers `pricing_mode_for` / `dominant_plan` / `config_declared_plan` (with the #106 global-config fallback) / `plan_tier_mix`. **Consumed by both the CLI (`cmd_optimize`, `cmd_tokenmaxx`, `cmd_cost` — both the `--compare` diff and the bare cost table, #175) and the REST API** (which emits `Framing.to_dict()` as the `framing` block) — neither re-derives the rules. The bare `tj cost` table renders COST cells via `render_dollar()` (subscription → "% of cycle", local → "—", api/unknown → `format_cost`) with the qualifier surfaced above; under the daemon it reuses the `framing` block from `/api/v1/cost` via `ApiBackend.fetch_cost_framing`. This module *reads* plan-tier/pricing-mode; the canonical derivation still lives on `SessionRecord.pricing_mode` + `SUBSCRIPTION_PLAN_TIERS` (semconv). When adding a dollar-bearing surface, consume this — do not re-implement the suppression rules.
- **`tokenjam/sdk/agent.py`**: `@watch()` decorator creates session spans only. `record_llm_call()` and `record_tool_call()` create child spans for manual instrumentation. LLM call spans from provider clients require `patch_anthropic()`, `patch_openai()`, etc.
- **`tokenjam/sdk/transport.py`**: `HttpTransport` — buffers up to 1000 spans, retries with exponential backoff (3 attempts, 2s base). Used when `tj serve` runs as a separate process.
- **`tokenjam/sdk/bootstrap.py`**: `ensure_initialised()` — lazy, thread-safe, idempotent bootstrap of config -> DB -> IngestPipeline -> TracerProvider. Called automatically by `@watch()` and all `patch_*()` functions. Registers atexit flush.
- **`tokenjam/sdk/integrations/`**: `Integration` protocol in `base.py`. Provider patches (anthropic, openai, gemini, bedrock, litellm) monkey-patch client methods to create OTel spans with token usage. `litellm.py` covers 100+ providers via LiteLLM's unified interface and uses a `contextvars.ContextVar` (`_tj_litellm_active`) to suppress inner provider patches (openai, anthropic) when active — prevents double-counted spans. Framework patches (langchain, langgraph, crewai, autogen) wrap LLM/tool methods. `llamaindex.py` and `openai_agents_sdk.py` are thin wrappers around those SDKs' native OTel support. `nemoclaw.py` is a WebSocket observer for OpenShell Gateway sandbox events.
- **`tokenjam/otel/provider.py`**: `TjSpanExporter` (custom `SpanExporter` that feeds spans into `IngestPipeline`), `convert_otel_span()` (OTel `ReadableSpan` → `NormalizedSpan`), `build_tracer_provider()` (sets up global `TracerProvider` with local + optional OTLP exporters).
- **`tokenjam/otel/exporters.py`**: Prometheus metric reader setup via `build_prometheus_exporter()`.
- **`tokenjam/otel/otlp_parsing.py`**: Shared OTLP JSON → `NormalizedSpan` parser. Two callers: `api/routes/spans.py` (live `POST /api/v1/spans`) and `core/ingest_adapters/otlp.py` (`tj backfill otlp`). Keep parsing in this one place — the live receive path and the backfill adapter must agree on attribute extraction, billing_account derivation, and timestamp handling.
- **`tokenjam/otel/semconv.py`**: `GenAIAttributes`, `TjAttributes` (includes `BILLING_ACCOUNT` and `PLAN_TIER`), `VALID_PLAN_TIERS` and `SUBSCRIPTION_PLAN_TIERS` frozensets — OTel GenAI semantic convention constants plus tj-specific extensions.
- **`tokenjam/api/app.py`**: FastAPI app factory (OpenAPI title `"TokenJam Lens"`). `tj serve` starts it with uvicorn. Accepts `db`, `config`, `ingest_pipeline` for testability. Registers all routers under `/api/v1` plus `/metrics`, `/health`, and the SPA at `/`. **`index.html` is read into a module string once at `create_app()` time** (`_index_html`) — so editing `tokenjam/ui/index.html` requires a `tj serve` restart to take effect; tests read the file from disk directly and aren't affected. Mounts `/ui/vendor` as `StaticFiles`.
- **`tokenjam/api/middleware.py`**: `IngestAuthMiddleware` — protects `POST /api/v1/spans` with Bearer token. Returns `JSONResponse(401)` directly (not `HTTPException`, which doesn't propagate from `BaseHTTPMiddleware.dispatch`).
- **`tokenjam/api/deps.py`**: `require_api_key` — FastAPI dependency for optional API key auth on GET endpoints. Only enforced when `api.auth.enabled = true` in config.
- **`tokenjam/api/routes/`**: One file per resource — `spans.py` (OTLP JSON ingest), `traces.py`, `cost.py`, `cost_compare.py`, `tools.py`, `alerts.py`, `drift.py`, `optimize.py`, `reuse.py` (`GET /api/v1/reuse/clusters` — the Reuse finding plus skeleton-rendering extras `planning_texts` + `pricing_mode`; a dedicated endpoint (not bolted onto `/optimize`) so the per-cluster planner text, which can be many KB, isn't paid for on every Overview poll — #154), `budget.py`, `status.py`, `agents.py`, `metrics.py` (Prometheus text format from DB queries), `version.py` (unauthenticated `GET /health` → `{"status":"ok","version":...}` mounted with no prefix, plus `GET /api/v1/version`; the version is derived at runtime via `importlib.metadata.version("tokenjam")` — no hardcoded literal). **The dollar-bearing read routes (`/cost`, `/cost/compare`, `/optimize`, `/budget`) each return a `framing` block** (see `core/framing.py`) so the web UI renders plan-tier-aware figures without re-deriving the rules in JS. `/optimize` takes `?fast=true` to skip the expensive Trim analyzer (returns `skipped_analyzers`) for the polling Overview; `/cost` returns a window-bucketed `series` for the chart (see Web UI below). **Concurrency:** the sync (`def`) read routes (`/optimize`, `/cost/compare`) run in Starlette's threadpool, so concurrent requests reach the DB from multiple threads. `DuckDBBackend.conn` is a **per-thread DuckDB cursor** (`threading.local`) over one shared database — cursors are independent connections safe for concurrent use, so fan-out callers (the Overview) can fetch in parallel. Fixed in #124 (was a single shared connection that aborted under concurrent access); do not collapse `conn` back to one shared connection object.
- **`tokenjam/core/summarize/`**: Structure-aware prompt summarization (advisory; new feature). `detect.py` classifies prose vs. structure (fenced/inline code, tags, templates, tables); `candidates.py` (+ `catalog.py`/`estimate.py`) powers the `tj summarize list` scan; `wrap.py` is the pure protect→restore algorithm (wrap each structured span behind an id'd `<tj-keep>` marker, restore verbatim by id — structure is a hard guarantee); `session.py` is the no-scratch `prepare`/`check` lifecycle + staging (re-derives the wrap from the live file + a content hash; persists nothing but the staged result); `apply.py` writes a staged rewrite back to the file (default dry-run; `--go` writes) behind an owner + content-hash + symlink guard, with a gzip backup and `undo`; `backup.py` stores the gzipped original + metadata under `~/.tj/summary/backups/`; `delivery.py` is the CLI's automated rewrite step — `claude -p` (subprocess, timeout-guarded) or the Anthropic API (lazy `httpx` + the user's own `TJ_ANTHROPIC_API_KEY`) — plus the "pays for itself" amortization. Pure domain logic — no `tokenjam.cli`/`tokenjam.api` imports (delivery's API path lazily imports `httpx`, its lone outbound dependency).
- **`tokenjam/mcp/server.py`**: FastMCP stdio server exposing observability data to Claude Code (plus the summarize tools — `list_summarize_candidates`, `summarize_prep`, `summarize_check`, `summarize_apply`, `summarize_undo`; see `core/summarize/`). Uses either a read-only DuckDB connection or HTTP proxy to `tj serve`. Initialized via `init()` from `cmd_mcp.py`.
- **`tokenjam/cli/main.py`**: Root Click group with global options (`--config`, `--json`, `--no-color`, `--db`, `--agent`, `-v`). Registers all subcommands.

### CLI Commands

`tj --help` lists all commands; most are self-explanatory. Non-obvious ones:

- **`tj quickstart`** (`cmd_quickstart.py`) — the **zero-install / zero-config first run** (issue #6). Opens a **transient `InMemoryBackend`** (nothing written to `~/.tj`, no config read/written, no daemon started or contacted), backfills `~/.claude/projects/*.jsonl` into it via `ingest_claude_code`, then renders quota composition (reusing `core/context_diagnostic.py` from #4) + a session timeline (`core/session_timeline.py`). Output leads with ccusage-parity framing ("reads the same files ccusage does"). It is in `no_db_commands` so the CLI never opens the on-disk DB / trips the daemon lock for it. The intended zero-install invocations are `npx tokenjam` (via the `npm-wrapper/` package — the bare `tj` name is squatted on npm; the wrapper routes bare `npx tokenjam` to `quickstart`) and `uvx --from tokenjam tj quickstart`. `tj onboard` remains the opt-in "go deeper" (daemon/MCP/live) path. **Bare `tj` (no subcommand) prints the branded home screen** (`cli/home.py` `print_home()`, #240) — banner + next-best-action, reading only config presence (never the DB) — not quickstart; `--help`/`--version` are eager and still short-circuit.
- **`tj demo [scenario]`** (`cmd_demo.py`) — runs Agent Incident Library scenarios (zero-config, no API keys). `tj demo` lists all; `tj demo retry-loop` runs one.
- **`tj doctor`** (`cmd_doctor.py`) — health checks (config, DB, secrets, webhooks, drift readiness, schema-vs-capture consistency). Exit 0 = ok, 1 = warnings, 2 = errors.
- **`tj optimize`** (`cmd_optimize.py`) — seven analyzers, registry-driven. **Analyzers are positional args** (not `--finding <name>`): `tj optimize downsize cache trim` runs three; bare `tj optimize` runs all. Registered names: `downsize`, `cache`, `cache-recommend`, `script`, `reuse`, `trim`, `budget-projection`. Flags: `--since 30d`, `--budget <provider>`, `--budget-usd <amount>`, `--compare <period>` (window-cost diff vs prior period; accepts `previous` / `last-week` / `last-month` / `last-7d` / `last-30d` / `YYYY-MM-DD:YYYY-MM-DD`), `--export-config <target>` (writes a routing snippet — currently `claude-code` — under `~/.config/tokenjam/exports/`; no `--apply` flag by design). Plan-tier-aware rendering: subscription users see "implied API value" framing and token-share savings (never dollar "spend"); local users see token-only framing; unknown-plan users see dollar figures suppressed with a `tj onboard --reconfigure` hint. Works alongside a running `tj serve` via the `/api/v1/optimize` HTTP fallback when the DuckDB write lock is held by the daemon.
- **`tj tokenmaxx`** (`cmd_tokenmaxx.py`) — shareable **quota/efficiency card** (#7; reframed from the old spend-tier card for the June-2026 "tokenminimizing" shift — an efficiency brag, never a spend brag). Leads with the context-COMPOSITION headline pulled from #4's `compute_context_diagnostic` (`tokenjam/core/context_diagnostic.py`): what share of quota went to *overhead* (re-reading history / CLAUDE.md / tool output, i.e. `diag.reread_share`) vs *real work* (uncached input + output). Classifies into 5 efficiency tiers keyed on the overhead share (TokenMinimizer ≤30% / LeanOperator ≤50% / SteadyState ≤70% / ContextHeavy ≤85% / QuotaSink) — lower overhead = leaner = a better tier. **Quota-native (mirrors #5's polarity):** the headline renders as a token-share / "% of cycle tokens" via `core/framing` (the same `_quota_share` path `cmd_context` uses); dollars are demoted to a secondary "Implied API value" line shown ONLY for `api` plans and suppressed for subscription / local / unknown. Needs a **direct DuckDB connection** (reads context composition; can't run against a live `tj serve` — same as `tj context` / `tj quota-audit`). `--weekly` is a 7-day "Quota Wrapped" recap preset. Output is a bordered Panel designed for screenshotting; the action line points at `tj context` (reclaim overhead) or `tj optimize`. Honesty discipline (Rule 14): the efficiency number is a measured token share, never a guaranteed saving.
- **`tj cost`** (`cmd_cost.py`) — cost breakdown by `--group-by agent|model|day|tool`. Same `--compare <period>` flag as `tj optimize` for window-over-window diffs (▲/▼ indicators, per-agent and per-model top-shifts, dollar + token deltas).
- **`tj backfill <source>`** (`cmd_backfill.py`) — ingest historical telemetry from external sources. Subcommands: `claude-code` (parses `~/.claude/projects/*.jsonl`, auto-invoked at the end of `tj onboard --claude-code`), `langfuse` (live API or JSON dump), `helicone` (live API or JSON dump), `otlp` (raw OTLP JSON via URL or file — reuses the same parser as the live `POST /api/v1/spans` route). All idempotent via deterministic span IDs.
- **`tj onboard`** (`cmd_onboard.py`) — `--claude-code` and `--codex` flags trigger integration-specific flows (writing to the **global** config). All paths — including plain `tj onboard` — prompt for plan tier (api / pro / max_5x / max_20x for Anthropic; api / plus / team / enterprise for OpenAI) and write it to `[budget.<provider>] plan = "..."`; `--plan <tier>` sets it non-interactively (issue #4). The plain path is Claude-first: its interactive prompt offers the Anthropic tiers, and an OpenAI-only `--plan` (plus/team/enterprise) is routed to `[budget.openai]`. Supports `--reconfigure` to re-prompt against an existing config. Does NOT auto-write a default `usd = 200` cycle ceiling — subscription users get only the `plan` field; API users are explicitly asked whether they want a self-imposed ceiling.
- **`tj report`** (`cmd_report.py`) — generates standalone HTML visualizations of analyzer findings. `tj report --trim [<agent_id>]` renders the Trim analyzer's per-token significance (was `--bloat` pre-0.3.1, renamed alongside the analyzer's registry string); `tj report --reuse [<agent_id>]` renders the Reuse analyzer's per-cluster planning skeleton (HTML + Markdown sidecars). Writes to `~/.cache/tokenjam/reports/` (override via `TOKENJAM_REPORT_DIR`) and opens in the default browser. `--reuse` works when the daemon holds the DB lock: like `tj optimize`, it dispatches to `ApiBackend.fetch_reuse_clusters` (`GET /api/v1/reuse/clusters`) and renders from the HTTP payload (`write_reuse_report(..., planning_texts=...)`) instead of a direct DB connection (#154). `--trim` remains DB-direct only.
- **`tj policy list`** (`cmd_policy.py`) — read-only preview of the unified policy surface. Consolidates existing `[alerts]`, `[alerts.channels]`, `[defaults.budget]`, `[budget.<provider>]`, per-agent `budget`/`drift`/`sensitive_actions`/`output_schema`, and `[capture]` config into one table; each row carries its source TOML section. Supports `--json`. `tj policy add | edit | apply | remove | test` are intentionally absent this sprint — the unified config migration is next sprint's work. `policy` is in `no_db_commands` in `cli/main.py` so it doesn't open the DB. Rich source-section strings (`[budget.anthropic]`, `[[alerts.channels]]`) must be passed through `rich.markup.escape()` before rendering — otherwise Rich consumes them as style tags.

- **`tj summarize`** (`cmd_summarize.py`) — structure-aware prompt summarization (advisory; new feature). `list` scans for prompt files worth summarizing (catalog-default; any scope-widening flag opens it to `*.md`) and estimates the per-call token saving — read-only. `prep`/`check` are the mechanism: `prep` wraps a prompt's structure behind id'd `<tj-keep>` markers and emits it for rewriting (CLI manual/copy, or the MCP `summarize_prep`/`summarize_check` tools for in-session rewrites); `check` re-reads + hash-guards the file, restores every block verbatim by id, and stages the result **only if structure survives** (a hard gate). No scratch state — the file on disk is the source of truth; `summarize` is in `no_db_commands` (config-only). `apply` writes a staged rewrite back to the file — default **dry-run** (it prints the unified diff), `--go` writes — guarded by an owner + content-hash + symlink check and a gzip backup; `undo` restores from that backup (refusing on drift). `prep --via claude-p` (your local `claude -p`, no key) or `prep --via api` (Anthropic with your `TJ_ANTHROPIC_API_KEY` + the required `[summarize] api_model`, reporting a "pays for itself" amortization) runs the rewrite for you in one shot — both still pass the `check` gate before staging.

- **`tj pricing list`** (`cmd_pricing.py`) — read-only inspection of the resolved model pricing table. Prints one row per `(provider, model)` with `input` / `output` / `cache_read` / `cache_write` rates and a `source` column (`override` / `packaged`) showing which layer the rate resolved from. Supports `--json` and `--model <substring>` (case-insensitive filter). `pricing` is in `no_db_commands` (config-only). The source derivation lives in `core/pricing.load_pricing_sources()` (public) — the CLI reads it rather than re-deriving precedence; a listed row is always `packaged` or `override` (the `$0.50`/`$2.00` default applies only to models absent from the table, which never appear here). The `set` half of a future `tj pricing set|list` is not built yet (#282 shipped `list` only).

All commands support `--json` for machine-readable output. Commands that query alerts use exit code 1 if active (unacknowledged, unsuppressed) alerts exist.

**CLI testing pattern:** Tests use `click.testing.CliRunner` with `unittest.mock.patch` on `tokenjam.cli.main.load_config` and `tokenjam.cli.main.open_db` to inject an `InMemoryBackend` and test config. See `tests/integration/test_cli.py`. Note: `cmd_doctor` opens its own DuckDB connection via `config.storage.path` to verify writability — in tests you must set this to a real temp path (e.g. `tmp_path / "test.duckdb"`).

**`no_db_commands` in `cli/main.py`:** Commands that don't open the DB at startup — currently `{stop, uninstall, onboard, mcp, demo, policy, proxy, summarize, pricing}`. New commands that read only from config (or do their own DB connection later) should be added to this set so they work when `tj serve` holds the write lock. Tests for these commands can patch `open_db` with `side_effect=AssertionError(...)` to verify they never touch the DB.

**Test factories:** `tests/factories.py` provides `make_llm_span(billing_account="anthropic", ...)` and `make_session(plan_tier="api", ...)` with safe defaults that preserve existing test behavior. Tests exercising subscription / local / unknown plan-tier rendering paths should pass the field explicitly.

### REST API

The API has two auth layers:
1. **Ingest auth** (middleware): `POST /api/v1/spans` requires `Authorization: Bearer <ingest_secret>`. Handled by `IngestAuthMiddleware`, which returns a `JSONResponse` directly — do **not** use `HTTPException` in `BaseHTTPMiddleware.dispatch` as it won't be caught by FastAPI's exception handler.
2. **API key auth** (dependency): All GET endpoints use `Depends(require_api_key)`. Only enforced when `api.auth.enabled = true`.

`POST /api/v1/spans` accepts OTLP JSON (`{"resourceSpans": [...]}`). Partial failures return 200 with `ingested`/`rejected` counts — 400 only if the entire body is malformed. The route parses OTLP spans into `NormalizedSpan` and feeds each through `IngestPipeline.process()`. Key parsing details: resource attributes are merged with span attributes (span wins on conflict); OTLP timestamps are nanosecond strings; OTLP `intValue` fields are strings (per spec for large numbers); unknown attribute value types silently become `None`.

`GET /metrics` generates Prometheus text format by querying the DB on each request (not using the OTel Prometheus exporter), so data is accurate after restarts. No caching — expensive on large datasets.

For `GET /api/v1/drift`, if `agent_id` is missing, return `JSONResponse(status_code=400)` — do not use a union return type like `dict | JSONResponse` as FastAPI cannot generate a response model for it. Use `response_model=None` on the decorator instead.

Integration tests use `httpx.AsyncClient` with `httpx.ASGITransport(app=app)` against `InMemoryBackend`. Synthetic alert tests use `unittest.mock.MagicMock` for the DB — you must explicitly set up `db.get_recent_spans.return_value` before calling `engine.evaluate()`, and silence channels with `engine.dispatcher.channels = []`.

### Web UI ("TokenJam Lens")

`tokenjam/ui/index.html` is the served dashboard — a **single-file Preact + htm SPA** (no build step, no TypeScript, no client-side router). "TokenJam Lens" is the **brand only**: it appears in `<title>`, the sidebar wordmark, and the OpenAPI title, but never in module names, route paths, or config keys. Screens: **Overview** (the default landing route — a triage front door), Status, Traces, Cost, Alerts, Drift, Optimize, Budget.

- **Offline-first (Critical Rule 18):** every JS/CSS dep is vendored under `tokenjam/ui/vendor/` — Preact + hooks + htm (ESM via `<script type="importmap">`) and **uPlot** (vendored IIFE global `uPlot` + CSS, pinned in `docs/internal/lens-vendor-versions.md`). No render-time external HTTP. `tests/unit/test_ui_offline.py` enforces this; clickable `<a href>` links are the only allowed external URLs.
- **Single compute path:** the UI reads everything from the REST API and **never re-implements analysis, aggregation, or plan-tier framing in JS** — it consumes the `framing` block (see `core/framing.py`). If the UI needs a number, extend the endpoint; don't compute it client-side.
- **URL is the source of truth for filters:** state lives in the hash + query params (`#/cost?since=7d&group_by=model`); `getRoute()` parses it, `navigate()` writes it back omitting defaults. Window vocabulary matches the CLI (`1h`/`24h`/`7d`/`30d`/`90d` + `YYYY-MM-DD:YYYY-MM-DD`). The default landing route is Overview (empty hash → `getRoute()` returns `overview`; do **not** re-introduce a render-time `location.hash = ...` redirect — it raced the first render, issue #132).
- **Charts:** `SpendChart` wraps uPlot, reads CSS custom properties (`--chart-1..5`) so it re-themes, and has a cursor tooltip. The spend chart spans the **full selected window** with zero-fill: `/api/v1/cost` returns a window-bucketed `series` (hourly buckets for ≤2-day windows, daily otherwise; epoch-second `bucket` keys) plus `series_bucket` + `window_start`/`window_end`, and the UI builds a continuous grid + pins the x-scale to the window (issues #133/#136).
- **Run-rate** is a single linear figure projected to the end of the current calendar cycle (`daily_rate × days-remaining`), captioned "not a forecast". The forecasting boundary is deliberate: linear run-rate only — no EWMA, seasonality, or anomaly detection.
- **Polling:** the Overview auto-refreshes every 30s only while the tab is visible (`document.visibilityState`) and **fetches its endpoints in parallel** via `Promise.all` (the daemon DB layer is concurrency-safe since #124 — per-thread cursors). The error handling is deliberately asymmetric: `/cost` is load-bearing (no `.catch` — its failure surfaces the error state), while the other five panels each carry a `.catch` fallback so one failing panel renders empty rather than blanking the Overview. Don't unify them. Detail screens refresh on user action.
- **Testing the UI (no JS runner in CI):** the Python `test` job can't run JS, so UI fixes are guarded by **static-grep regression tests** in `tests/unit/test_lens_ui_regression.py` (assert buggy patterns are *gone* and new helpers are present) plus `test_ui_offline.py`. When iterating locally, validate syntax with `node --check` on the extracted `<script type="module">` block, and verify visually by running `tj serve` (or a seeded `create_app` + uvicorn on an alt port) and screenshotting with headless Chrome — there is intentionally no Playwright/Cypress.

### Session Continuity

When a span has a `conversation_id` matching an existing session, it's attributed to that session (even across process restarts). New `conversation_id` = new session.

## Critical Rules

1. **DuckDB only** — never import `sqlite3` or write SQLite-style queries. Use `TIMESTAMPTZ` not `TEXT` for timestamps, `JSON` not `TEXT` for JSON. When extracting dates from `TIMESTAMPTZ` columns, always use `CAST(col AT TIME ZONE 'UTC' AS DATE)` — bare `CAST(col AS DATE)` converts to the local timezone first, causing mismatches with Python's `utcnow().date()`.
2. **TOML binary mode** — `tomllib.load()` requires `open(path, "rb")` not `"r"`. Text mode raises `TypeError` at runtime. Use the conditional import: `tomllib` (3.11+) or `tomli` (3.10). Writing config uses `tomli_w`.
3. **`@watch()` alone does NOT create LLM spans** — only session start/end. Provider patches (`patch_anthropic()`, `patch_openai()`, etc.) are needed for individual LLM call spans.
4. **Ingest auth** — `POST /api/v1/spans` requires `Authorization: Bearer <ingest_secret>` from `security.ingest_secret` in `tj.toml`.
5. **Alert content stripping** — remove `gen_ai.prompt.content`, `gen_ai.completion.content`, `gen_ai.tool.input`, `gen_ai.tool.output` from alert payloads sent to external channels unless `alerts.include_captured_content = true`. Stdout and file channels always get full payload. Note: content is also stripped at *ingest* (before DB write) by `strip_captured_content()` in `core/ingest.py` per the four `[capture]` toggles (`prompts` / `completions` / `tool_inputs` / `tool_outputs`) — so the alert flag is moot when the corresponding capture flag is off.
6. **No unicode bullets** — never hardcode `•` or `\u2022`; Rich handles bullet formatting.
7. **Parameterised SQL only** — never use f-string SQL.
8. **All test spans via factory** — never construct `NormalizedSpan` directly in tests; use `tests/factories.py` (`make_llm_span`, `make_session`, `make_tool_span`, `make_session_with_spans`).
9. **Use `utcnow()` for timestamps** — always use `tokenjam.utils.time_parse.utcnow()` instead of `datetime.now()` or `datetime.utcnow()`. It returns timezone-aware UTC datetimes.
10. **Use semconv constants** — reference `GenAIAttributes` and `TjAttributes` from `tokenjam/otel/semconv.py` instead of hardcoding OTel attribute name strings.
11. **OTel TracerProvider is global and set-once** — `trace.set_tracer_provider()` only works once per process. In tests, set the provider once at module level (not per-test in a fixture) and clear spans between tests. Use a custom `_CollectingExporter(SpanExporter)` since `InMemorySpanExporter` is not available in the installed OTel version. See `tests/agents/test_mock_scenarios.py` for the SDK test pattern and `tests/integration/test_full_pipeline.py` for the pipeline pattern.
12. **New SDK integrations must call `ensure_initialised()`** — every `patch_*()` convenience function must call `from tokenjam.sdk.bootstrap import ensure_initialised; ensure_initialised()` before installing hooks. This lazily bootstraps the TracerProvider + IngestPipeline on first use.
13. **PyPI package name is `tokenjam`, not `ocw`** — the package on PyPI is `tokenjam`. The CLI command is `tj`. The Python package directory is `tokenjam/`. **Recommended install: `pipx install tokenjam`** (sidesteps PEP 668 on Homebrew Python and Debian 12+/Ubuntu 24+). `pip install tokenjam` works inside a clean venv but fails on system Python with a misleading externally-managed-environment error. Never write `pip install ocw` in docs, examples, or comments.
14. **`tj optimize` output must never claim quality equivalence** — the `downsize` finding flags structural candidates only. Every user-visible string says "looks like" / "candidate" / "review before switching" — never "safe to downgrade" or "would have worked." The `MODEL_DOWNGRADE_CAVEAT` constant lives on `DowngradeFinding` as a dataclass default so it can't be removed by accident; it must also appear in human-readable CLI output. The same honesty discipline applies to all other analyzers — `cache` ("you're getting X% of available caching"), `cache-recommend` (Anthropic-only, structural prefix detection), `script` ("structural shape matches", "review before replacing with a script"), `trim` ("predicted low-significance regions; review before editing"). `tj optimize --export-config` snippets bake the caveat block into the JSONC output as comments.
15. **Version bump on release** — both `pyproject.toml` (`version = "X.Y.Z"`) and `sdk-ts/package.json` (`"version": "X.Y.Z"`) must be bumped to the new version before creating a GitHub release. The publish workflows (`publish-pypi.yml`, `publish-npm.yml`) trigger on `release published` events and will fail with 403 if the version already exists on PyPI/npm.
16. **New optimize analyzers self-register** — drop a `.py` file under `tokenjam/core/optimize/analyzers/` with a function decorated `@register("name")` taking `AnalyzerContext`. Auto-discovery in `analyzers/__init__.py` walks the directory at import time. `cmd_optimize.py`'s positional `findings` Click choices read from `ANALYZER_REGISTRY.keys()` at decoration — no edits needed there. If your analyzer depends on (or is depended on by) another, append it to `ANALYZER_ORDER` in `runner.py` at the right position. Wave-2 analyzers attach their findings to `OptimizeReport.findings[name]` (generic dict); the older `downsize` (registered name; file is `model_downgrade.py`) and `budget-projection` analyzers retain typed slots on `OptimizeReport` for backwards compat with `cmd_optimize` and the MCP server.
17. **OTLP parsing has one home** — `tokenjam/otel/otlp_parsing.py`. Both the live `POST /api/v1/spans` route and the `tj backfill otlp` adapter import `parse_otlp_span` and `extract_resource_attrs` from there. If you need to extend OTLP attribute extraction, do it once in that module; do not copy-paste into either caller.
18. **Web UI must work fully offline** — `tokenjam/ui/index.html` is the served dashboard ("TokenJam Lens"; see Architecture → Web UI). It is intentionally a single-file SPA with **zero external HTTP loads at render time**. Preact + hooks + htm + **uPlot** are vendored under `tokenjam/ui/vendor/` (ESM via `<script type="importmap">`; uPlot as a plain `<script>` IIFE global); fonts use system-font fallbacks (no Google Fonts); the favicon is inlined as a `data:` URL. The FastAPI app mounts `/ui/vendor` as `StaticFiles`. The `tests/unit/test_ui_offline.py` regression test asserts no render-time external URLs exist anywhere outside `<a href>` (clickable links to github.com are fine — they only fetch on click) and that vendored CSS has no external `url()`. If you add a CDN font, script, or stylesheet, that test will fail. Vendor the asset locally instead. See issue #87 + PR #88.
19. **Analyzer registry names ≠ file names** — registry strings (`downsize`, `cache`, `script`, `trim`) are decoupled from Python module filenames (`model_downgrade.py`, `cache_efficacy.py`, `workflow_restructure.py`, `prompt_bloat.py`). The 0.3.1 rename only changed `@register("...")` strings; file names stayed for git-blame continuity. When grepping for an analyzer, search both the registry string AND the older file-name keyword.
20. **`.tj/config.toml` is untracked and must stay that way** — the file contains a live per-install `ingest_secret` and is regenerated by `tj onboard` / `tj serve`. It was committed in error from v0.2.0 through v0.3.5 (leaked secret in git history; see PR #145 + issue #141 finding #6). `.gitignore` covers it, and `tests/unit/test_no_tracked_dev_secrets.py` fails CI if it's re-added to the index. If you see `.tj/config.toml` in your `git status` as modified or new, that's expected — just don't `git add` it.

## Config

Config is TOML, discovered at: `tj.toml` -> `.tj/config.toml` -> `~/.config/tj/config.toml`. Override with `--config` or `TJ_CONFIG` env var. Full config hierarchy is in `tokenjam/core/config.py` (`TjConfig` dataclass).

Two distinct budget concepts coexist — do not conflate:
- **`[defaults.budget]` / `[agents.<id>.budget]`** (`daily_usd`, `session_usd`) — per-agent alert thresholds checked on every span by `AlertEngine`.
- **`[budget.<provider>]`** (`plan`, `usd`, `cycle_start_day`, `applies_to_services`) — per-provider budget config. `plan` is the user's declared plan tier (api / pro / max_5x / max_20x / plus / team / enterprise / local), prompted for by `tj onboard` and used by `IngestPipeline` to populate `SessionRecord.plan_tier` at session creation. `usd` is a periodic monthly ceiling used only by `tj optimize` budget-projection (read-only; no alerts fire from it). Onboard does NOT auto-write `usd = 200` — subscription users get only the `plan` field; API users are explicitly asked whether they want a self-imposed ceiling. The budget-projection analyzer scopes spend by `provider` column and (optionally) by `agent_id IN applies_to_services`.

`tj onboard --claude-code` and `tj onboard --codex` always write to the **global** config (`~/.config/tj/config.toml`) regardless of cwd. This is intentional: each coding-agent integration reads one ingest secret from a single global location (`~/.claude/settings.json` or `~/.codex/config.toml`), and per-project configs would rotate that secret on every onboard, breaking auth for previously onboarded projects. Onboarded Claude Code project paths are tracked in `~/.config/tj/projects.json` for clean uninstall. Codex onboarding is fully project-agnostic — Codex hardcodes `service.name=codex_exec` in its binary, so there is one Codex agent ID for all projects.

## Daemon (launchd / systemd)

`tj onboard` (and `tj onboard --claude-code` / `--codex`) installs a background daemon that runs `tj serve` on login:
- **macOS**: `~/Library/LaunchAgents/com.tokenjam.serve.plist` — loaded via `launchctl load`. Logs at `/tmp/tj-serve.{out,err}`.
- **Linux**: `~/.config/systemd/user/tokenjam.service` — enabled via `systemctl --user enable --now tokenjam`.
- **Other**: skipped with a notice; user runs `tj serve` manually.

Reinstall behavior: `--claude-code` and `--codex` onboard check `_daemon_already_running()` (launchctl list / systemctl is-active) and skip reinstall when the daemon is up unless `--force` is passed. This avoids spurious "Background Items Added" prompts on macOS during second-project onboards. The launchd path always uses `launchctl unload -w` then `launchctl load -w` — the `-w` flag clears any Disabled=true entry from the launchd database (`tj stop` writes Disabled=true via `launchctl unload -w`), without which a subsequent plain `launchctl load` is a silent no-op. Use `tj stop` to halt the daemon, `tj uninstall` to remove unit files. `tj stop` also sweeps for any orphan foreground `tj serve` processes (e.g. from a manual `tj serve &`) so it reliably frees port 7391.

`tj serve` writes its resolved config path to `~/.local/share/tj/server.state` at startup. This is informational — onboarding flows (`--claude-code` and `--codex`) always write to the global config, so server.state is not used for secret-sync.

## MCP Server

**The MCP is an SDK / API surface, not a Claude Code / Codex one (#59).** It puts tj *in the request path* — the right place for SDK / API integrations doing real-time enforcement/policy/budgets. It is deliberately **not** wired for Claude Code / Codex subscription users: an in-loop MCP is a per-turn quota burden on them (a measured A/B showed **+36%** model-weighted quota vs a no-tj control). Those users get tj **out-of-band**: the zero-token statusline (`tj statusline`, wired by `tj onboard --claude-code`) plus OTel telemetry ingest. `tj mcp` still works for anyone who invokes it; onboarding just no longer defaults CC/Codex users into it.

`tj mcp` starts a FastMCP stdio server. The connection mode is chosen at startup by `cmd_mcp.py`:
1. If `tj serve` is reachable on `config.api.{host,port}`, MCP proxies to it via HTTP (live ingest visible).
2. Otherwise it tries to spawn `tj serve` in the background and waits up to 10s for the port.
3. If neither works, it falls back to a **read-only DuckDB connection** — read tools still work, but newly ingested spans won't appear until restart.
4. If no config file is found, `init()` is skipped and tools return a no-config sentinel.

SDK / API users who want the in-loop tools can wire it manually: `claude mcp add tj --scope user -- tj mcp`. The `--claude-code` and `--codex` onboard flows **no longer** register the MCP (they wire the out-of-band statusline / OTel instead), and a re-onboard retires any tj-managed `[mcp_servers.tj]` block a previous version wrote to `~/.codex/config.toml`.

## Codex CLI Integration

`tj onboard --codex` writes an `[otel]` block to `~/.codex/config.toml` (out-of-band telemetry only). It does **not** register the tj MCP for Codex (#59) — Codex has no statusline surface, so tj stays fully out-of-band via OTel + the `tj` CLI; a re-onboard retires any `[mcp_servers.tj]` block a previous version wrote. Notes:
- Codex hardcodes `service.name=codex_exec` in its binary and silently ignores `[otel.resource]`, so onboarding does **not** write that block — all Codex traces land under the `codex_exec` agent ID regardless of project. Onboarding is one-time global, not per-project.
- Codex emits OTLP **logs** (not spans) to `/v1/logs`. `tokenjam/api/routes/logs.py` converts Codex events (`sse_event`, `user_prompt`, `tool_decision`, `tool_result`, `api_request`) into normalized spans for cost/drift/alerting. Event name is read from `attrs["event.name"]` when the OTLP body is empty (Codex schema quirk); epoch `timeUnixNano=0` falls back to `attrs["event.timestamp"]` ISO-8601. The `/v1/logs` endpoint also silently accepts `resourceSpans`/`resourceMetrics` because Codex's exporter reuses one endpoint for all signal types.
- Re-running `tj onboard --codex` is a no-op only when both `[otel]` and `[mcp_servers.tj]` are present in `~/.codex/config.toml`. Re-onboarding either Codex or Claude Code cross-syncs the ingest secret into the other's config if it's already configured.

## Examples Convention

Each provider integration in `examples/single_provider/` and each framework in `examples/single_framework/` lives in **its own file** — when adding a new SDK integration, mirror this layout (one demo file per integration) so the examples directory stays a 1:1 map of supported integrations. Multi-provider/framework demos go in `examples/multi/`; alert and drift demos that need no API keys go in `examples/alerts_and_drift/`.

The Agent Incident Library at `incidents/` is separate: each scenario is a `scenario.py` + `README.md` pair, invoked via `tj demo <scenario>`. Scenarios inject synthetic spans through `tokenjam/demo/env.py` to simulate real failures (retry-loop, surprise-cost, hallucination-drift) without API keys or a live server.

## Pricing

Model pricing lives in `tokenjam/pricing/models.toml` (USD per million tokens) — the packaged file `core/pricing.py` loads via `PRICING_FILE = Path(__file__).parent.parent / "pricing" / "models.toml"`. There is no repo-root `pricing/` copy (it was moved into the package in v0.1.x so it ships in the wheel; editing a repo-root file would have no runtime effect). Structure: `[provider.model_name]` with `input_per_mtok`, `output_per_mtok`, and optional `cache_read_per_mtok`/`cache_write_per_mtok`. Unknown models fall back to default rates ($0.50/$2.00 per MTok) with a logged warning. The pricing table is LRU-cached at process startup — restart to pick up changes.

The packaged table is community-maintained: submit a PR editing `tokenjam/pricing/models.toml` when provider prices change. No code changes needed — the file is loaded at runtime.

**Local user overrides (no PR needed)** — users correct or add rates *for their own install* via override layers that `core/pricing.py` merges over the packaged table. Two sources, two key forms (see `docs/configuration.md` → "Pricing overrides" for the user-facing version):

- **Sources** (lowest priority first; later wins): the packaged `models.toml`, then a standalone file (`~/.config/tj/pricing.toml`, or `TJ_PRICING_FILE`), then a `[pricing]` section in the main config (`tj.toml`). The project-local config `[pricing]` wins over the global standalone file.
- **Key forms** — told apart deterministically by section name in `_split_pricing_raw()` (the reserved `models` section vs everything-else-is-a-provider; no value-shape guessing, no ordering dependency):
  - **Provider-keyed** (`[pricing.anthropic]` / `[anthropic]` whose values are model sub-tables) — merged per `(provider, model)` over the packaged table. This is the long-standing `[provider.model]` format.
  - **Model-keyed** (the reserved `[pricing.models]` section in tj.toml, or `[models]` in the standalone file) — keyed by **bare model name**, applied **regardless of inferred provider**. This is the attribution-proof path: it prices a span even when the provider resolved to `"unknown"` (the #194 open-weight class). `models` is a reserved key (`MODEL_SECTION_KEY`), never a provider, so the forms never collide.
- **`get_rates(provider, model)` lookup order** (first match wins): model-keyed override → provider-keyed table (user `[provider.model]` over packaged) → `None` (→ `calculate_cost` applies the `$0.50`/`$2.00` default and logs once). Each step tries an exact match, then strips a trailing `-YYYYMMDD` suffix.
- Both layers are LRU-cached (`load_pricing_table` + `load_model_pricing_overrides`); call `clear_pricing_cache()` or **restart the daemon** to pick up an edit.

The packaged table stays the zero-config default — the override is a *layer*, never a replacement; no user ever has to declare a rate to get started. Read-only inspection ships via `tj pricing list` (see CLI Commands; #282). The `set` half — a `tj pricing set` to edit overrides without hand-writing TOML — is not built yet.

## CI

GitHub Actions workflow at `.github/workflows/ci.yml` runs on push/PR to `main`:
- **`test`** job: Python 3.10/3.11/3.12 matrix — `ruff check` and `mypy` (continue-on-error), then `pytest tests/unit/ tests/synthetic/ tests/agents/ tests/integration/` (blocking)
- **`test-ts`** job: Node 20 — `npm install && npm test` in `sdk-ts/`

All steps are blocking — lint, typecheck, and tests must pass for CI to go green.

There is no pre-commit configuration in this repo; `ruff` and `mypy` only run in CI. Run them locally before pushing.

## Growth instrumentation — weekly traffic archive

A separate `.github/workflows/traffic-archive.yml` runs every Sunday at 12:00 UTC (also `workflow_dispatch`) and archives the GitHub Traffic API (views / clones / referrers / paths — the only sources with the 14-day retention problem) into a JSON file at `traffic/<year>-W<week>.json`. **The archive lives on its own orphan branch `traffic-data`, not on `main`** — code and telemetry are deliberately separated, and `main`'s branch protection stays intact.

Key facts:
- The workflow uses `${{ secrets.TRAFFIC_PAT }}` (fine-grained PAT, resource owner = `Metabuilder-Labs` org, `Administration: Read` on this repo) for the Traffic API read — the default `GITHUB_TOKEN` returns 403 there. The push step uses the default `GITHUB_TOKEN` since `traffic-data` is unprotected.
- The orphan-branch design was the close-out after several other paths (PAT-as-owner push, classic-protection bypass, rulesets bot-bypass) hit GitHub UI / API limitations. See PRs #171 / #172 / #173 for the trail. Don't try to move the archive back onto `main` — that path was explored and dead-ended.
- The archive is the only growth-instrumentation surface in this repo. Health-check + spreadsheet-fill workflows were considered but descoped (Cowork sandbox couldn't reach external APIs, and GH Actions for the same was deemed overkill for a solo project). Manual review is the current model — fetch the JSON when wanted, hand it to an agent for analysis.
- `growth/README.md` documents the schema + read-it-manually flow.
- TRAFFIC_PAT expires Jun 20 2027 — renew before then.

When adding any new automation that needs to *write* to the repo on a schedule, follow the same pattern: data on its own unprotected ref, code on `main`. Don't try to make `main` accept bot commits.

## Releases

PyPI and npm publishes are triggered by GitHub Release events (`.github/workflows/publish-pypi.yml`, `publish-npm.yml`, both `on: release: types: [published]`). Release flow:

1. Bump both `pyproject.toml` `version` and `sdk-ts/package.json` `"version"` to the new `X.Y.Z` (see Critical Rule 15).
2. Merge to `main`.
3. Create a GitHub Release with tag `vX.Y.Z` (e.g. via `gh release create vX.Y.Z --generate-notes`). Publishing the release fires both workflows.

If a version already exists on PyPI or npm, the publish workflow fails with 403 — bump again rather than retrying.

## Packaging

Build system is hatchling. `[tool.hatch.build.targets.wheel] packages = ["tokenjam"]` — the package directory is `tokenjam/` (matching the PyPI name); only the *CLI command* is `tj` (`[project.scripts] tj = "tokenjam.cli.main:cli"`). Non-`.py` assets under the package ship in the wheel automatically — this is how the vendored UI (`tokenjam/ui/index.html`, `tokenjam/ui/vendor/*`) and `tokenjam/pricing/models.toml` reach users.

Key runtime dependency: `pytz` is required by DuckDB for `TIMESTAMPTZ` column handling — it's listed explicitly in `dependencies` because DuckDB doesn't declare it on all platforms.

**The `tj` npm wrapper** (`npm-wrapper/`, issue #6) is a separate, dependency-free npm package named `tokenjam` (unscoped, distinct from the `@tokenjam/sdk` SDK package; the bare `tj` name is already taken on npm by an unrelated pub/sub library, so the PACKAGE is `tokenjam` while the installed BIN is still `tj`) whose only job is to make `npx tokenjam` work. `bin/tj.js` shells out to the Python CLI via the first available runner (`uvx --from tokenjam tj` → `pipx run --spec tokenjam tj` → an installed `tj` on PATH) and passes every arg through. Bump its `version` alongside `pyproject.toml`/`sdk-ts` on release. **It is NOT yet published to npm** — building + documenting only (Critical Rule 15's publish flow covers it when the time comes). `npm-wrapper/` has no CI test (no Python to drive in the JS lane); validate it locally with `node -c npm-wrapper/bin/tj.js`.

**Optional extras** (declared under `[project.optional-dependencies]`):
- `tokenjam[bloat]` — `llmlingua>=0.2`, used by the Trim analyzer. Pulls PyTorch + transformers (~2GB). Kept out of base install. The analyzer self-registers without the extra installed; the deferred `import llmlingua` inside the analysis function body raises a typed message pointing the user at the install command.
- Framework extras `[langchain]`, `[crewai]`, `[autogen]`, `[litellm]` for SDK patches.
- `[dev]` for local development (`pytest`, `ruff`, `mypy`, `httpx`).
- `[mcp]` — empty no-op alias. `fastmcp` moved into the base install in v0.3.5 (#101), so the FastMCP stdio server (`tj mcp`) works on a plain `pipx install tokenjam`. The extra is kept so old `tokenjam[mcp]` install commands still succeed; it pulls nothing extra.

## Further Reading

- **[docs/architecture.md](docs/architecture.md)** — design principles, system overview, data flow, SDK internals, alert system, drift detection, MCP server, Claude Code integration, budget system, testing architecture, and the **OTel semconv extensions** section documenting `tokenjam.billing_account` (span attribute) and `tokenjam.plan_tier` (session-level), the `pricing_mode` derivation rules, and why `plan_tier` lives on `SessionRecord` rather than each span.
- **[docs/installation.md](docs/installation.md)** — base install vs optional extras matrix. Documents `tokenjam[bloat]` (the ~2GB torch + transformers extra used by the Trim analyzer), framework adapter extras (`[langchain]` / `[crewai]` / `[autogen]` / `[litellm]`), and the MCP / dev extras.
- **[docs/configuration.md](docs/configuration.md)** — full TOML config surface plus the "Content capture and privacy" section explaining the four `[capture]` toggles and how they interact with `alerts.include_captured_content`.
- **Optimize product pages** — one per user-facing product, all under `docs/optimize/`:
  - [`downsize.md`](docs/optimize/downsize.md) — cheaper-model candidate flagging (registry: `downsize`, file: `model_downgrade.py`)
  - [`cache.md`](docs/optimize/cache.md) — `cache` (current caching ratio) + `cache-recommend` (Anthropic-only breakpoint suggestions)
  - [`script.md`](docs/optimize/script.md) — `script` clustering by `(tool_name, arg_shape)` signature (file: `workflow_restructure.py`)
  - [`trim.md`](docs/optimize/trim.md) — LLMLingua-2 token-significance classifier (`trim`, file: `prompt_bloat.py`), install + capture requirements, performance numbers
- **[AGENTS.md](AGENTS.md)** — codebase conventions for contributors (referenced from the top-level README).
- **Backfill adapters** — `docs/backfill/overview.md` lists the four sources (`claude-code` / `langfuse` / `helicone` / `otlp`) with the partnership-posture framing; per-adapter pages document modes (URL / file), field mapping, idempotency, and v1 limitations.
- **[docs/policy/overview.md](docs/policy/overview.md)** — read-only preview of the unified policy surface (`tj policy list`). Notes that the `add` / `edit` / `apply` subcommands and the underlying `[policy]` config migration land next sprint.
- **Internal specs** — `docs/internal/specs/` is reserved for canonical specs that production code references at long-term. Currently empty (sprint specs have been cleaned up after merge); add new ones here when a feature needs a stable, code-referenced source of truth.
- **[docs/internal/release-smoke-checklist.md](docs/internal/release-smoke-checklist.md)** — fresh-install pre-release gate (clean env → `pipx install` → `tj onboard --claude-code` → verify Lens plan badge + `tj optimize` agree → sane session count). Catches plan-tier / framing regressions a real first-run hits that synthetic-factory tests don't. Its automated counterpart is `tests/integration/test_first_run_roundtrip_239.py`.
