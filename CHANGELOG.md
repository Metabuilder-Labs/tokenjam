# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## Unreleased

This release pivots TokenJam toward cost-optimization for autonomous agents. Four `tj optimize` analyzers — **Downsize**, **Trim**, **Cache**, and **Script** — read the same spans you'd otherwise pay LangSmith or Langfuse to host, then point at concrete cost cuts. Plan-tier-aware rendering means subscription users (Claude Pro, Max, ChatGPT Plus/Team) see "implied API value" framing instead of dollar "spend" claims they didn't pay. New backfill adapters for Langfuse, Helicone, and raw OTLP make the analyzers work against telemetry you already have elsewhere.

### Added

**Cost-optimization analyzers (four new `tj optimize` findings).**

- **Downsize** (`--finding model-downgrade`, ships in base install). Existing analyzer, now with the v1.1 honest-output rendering described below. Flags sessions whose structural shape matches a class of work where a cheaper alternative model is worth reviewing — never claims quality equivalence.
- **Cache** — two findings under one product:
  - `--finding cache-efficacy` (no content capture required). Measures current prompt-caching usage per (provider, model). Anthropic fully supported; OpenAI / Gemini best-effort; Bedrock / LiteLLM / Cohere unsupported in v1. Flags rows with ≥100K input tokens and <30% caching efficacy.
  - `--finding cache-recommend` (Anthropic-only v1, requires `[capture] prompts = true`). Walks captured prompts, hashes the first ~2000 chars, flags prefixes shared by ≥3 calls as `cache_control` breakpoint candidates.
- **Script** (`--finding workflow-restructure`). Clusters sessions by `(tool_name, arg_shape)` signature. Flags clusters with ≥20 instances as candidates for replacement with deterministic shell scripts. `arg_shape` classifies args by type (`file_path` / `command_string` / `json_object` / `array` / `number` / `boolean` / `string`) so structural patterns cluster even when values vary. Degrades to tool-names-only when `capture.tool_inputs = false`.
- **Trim** (`--finding prompt-bloat`, requires optional `tokenjam[bloat]` extra). LLMLingua-2 token-significance classifier identifies long low-significance regions in captured prompts. Self-registers without the extra installed and surfaces a clear install hint on first run if missing. Model downloads on first use (~110MB) and caches under `~/.cache/tokenjam/models/`. Never auto-rewrites prompts — recommendations only.

**Plan-tier-aware rendering (v1.1 honest output).**

- New `plan_tier` column on `SessionRecord` (`api` / `pro` / `max_5x` / `max_20x` / `plus` / `team` / `enterprise` / `local` / `unknown`). `tj onboard --claude-code` and `tj onboard --codex` prompt for the user's plan and write it to `[budget.<provider>] plan = "..."`. `tj onboard --reconfigure` re-runs the prompts against an existing config. `--plan` CLI flag bypasses the interactive prompt for scripted onboards.
- New `billing_account` attribute on every span (`anthropic` / `openai` / `google` / `bedrock` / `local.ollama`). Provider-only identifier set by every integration that writes spans (OTel patches, Claude Code JSONL backfill, OTLP HTTP ingest, OTLP logs ingest, Langfuse / Helicone backfill adapters).
- New `pricing_mode` derived property on `SessionRecord` (`local` / `subscription` / `api` / `unknown`). Single source of truth for renderer branching.
- `tj optimize` renderers branch on `pricing_mode`: subscription users see "implied API value" framing — never a dollar "spend" figure they didn't pay. Header surfaces plan label + monthly fee multiplier; downgrade body reframes savings as token-share against the plan's allocation. Local users see token-only framing. API users see the existing dollar-denominated rendering. JSON output carries top-level `plan` and `pricing_mode` fields plus a `monthly_tokens_freed` field on downgrade findings for non-API plans. Budget projections suppressed for subscription users (no dollar-denominated cap).

**Backfill adapters.**

- `tj backfill langfuse` — `--source-url` (live API) and `--source-file` (JSON dump) modes. Maps Langfuse `Observation` records onto `NormalizedSpan` with deterministic span IDs for idempotent re-runs. `billing_account` derived from model name. Supports `{data: [...]}`, bare list, and NDJSON input shapes. See `docs/backfill/langfuse.md`.
- `tj backfill helicone` — same two-mode pattern. POSTs `/v1/request/query` against Helicone with Bearer auth in live mode. See `docs/backfill/helicone.md`.
- `tj backfill otlp` — generic OTLP-JSON ingestion for sources Langfuse and Helicone don't cover. Reuses the same parser as the live `POST /api/v1/spans` route via `tokenjam/otel/otlp_parsing.py`. See `docs/backfill/otlp.md`.

**Other surfaces.**

- `tj report --bloat [<agent_id>]` — generates an HTML visualization of the Trim analyzer's findings. Output saved under `~/.cache/tokenjam/reports/` and opened in the user's default browser. `--no-open` writes without opening.
- `tj policy list` — read-only preview of a unified `policy` view over existing alerts, drift, schema, sensitive-actions, capture, and per-provider budget configuration. Each row points back to the TOML section it was read from. `--json` supported. The full `tj policy add | edit | apply | remove | test` surface (and the underlying unified `[policy]` config migration) lands next sprint. See `docs/policy/overview.md`.
- `--compare` flag on `tj cost` and `tj optimize` — surfaces a window-cost diff against a prior period. Accepts `previous` / `last-week` / `last-month` / `last-7d` / `last-30d` keywords (equal-length prior window) or `YYYY-MM-DD:YYYY-MM-DD` for explicit ranges. Output includes spend delta, token delta, and top per-agent/per-model shifts with ▲/▼ indicators. `--json` returns a structured `CostDiff` payload.

**Documentation.**

- `docs/installation.md` documents the new `tokenjam[bloat]` optional extra (pulls torch + transformers, ~2GB) and the rest of the optional-extras matrix.
- `docs/configuration.md` now documents the four-flag `[capture]` config (`prompts` / `completions` / `tool_inputs` / `tool_outputs`), the strip-on-ingest gate in `IngestPipeline.process()`, and its precedence with `alerts.include_captured_content`.
- Per-analyzer pages: `docs/optimize/cache.md` (per-provider support table), `docs/optimize/script.md` (worked example of signature definition), `docs/optimize/trim.md` (install + capture requirements + performance numbers).
- `docs/architecture.md` gains a section on the `billing_account` and `plan_tier` semconv extensions, including the `pricing_mode` derivation rules.
- `docs/backfill/overview.md` lists all four backfill sources (claude-code, langfuse, helicone, otlp) with a partnership-posture note (we ingest from these tools; we don't replace them).
- `docs/internal/specs/v1.1-honest-output.md` committed as the canonical reference for Wave-1 rendering decisions.

### Changed
- **`tj optimize --finding` replaces `--only`.** Registry-driven valid choices (`model-downgrade`, `budget-projection`, plus any new analyzer). Repeatable. The `--only model|budget` flag has been removed (no backwards-compat per the no-external-users decision).
- **`tj onboard` no longer auto-writes `[budget.anthropic] usd = 200`.** Subscription users see no auto-written ceiling; API users are explicitly prompted for an optional self-imposed monthly ceiling.
- **`tj status` surfaces unknown plan tiers.** When sessions exist with `plan_tier = 'unknown'`, prints a one-line note pointing the user at `tj onboard --reconfigure`. Exit code unchanged.
- **`tj optimize` plan-tier-aware rendering.** When every session in the window has `plan_tier = 'unknown'`, dollar figures are suppressed and a header note explains why. Mixed / partial-unknown windows render normally with an advisory note.
- **MCP `get_optimize_report` tool.** Now accepts `findings: list[str]` (was `only: str`). Docstring surfaces for both API-billing and subscription-plan-efficiency phrasings.
- **`tj tokenmaxx` tier ladder expanded to six tiers and renamed.** Two highest tiers renamed from `TokenChad` / `TokenGigaChad` to `TokenSuperMaxxer` / `TokenGigaMaxxer`, and a new `TokenMegaMaxxer` tier slots between them covering the 20× – 50× multiplier range. The previous top tier started at 20×; the new top tier starts at 50×, so the absolute headline for very-heavy users is more meaningful. Fire-emoji escalation matches the new tier count: 🔥 → 🔥🔥 → 🔥🔥🔥. JSON output's `tier` field carries the new label string verbatim; consumers reading the `tier` value must update accordingly.

### Fixed
- **Cache-only spans were costed at $0.** A prompt-cache hit (0 new input/output tokens but non-zero cache-read tokens) bills the cache-read rate, but both `calculate_cost` and `CostEngine.process_span` short-circuited on input/output alone, dropping the span as a no-op and under-reporting spend. The early-return guards now fire only when *all* token counts are zero, so cache-read (and cache-write) costs are charged correctly.
- **Cache-write (cache-creation) tokens were dropped on the live ingest path.** The SDK integrations emit `gen_ai.usage.cache_creation_tokens`, the pricing table carries a `cache_write_per_mtok` rate, and `calculate_cost` already priced it — but the OTLP span parser and provider reader only read cache-read tokens, so cache-creation tokens never reached `CostEngine.process_span` and their (higher-rate) cost was never charged. `NormalizedSpan` now carries `cache_write_tokens`, both parsers populate it, and `process_span` charges it. Only the backfill path costed cache-write before; the live path now matches.

### Internal
- **Registry-driven optimize analyzers.** `tokenjam/core/optimize.py` split into `tokenjam/core/optimize/` package with `registry.py`, `runner.py`, `types.py`, and `analyzers/` subpackage using `pkgutil` auto-discovery. New analyzers drop a file under `analyzers/` with a `@register("name")` decorator — nothing else needs editing. See `tokenjam/core/optimize/README.md`.
- **`OptimizeReport.findings` generic dict.** Wave 2 analyzers attach their results here keyed by registration name. Adding a new analyzer no longer requires a typed slot on `OptimizeReport`. The existing typed slots (`downgrade`, `budgets`) stay for backwards compatibility with `cmd_optimize` and the MCP server.
- **`TjAttributes.BILLING_ACCOUNT` and `TjAttributes.PLAN_TIER`** semconv constants, plus `VALID_PLAN_TIERS` / `SUBSCRIPTION_PLAN_TIERS` frozensets in `tokenjam.otel.semconv`.
- **`tokenjam[bloat]` optional dependency.** Pulls `llmlingua>=0.2` (and transitively torch + transformers, ~2GB). Kept out of the base install.
- **DuckDB migration 4** adds `spans.billing_account TEXT` and `sessions.plan_tier TEXT DEFAULT 'unknown'`. New columns use `ALTER TABLE ADD COLUMN`; no backfill heuristics (product has no external users).
- **`IngestPipeline._build_or_update_session`** late-resolves `plan_tier` when a session starts on a tool span (no billing signal) and a later LLM span carries `billing_account`. Once set to a known value, `plan_tier` is never demoted back to `unknown`.
- **Test factories** `make_session(plan_tier="api")` and `make_llm_span(billing_account="anthropic")` carry safe defaults so existing tests behave as before.

## [0.1.7] - 2026-04-13

### Added
- **MCP server (`tj mcp`)** — stdio-based Model Context Protocol server giving Claude Code direct access to TokenJam observability data. 13 tool handlers: status, traces, alerts, budget headroom, cost summary, drift report, tool stats, trace detail, acknowledge alerts, setup project, list sessions, open dashboard. Dual-mode operation: routes queries through REST API when `tj serve` is running, falls back to read-only DuckDB otherwise. Auto-starts `tj serve` on demand.
- **Claude Code integration (`tj onboard --claude-code`)** — one-command setup for Claude Code telemetry. Configures OTLP log exporter in `~/.claude/settings.json`, sets project-level `OTEL_RESOURCE_ATTRIBUTES`, adds Docker-compatible endpoint to shell env, and optionally installs background daemon. Re-runs resync the auth header to fix 401s without manual setup.
- **Logs ingestion (`POST /v1/logs`)** — new OTLP log endpoint that converts Claude Code log events (`api_request`, `tool_result`, `api_error`, `user_prompt`, `tool_decision`) into NormalizedSpans with deterministic trace/span IDs. Spans flow through the standard ingest pipeline for cost, alerts, and drift.
- **`tj drift` CLI** — behavioral drift report with Rich table output showing baseline vs latest session Z-scores per dimension (input tokens, output tokens, duration, tool call count, tool sequence similarity). Color-coded thresholds, `--json` support, exit code 1 if drift detected.
- **`tj budget` CLI + API** — view and set per-agent daily/session cost limits. `GET/POST /api/v1/budget` endpoints. `resolve_effective_budget()` with per-field fallback so each budget dimension independently falls back to defaults.
- **Architecture documentation** (`docs/architecture.md`) — comprehensive architecture doc covering design principles, data flow, SDK internals, alert system, drift detection, MCP server, Claude Code pipeline, and testing architecture.
- `ClaudeCodeEvents` semantic conventions in `tokenjam/otel/semconv.py` for Claude Code log event attributes

### Fixed
- Budget resolution inconsistency between AlertEngine enforcement and CLI display — both now use `resolve_effective_budget()` with field-level merge
- Drift display threshold bug in Z-score comparison
- `tj stop` now passes `-w` to `launchctl unload` to prevent auto-restart on macOS; added Linux systemd support
- Waterfall tooltip clipping for right-edge spans in web UI
- CLAUDE.md install command corrected from `pip install tokenjam` to `pip install tokenjam`

### Improved
- README updated with Claude Code integration section, budget/drift CLI references, MCP server docs
- Web UI: budget headroom display, cost-today column in active sessions table, tooltip polish
- Onboard wizard: expanded for Claude Code workflow, status command enhancements
- MCP tool descriptions optimized for better agent tool selection

### Changed
- Removed historical task specs from `.claude/specs/` (design intent preserved in `docs/architecture.md`)
- CLAUDE.md "Task Specs" section replaced with "Further Reading" linking to architecture doc
- 338 tests passing (up from 223)

## [0.1.6] - 2026-04-08

### Improved
- **`tj onboard` UX overhaul**
  - Removed agent ID prompt — agents are auto-discovered when spans arrive
  - Budget is now a global default (`[defaults.budget]`) that applies to all agents
  - Per-agent `[agents.X.budget]` overrides the default when configured
  - Cleaner budget prompt: "Daily budget in USD per agent (0 = no limit, default 5)"
  - Daemon installs automatically (skip with `--no-daemon`)
  - Rich next-steps output with instrumentation code example
  - Minimal config file with commented per-agent example

## [0.1.5] - 2026-04-08

### Fixed
- **Pricing file missing from pip wheel** — `pricing/models.toml` was at the repo root, outside the `tokenjam/` package. Moved to `tokenjam/pricing/models.toml` so it's included in the wheel. All costs showed `$0.000000` in v0.1.4.

### Improved
- **Web UI polish** — custom hover tooltips on waterfall bars (cost, duration, model), back arrow on trace detail, agent name heading, tighter layout, hint text on Status and Traces views
- **Waterfall bar labels** — now show cost alongside duration and model name

### Added
- Manual release testing checklist (`tests/manual-new-release-tests.md`)
- Pre-release testing checklist (`tests/manual-pre-release-testing.md`)

### Changed
- Task specs moved from `.claude/` to `.claude/specs/`

## [0.1.4] - 2026-04-08

### Fixed
- SDK DuckDB lock error when `tj serve` is running — bootstrap now detects the server and sends spans via HTTP (`TjHttpExporter`) instead of opening DuckDB directly
- LiteLLM model names no longer include provider prefix (`gpt-4o-mini` not `openai/gpt-4o-mini`), fixing pricing lookup failures
- LiteLLM streaming wrappers now correctly attribute provider and stripped model name

### Added
- OpenClaw integration — zero-code OTLP ingestion for OpenClaw agents (PR #15)
- Web UI restyled to opencla.watch palette (deep navy + electric blue, IBM Plex Mono, Bricolage Grotesque)
- Inline SVG logo in web UI sidebar

### Changed
- Node.js upgraded from 20 to 22 in CI and publish workflows
- npm SDK bumped to 0.1.4 (matching Python release)
- README: added Web UI section, updated roadmap (4 items complete, 5 new)

## [0.1.3] - 2026-04-07

### Added
- **Web UI** — local dashboard served by `tj serve` at `http://127.0.0.1:7391/`
  - Status view with agent cards, cost, tokens, alerts (auto-refresh 5s)
  - Traces view with span waterfall visualization and click-to-inspect detail
  - Cost view with breakdown by day/agent/model/tool and summary totals
  - Alerts view with severity filtering and expandable JSON detail
  - Drift view with baseline vs latest session Z-score pass/fail
- `GET /api/v1/status` endpoint — agent status data (mirrors `tj status --json`)
- Drift endpoint now lists all agents when `agent_id` is omitted
- LiteLLM provider integration (`patch_litellm()`)
- Single-file Preact SPA — no build step, dark theme, JetBrains Mono

### Changed
- CORS updated to regex matching for `localhost:*` ports
- API key injected into UI via `<meta>` tag (no user prompt needed)

## [0.1.2] - 2026-04-07

### Fixed
- `tj serve` printing wrong metrics port (9464 instead of 7391)
- `tj onboard` launchd daemon install now degrades gracefully on failure instead of crashing
- CLI commands now fall back to REST API when DuckDB is locked by `tj serve`

### Added
- `tj stop` command — graceful shutdown of daemon or background process
- `tj uninstall` command — clean removal of all TokenJam data, config, and daemon
- 16 runnable example agents across 4 tiers: single provider, single framework, multi-agent, and alerts/drift demos
- API fallback backend (`ApiBackend`) so CLI works while `tj serve` holds the DB lock

### Changed
- README: added toy agent quick-start, example agents section, corrected metrics URL, updated CLI reference
- CLAUDE.md: updated CLI command table, repo layout, added PyPI package name rule

## [0.1.1] - 2026-04-06

### Fixed
- `tj export` returning empty output due to corrupted DuckDB span indexes
- `tj status` showing `?` instead of `●` for completed sessions
- `tj status` showing `$0.000000` cost due to `date.today()` vs UTC date mismatch
- `tj cost` showing spurious `$0.000000` row from session-level spans with no model

### Added
- `tj trace` prefix matching — short trace IDs now resolve like git short hashes
- PyPI and npm publish workflows (`publish-pypi.yml`, `publish-npm.yml`)
- PyPI metadata: README as long description, classifiers, project URLs
- `CODEOWNERS` requiring review from @anilmurty

### Changed
- Renamed npm package from `@tokenjam/sdk` to `@tokenjam/sdk`
- Consolidated `AGENTS.md` to point at `CLAUDE.md` as source of truth

## [0.1.0] - 2026-04-05

### Added
- Core observability pipeline: span ingestion, session tracking, cost calculation
- DuckDB storage backend with migration runner
- 13 alert types with 6 dispatch channels (stdout, file, ntfy, webhook, Discord, Telegram)
- Z-score behavioral drift detection with automatic baseline building
- JSON Schema validation for tool outputs (declared or genson-inferred)
- CLI commands: `onboard`, `status`, `traces`, `cost`, `alerts`, `drift`, `tools`, `export`, `serve`, `doctor`
- REST API with OTLP JSON ingest endpoint and Prometheus metrics
- Python SDK: `@watch()` decorator, `patch_anthropic()`, `patch_openai()`, and 9 more provider/framework integrations
- TypeScript SDK (`@tokenjam/sdk`): `TjClient` and `SpanBuilder` for Node.js agents
- Auto-bootstrap: TracerProvider initializes lazily on first `@watch()` or `patch_*()` call
- Community-maintained model pricing table (`pricing/models.toml`)
- Session continuity via `conversation_id` across process restarts
- GitHub Actions CI (Python 3.10/3.11/3.12 + TypeScript)
