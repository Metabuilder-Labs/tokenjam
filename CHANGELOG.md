# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [0.1.7] - 2026-04-13

### Added
- **MCP server (`tj mcp`)** — stdio-based Model Context Protocol server giving Claude Code direct access to OCW observability data. 13 tool handlers: status, traces, alerts, budget headroom, cost summary, drift report, tool stats, trace detail, acknowledge alerts, setup project, list sessions, open dashboard. Dual-mode operation: routes queries through REST API when `tj serve` is running, falls back to read-only DuckDB otherwise. Auto-starts `tj serve` on demand.
- **Claude Code integration (`tj onboard --claude-code`)** — one-command setup for Claude Code telemetry. Configures OTLP log exporter in `~/.claude/settings.json`, sets project-level `OTEL_RESOURCE_ATTRIBUTES`, adds Docker-compatible endpoint to shell env, and optionally installs background daemon. Re-runs resync the auth header to fix 401s without manual setup.
- **Logs ingestion (`POST /v1/logs`)** — new OTLP log endpoint that converts Claude Code log events (`api_request`, `tool_result`, `api_error`, `user_prompt`, `tool_decision`) into NormalizedSpans with deterministic trace/span IDs. Spans flow through the standard ingest pipeline for cost, alerts, and drift.
- **`tj drift` CLI** — behavioral drift report with Rich table output showing baseline vs latest session Z-scores per dimension (input tokens, output tokens, duration, tool call count, tool sequence similarity). Color-coded thresholds, `--json` support, exit code 1 if drift detected.
- **`tj budget` CLI + API** — view and set per-agent daily/session cost limits. `GET/POST /api/v1/budget` endpoints. `resolve_effective_budget()` with per-field fallback so each budget dimension independently falls back to defaults.
- **Architecture documentation** (`docs/architecture.md`) — comprehensive architecture doc covering design principles, data flow, SDK internals, alert system, drift detection, MCP server, Claude Code pipeline, and testing architecture.
- `ClaudeCodeEvents` semantic conventions in `tj/otel/semconv.py` for Claude Code log event attributes

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
- **Pricing file missing from pip wheel** — `pricing/models.toml` was at the repo root, outside the `tj/` package. Moved to `tj/pricing/models.toml` so it's included in the wheel. All costs showed `$0.000000` in v0.1.4.

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
- `tj uninstall` command — clean removal of all OCW data, config, and daemon
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
