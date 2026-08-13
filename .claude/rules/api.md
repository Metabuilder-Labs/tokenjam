---
description: REST API auth rules and architecture for the local tj serve surface, plus the MCP stdio server.
paths:
  - "tokenjam/api/**"
  - "tokenjam/mcp/**"
  - "tokenjam/cli/cmd_serve.py"
---

# API rules

### Critical Rule 4 — Ingest auth

`POST /api/v1/spans` requires `Authorization: Bearer <ingest_secret>` from
`security.ingest_secret` in `tj.toml`. It is enforced by `IngestAuthMiddleware`, which returns a
`JSONResponse` directly — `HTTPException` does not propagate from `BaseHTTPMiddleware.dispatch`.

## `tokenjam/api/` — local REST API ("TokenJam Lens" backend)

- **`app.py`**: FastAPI app factory (OpenAPI title `"TokenJam Lens"`). `tj serve` starts it with uvicorn. Accepts `db`, `config`, `ingest_pipeline` for testability. Registers all routers under `/api/v1` plus `/metrics`, `/health`, and the SPA at `/`. **`index.html` is read into a module string once at `create_app()` time** (`_index_html`) — so editing `tokenjam/ui/index.html` requires a `tj serve` restart to take effect; tests read the file from disk directly and aren't affected. Mounts `/ui/vendor` as `StaticFiles`.
- **`middleware.py`**: `IngestAuthMiddleware` — protects `POST /api/v1/spans` with a Bearer token. Returns `JSONResponse(401)` directly (not `HTTPException`, which doesn't propagate from `BaseHTTPMiddleware.dispatch`).
- **`deps.py`**: `require_api_key` — FastAPI dependency for optional API key auth on GET endpoints. Only enforced when `api.auth.enabled = true` in config.
- **`routes/`**: one file per resource.

### Auth

Two layers:
1. **Ingest auth** (middleware): `POST /api/v1/spans` requires `Authorization: Bearer <ingest_secret>` from `security.ingest_secret` (Critical Rule 4). Handled by `IngestAuthMiddleware`, which returns a `JSONResponse` directly — do **not** use `HTTPException` in `BaseHTTPMiddleware.dispatch` as it won't be caught by FastAPI's exception handler.
2. **API key auth** (dependency): all GET endpoints use `Depends(require_api_key)`. Only enforced when `api.auth.enabled = true`.

### Route behaviour worth knowing

- **`POST /api/v1/spans`** accepts OTLP JSON (`{"resourceSpans": [...]}`). Partial failures return 200 with `ingested`/`rejected` counts — 400 only if the entire body is malformed. The route parses OTLP spans into `NormalizedSpan` (via `otel/otlp_parsing.py`) and feeds each through `IngestPipeline.process()`. Key parsing details: resource attributes are merged with span attributes (span wins on conflict); OTLP timestamps are nanosecond strings; OTLP `intValue` fields are strings (per spec for large numbers); unknown attribute value types silently become `None`.
- **`/optimize`** serves a **precomputed** report. No analyzer runs on any request path: reports are produced by a background daemon pass (`core/optimize/report_store.py` + `scan_cycle.py`) at boot, on an interval, and when a user presses Rescan (`POST /optimize/rescan`, rate-limited by `[optimize] scan_min_rescan_seconds`). The `?fast=true` query param and the `skipped_analyzers` response key survive only for wire back-compat — `fast` no longer skips anything and `skipped_analyzers` is unconditionally `[]`.
- **`/reuse/clusters`** (`reuse.py`) returns the Reuse finding plus skeleton-rendering extras `planning_texts` + `pricing_mode`. It is a dedicated endpoint rather than a field bolted onto `/optimize` so the per-cluster planner text, which can be many KB, isn't paid for on every Overview poll (#154).
- **`/health`** (`version.py`) is unauthenticated and mounted with no prefix → `{"status":"ok","version":...}`, plus `GET /api/v1/version`. The version is derived at runtime via `importlib.metadata.version("tokenjam")` — no hardcoded literal.
- **`GET /metrics`** generates Prometheus text format by querying the DB on each request (not via the OTel Prometheus exporter), so data is accurate after restarts. No caching — expensive on large datasets.
- **`GET /api/v1/drift`**: if `agent_id` is missing, return `JSONResponse(status_code=400)` — do not use a union return type like `dict | JSONResponse`, as FastAPI cannot generate a response model for it. Use `response_model=None` on the decorator instead.
- **`/v1/logs`** (`logs.py`) converts Codex OTLP **log** events (`sse_event`, `user_prompt`, `tool_decision`, `tool_result`, `api_request`) into normalized spans for cost/drift/alerting. Event name is read from `attrs["event.name"]` when the OTLP body is empty (Codex schema quirk); epoch `timeUnixNano=0` falls back to `attrs["event.timestamp"]` ISO-8601. The endpoint also silently accepts `resourceSpans`/`resourceMetrics` because Codex's exporter reuses one endpoint for all signal types.
- **`/cost`** returns a window-bucketed `series` for the Lens spend chart (hourly buckets for ≤2-day windows, daily otherwise; epoch-second `bucket` keys) plus `series_bucket` + `window_start`/`window_end`.
- **The dollar-bearing read routes (`/cost`, `/cost/compare`, `/optimize`, `/budget`) each return a `framing` block** (see `core/framing.py`) so the web UI renders plan-tier-aware figures without re-deriving the rules in JS.

### Concurrency

The sync (`def`) read routes (`/optimize`, `/cost/compare`) run in Starlette's threadpool, so
concurrent requests reach the DB from multiple threads. `DuckDBBackend.conn` is a **per-thread
DuckDB cursor** (`threading.local`) over one shared database — cursors are independent connections
safe for concurrent use, so fan-out callers (the Overview) can fetch in parallel. Fixed in #124 (it
was a single shared connection that aborted under concurrent access); **do not collapse `conn` back
to one shared connection object.**

### Testing

Integration tests use `httpx.AsyncClient` with `httpx.ASGITransport(app=app)` against
`InMemoryBackend`. Synthetic alert tests use `unittest.mock.MagicMock` for the DB — you must
explicitly set up `db.get_recent_spans.return_value` before calling `engine.evaluate()`, and silence
channels with `engine.dispatcher.channels = []`.

## `tokenjam/mcp/`

**The MCP is an SDK / API surface, not a Claude Code / Codex one.** It puts tj *in the request path* — the right place for SDK / API integrations doing real-time enforcement/policy/budgets. It is deliberately **not** wired for Claude Code / Codex subscription users: an in-loop MCP is a per-turn token burden on them (an A/B against a no-tj control measured a **materially higher** model-weighted token count; the figure itself is deliberately not restated here — it lives in the shipped product string that quotes it, and in the test pinning that string). Those users get tj **out-of-band**: the zero-token statusline (`tj statusline`, wired by `tj onboard --claude-code`) plus OTel telemetry ingest. `tj mcp` still works for anyone who invokes it; onboarding just no longer defaults CC/Codex users into it.

`server.py` is a FastMCP stdio server exposing observability data (plus the summarize tools —
`list_summarize_candidates`, `summarize_prep`, `summarize_check`, `summarize_apply`,
`summarize_undo`; see `core/summarize/`). It uses either a read-only DuckDB connection or an HTTP
proxy to `tj serve`, and is initialized via `init()` from `cli/cmd_mcp.py`.

`tj mcp` starts the server. The connection mode is chosen at startup by `cmd_mcp.py`:
1. If `tj serve` is reachable on `config.api.{host,port}`, MCP proxies to it via HTTP (live ingest visible).
2. Otherwise it tries to spawn `tj serve` in the background and waits for the port up to `_start_and_wait`'s `timeout` default (`cmd_mcp.py`).
3. If neither works, it falls back to a **read-only DuckDB connection** — read tools still work, but newly ingested spans won't appear until restart.
4. If no config file is found, `init()` is skipped and tools return a no-config sentinel.

SDK / API users who want the in-loop tools can wire it manually: `claude mcp add tj --scope user -- tj mcp`. The `--claude-code` and `--codex` onboard flows **no longer** register the MCP (they wire the out-of-band statusline / OTel instead), and a re-onboard retires any tj-managed `[mcp_servers.tj]` block a previous version wrote to `~/.codex/config.toml`.
