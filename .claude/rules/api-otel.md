---
paths:
  - "tokenjam/api/**"
  - "tokenjam/otel/**"
---

# REST API, OTel plumbing, storage-backend parity

> Moved verbatim out of `CLAUDE.md` so it loads only when you touch these files.
> Cross-cutting rules stay in `CLAUDE.md`.

- **`tokenjam/otel/provider.py`**: `TjSpanExporter` (custom `SpanExporter` that feeds spans into `IngestPipeline`), `convert_otel_span()` (OTel `ReadableSpan` → `NormalizedSpan`), `build_tracer_provider()` (sets up global `TracerProvider` with local + optional OTLP exporters).
- **`tokenjam/otel/otlp_parsing.py`**: Shared OTLP JSON → `NormalizedSpan` parser. Two callers: `api/routes/spans.py` (live `POST /api/v1/spans`) and `core/ingest_adapters/otlp.py` (`tj backfill otlp`). Keep parsing in this one place — the live receive path and the backfill adapter must agree on attribute extraction, billing_account derivation, and timestamp handling.
- **`tokenjam/otel/semconv.py`**: `GenAIAttributes`, `TjAttributes` (includes `BILLING_ACCOUNT` and `PLAN_TIER`), `VALID_PLAN_TIERS` and `SUBSCRIPTION_PLAN_TIERS` frozensets — OTel GenAI semantic convention constants plus tj-specific extensions.
- **`tokenjam/api/app.py`**: FastAPI app factory (OpenAPI title `"TokenJam Lens"`). `tj serve` starts it with uvicorn. Accepts `db`, `config`, `ingest_pipeline` for testability. Registers all routers under `/api/v1` plus `/metrics`, `/health`, and the SPA at `/`. **`index.html` is read into a module string once at `create_app()` time** (`_index_html`) — so editing `tokenjam/ui/index.html` requires a `tj serve` restart to take effect; tests read the file from disk directly and aren't affected. Mounts `/ui/vendor` as `StaticFiles`.
- **`tokenjam/api/middleware.py`**: `IngestAuthMiddleware` — protects `POST /api/v1/spans` with Bearer token. Returns `JSONResponse(401)` directly (not `HTTPException`, which doesn't propagate from `BaseHTTPMiddleware.dispatch`).
- **`tokenjam/api/deps.py`**: `require_api_key` — FastAPI dependency for optional API key auth on GET endpoints. Only enforced when `api.auth.enabled = true` in config.
- **`tokenjam/api/routes/`**: One file per resource — `spans.py` (OTLP JSON ingest), `traces.py`, `cost.py`, `cost_compare.py`, `tools.py`, `alerts.py`, `drift.py`, `optimize.py`, `reuse.py` (`GET /api/v1/reuse/clusters` — the Reuse finding plus skeleton-rendering extras `planning_texts` + `pricing_mode`; a dedicated endpoint (not bolted onto `/optimize`) so the per-cluster planner text, which can be many KB, isn't paid for on every Overview poll — #154), `budget.py`, `status.py`, `agents.py`, `metrics.py` (Prometheus text format from DB queries), `version.py` (unauthenticated `GET /health` → `{"status":"ok","version":...}` mounted with no prefix, plus `GET /api/v1/version`; the version is derived at runtime via `importlib.metadata.version("tokenjam")` — no hardcoded literal). **The dollar-bearing read routes (`/cost`, `/cost/compare`, `/optimize`, `/budget`) each return a `framing` block** (see `core/framing.py`) so the web UI renders plan-tier-aware figures without re-deriving the rules in JS. `/optimize` takes `?fast=true` to skip the expensive Trim analyzer (returns `skipped_analyzers`) for the polling Overview; `/cost` returns a window-bucketed `series` for the chart (see Web UI below). **Concurrency:** the sync (`def`) read routes (`/optimize`, `/cost/compare`) run in Starlette's threadpool, so concurrent requests reach the DB from multiple threads. `DuckDBBackend.conn` is a **per-thread DuckDB cursor** (`threading.local`) over one shared database — cursors are independent connections safe for concurrent use, so fan-out callers (the Overview) can fetch in parallel. Fixed in #124 (was a single shared connection that aborted under concurrent access); do not collapse `conn` back to one shared connection object.

### REST API

The API has two auth layers:
1. **Ingest auth** (middleware): `POST /api/v1/spans` requires `Authorization: Bearer <ingest_secret>`. Handled by `IngestAuthMiddleware`, which returns a `JSONResponse` directly — do **not** use `HTTPException` in `BaseHTTPMiddleware.dispatch` as it won't be caught by FastAPI's exception handler.
2. **API key auth** (dependency): All GET endpoints use `Depends(require_api_key)`. Only enforced when `api.auth.enabled = true`.

`POST /api/v1/spans` accepts OTLP JSON (`{"resourceSpans": [...]}`). Partial failures return 200 with `ingested`/`rejected` counts — 400 only if the entire body is malformed. The route parses OTLP spans into `NormalizedSpan` and feeds each through `IngestPipeline.process()`. Key parsing details: resource attributes are merged with span attributes (span wins on conflict); OTLP timestamps are nanosecond strings; OTLP `intValue` fields are strings (per spec for large numbers); unknown attribute value types silently become `None`.

`GET /metrics` generates Prometheus text format by querying the DB on each request (not using the OTel Prometheus exporter), so data is accurate after restarts. No caching — expensive on large datasets.

For `GET /api/v1/drift`, if `agent_id` is missing, return `JSONResponse(status_code=400)` — do not use a union return type like `dict | JSONResponse` as FastAPI cannot generate a response model for it. Use `response_model=None` on the decorator instead.

Integration tests use `httpx.AsyncClient` with `httpx.ASGITransport(app=app)` against `InMemoryBackend`. Synthetic alert tests use `unittest.mock.MagicMock` for the DB — you must explicitly set up `db.get_recent_spans.return_value` before calling `engine.evaluate()`, and silence channels with `engine.dispatcher.channels = []`.

### StorageBackend parity (serve shim vs DB)

`tj` ships three `StorageBackend` implementations: `DuckDBBackend`, `InMemoryBackend` (tests), and `ApiBackend` — the read-only HTTP shim the CLI uses when `tj serve` holds the DB write-lock. The shim reconstructs objects from JSON, so it repeatedly diverged from the DB backend *silently* (e.g. dropped `cache_write_tokens` on daemon-fetched spans, cache columns zeroed, `get_daily_cost` returning a cumulative total). That whole class of bug (#51) is now guarded by `tests/integration/test_storage_backend_parity.py`, which runs each faithfully-mirrored read method against both the DuckDB backend and the `ApiBackend` shim (over a **real in-process uvicorn server**, not the ASGI shortcut a sync `ApiBackend` can't use) and fails on any divergence.

**Rule: every `StorageBackend` protocol method must be classified in that file** — as parity-covered (`SHIM_PARITY_METHODS` + a spec in `_parity_specs`), a documented `SHIM_KNOWN_GAPS`, deliberately unimplemented (`SHIM_NOT_IMPLEMENTED`), or lifecycle. Adding a protocol method, or teaching the shim a new method, fails CI (`test_every_protocol_method_is_classified` / `test_unimplemented_methods_have_no_silent_shim`) until you classify it — which for a new read method means adding a parity assertion. Don't skip the classification to make CI green; that reopens the exact hole this file closes.
