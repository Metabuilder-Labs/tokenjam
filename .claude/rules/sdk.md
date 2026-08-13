---
description: Python SDK instrumentation and integration rules, plus @watch(), attribution, transport, bootstrap and the provider/framework integrations.
paths:
  - "tokenjam/sdk/**"
  - "examples/single_provider/**"
  - "examples/single_framework/**"
---

# SDK rules

### Critical Rule 3 — `@watch()` alone does NOT create LLM spans

`@watch()` creates only session start/end spans. Provider patches (`patch_anthropic()`,
`patch_openai()`, etc.) are needed for individual LLM call spans.

### Critical Rule 12 — New SDK integrations must call `ensure_initialised()`

Every `patch_*()` convenience function must call
`from tokenjam.sdk.bootstrap import ensure_initialised; ensure_initialised()` before installing
hooks. This lazily bootstraps the TracerProvider + IngestPipeline on first use.

## `tokenjam/sdk/`

- **`agent.py`**: `@watch()` decorator creates session spans only. `record_llm_call()` and `record_tool_call()` create child spans for manual instrumentation. LLM call spans from provider clients require `patch_anthropic()`, `patch_openai()`, etc. (Critical Rule 3). `@watch(tenant_id=..., feature=...)` and both `record_*` functions accept the SDK cost-attribution dimensions (see `attribution.py` and `docs/architecture.md` → "SDK cost-attribution dimensions").
- **`attribution.py`**: `attribution()` — a `contextvars`-based context manager that attaches tenant_id/feature/prompt_template_id/prompt_template_version to every span created within it, including auto-instrumented provider-patch spans that have no per-call kwarg to pass these through (the patched method signature belongs to the third-party client, not tj). `stamp_span_attribution(span, ...)` is the shared helper every provider integration (anthropic/openai/gemini/bedrock/litellm/langchain) calls right after `start_span`: explicit kwarg wins, else the ambient context, else nothing is stamped.
- **`transport.py`**: `HttpTransport` — buffers up to 1000 spans, retries with exponential backoff (3 attempts, 2s base). Used when `tj serve` runs as a separate process.
- **`bootstrap.py`**: `ensure_initialised()` — lazy, thread-safe, idempotent bootstrap of config -> DB -> IngestPipeline -> TracerProvider. Called automatically by `@watch()` and all `patch_*()` functions (Critical Rule 12). Registers atexit flush.
- **`integrations/`**: `Integration` protocol in `base.py`. Provider patches (anthropic, openai, gemini, bedrock, litellm) monkey-patch client methods to create OTel spans with token usage. `litellm.py` covers 100+ providers via LiteLLM's unified interface and uses a `contextvars.ContextVar` (`_tj_litellm_active`) to suppress inner provider patches (openai, anthropic) when active — prevents double-counted spans. Framework patches (langchain, langgraph, crewai, autogen) wrap LLM/tool methods. `llamaindex.py` and `openai_agents_sdk.py` are thin wrappers around those SDKs' native OTel support. `nemoclaw.py` is a WebSocket observer for OpenShell Gateway sandbox events.

`anthropic.py` stamps `gen_ai.response.id` — one of the cross-observer call ids Critical Rule 38
covers. `sdk-ts/` is fully independent from Python and communicates only via HTTP.

Examples convention: each provider integration in `examples/single_provider/` and each framework in
`examples/single_framework/` lives in **its own file** — when adding a new SDK integration, mirror
that layout so the examples directory stays a 1:1 map of supported integrations.
