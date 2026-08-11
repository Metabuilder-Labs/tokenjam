---
paths:
  - "tokenjam/sdk/**"
  - "examples/**"
  - "incidents/**"
---

# Python SDK and provider integrations

> Moved verbatim out of `CLAUDE.md` so it loads only when you touch these files.
> Cross-cutting rules stay in `CLAUDE.md`.

- **`tokenjam/sdk/agent.py`**: `@watch()` decorator creates session spans only. `record_llm_call()` and `record_tool_call()` create child spans for manual instrumentation. LLM call spans from provider clients require `patch_anthropic()`, `patch_openai()`, etc. `@watch(tenant_id=..., feature=...)` and both `record_*` functions accept the SDK cost-attribution dimensions (see `tokenjam/sdk/attribution.py` below and docs/architecture.md → "SDK cost-attribution dimensions").
- **`tokenjam/sdk/attribution.py`**: `attribution()` — a `contextvars`-based context manager that attaches tenant_id/feature/prompt_template_id/prompt_template_version to every span created within it, including auto-instrumented provider-patch spans that have no per-call kwarg to pass these through (the patched method signature belongs to the third-party client, not tj). `stamp_span_attribution(span, ...)` is the shared helper every provider integration (anthropic/openai/gemini/bedrock/litellm/langchain) calls right after `start_span`: explicit kwarg wins, else the ambient context, else nothing is stamped.
- **`tokenjam/sdk/transport.py`**: `HttpTransport` — buffers up to 1000 spans, retries with exponential backoff (3 attempts, 2s base). Used when `tj serve` runs as a separate process.
- **`tokenjam/sdk/bootstrap.py`**: `ensure_initialised()` — lazy, thread-safe, idempotent bootstrap of config -> DB -> IngestPipeline -> TracerProvider. Called automatically by `@watch()` and all `patch_*()` functions. Registers atexit flush.
- **`tokenjam/sdk/integrations/`**: `Integration` protocol in `base.py`. Provider patches (anthropic, openai, gemini, bedrock, litellm) monkey-patch client methods to create OTel spans with token usage. `litellm.py` covers 100+ providers via LiteLLM's unified interface and uses a `contextvars.ContextVar` (`_tj_litellm_active`) to suppress inner provider patches (openai, anthropic) when active — prevents double-counted spans. Framework patches (langchain, langgraph, crewai, autogen) wrap LLM/tool methods. `llamaindex.py` and `openai_agents_sdk.py` are thin wrappers around those SDKs' native OTel support. `nemoclaw.py` is a WebSocket observer for OpenShell Gateway sandbox events.

## Examples Convention

Each provider integration in `examples/single_provider/` and each framework in `examples/single_framework/` lives in **its own file** — when adding a new SDK integration, mirror this layout (one demo file per integration) so the examples directory stays a 1:1 map of supported integrations. Multi-provider/framework demos go in `examples/multi/`; alert and drift demos that need no API keys go in `examples/alerts_and_drift/`.

The Agent Incident Library at `incidents/` is separate: each scenario is a `scenario.py` + `README.md` pair, invoked via `tj demo <scenario>`. Scenarios inject synthetic spans through `tokenjam/demo/env.py` to simulate real failures (retry-loop, surprise-cost, hallucination-drift) without API keys or a live server.
