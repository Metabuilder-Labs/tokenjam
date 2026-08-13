---
description: OTel semantic-convention constant discipline, the exporter, and the one home for OTLP parsing.
paths:
  - "tokenjam/otel/**"
  - "tokenjam/core/ingest.py"
  - "tokenjam/core/ingest_adapters/**"
  - "tokenjam/api/routes/spans.py"
  - "tokenjam/api/routes/logs.py"
  - "tokenjam/sdk/integrations/**"
---

# OTel rules

### Critical Rule 10 — Use semconv constants

Reference `GenAIAttributes` and `TjAttributes` from `tokenjam/otel/semconv.py` instead of hardcoding
OTel attribute name strings. `semconv.py` is pure constants with no internal imports, so importing it
from anywhere is free of layering risk.

OTLP parsing has exactly one home, `tokenjam/otel/otlp_parsing.py` — see the package notes below.

## `tokenjam/otel/`

- **`provider.py`**: `TjSpanExporter` (custom `SpanExporter` that feeds spans into `IngestPipeline`), `convert_otel_span()` (OTel `ReadableSpan` → `NormalizedSpan`), `build_tracer_provider()` (sets up the global `TracerProvider` with local + optional OTLP exporters).
- **`otlp_parsing.py`**: **OTLP parsing has one home.** Shared OTLP JSON → `NormalizedSpan` parser. Both callers — `api/routes/spans.py` (live `POST /api/v1/spans`) and `core/ingest_adapters/otlp.py` (`tj backfill otlp`) — import `parse_otlp_span` and `extract_resource_attrs` from here. The live receive path and the backfill adapter must agree on attribute extraction, `billing_account` derivation, and timestamp handling, so extend OTLP attribute extraction once in this module; never copy-paste into either caller.
- **`semconv.py`**: `GenAIAttributes`, `TjAttributes` (includes `BILLING_ACCOUNT` and `PLAN_TIER`), `VALID_PLAN_TIERS` and `SUBSCRIPTION_PLAN_TIERS` frozensets, and `ClaudeCodeEvents` — OTel GenAI semantic convention constants plus tj-specific extensions. **Pure constants, no internal imports.** Reference these instead of hardcoding OTel attribute name strings (Critical Rule 10). Read `ClaudeCodeEvents` for what Claude Code's `api_request` event actually carries before designing around a per-call id (Critical Rule 38 — it carries none).
