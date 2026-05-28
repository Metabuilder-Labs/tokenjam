# `tj backfill otlp`

Generic OTLP-JSON ingestion. The "works with anything" adapter for sources Langfuse and Helicone don't cover: OTel SDKs writing JSON files, observability tools that emit OTLP-shaped exports, OTLP HTTP collectors that publish JSON dumps.

Like the other backfill adapters, this is a partnership move — keep whatever OTel-emitting tool you already use, then point `tj backfill otlp` at a dump to run `tj optimize` against the same data locally.

## File ingestion

```bash
tj backfill otlp --source-file ./traces.json
```

Accepts:

1. A single `{"resourceSpans": [...]}` envelope (the OTLP JSON wire format).
2. NDJSON — one OTLP envelope per line.

This is the recommended mode. Export from your OTel SDK or collector, then ingest.

## URL ingestion

```bash
tj backfill otlp --source-url https://example.com/traces.json --since 7d
```

GET-fetches a JSON dump from a URL. **Not** for live push-style OTLP — for that, configure your collector to send to the running `tj serve` endpoint at `POST /api/v1/spans`, which uses the same OTLP parser as this adapter.

`--since` filters spans by `start_time` and accepts `30d` / `24h` / ISO-8601, matching `tj cost --since`.

## What gets mapped

The adapter uses `tokenjam.otel.otlp_parsing.iter_otlp_spans()` — the same parser that handles live `POST /api/v1/spans` ingest. Resource attributes are merged into per-span attributes (span wins on conflict), OTLP timestamps (nanosecond strings) are converted to UTC datetimes, and OTLP `intValue` fields (strings per the OTLP spec) are coerced to ints.

Standard GenAI semconv attributes (`gen_ai.system`, `gen_ai.request.model`, `gen_ai.usage.input_tokens`, etc.) populate the corresponding `NormalizedSpan` fields. The `tokenjam.billing_account` extension attribute is honored when present; otherwise `billing_account` is derived from the model name.

## Idempotency

`span_id` is read directly from the OTLP payload (each span carries a unique `(trace_id, span_id)`). The DB's `PRIMARY KEY` on `spans.span_id` causes re-runs to skip already-present rows. Safe to schedule nightly.

The CLI reports four counters: `spans_seen`, `spans_written`, `spans_skipped`, `spans_rejected`. Rejected spans are those that fail sanitization (oversized attributes, malformed timestamps) — the same gate the live ingest path uses.

## Limitations in v1

- No support for OTLP/protobuf wire format. JSON only. If you need protobuf, convert via the OTel collector first.
- Histogram and metric records in mixed payloads are ignored — only `resourceSpans` is processed.
- Sources that don't emit GenAI semconv attributes will still ingest, but with empty token counts and zero cost (no provider/model to look up in `pricing/models.toml`).
