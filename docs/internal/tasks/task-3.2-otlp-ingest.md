# Task 3.2 — Raw OTLP ingest adapter

**Wave 3. Dispatch after Wave 2 fully merges. Runs in parallel with Tasks 3.1, 3.3, and (optionally) Task 4.**

## Prerequisites

- Read `CLAUDE.md`, `docs/internal/tasks/decisions-locked.md`.
- Read `docs/backfill/overview.md` and `tokenjam/core/ingest_adapters/langfuse.py` as the reference implementation.

## Summary

Generic OTLP ingestion. The "works with anything" adapter that covers Langfuse/Helicone-incompatible sources. Accepts an OTLP JSON file or a live OTLP HTTP endpoint.

## Scope

- New module: `tokenjam/core/ingest_adapters/otlp.py`
- Subcommand: `tj backfill otlp`
- Two input modes:
  - `--source-url <url>` for pulling from a live OTLP HTTP endpoint
  - `--source-file <path>` for local OTLP JSON file ingestion
- Accept OTLP JSON in the standard `{"resourceSpans": [...]}` envelope.
- Map OTLP spans into `NormalizedSpan`. Reuse the existing OTLP parsing logic in `tokenjam/api/routes/spans.py` (the live `POST /api/v1/spans` endpoint already does this) — extract into a shared utility if it isn't one yet, so both code paths use the same mapping.
- Idempotent via deterministic span IDs (use the OTLP `trace_id` + `span_id`).
- `billing_account` is set from resource attributes if present (e.g. `tj.billing_account`), or inferred from `gen_ai.system` if not, falling back to `unknown` provider — never crash on missing fields.
- Backfilled sessions get `plan_tier = unknown` per Task 0 default.
- Fixture: a real OTLP JSON dump (can be generated with any OTel SDK by capturing the exporter payload).

## Files touched

- New: `tokenjam/core/ingest_adapters/otlp.py`
- Possibly refactor: `tokenjam/api/routes/spans.py` (extract OTLP mapping into shared utility if not already shared)
- `tokenjam/cli/cmd_backfill.py` (register the `otlp` subcommand)
- New: `tests/unit/test_ingest_otlp.py`
- New: `tests/fixtures/otlp_sample.json`
- `docs/backfill/otlp.md`
- `CHANGELOG.md`

## Coordination

- Registers a subcommand in `cmd_backfill.py` alongside `langfuse` (Task 1.2, already merged) and `helicone` (Task 3.1, parallel). Different subcommand names — trivial conflict only.
- If extracting shared OTLP mapping logic, coordinate carefully with anything else touching `api/routes/spans.py` (nothing else in this sprint should be — confirm before refactoring).

## Done-when

- `tj backfill otlp --source-file tests/fixtures/otlp_sample.json` populates the local DuckDB with normalized spans.
- `tj optimize` analyzes those spans identically to natively-captured spans.
- `tj backfill otlp --source-url <otlp-endpoint>` works against a live OTLP HTTP endpoint (smoke test).
- Re-running the same backfill is a no-op (deterministic span IDs).
- The OTLP parsing logic is shared between the backfill adapter and the live `POST /api/v1/spans` endpoint — one implementation, two callers.
- `billing_account` is set from resource attributes when present, inferred from `gen_ai.system` when not, never crashes on missing fields.
