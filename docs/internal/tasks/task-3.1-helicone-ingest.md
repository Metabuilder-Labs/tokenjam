# Task 3.1 — Helicone ingest adapter

**Wave 3. Dispatch after Wave 2 fully merges. Runs in parallel with Tasks 3.2, 3.3, and (optionally) Task 4.**

## Prerequisites

- Read `CLAUDE.md`, `docs/internal/tasks/decisions-locked.md`.
- Read `docs/backfill/overview.md` (committed by Task 1.2) — this is the canonical pattern for backfill adapters.
- Read `tokenjam/core/ingest_adapters/langfuse.py` (committed by Task 1.2) as the reference implementation.

## Summary

Second ingest adapter. Same shape as Task 1.2 (Langfuse), different source.

## Scope

- New module: `tokenjam/core/ingest_adapters/helicone.py`
- Map Helicone's `/v1/requests/query` response format to TokenJam normalized spans.
- Subcommand: `tj backfill helicone`
- Two input modes (same pattern as Langfuse):
  - `--source-url <url>` for live Helicone API ingestion
  - `--source-file <path>` for local JSON ingestion (testing, offline)
- Optional `--api-key <key>` for live mode; `--since <date>` for both.
- Idempotent via deterministic span IDs derived from `(helicone_request_id, ...)`.
- Backfilled sessions get `plan_tier = unknown`. Backfilled spans get `billing_account` set based on the Helicone request's `provider` or `model` field.
- **Fixture rule:** sanitized real Helicone response. No inventing fields from documentation.
- Documentation in `docs/backfill/helicone.md`.

## Files touched

- New: `tokenjam/core/ingest_adapters/helicone.py`
- `tokenjam/cli/cmd_backfill.py` (register the `helicone` subcommand)
- New: `tests/unit/test_ingest_helicone.py`
- New: `tests/fixtures/helicone_real_response.json`
- `docs/backfill/helicone.md`
- `CHANGELOG.md`

## Coordination

- Registers a subcommand in `cmd_backfill.py` alongside `langfuse` (Task 1.2, already merged) and `otlp` (Task 3.2, parallel). Different subcommand names — trivial conflict only.

## Done-when

- `tj backfill helicone --source-file tests/fixtures/helicone_real_response.json` populates the local DuckDB with normalized spans.
- `tj optimize` analyzes those spans identically to natively-captured spans.
- `tj backfill helicone --source-url <real-helicone-url> --api-key <key>` works against a live Helicone instance (manual smoke test).
- Re-running the same backfill is a no-op (deterministic span IDs).
- `billing_account` is set on backfilled spans based on the response's provider/model field.
