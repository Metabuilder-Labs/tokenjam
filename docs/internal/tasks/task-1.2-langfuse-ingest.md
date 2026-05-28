# Task 1.2 — Langfuse ingest adapter

**Wave 1. Dispatch after Task 0 fully merges. Runs in parallel with Tasks 1.1 and 1.3.**

## Prerequisites

- Read `CLAUDE.md`, `docs/internal/tasks/decisions-locked.md`.

## Summary

First ingest adapter. Reads a Langfuse export and produces TokenJam normalized spans written to the local DuckDB via the existing ingest pipeline. Establishes the unified `tj backfill <source>` pattern that Tasks 3.1 (Helicone) and 3.2 (OTLP) will reuse.

## Design choices (already decided)

- **Package location:** `tokenjam/core/ingest_adapters/`. NOT `tokenjam/sdk/integrations/` — the SDK integrations package is for outbound monkey-patches; ingest adapters are inbound.
- **CLI surface:** subcommand of existing `tj backfill`, not a new `tj ingest` command. Pattern: `tj backfill langfuse ...`. The existing `tj backfill claude-code` is the canonical example.
- **Fixture rule:** the test fixture must be a real captured Langfuse `/api/public/observations` response, sanitized of any user data. Do not invent fields from Langfuse documentation. If a real response isn't available, set up a local Langfuse instance briefly to capture one.

## Scope

- New module: `tokenjam/core/ingest_adapters/langfuse.py`
- New CLI subcommand registered under existing `tokenjam/cli/cmd_backfill.py`
- **Two input modes:**
  - `--source-url <url>` for live Langfuse API ingestion (production use)
  - `--source-file <path>` for local JSON file ingestion (testing, offline, scripted)
- Optional `--api-key <key>` for live mode; `--since <date>` for both.
- Map Langfuse's `Observation` type to TokenJam's `NormalizedSpan` schema.
- Handle pagination, rate limiting, errors gracefully.
- **Idempotent:** same `(langfuse_trace_id, observation_id)` produces deterministic TokenJam `span_id`.
- Progress indicator for large exports.
- Backfilled sessions get `plan_tier = unknown` per Task 0 default. User resolves via `tj onboard --reconfigure`.
- Backfilled spans should get `billing_account` set based on the Langfuse observation's `model` field (e.g. `claude-*` → `anthropic`, `gpt-*` → `openai`).
- Documentation in `docs/backfill/langfuse.md`.
- Also create `docs/backfill/overview.md` documenting the unified `tj backfill <source>` pattern (this is the first adapter; this file will be referenced by 3.1, 3.2).
- Tests against the sanitized real fixture using `--source-file`. No live API key required for CI.

## Files touched

- New: `tokenjam/core/ingest_adapters/__init__.py`
- New: `tokenjam/core/ingest_adapters/langfuse.py`
- `tokenjam/cli/cmd_backfill.py` (register the `langfuse` subcommand with both `--source-url` and `--source-file` flags)
- New: `tests/unit/test_ingest_langfuse.py`
- New: `tests/fixtures/langfuse_real_response.json` (sanitized real response)
- `docs/backfill/langfuse.md`
- `docs/backfill/overview.md` (new — documents the unified backfill pattern)
- `CHANGELOG.md`

## Coordination

- Registers a subcommand in `cmd_backfill.py`. Tasks 3.1 and 3.2 will add `helicone` and `otlp` subcommands later — different names, trivial conflict only.

## Done-when

- `tj backfill langfuse --source-file tests/fixtures/langfuse_real_response.json` populates the local DuckDB with normalized spans.
- `tj optimize` analyzes those spans identically to natively-captured spans.
- `tj backfill langfuse --source-url <real-langfuse-url> --api-key <key>` works against a live Langfuse instance (manual smoke test; not in CI).
- Re-running the same backfill is a no-op (deterministic span IDs).
- `billing_account` is set on backfilled spans based on model name.
- `docs/backfill/overview.md` documents the unified pattern for future adapters.
