# `tj backfill` — ingest historical telemetry

TokenJam's `backfill` command imports historical telemetry from external sources into the local DuckDB, where the standard analyzers (`tj cost`, `tj optimize`, etc.) can read it. Every backfill source maps records onto the same internal `NormalizedSpan` schema, so once imported the data is indistinguishable from natively-captured telemetry.

TokenJam's posture toward upstream tools like Langfuse, Helicone, Phoenix, and LangSmith is partnership, not displacement: keep using whatever you use for live tracing, then point `tj backfill <source>` at it to run the local cost-optimization analyzers (`tj optimize`) against the same data.

## Supported sources

| Source | Command | Status |
|---|---|---|
| Claude Code (on-disk session JSONL) | `tj backfill claude-code` | Stable |
| Codex CLI (on-disk rollout JSONL) | `tj backfill codex` | Stable |
| Langfuse (live API or JSON dump) | `tj backfill langfuse` | Stable |
| Helicone (live API or JSON dump) | `tj backfill helicone` | Stable |
| Raw OTLP JSON (file or HTTP dump) | `tj backfill otlp` | Stable |

Every adapter accepts `--source-url` (live endpoint) or `--source-file` (offline JSON dump), plus optional `--since` for time-windowed ingest. Re-running an ingest is a no-op — see the idempotency note below.

## Idempotency

Every adapter derives deterministic `span_id`s from the source's identifiers (e.g. `(langfuse_trace_id, observation_id)` for Langfuse, `(helicone_request_id)` for Helicone, `(session_id, message_uuid)` for Claude Code, `(session_id, per-turn index)` / `(session_id, call_id)` for Codex, the OTLP-payload's own `trace_id`+`span_id` for raw OTLP). Re-running an ingest is a no-op — rows that already exist are skipped via the spans table's `PRIMARY KEY` on `span_id`.

This means you can:

- Re-run backfills nightly via cron without duplicating data.
- Combine multiple `--since` windows; overlapping spans collapse on the first import.
- Run `tj backfill <source>` *and* keep TokenJam's own daemon collecting live; the daemon's spans and backfilled spans share the same DB without conflict (as long as the source IDs differ, which they will).

## Plan tier on backfilled sessions

Backfilled sessions get `SessionRecord.plan_tier = 'unknown'` because the source data doesn't carry a plan-tier identifier. Run `tj onboard --reconfigure` to set your plan; `tj optimize` will then render dollar figures correctly for those sessions.

## See also

- [`tj backfill codex`](codex.md) — Codex CLI on-disk rollout ingestion
- [`tj backfill langfuse`](langfuse.md) — Langfuse observation ingestion
- [`tj backfill helicone`](helicone.md) — Helicone request-record ingestion
- [`tj backfill otlp`](otlp.md) — generic OTLP JSON ingestion
