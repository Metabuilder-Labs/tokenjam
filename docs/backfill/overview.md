# `tj backfill` — ingest historical telemetry

TokenJam's `backfill` command imports historical telemetry from external sources into the local DuckDB, where the standard analyzers (`tj cost`, `tj optimize`, etc.) can read it. Every backfill source maps records onto the same internal `NormalizedSpan` schema, so once imported the data is indistinguishable from natively-captured telemetry.

## Supported sources

| Source | Command | Status |
|---|---|---|
| Claude Code (on-disk session JSONL) | `tj backfill claude-code` | Stable |
| Langfuse (live API or JSON dump) | `tj backfill langfuse` | Stable |

Helicone and raw OTLP adapters land in Wave 3 of the current sprint.

## Idempotency

Every adapter derives deterministic `span_id`s from the source's identifiers (e.g. `(langfuse_trace_id, observation_id)` for Langfuse, `(session_id, message_uuid)` for Claude Code). Re-running an ingest is a no-op — rows that already exist are skipped via the spans table's `PRIMARY KEY` on `span_id`.

This means you can:

- Re-run backfills nightly via cron without duplicating data.
- Combine multiple `--since` windows; overlapping spans collapse on the first import.
- Run `tj backfill langfuse --source-url ...` *and* keep TokenJam's own daemon collecting live; the daemon's spans and backfilled spans share the same DB without conflict (as long as the source IDs differ, which they will).

## Plan tier on backfilled sessions

Backfilled sessions get `SessionRecord.plan_tier = 'unknown'` because the source data doesn't carry a plan-tier identifier. Run `tj onboard --reconfigure` to set your plan; `tj optimize` will then render dollar figures correctly for those sessions.

## See also

- [`tj backfill claude-code`](claude-code.md) — Claude Code session JSONL ingestion
- [`tj backfill langfuse`](langfuse.md) — Langfuse observation ingestion
