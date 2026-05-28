# `tj backfill langfuse`

Imports [Langfuse](https://langfuse.com) observations into the local TokenJam DB. Two input modes — live API or local JSON dump.

## Live API ingestion

```bash
tj backfill langfuse \
  --source-url https://cloud.langfuse.com \
  --api-key lf_pk_... \
  --since 30d
```

Reads from Langfuse's public REST endpoint (`/api/public/observations`) with Bearer-auth. Follows pagination automatically.

The `--source-url` value can be the bare base URL (`https://cloud.langfuse.com`) — the adapter appends `/api/public/observations`. Self-hosted Langfuse instances work the same way.

`--since` accepts the same syntax as `tj cost --since`: `30d`, `24h`, or an ISO-8601 timestamp.

## File ingestion

```bash
tj backfill langfuse --source-file ./langfuse-export.json
```

Accepts three input shapes:

1. `{"data": [...], "meta": {...}}` — the format returned by the live API.
2. `[...]` — a bare JSON array of observations.
3. NDJSON — one JSON observation per line.

The file mode is the right choice for testing, offline analysis, or scripted ingestion from a snapshot.

## What gets mapped

Each Langfuse `Observation` becomes one TokenJam span:

| Langfuse field | TokenJam field |
|---|---|
| `id` + `traceId` | `span_id` (deterministic hash) |
| `traceId` | `trace_id` (deterministic hash) |
| `parentObservationId` | `parent_span_id` |
| `sessionId` (fallback: `traceId`) | `conversation_id` |
| `userId` (fallback: `"langfuse"`) | `agent_id` |
| `type` (`GENERATION` / `SPAN` / `EVENT`) | `kind` + `name` |
| `startTime` / `endTime` | `start_time` / `end_time` |
| `model` | `model`, `provider` (derived), `billing_account` (derived) |
| `usage.input` / `usage.output` | `input_tokens` / `output_tokens` |
| `usageDetails.input_cache_read` | `cache_tokens` |
| `calculatedTotalCost` | `cost_usd` |
| `level` / `statusMessage` | `status_code` / `status_message` |

`billing_account` is derived from the model name (`claude-*` → `anthropic`, `gpt-*`/`o3`/`o4` → `openai`, `gemini-*` → `google`). Unknown models leave it `NULL`; affected sessions will surface as `plan_tier = 'unknown'` in `tj optimize`.

## Idempotency

The TokenJam `span_id` is a deterministic SHA-256 hash of `("langfuse-obs", traceId, id)`. Re-running the same backfill skips rows already present — the output reports `spans_written` vs `spans_skipped`. Safe to schedule nightly.

## Limitations in v1

- Tool-output content from Langfuse `SPAN`-type observations is not extracted into TokenJam's `gen_ai.tool.output` attribute. The `name` and parent linkage are preserved.
- Custom Langfuse cost data outside `calculatedTotalCost` is ignored; TokenJam recomputes cost from `pricing/models.toml` if the field is missing.
- Multi-tenant Langfuse instances aren't filtered by project — the API key's scope determines what's returned.
