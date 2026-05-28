# `tj backfill helicone`

Imports [Helicone](https://helicone.ai) request records into the local TokenJam DB. Two input modes — live API or local JSON dump.

TokenJam doesn't replace Helicone. Keep Helicone wherever you have it; point `tj backfill helicone` at the same data so the local cost-optimization analyzers (`tj optimize`) can read it.

## Live API ingestion

```bash
tj backfill helicone \
  --source-url https://api.helicone.ai \
  --api-key hc_pk_... \
  --since 30d
```

POSTs `/v1/request/query` against Helicone with Bearer auth and follows pagination. Self-hosted Helicone instances work the same way — point `--source-url` at the base URL of your deployment.

`--since` accepts the same syntax as `tj cost --since`: `30d`, `24h`, or an ISO-8601 timestamp.

## File ingestion

```bash
tj backfill helicone --source-file ./helicone-export.json
```

Accepts three input shapes:

1. `{"data": [...]}` — the format returned by the live `/v1/request/query` endpoint.
2. `[...]` — a bare JSON array of records.
3. NDJSON — one JSON record per line.

The file mode is the right choice for testing, offline analysis, or scripted ingestion from a snapshot.

## What gets mapped

Each Helicone request record becomes one TokenJam span:

| Helicone field | TokenJam field |
|---|---|
| `request.id` | `span_id` (deterministic hash) |
| `Helicone-Property-Session` (fallback: `request.id`) | `conversation_id` |
| `request.user_id` (fallback: `"helicone"`) | `agent_id` |
| `request.model` | `model` |
| `request.provider` (fallback: derived from model) | `provider` + `billing_account` |
| `request.created_at` | `start_time` |
| `request.created_at + response.delay_ms` | `end_time` |
| `request.prompt_tokens` | `input_tokens` |
| `response.completion_tokens` | `output_tokens` |
| `cost_usd` / `costUSD` | `cost_usd` |
| `properties` | merged into `attrs` |

`billing_account` is derived from `request.provider` when present, or from the model name otherwise. Unknown providers leave it `NULL`; affected sessions will surface as `plan_tier = 'unknown'` in `tj optimize`.

## Idempotency

The TokenJam `span_id` is a deterministic SHA-256 hash of `("helicone", request.id)`. Re-running the same backfill skips rows already present — the output reports `spans_written` vs `spans_skipped`. Safe to schedule nightly.

## Limitations in v1

- Helicone's per-request prompt/response bodies are not extracted into `gen_ai.prompt.content` / `gen_ai.completion.content`. Token counts and structural metadata only.
- Multi-tenant Helicone instances aren't filtered by org — the API key's scope determines what's returned.
- If `cost_usd` is missing from a record, TokenJam recomputes cost from `pricing/models.toml` using the model name.
