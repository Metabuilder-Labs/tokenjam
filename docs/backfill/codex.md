# `tj backfill codex` — ingest Codex CLI sessions

The OpenAI Codex CLI persists every session as an on-disk **rollout** JSONL file. `tj backfill codex` parses those files into `NormalizedSpan`s so the local analyzers (`tj cost`, `tj optimize`) can read your Codex history — the same way `tj backfill claude-code` reads Claude Code transcripts.

This is the historical counterpart to the live Codex path: `tj onboard --codex` wires out-of-band OTel telemetry going *forward*, and `tj backfill codex` fills in the sessions that ran before onboarding.

## Location and format

Codex writes rollout files under:

```
~/.codex/sessions/YYYY/MM/DD/rollout-<timestamp>-<uuid>.jsonl
```

Each line is a `RolloutLine`: `{"timestamp": "<iso8601>", "type": "<kind>", "payload": {...}}`. The kinds the adapter consumes:

| Line `type` | Payload | Used for |
|---|---|---|
| `session_meta` | `session_id` (or `id`), `timestamp`, `cwd`, `cli_version`, `model_provider` | session identity |
| `turn_context` | `model` | the effective model for the turns that follow |
| `event_msg` → `token_count` | `info.last_token_usage` (per-turn delta) and `info.total_token_usage` (cumulative) | LLM span + token counts |
| `response_item` → `function_call` | `name`, `call_id` | tool spans |

`TokenUsage` fields: `input_tokens`, `cached_input_tokens`, `output_tokens`, `reasoning_output_tokens`, `total_tokens`.

## Usage

```bash
# Ingest all Codex sessions
tj backfill codex

# Only sessions from the last 30 days
tj backfill codex --since 30d

# Point at a non-default sessions root
tj backfill codex --root /path/to/.codex/sessions
```

## Field mapping

- **Provider / billing account** — always `openai` (Codex is OpenAI-only), matching the live logs path.
- **Agent id** — `codex_exec`. Codex hardcodes `service.name=codex_exec` in its binary, so live-ingested Codex spans all land under that agent id; backfill uses the same id so history and live telemetry attribute to one agent.
- **Model** — tracked from the most recent `turn_context.model`.
- **Tokens** — one LLM span per `token_count` event, built from the per-turn `last_token_usage` *delta* so the summed session cost equals the cumulative `total_token_usage` (no double-count). `cached_input_tokens` is a subset of `input_tokens`, so billable input is `input − cached`; `reasoning_output_tokens` are folded into output (billed at the output rate).
- **Cache** — `cached_input_tokens` map to `cache_tokens` (read hits). OpenAI's automatic prompt caching has no separate cache-creation charge, so `cache_write_tokens` is always 0 (mirrors the live path).
- **Cost** — recomputed from `tokenjam/pricing/models.toml`. Models not yet in the packaged table fall back to the default rate; add a [pricing override](../configuration.md) for exact figures.

## Idempotency

Span IDs are deterministic — `(session_id, per-turn index)` for LLM spans and `(session_id, call_id)` for tool spans. Re-running is a no-op: spans already present are skipped via the spans table's `PRIMARY KEY` on `span_id`.

## Plan tier

Backfilled Codex sessions get their `plan_tier` from `[budget.openai] plan` in your config (set by `tj onboard`), matching the live ingest path. When no plan is configured they fall back to `unknown`.

## v1 limitations

- No content capture — the adapter extracts token/cost/tool-shape only, not prompt or completion text.
- Sub-agent attribution is not modeled (Codex rollouts don't carry a Claude-Code-style sidechain marker).
