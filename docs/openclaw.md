# Using TokenJam with OpenClaw

OpenClaw has built-in OpenTelemetry support via its `diagnostics-otel` plugin. Point it at `tj serve` and all your agent telemetry flows in automatically — no SDK code required.

## Setup

1. Start tj serve:

   ```bash
   tj onboard
   tj serve &
   ```

2. Add to your `openclaw.json`:

   ```json
   {
     "diagnostics": {
       "enabled": true,
       "otel": {
         "enabled": true,
         "endpoint": "http://127.0.0.1:7391",
         "serviceName": "my-openclaw-agent",
         "traces": true,
         "metrics": true,
         "captureContent": false
       }
     },
     "plugins": {
       "allow": ["diagnostics-otel"],
       "entries": {
         "diagnostics-otel": {
           "enabled": true
         }
       }
     }
   }
   ```

3. Restart your OpenClaw gateway. Traces appear immediately:

   ```bash
   tj status
   tj traces
   tj cost --since 1h
   ```

## What gets captured

- Every agent turn with full tool call history
- Token usage and cost per model call
- Tool executions (file reads, shell commands, web searches, file writes)
- Session continuity across multi-turn conversations

## How it works

OpenClaw's `diagnostics-otel` plugin exports standard OTLP/HTTP JSON to `{endpoint}/v1/traces`. OCW accepts this at `POST /v1/traces` and maps OpenClaw-specific span patterns:

| OpenClaw span name | OCW interpretation |
|---|---|
| `openclaw.request` | Root agent session span |
| `openclaw.agent.turn` | Agent turn (child of session) |
| `tool.Read`, `tool.exec`, `tool.Write`, etc. | Tool call — tool name extracted from span name |
| `openclaw.model.usage` | LLM call — token counts extracted for cost tracking |

The `serviceName` field in your OpenClaw config becomes the `agent_id` in OCW (used for filtering, budgets, and alerts).

## Sensitive action alerts

Configure alerts for OpenClaw tool calls in `.tj/config.toml`:

```toml
[agents.my-openclaw-agent]
  [[agents.my-openclaw-agent.sensitive_actions]]
  name = "Write"
  severity = "warning"

  [[agents.my-openclaw-agent.sensitive_actions]]
  name = "exec"
  severity = "critical"
```

This fires alerts when your OpenClaw agent writes files or executes shell commands.

## Budget limits

```toml
[agents.my-openclaw-agent.budget]
daily_usd = 10.00
session_usd = 2.00
```

## Troubleshooting

**No spans appearing?**
- Verify `tj serve` is running: `curl http://127.0.0.1:7391/api/v1/status`
- Check that `diagnostics-otel` plugin is enabled and allowed in `openclaw.json`
- If using an ingest secret, add it to the OpenClaw OTLP headers config

**Model costs showing default rates?**
- Ensure the model name in OpenClaw matches an entry in `pricing/models.toml`
- Run `tj cost` to see which models are being used
