# OpenClaw + TokenJam Example

This is a config-only integration — no Python code needed. OpenClaw's built-in OTel exporter sends traces directly to `tj serve`.

## Step 1: Start TokenJam

```bash
pipx install tokenjam
tj onboard
tj serve &
```

## Step 2: Configure OpenClaw

Add this to your `openclaw.json`:

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

## Step 3: Configure sensitive action alerts (optional)

Add to `.tj/config.toml`:

```toml
[agents.my-openclaw-agent]

  [[agents.my-openclaw-agent.sensitive_actions]]
  name = "Write"
  severity = "warning"

  [[agents.my-openclaw-agent.sensitive_actions]]
  name = "exec"
  severity = "critical"

  [[agents.my-openclaw-agent.sensitive_actions]]
  name = "web_search"
  severity = "info"

  [agents.my-openclaw-agent.budget]
  daily_usd = 10.00
  session_usd = 2.00
```

## Step 4: Run OpenClaw and verify

Start your OpenClaw gateway, then check TokenJam:

```bash
# Agent overview — should show your openclaw agent
tj status

# Trace listing — shows openclaw.request, openclaw.agent.turn, tool.* spans
tj traces

# Span waterfall for a specific trace
tj trace <trace-id>

# Cost breakdown — token usage from openclaw.model.usage spans
tj cost --since 1h

# Tool call summary — shows Read, exec, Write, web_search counts
tj tools

# Alerts — shows any sensitive action or budget alerts
tj alerts
```

## Expected output from `tj traces`

```
TRACE ID         AGENT                  SPANS  DURATION  STATUS
a1b2c3d4e5f6...  my-openclaw-agent      12     4.2s      OK

  openclaw.request                          4.2s
  ├── openclaw.agent.turn                   3.8s
  │   ├── tool.Read                         0.1s
  │   ├── tool.exec                         1.2s
  │   └── tool.Write                        0.3s
  └── openclaw.model.usage                  —
```

See [docs/openclaw.md](../../docs/openclaw.md) for the full setup guide.
