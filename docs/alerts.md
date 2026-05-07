# Alerts

`ocw` fires alerts the moment something happens — sensitive tool calls, budget breaches, behavioral drift, sandbox violations. Alerts are evaluated after every span ingest and dispatched to your configured channels in real time.

## Alert types

| Type | Trigger | Default severity |
|---|---|---|
| `sensitive_action` | Configured tool name called (e.g. `send_email`, `delete_file`) | Configured per action |
| `cost_budget_daily` | Agent's daily spend exceeds configured limit | critical |
| `cost_budget_session` | Session spend exceeds configured limit | critical |
| `retry_loop` | 4+ identical tool calls in the last 6 spans | warning |
| `token_anomaly` | Token usage deviates significantly from baseline | warning |
| `schema_violation` | Tool output fails JSON Schema validation | warning |
| `drift_detected` | Behavioral Z-score exceeds threshold (default 2.0) | warning |
| `session_duration` | Session wall time exceeds threshold (default 3600s) | warning |
| `failure_rate` | >20% errors in last 20 spans (checked every 5th error) | warning |
| `network_egress_blocked` | NemoClaw blocked an outbound network request | critical |
| `filesystem_access_denied` | NemoClaw denied a filesystem operation | critical |
| `syscall_denied` | NemoClaw denied a system call | critical |
| `inference_rerouted` | NemoClaw rerouted an inference request | warning |

## Channels

Configure where alerts go in `.ocw/config.toml`. Multiple channels work simultaneously — you can get push notifications on your phone and a Discord message at the same time.

```toml
# Push notification (free, no account required)
[[alerts.channels]]
type = "ntfy"
topic = "my-agent-alerts"

# Discord webhook
[[alerts.channels]]
type = "discord"
webhook_url = "https://discord.com/api/webhooks/..."

# Telegram bot
[[alerts.channels]]
type = "telegram"
bot_token = "123456:ABC-DEF..."
chat_id = "-1001234567890"

# Generic webhook (POST JSON payload)
[[alerts.channels]]
type = "webhook"
url = "https://your-endpoint.com/alerts"

# Local file log
[[alerts.channels]]
type = "file"
path = "~/.ocw/alerts.log"

# Stdout (always enabled by default)
[[alerts.channels]]
type = "stdout"
```

## Sensitive actions

Define which tool calls should trigger immediate alerts:

```toml
[agents.my-email-agent]
  [[agents.my-email-agent.sensitive_actions]]
  name     = "send_email"
  severity = "critical"

  [[agents.my-email-agent.sensitive_actions]]
  name     = "delete_file"
  severity = "critical"

  [[agents.my-email-agent.sensitive_actions]]
  name     = "submit_form"
  severity = "warning"
```

## Cooldown

To prevent alert storms, `ocw` tracks a cooldown per agent + alert type. Repeat alerts within the cooldown window are suppressed — still persisted to the database, but not dispatched to channels.

```toml
[alerts]
cooldown_seconds = 300   # 5 minutes between repeated alerts of the same type
```

## Content stripping

By default, alert payloads sent to external channels (Discord, Telegram, webhook, ntfy) have sensitive content stripped: `prompt_content`, `completion_content`, `tool_input`, `tool_output`. To include full content:

```toml
[alerts]
include_captured_content = true
```

Stdout and file channels always include the full payload regardless of this setting.

## CLI

```bash
tj alerts                    # all alert history
tj alerts --severity critical   # filter by severity
tj alerts --type sensitive_action   # filter by type
tj alerts --since 1h         # recent alerts only
```

## REST API

```
GET  /api/v1/alerts                    # list alerts (supports filtering)
PATCH /api/v1/alerts/{id}/acknowledge  # mark alert as acknowledged
```

## MCP (Claude Code)

The MCP server exposes `list_alerts` and `acknowledge_alert` tools, so Claude Code can check and manage alerts directly within a session.
