# Alerts

`tj` fires alerts the moment something happens — sensitive tool calls, budget breaches, behavioral drift, sandbox violations. Alerts are evaluated after every span ingest and dispatched to your configured channels in real time.

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

Configure where alerts go in `.tj/config.toml`. Multiple channels work simultaneously — you can get push notifications on your phone and a Discord message at the same time.

```toml
# Push notification (free, no account required)
[[alerts.channels]]
type = "ntfy"
topic = "my-agent-alerts"

# Mast pager app for iPhone and Android (free, no account required; the key in the URL is the only credential)
[[alerts.channels]]
type = "mast"
channel_url = "https://mast.tissue.dev/mk_0123456789abcdef0123456789abcdef01234567"

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
path = "~/.tj/alerts.log"

# Stdout (always enabled by default)
[[alerts.channels]]
type = "stdout"
```

### Mast

[Mast](https://tissue.systems/mast) is a pager app for iPhone and Android. Install it, make a
channel, and paste the channel URL into the config above. It works without an account, but a
channel made that way is lost if the app is reinstalled, and the only sign on this side is a
"No such channel." line in the log. Sign in to a free Tissue account if you want the channel
to survive that.

`tj` maps severity onto three of Mast's four priorities: `info` sends `quiet`, `warning`
sends `normal`, `critical` sends `loud`. Nothing maps to the fourth, `page`. It repeats until
someone acks it and ignores mute and quiet hours, so you have to ask for it, which is what
you'd do on a channel that's meant to wake someone up:

```toml
[[alerts.channels]]
type         = "mast"
channel_url  = "https://mast.tissue.dev/mk_..."
priority     = "page"
min_severity = "critical"
```

Each channel also has a ceiling, shown as **Highest priority** in the app, and a send that
asks for more than that is delivered at the ceiling. It's set when the channel is created. The
iOS app doesn't go below `loud`, the Android app uses whatever default priority you picked, and
a channel made from an API token starts at `loud`. So a channel created on Android with the
default of `Normal` will deliver `critical` alerts as `normal`, and `priority = "page"` won't
page anyone until you raise the ceiling on the phone. Worth checking before you rely on it.

Repeat alerts carry a dedupe key of `<type>:<agent>`, so an alert that re-fires after the
cooldown lapses updates the card already on the phone instead of stacking another one.

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

To prevent alert storms, `tj` tracks a cooldown per agent + alert type. Repeat alerts within the cooldown window are suppressed — still persisted to the database, but not dispatched to channels.

```toml
[alerts]
cooldown_seconds = 300   # 5 minutes between repeated alerts of the same type
```

Cooldown state survives a daemon restart: `AlertEngine` rehydrates it from the persisted `alerts` table on startup, so a restart mid-cooldown does not re-fire a duplicate alert or resend a duplicate ntfy/Discord push.

## Content stripping

By default, alert payloads sent to external channels (Discord, Telegram, webhook, ntfy, Mast) have sensitive content stripped: `prompt_content`, `completion_content`, `tool_input`, `tool_output`. To include full content:

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

## MCP (SDK / API integrations)

The MCP server exposes `list_alerts` and `acknowledge_alert` tools for SDK / API integrations that already sit in the request path. It is not wired up for Claude Code or Codex (an in-loop MCP is a per-turn token tax there, +36% measured) — those agents check alerts via `tj` CLI reports or the dashboard instead.
