# NemoClaw Integration

Running OpenClaw inside [NVIDIA NemoClaw](https://github.com/NVIDIA/NemoClaw)? `ocw` connects to the OpenShell Gateway WebSocket and turns every sandbox event — blocked network requests, filesystem denials, inference reroutes — into a first-class alert.

```python
from ocw.sdk.integrations.nemoclaw import watch_nemoclaw

observer = watch_nemoclaw()
asyncio.create_task(observer.connect())  # non-blocking, runs alongside your agent
```

This is the observability layer that NemoClaw doesn't ship with.

## What gets captured

The NemoClaw observer listens on the OpenShell Gateway WebSocket for sandbox enforcement events and converts them into OCW alerts:

| NemoClaw event | OCW alert type | Severity |
|---|---|---|
| Network egress blocked | `network_egress_blocked` | critical |
| Filesystem access denied | `filesystem_access_denied` | critical |
| Syscall denied | `syscall_denied` | critical |
| Inference rerouted | `inference_rerouted` | warning |

These alerts flow through the standard alert pipeline — cooldown, dispatch to configured channels (ntfy, Discord, Telegram, webhook), and persistence to the database.

## Configuration

Configure alert channels in `.ocw/config.toml` to get notified when sandbox events fire:

```toml
[[alerts.channels]]
type = "ntfy"
topic = "my-agent-sandbox-alerts"

[[alerts.channels]]
type = "discord"
webhook_url = "https://discord.com/api/webhooks/..."
```

## Combining with OpenClaw observability

NemoClaw wraps OpenClaw, so you can use both integrations together:

1. Point OpenClaw's `diagnostics-otel` plugin at `tj serve` for trace/cost/drift observability (see [OpenClaw integration](openclaw.md))
2. Add `watch_nemoclaw()` for sandbox enforcement alerts

This gives you full coverage: LLM costs, tool call traces, behavioral drift detection, and real-time sandbox violation alerts.
