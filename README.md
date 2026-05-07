<div align="center">

<img src="https://tokenjam.dev/icon.svg" alt="TokenJam" width="72" height="72">

# TokenJam

The open-source LLM observability tool for autonomous agents.

No cloud. No signup. No surprises.

[![CI](https://github.com/Metabuilder-Labs/tokenjam/actions/workflows/ci.yml/badge.svg)](https://github.com/Metabuilder-Labs/tokenjam/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/tokenjam?color=3d8eff&labelColor=0d1117)](https://pypi.org/project/tokenjam/)
[![Python](https://img.shields.io/badge/python-3.10%2B-3d8eff?labelColor=0d1117)](https://pypi.org/project/tokenjam/)
[![npm](https://img.shields.io/npm/v/@tokenjam/sdk?color=3d8eff&labelColor=0d1117)](https://www.npmjs.com/package/@tokenjam/sdk)
[![License: MIT](https://img.shields.io/badge/license-MIT-3d8eff?labelColor=0d1117)](LICENSE)
[![OTel](https://img.shields.io/badge/OTel-GenAI%20SemConv-3d8eff?labelColor=0d1117)](https://opentelemetry.io/docs/specs/semconv/gen-ai/)

```
pip install tokenjam
```

</div>

---

Your agent sends emails, writes files, calls APIs, and spends your money — all while you're away. Most observability tools were built for LLM developers building chat products. `tj` was built for **agents with real-world consequences**: real-time cost tracking, safety alerts, behavioral drift detection, all running locally on your machine.

---

## What you get

**Real-time cost tracking.** Every LLM call is priced as it happens — by agent, model, session, and tool. Budget alerts fire before you hit the limit, not after.

**Safety alerts.** Configure any tool call as a sensitive action (`send_email`, `delete_file`, `submit_form`) and get notified instantly via ntfy, Discord, Telegram, webhook, or stdout.

**Behavioral drift detection.** `tj` builds a statistical baseline from your agent's real behavior and alerts when something deviates — a prompt tweak, a model update, a dependency bump. No LLM required.

**Tool output validation.** Declare a JSON Schema for your tools or let `tj` infer one automatically. Schema violations are caught the moment they occur.

**100% local.** DuckDB. Local REST API. No cloud backend. No API key for `tj` itself. Your telemetry never leaves your machine unless you explicitly export it.

---

## Get started

`tj` works four ways. Pick the one that fits.

### Coding agents — zero code

For **Claude Code**, **Codex**, and any agent that already emits OpenTelemetry. No SDK, no code changes.

```bash
pip install "tokenjam[mcp]"
tj onboard --claude-code    # or: tj onboard --codex
# Restart your coding agent
```

Every session, API call, tool use, and error is now a tracked span with cost and alert evaluation. The MCP server gives your coding agent 13 tools to query its own telemetry mid-session — just ask "how much have I spent today?" or "are there any active alerts?"

[Full Claude Code & Codex setup →](#claude-code--coding-agents)

### Python SDK

For any Python agent — Anthropic, OpenAI, Gemini, Bedrock, LangChain, CrewAI, and [10+ more](#supported-frameworks).

```bash
pip install tokenjam
tj onboard    # creates config, generates ingest secret
ocw doctor     # verify your setup
```

```python
from tokenjam.sdk import watch
from tokenjam.sdk.integrations.anthropic import patch_anthropic

patch_anthropic()    # auto-intercepts all Anthropic API calls

@watch(agent_id="my-agent")
def run(task: str) -> str:
    # your agent code — nothing else to change
    ...
```

One-line patches exist for every major provider and framework. [See all integrations →](#supported-frameworks)

### TypeScript SDK

For any Node.js / TypeScript agent. Sends spans to `tj serve` over HTTP.

```bash
npm install @tokenjam/sdk
```

```typescript
import { TjClient, SpanBuilder } from "@tokenjam/sdk";

const client = new TjClient({
  baseUrl:      "http://127.0.0.1:7391",
  ingestSecret: process.env.TJ_INGEST_SECRET ?? "",
});

const span = new SpanBuilder("invoke_agent")
  .agentId("my-ts-agent")
  .model("gpt-4o-mini")
  .provider("openai")
  .inputTokens(450)
  .outputTokens(120)
  .build();

await client.send([span]);
```

### Any OTel-compatible agent

Already emitting OpenTelemetry? Point your OTLP exporter at `tj serve` — no SDK needed:

```bash
tj serve &
export OTEL_EXPORTER_OTLP_ENDPOINT=http://127.0.0.1:7391
# run your agent as usual
```

| Framework | OTel support |
|---|---|
| **Claude Code** | Built-in — `tj onboard --claude-code` |
| **OpenClaw** | Built-in (`diagnostics-otel` plugin) — [setup guide](docs/openclaw.md) |
| LlamaIndex | `opentelemetry-instrumentation-llama-index` |
| OpenAI Agents SDK | Built-in |
| Google ADK | Built-in |
| Strands Agent SDK (AWS) | Built-in |
| Haystack | Built-in |
| Pydantic AI | Built-in |
| Semantic Kernel | Built-in |

---

## CLI

```
tj status
```

```
● my-email-agent   completed   (2m 14s)

  Cost today:     $0.0340 / $5.0000 limit
  Tokens:         12.4k in / 3.8k out
  Tool calls:     47
  Active session: sess-a1b2c3

  send_email called (sensitive action: critical)
```

https://github.com/user-attachments/assets/b94d13f6-1432-40d4-b093-6958d74f0e65

```bash
tj status           # current state, cost, active alerts
ocw traces           # full span history with waterfall view
ocw cost --since 7d  # cost breakdown by agent, model, day
ocw alerts           # everything that fired while you were away
ocw budget           # view and set daily/session cost limits
ocw drift            # behavioral drift Z-scores vs baseline
ocw tools            # tool call history with error rates
tj serve            # start the web UI + REST API
```

---

## Web UI

`tj serve` starts a local dashboard at `http://127.0.0.1:7391/`.

https://github.com/user-attachments/assets/ff09caec-3487-4542-8628-d62b7d92591f

- **Status** — agent overview with cost, tokens, tool calls, and active alerts
- **Traces** — trace list with span waterfall visualization
- **Cost** — breakdown by agent, model, day, or tool
- **Alerts** — alert history with severity filtering
- **Budget** — view and edit daily/session cost limits per agent
- **Drift** — behavioral drift report with Z-score analysis

No signup, no cloud — runs entirely on your machine.

---

## ocw vs LangSmith vs Langfuse

LangSmith and Langfuse are excellent for tracing LLM API calls and running evals on chat outputs. `tj` solves a different problem: **autonomous agents running unsupervised with real-world consequences**.

| | `tj` | LangSmith | Langfuse | Datadog LLM Obs |
|---|---|---|---|---|
| Signup required | ❌ | ✅ | ✅ | ✅ |
| Data leaves your machine | ❌ | ✅ | cloud only | ✅ |
| Real-time sensitive action alerts | ✅ | ❌ | ❌ | ❌ |
| Behavioral drift detection | ✅ | ❌ | ❌ | ❌ |
| Local-first, no cloud required | ✅ | ❌ | self-host only | ❌ |
| OTel GenAI SemConv native | ✅ | partial | partial | partial |
| NemoClaw sandbox events | ✅ | ❌ | ❌ | ❌ |
| Works with any agent/framework | ✅ | LangChain-first | partial | ❌ |
| Free, MIT licensed | ✅ | freemium | freemium | paid |

---

## Claude Code + coding agents

### Claude Code

Monitor every Claude Code session — costs, tool calls, API requests, errors — with two commands:

```bash
pip install "tokenjam[mcp]"
tj onboard --claude-code
# Restart Claude Code, then:
tj status --agent claude-code-<project>
```

`tj onboard --claude-code` does everything in one shot:
- Creates a shared config at `~/.config/tj/config.toml` (one config for all projects)
- Writes OTLP exporter vars to `~/.claude/settings.json`
- Tags this project by writing `OTEL_RESOURCE_ATTRIBUTES` to `.claude/settings.json`
- Registers the MCP server globally (`claude mcp add --scope user tj -- tj mcp`)
- Installs a background daemon (launchd on macOS, systemd on Linux)
- Adds Docker harness-compatible OTLP env vars to `~/.zshrc`

**Claude Code must be restarted** after running `tj onboard --claude-code`.

**Adding more projects** — run once per project directory:

```bash
cd /path/to/other-project
tj onboard --claude-code   # tags this project, no reinstall needed
# Restart Claude Code
```

Each project gets its own agent ID (`claude-code-<repo-name>`), all sharing one server and one ingest secret.

### MCP server

The MCP server gives Claude Code direct access to your observability data inside the session. 13 tools available after restart:

| Tool | What it does |
|---|---|
| `get_status` | Current agent state — tokens, cost, active alerts |
| `get_budget_headroom` | Budget limit vs spend |
| `list_active_sessions` | All running sessions across agents |
| `list_agents` | All known agents with lifetime cost |
| `get_cost_summary` | Cost breakdown by day / agent / model |
| `list_alerts` | Alert history with severity filtering |
| `list_traces` | Recent traces with cost and duration |
| `get_trace` | Full span waterfall for a trace |
| `get_tool_stats` | Tool call counts and average duration |
| `get_drift_report` | Drift baseline vs latest session |
| `acknowledge_alert` | Mark an alert as acknowledged |
| `setup_project` | Configure a project for OCW telemetry |
| `open_dashboard` | Open the web UI (starts `tj serve` if needed) |

The MCP server opens DuckDB read-only — no lock conflicts with `tj serve`.

**Per-project tagging** — after installing globally, ask Claude Code:

> "Set up OCW for this project"

Claude calls `setup_project`, which writes `.claude/settings.json` with the right `OTEL_RESOURCE_ATTRIBUTES` for this project.

### Codex

Monitor every Codex session — run once, globally:

```bash
pip install "tokenjam[mcp]"
tj onboard --codex
```

`tj onboard --codex` is project-agnostic. It writes to `~/.codex/config.toml` (Codex's single global config), so you only run it once — not once per project. Codex hardcodes `service.name="codex_exec"` in its binary, so all sessions appear under the same agent ID regardless of which repo you're working in.

`tj onboard --codex`:
- Writes an `[otel]` block and `[mcp_servers.ocw]` to `~/.codex/config.toml`
- Registers the MCP server so Codex can call OCW tools directly
- Installs the background daemon (launchd / systemd)

**Codex must be restarted** after running `tj onboard --codex`.

```bash
tj status --agent codex_exec   # check it's working
```

The same 13 MCP tools available to Claude Code are available to Codex after restart.

### Uninstalling

```bash
# Remove all OCW data, config, daemon, MCP registration, and env vars:
ocw uninstall --yes

# Then remove the package:
pip uninstall tokenjam -y
```

`tj uninstall` cleans up everything set by `tj onboard --claude-code`: daemon, MCP server, `~/.ocw/`, `~/.config/ocw/`, OTLP env vars in `~/.claude/settings.json`, `OTEL_RESOURCE_ATTRIBUTES` in every onboarded project's `.claude/settings.json`, and the harness env block in `~/.zshrc`.

---

## Supported frameworks

### Python — provider patches

Intercept at the API level. Framework-agnostic.

```python
from tokenjam.sdk.integrations.anthropic import patch_anthropic   # Anthropic
from tokenjam.sdk.integrations.openai    import patch_openai      # OpenAI
from tokenjam.sdk.integrations.gemini    import patch_gemini      # Google Gemini
from tokenjam.sdk.integrations.bedrock   import patch_bedrock     # AWS Bedrock
from tokenjam.sdk.integrations.litellm   import patch_litellm     # LiteLLM (100+ providers)
```

`patch_litellm()` covers all providers LiteLLM routes to (OpenAI, Anthropic, Bedrock, Vertex, Cohere, Mistral, Ollama, etc.). If you use LiteLLM, you don't need individual patches.

OpenAI-compatible providers (Groq, Together, Fireworks, xAI, Azure OpenAI) work via `patch_openai(base_url=...)`.

### Python — framework patches

Instrument the framework's own abstractions:

```python
from tokenjam.sdk.integrations.langchain         import patch_langchain        # BaseLLM + BaseTool
from tokenjam.sdk.integrations.langgraph         import patch_langgraph        # CompiledGraph
from tokenjam.sdk.integrations.crewai            import patch_crewai           # Task + Agent
from tokenjam.sdk.integrations.autogen           import patch_autogen          # ConversableAgent
from tokenjam.sdk.integrations.llamaindex        import patch_llamaindex       # Native OTel
from tokenjam.sdk.integrations.openai_agents_sdk import patch_openai_agents    # Native OTel
from tokenjam.sdk.integrations.nemoclaw          import watch_nemoclaw         # NemoClaw Gateway
```

Full framework support guide: [docs/framework-support.md](docs/framework-support.md)

---

## Alert channels

Configure where alerts go. Multiple channels work simultaneously.

```toml
# .ocw/config.toml

[[alerts.channels]]
type = "ntfy"
topic = "my-agent-alerts"   # push to your phone, free, no account

[[alerts.channels]]
type = "discord"
webhook_url = "https://discord.com/api/webhooks/..."

[[alerts.channels]]
type = "webhook"
url = "https://your-endpoint.com/alerts"
```

Alert types: `sensitive_action` · `cost_budget_daily` · `cost_budget_session` · `session_duration` · `retry_loop` · `token_anomaly` · `schema_violation` · `drift_detected` · `failure_rate` · `network_egress_blocked` · `filesystem_access_denied` · `syscall_denied` · `inference_rerouted`

Full alert reference — trigger conditions, cooldown config, content stripping, all 6 channel types: [docs/alerts.md](docs/alerts.md)

---

## NemoClaw support

Running OpenClaw inside [NVIDIA NemoClaw](https://github.com/NVIDIA/NemoClaw)? `tj` connects to the OpenShell Gateway WebSocket and turns sandbox events — blocked network requests, filesystem denials, inference reroutes — into alerts.

```python
from tokenjam.sdk.integrations.nemoclaw import watch_nemoclaw

observer = watch_nemoclaw()
asyncio.create_task(observer.connect())
```

This is the observability layer that NemoClaw doesn't ship with.

Full event table and configuration: [docs/nemoclaw-integration.md](docs/nemoclaw-integration.md)

---

## Export and integrate

```bash
ocw export --format otlp       # forward to Grafana, Datadog, any OTel backend
ocw export --format openevals  # openevals / agentevals trajectory evaluation
ocw export --format json       # NDJSON
ocw export --format csv
```

Prometheus metrics at `http://127.0.0.1:7391/metrics` when `tj serve` is running.

Export filtering flags, REST API endpoints, and API docs: [docs/export.md](docs/export.md)

---

## Architecture

```mermaid
flowchart TD
    Agent["Your agent"]

    Agent --> Terminal["Coding agents\nClaude Code · Codex"]
    Agent --> PythonSDK["Python SDK\n@watch + patch_*"]
    Agent --> TypeScriptSDK["TypeScript SDK\n@tokenjam/sdk"]

    Terminal --> OTLP["OTLP export"]
    PythonSDK --> Exporter["TjSpanExporter"]
    TypeScriptSDK --> HTTP["POST /api/v1/spans"]
    OTLP --> HTTP

    Exporter --> Ingest
    HTTP --> Ingest

    Ingest["IngestPipeline\nSanitize · Session continuity · Extract"]

    Ingest --> Cost["CostEngine\npricing.toml"]
    Ingest --> Alerts["AlertEngine\n13 types · 6 channels"]
    Ingest --> Schema["SchemaValidator\nJSON Schema + infer"]

    Cost --> DB["DuckDB\nlocal · embedded"]
    Alerts --> DB
    Schema --> DB

    DB --> CLI["ocw CLI"]
    DB --> API["REST API + Web UI\n:7391"]
    DB --> MCP["MCP Server\n13 tools"]
    DB --> Prom["Prometheus\n:7391/metrics"]
```

Full architecture deep-dive — design principles, SDK internals, alert system, testing: [docs/architecture.md](docs/architecture.md)

---

## Configuration

```toml
# .ocw/config.toml — generated by tj onboard

[defaults.budget]
daily_usd = 10.00

[agents.my-email-agent]
description = "Personal email management agent"

  [agents.my-email-agent.budget]
  daily_usd   = 5.00
  session_usd = 1.00

  [[agents.my-email-agent.sensitive_actions]]
  name     = "send_email"
  severity = "critical"

  [agents.my-email-agent.drift]
  enabled           = true
  baseline_sessions = 10
  token_threshold   = 2.0

[capture]
prompts      = false
completions  = false
tool_outputs = false

[storage]
path           = "~/.ocw/telemetry.duckdb"
retention_days = 90
```

Budget limits merge per-field: each agent inherits defaults unless overridden. Set via CLI (`tj budget --daily 10`), API, or web UI. Run `tj doctor` to verify.

Config file discovery order, full config schema, API auth, capture settings: [docs/configuration.md](docs/configuration.md)

---

## CLI reference

16 commands: `onboard`, `doctor`, `status`, `traces`, `cost`, `alerts`, `budget`, `drift`, `tools`, `demo`, `export`, `mcp`, `serve`, `stop`, `uninstall`. All support `--json` for machine-readable output.

Global flags, per-command options, exit codes: [docs/cli-reference.md](docs/cli-reference.md)

---

## Examples

The [`examples/`](examples/) directory has runnable agents for every integration:

- **Single provider** — Anthropic, OpenAI, Gemini, Bedrock, OpenAI Agents SDK
- **Single framework** — LangChain, LangGraph, CrewAI, AutoGen, LlamaIndex
- **Multi-integration** — provider router, CrewAI + LangChain, RAG with fallback
- **Alerts and drift** — sensitive action alerts, budget breach, drift detection (no API keys needed)

```bash
python examples/single_provider/anthropic_agent.py
python examples/alerts_and_drift/drift_demo.py     # no API key needed
```

See [`examples/README.md`](examples/README.md) for the full list.

---

## Agent Incident Library

Reproducible AI agent failures you can run in 30 seconds. No API keys, no config, no setup.

```bash
ocw demo                     # list all scenarios
ocw demo retry-loop          # run one
ocw demo retry-loop --json   # machine-readable output
```

| Scenario | What goes wrong | What OCW catches |
|---|---|---|
| [`retry-loop`](incidents/retry-loop/README.md) | Agent retries a failing tool in a loop, burning time and tokens | `retry_loop` + `failure_rate` alerts fire automatically |
| [`surprise-cost`](incidents/surprise-cost/README.md) | Model silently escalates from Haiku to Opus mid-chain | Per-model cost breakdown shows the $3+ you didn't expect |
| [`hallucination-drift`](incidents/hallucination-drift/README.md) | Agent behavior shifts — different tokens, different tools | `drift_detected` alert fires with Z-scores at session end |

Each scenario runs against an in-memory backend and produces a side-by-side comparison: what `print()` shows vs. what OCW reveals.

---

## Architecture

See [AGENTS.md](AGENTS.md) for codebase conventions.

PRs welcome. If you're adding a framework integration, open an issue first.

---

## Roadmap

**Shipped:**

- [x] `tj serve` background daemon (launchd / systemd)
- [x] Web UI with auto-polling (status, traces, cost, alerts, budget, drift)
- [x] LiteLLM provider patch (100+ providers)
- [x] `tj stop` and `tj uninstall`
- [x] Claude Code integration (`tj onboard --claude-code`)
- [x] Codex integration (`tj onboard --codex`)
- [x] OpenClaw integration (zero-code via `diagnostics-otel` plugin)
- [x] NemoClaw sandbox observer (WebSocket gateway events)
- [x] OTLP log-to-span pipeline (Claude Code log events)
- [x] `tj budget` CLI, API, and web UI
- [x] `tj drift` with Z-score reporting
- [x] Full pipeline wiring (alerts, schema, drift in `tj serve`)
- [x] MCP server — 13 tools for Claude Code

**Up next:**

- [ ] `tj watch` — live tail mode for spans
- [ ] `tj replay` — replay captured sessions against new model versions
- [ ] TypeScript framework patches (LangChain JS, OpenAI Agents SDK)
- [ ] Vercel AI SDK integration (TypeScript)
- [ ] Mastra integration (TypeScript)
- [ ] Azure AI Agent Service integration
- [ ] Docker image
- [ ] GitHub Actions for CI drift/cost checks

---

<div align="center">

**[opencla.watch](https://opencla.watch)** · [PyPI](https://pypi.org/project/tokenjam/) · [npm](https://www.npmjs.com/package/@tokenjam/sdk)

MIT License · Built by [Metabuilder Labs](https://github.com/Metabuilder-Labs)

</div>
