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

**Cost optimization for Claude Code — out of the box.** Run `tj onboard --claude-code` and TokenJam reads your existing Claude Code session logs (up to 30 days, whatever your local retention has kept) so you can run `tj optimize` immediately:

```
$ tj optimize
Analyzing 39 sessions, 3.9M tokens, $237.39 spend (last 30d, claude-code-myproj)…

  ① Model downgrade: 12% of sessions match a smaller-model candidate shape
     • 5 of 39 sessions matched structural heuristics
     • Would have cost ~$1.47 on the smaller model vs $42.18 actual (in window)
     • Projected savings if pattern holds: $51.30/mo
     ! Candidate-flagging heuristic, not a quality judgment.
       Review the example sessions before changing models.

  ② Budget projection (anthropic, $200/cycle): projected to exceed cycle budget
     • Monthly run rate: $237.39 (1.2× the budget)
     • At current pace, exhausted on 2026-05-26 (11 days from now)
     • Projected overage: $37.39
```

Two analyzers reading the same spans you'd otherwise pay LangSmith to host: structural model-downgrade candidate flagging (never claims quality equivalence — surfaces examples to review) and per-provider monthly budget projection. Works with **any** agent already sending TokenJam data, not just Claude Code.

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
tj optimize                 # see cost-saving candidates + budget projection
# Restart your coding agent for live telemetry
```

`tj onboard --claude-code` auto-backfills your existing session logs from `~/.claude/projects/` so `tj optimize` works on the first run — no waiting for new data to accumulate. The MCP server gives your coding agent 14 tools to query its own telemetry mid-session — just ask "how much have I spent today?" or "where could I save money?"

[Full Claude Code & Codex setup →](#claude-code--coding-agents)

### Python SDK

For any Python agent — Anthropic, OpenAI, Gemini, Bedrock, LangChain, CrewAI, and [10+ more](#supported-frameworks).

```bash
pip install tokenjam
tj onboard    # creates config, generates ingest secret
tj doctor     # verify your setup
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
tj status              # current state, cost, active alerts
tj traces              # full span history with waterfall view
tj cost --since 7d     # cost breakdown by agent, model, day
tj optimize            # cost-saving candidates + budget projection
tj backfill claude-code  # ingest historical sessions from ~/.claude/projects/
tj alerts              # everything that fired while you were away
tj budget              # view and set daily/session cost limits
tj drift               # behavioral drift Z-scores vs baseline
tj tools               # tool call history with error rates
tj serve               # start the web UI + REST API
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

## tj vs LangSmith vs Langfuse

LangSmith and Langfuse are excellent for tracing LLM API calls and running evals on chat outputs. `tj` solves a different problem: **autonomous agents running unsupervised with real-world consequences**.

| | `tj` | LangSmith | Langfuse | Datadog LLM Obs |
|---|---|---|---|---|
| Signup required | ❌ | ✅ | ✅ | ✅ |
| Data leaves your machine | ❌ | ✅ | cloud only | ✅ |
| Real-time sensitive action alerts | ✅ | ❌ | ❌ | ❌ |
| Model-downgrade cost recommendations | ✅ | ❌ | ❌ | ❌ |
| Behavioral drift detection | ✅ | ❌ | ❌ | ❌ |
| Local-first, no cloud required | ✅ | ❌ | self-host only | ❌ |
| OTel GenAI SemConv native | ✅ | partial | partial | partial |
| NemoClaw sandbox events | ✅ | ❌ | ❌ | ❌ |
| Works with any agent/framework | ✅ | LangChain-first | partial | ❌ |
| Free, MIT licensed | ✅ | freemium | freemium | paid |

---

## Claude Code + coding agents

### Claude Code

Monitor every Claude Code session and get cost-optimization recommendations from your existing usage in three commands:

```bash
pip install "tokenjam[mcp]"
tj onboard --claude-code   # auto-backfills your existing session logs
tj optimize                # cost-saving candidates + budget projection
# Then restart Claude Code so live telemetry starts flowing
```

`tj onboard --claude-code` does everything in one shot:
- Creates a shared config at `~/.config/tj/config.toml` (one config for all projects)
- Writes OTLP exporter vars to `~/.claude/settings.json`
- Tags this project by writing `OTEL_RESOURCE_ATTRIBUTES` to `.claude/settings.json`
- Registers the MCP server globally (`claude mcp add --scope user tj -- tj mcp`)
- Installs a background daemon (launchd on macOS, systemd on Linux)
- Adds Docker harness-compatible OTLP env vars to `~/.zshrc`
- **Reads your existing `~/.claude/projects/*.jsonl` session logs** and ingests them into the local DB so `tj optimize` returns real numbers on first run (idempotent — safe to re-run)
- Writes a sensible default `[budget.anthropic] usd = 200` for the budget projector to project against — edit `~/.config/tj/config.toml` to change

**Claude Code must be restarted** after running `tj onboard --claude-code`.

#### `tj optimize` — what you actually get

Two analyzers run over the spans TokenJam has captured. The output is read-only recommendations — `tj optimize` never changes how your agent runs.

**① Model-downgrade candidates.** Flags sessions whose structural shape (short input, short output, few tool calls) matches a class of work where a cheaper model in the same provider family is worth reviewing. Never asserts the cheaper model *would have produced the same answer* — only that the shape is worth a look. Real examples are surfaced so you can spot-check before changing models.

**② Budget projection.** Per-provider monthly projection against any `[budget.<provider>]` ceiling you've configured. Scopes spend by provider — an Anthropic budget excludes OpenAI spend. Shows exhaustion date, projected overage, and what the run rate would drop to if you acted on the downgrade candidates.

```bash
tj optimize                                # both analyzers, last 30 days
tj optimize --only budget                  # just the projection
tj optimize --budget anthropic --budget-usd 50   # test a different ceiling
tj optimize --json                         # machine-readable for piping
```

Works alongside a running `tj serve` (read-only fallback). Also exposed as the `get_optimize_report` MCP tool — your coding agent can ask itself "where could I save money?" mid-session.

**Adding more projects** — run once per project directory:

```bash
cd /path/to/other-project
tj onboard --claude-code   # tags this project, no reinstall needed
# Restart Claude Code
```

Each project gets its own agent ID (`claude-code-<repo-name>`), all sharing one server and one ingest secret.

### MCP server

The MCP server gives Claude Code direct access to your observability data inside the session. 14 tools available after restart:

| Tool | What it does |
|---|---|
| `get_optimize_report` | Cost-saving candidates and budget projection — fires for either question (e.g. "where could I save money?" / "will I exceed my budget?") |
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
| `setup_project` | Configure a project for TokenJam telemetry |
| `open_dashboard` | Open the web UI (starts `tj serve` if needed) |

The MCP server opens DuckDB read-only — no lock conflicts with `tj serve`.

**Per-project tagging** — after installing globally, ask Claude Code:

> "Set up TokenJam for this project"

Claude calls `setup_project`, which writes `.claude/settings.json` with the right `OTEL_RESOURCE_ATTRIBUTES` for this project.

### Codex

Monitor every Codex session — run once, globally:

```bash
pip install "tokenjam[mcp]"
tj onboard --codex
```

`tj onboard --codex` is project-agnostic. It writes to `~/.codex/config.toml` (Codex's single global config), so you only run it once — not once per project. Codex hardcodes `service.name="codex_exec"` in its binary, so all sessions appear under the same agent ID regardless of which repo you're working in.

`tj onboard --codex`:
- Writes an `[otel]` block and `[mcp_servers.tj]` to `~/.codex/config.toml`
- Registers the MCP server so Codex can call TokenJam tools directly
- Installs the background daemon (launchd / systemd)

**Codex must be restarted** after running `tj onboard --codex`.

```bash
tj status --agent codex_exec   # check it's working
```

The same 13 MCP tools available to Claude Code are available to Codex after restart.

### Uninstalling

```bash
# Remove all TokenJam data, config, daemon, MCP registration, and env vars:
tj uninstall --yes

# Then remove the package:
pip uninstall tokenjam -y
```

`tj uninstall` cleans up everything set by `tj onboard --claude-code`: daemon, MCP server, `~/.tj/`, `~/.config/tj/`, OTLP env vars in `~/.claude/settings.json`, `OTEL_RESOURCE_ATTRIBUTES` in every onboarded project's `.claude/settings.json`, and the harness env block in `~/.zshrc`.

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
# .tj/config.toml

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
tj export --format otlp       # forward to Grafana, Datadog, any OTel backend
tj export --format openevals  # openevals / agentevals trajectory evaluation
tj export --format json       # NDJSON
tj export --format csv
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

    DB --> CLI["tj CLI"]
    DB --> API["REST API + Web UI\n:7391"]
    DB --> MCP["MCP Server\n13 tools"]
    DB --> Prom["Prometheus\n:7391/metrics"]
```

Full architecture deep-dive — design principles, SDK internals, alert system, testing: [docs/architecture.md](docs/architecture.md)

---

## Configuration

```toml
# .tj/config.toml — generated by tj onboard

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
path           = "~/.tj/telemetry.duckdb"
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
tj demo                     # list all scenarios
tj demo retry-loop          # run one
tj demo retry-loop --json   # machine-readable output
```

| Scenario | What goes wrong | What TokenJam catches |
|---|---|---|
| [`retry-loop`](incidents/retry-loop/README.md) | Agent retries a failing tool in a loop, burning time and tokens | `retry_loop` + `failure_rate` alerts fire automatically |
| [`surprise-cost`](incidents/surprise-cost/README.md) | Model silently escalates from Haiku to Opus mid-chain | Per-model cost breakdown shows the $3+ you didn't expect |
| [`hallucination-drift`](incidents/hallucination-drift/README.md) | Agent behavior shifts — different tokens, different tools | `drift_detected` alert fires with Z-scores at session end |

Each scenario runs against an in-memory backend and produces a side-by-side comparison: what `print()` shows vs. what TokenJam reveals.

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
