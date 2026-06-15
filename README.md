<div align="center">

<img src="https://tokenjam.dev/icon.svg" alt="TokenJam" width="72" height="72">

# TokenJam

### Token Efficiency For AI Agents

TokenJam reads your agent's telemetry and tells you when to downsize, when to trim prompts, what to cache, and what to script. The result is a lower AI bill. Runs entirely on your machine.

[![CI](https://github.com/Metabuilder-Labs/tokenjam/actions/workflows/ci.yml/badge.svg)](https://github.com/Metabuilder-Labs/tokenjam/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/tokenjam?color=3d8eff&labelColor=0d1117)](https://pypi.org/project/tokenjam/)
[![Python](https://img.shields.io/badge/python-3.10%2B-3d8eff?labelColor=0d1117)](https://pypi.org/project/tokenjam/)
[![npm](https://img.shields.io/npm/v/@tokenjam/sdk?color=3d8eff&labelColor=0d1117)](https://www.npmjs.com/package/@tokenjam/sdk)
[![License: MIT](https://img.shields.io/badge/license-MIT-3d8eff?labelColor=0d1117)](LICENSE)
[![OTel](https://img.shields.io/badge/OTel-GenAI%20SemConv-3d8eff?labelColor=0d1117)](https://opentelemetry.io/docs/specs/semconv/gen-ai/)

```
pipx install tokenjam
```

<sub>Don't have pipx? `brew install pipx` on macOS, `apt install pipx` on Debian/Ubuntu, or see [docs/installation.md](docs/installation.md). `pip install tokenjam` also works in a clean venv.</sub>

**No cloud · No signup · No vendor lock-in**

</div>

---

## Four Analyzers. One Install.

TokenJam reads telemetry from every major agent runtime, framework, provider, and observability tool and surfaces savings across four areas.

<table>
<tr>
<td width="50%" valign="top">

### 🪶 Downsize

Flags sessions where a cheaper model in the same family is worth a look. Never claims quality equivalence — surfaces examples so you can spot-check.

<pre><code>tj optimize downsize</code></pre>

[Details →](docs/optimize/downsize.md)

</td>
<td width="50%" valign="top">

### 💾 Cache

Shows your current caching ratio per (provider, model) and suggests Anthropic prompt-cache breakpoints from stable prefixes in your real usage.

<pre><code>tj optimize cache</code></pre>

[Details →](docs/optimize/cache.md)

</td>
</tr>
<tr>
<td width="50%" valign="top">

### 📜 Script

Finds clusters of deterministic `(tool_name, arg_shape)` sequences that match the shape of work a plain script could replace.

<pre><code>tj optimize script</code></pre>

[Details →](docs/optimize/script.md)

</td>
<td width="50%" valign="top">

### ✂️ Trim

Predicts which regions of your prompts the model gives little weight to. Surfaces what's safe to cut.

<pre><code>tj optimize trim</code></pre>

[Details →](docs/optimize/trim.md)

</td>
</tr>
</table>

Run all four with `tj optimize`. Run several with `tj optimize downsize cache trim`.

---

## 30-second quickstart

For **Claude Code** users — zero code, auto-backfills your last 30 days:

```bash
pipx install 'tokenjam[mcp]'
tj onboard --claude-code
tj optimize          # cost-saving candidates from your actual usage
```

To upgrade later: `pipx upgrade tokenjam` (then `tj stop && tj serve &` to reload the daemon, and `tj --version` to verify). See [docs/installation.md](docs/installation.md#upgrading).

For any Python agent:

```python
from tokenjam.sdk import watch
from tokenjam.sdk.integrations.anthropic import patch_anthropic

patch_anthropic()

@watch(agent_id="my-agent")
def run(task: str) -> str:
    ...
```

→ [Python SDK](docs/python-sdk.md) · [TypeScript SDK](docs/typescript-sdk.md) · [Codex](docs/claude-code-integration.md#codex) · [OTel-compatible agents](docs/framework-support.md)

---

## Why local-first matters

Your spans contain prompts, completions, tool inputs, and customer data. Shipping that to a SaaS vendor for "observability" is a data-egress decision most teams aren't ready to make.

|                                            | TokenJam | LangSmith | Langfuse | Datadog LLM Obs |
|---|---|---|---|---|
| Signup required                            | ❌       | ✅        | ✅       | ✅              |
| Data leaves your machine                   | ❌       | ✅        | cloud only | ✅           |
| Cost-optimization analyzers (Downsize, Cache, Script, Trim) | ✅ | ❌ | ❌ | ❌ |
| Real-time sensitive-action alerts          | ✅       | ❌        | ❌       | ❌              |
| Behavioral drift detection                 | ✅       | ❌        | ❌       | ❌              |
| OTel GenAI SemConv native                  | ✅       | partial   | partial  | partial         |
| Works with any agent / framework           | ✅       | LangChain-first | partial | ❌            |
| Free, MIT licensed                         | ✅       | freemium  | freemium | paid            |

---

## Web UI

`tj serve` runs a local dashboard at `http://127.0.0.1:7391/` with status, traces, cost breakdown, alerts, budget, and drift.

<table>
<tr>
<td width="50%"><img src="docs/screenshots/tj-status.png" alt="tj status page" /></td>
<td width="50%"><img src="docs/screenshots/tj-cost.png" alt="tj cost page" /></td>
</tr>
<tr>
<td width="50%"><img src="docs/screenshots/tj-traces.png" alt="tj traces page" /></td>
<td width="50%"><img src="docs/screenshots/tj-alerts.png" alt="tj alerts page" /></td>
</tr>
</table>

---

## Beyond optimization

TokenJam is also a full observability stack. The four analyzers ride on top.

- **Real-time cost tracking** — every LLM call priced as it happens
- **Safety alerts** — 13 alert types, 6 channels (ntfy, Discord, Telegram, webhook, file, stdout)
- **Behavioral drift detection** — Z-score baselines, no LLM required
- **Schema validation** — declare or infer JSON Schema for tool outputs
- **OTel-native** — point any OTLP exporter at `tj serve` and you're done
- **MCP server** — 14 tools letting Claude Code query its own telemetry mid-session

---

## CLI

```bash
tj optimize            # all four cost-optimization analyzers
tj optimize downsize   # one analyzer
tj status              # current cost, tokens, active alerts
tj cost --since 7d     # spend by agent / model / day / tool
tj alerts              # everything that fired while you were away
tj drift               # behavioral drift Z-scores
tj backfill claude-code # ingest historical ~/.claude/projects/ sessions
tj serve               # start the web UI + REST API
```

[Full CLI reference →](docs/cli-reference.md)

---

## Documentation

| Topic | Where |
|---|---|
| 🪶 Downsize / Cache / Script / Trim deep-dives | [docs/optimize/](docs/optimize/) |
| Claude Code & Codex integration | [docs/claude-code-integration.md](docs/claude-code-integration.md) |
| Python SDK reference | [docs/python-sdk.md](docs/python-sdk.md) |
| TypeScript SDK reference | [docs/typescript-sdk.md](docs/typescript-sdk.md) |
| Framework support (LangChain / CrewAI / etc.) | [docs/framework-support.md](docs/framework-support.md) |
| Alert channels & rule reference | [docs/alerts.md](docs/alerts.md) |
| Backfill from Langfuse / Helicone / OTLP | [docs/backfill/](docs/backfill/) |
| Configuration | [docs/configuration.md](docs/configuration.md) |
| Architecture deep-dive | [docs/architecture.md](docs/architecture.md) |
| Installation extras (Trim, framework patches) | [docs/installation.md](docs/installation.md) |
| Export to Grafana / Datadog / NDJSON | [docs/export.md](docs/export.md) |
| NemoClaw sandbox observer | [docs/nemoclaw-integration.md](docs/nemoclaw-integration.md) |

---

## Roadmap

**Shipped in 0.3.x:** Downsize · Cache · Script · Trim · Claude Code + Codex onboarding · MCP server · Web UI · Backfill adapters (Langfuse, Helicone, OTLP) · Period comparison · Routing-config export · Read-only policy preview

**Up next:**
- [ ] `tj policy add | edit | apply` — unified rule surface
- [ ] `tj replay` — replay captured sessions against new model versions
- [ ] TypeScript framework patches (LangChain JS, OpenAI Agents SDK)
- [ ] Vercel AI SDK & Mastra integrations
- [ ] Docker image
- [ ] GitHub Actions for CI drift/cost checks

---

<div align="center">

**[tokenjam.dev](https://tokenjam.dev)** · [PyPI](https://pypi.org/project/tokenjam/) · [npm](https://www.npmjs.com/package/@tokenjam/sdk) · [Issues](https://github.com/Metabuilder-Labs/tokenjam/issues)

MIT License · Built by [Metabuilder Labs](https://github.com/Metabuilder-Labs)

</div>
