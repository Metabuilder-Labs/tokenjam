<div align="center">

<img src="docs/brand/tokenjam-banner.png" alt="TokenJam" width="440">

### Token Efficiency For AI Agents

TokenJam reads your agent's telemetry and tells you when to downsize, when to trim prompts, what to cache, what to script, and what plans you've already paid to figure out — then shows it all in a local browser dashboard. Runs entirely on your machine.

[![CI](https://github.com/Metabuilder-Labs/tokenjam/actions/workflows/ci.yml/badge.svg)](https://github.com/Metabuilder-Labs/tokenjam/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/tokenjam?color=3d8eff&labelColor=0d1117)](https://pypi.org/project/tokenjam/)
[![Downloads](https://img.shields.io/pypi/dm/tokenjam?color=3d8eff&labelColor=0d1117&label=downloads)](https://pypi.org/project/tokenjam/)
[![Python](https://img.shields.io/badge/python-3.10%2B-3d8eff?labelColor=0d1117)](https://pypi.org/project/tokenjam/)
[![npm](https://img.shields.io/npm/v/@tokenjam/sdk?color=3d8eff&labelColor=0d1117)](https://www.npmjs.com/package/@tokenjam/sdk)
[![License: MIT](https://img.shields.io/badge/license-MIT-3d8eff?labelColor=0d1117)](LICENSE)
[![OTel](https://img.shields.io/badge/OTel-GenAI%20SemConv-3d8eff?labelColor=0d1117)](https://opentelemetry.io/docs/specs/semconv/gen-ai/)

**No cloud · No signup · No vendor lock-in**

<img src="docs/assets/tj-quickstart-hero.png" alt="tj quickstart output: a quota-composition panel showing what share of tokens went to re-reading context vs. net-new work, plus a session timeline with per-session token counts and re-read percentages" width="720">

</div>

---

## Get started

**Install TokenJam and wire it into Claude Code** — one command sets up live capture, all six analyzers, Lens (the local dashboard), and the zero-token statusline:

```bash
pipx install tokenjam && tj onboard --claude-code
```

This installs the CLI, backfills your recent history, and wires the statusline and hooks. Restart Claude Code and you're live.

**Just looking?** `npx tokenjam` prints a 15-second read-only report over the logs you already have — no install, nothing kept.

Building your own agent with the SDK? Install *in your project* (`pip install tokenjam` + `tj onboard`) — see the table below.

<sub>`npx tokenjam` and `uvx tokenjam` launch the Python CLI via `uvx`/`pipx` under the hood — see [docs/installation.md](docs/installation.md) for the runner requirements and the full install matrix.</sub>

---

## Which path are you?

| You are | Run this | What you get |
|---|---|---|
| **Claude Code user** | `pipx install tokenjam && tj onboard --claude-code` | Auto-backfills your last 30 days, wires a zero-token statusline, unlocks all six analyzers + Lens |
| **Codex CLI user** | `pipx install tokenjam && tj onboard --codex` | Same onboarding flow, wired for Codex's session logs |
| **Python SDK / API agent dev** | `pipx install tokenjam && tj onboard` + `@watch()` in your code (below) | Live capture from your own agent process, no CLI-specific backfill |
| **Framework user** (LangChain / CrewAI / AutoGen) | `pip install tokenjam[langchain]` (or `[crewai]` / `[autogen]`) + one `patch_*()` call | Framework-level spans with no manual instrumentation |
| **Already on Langfuse / Helicone** | `tj backfill langfuse --source-url <url> --api-key <key>`<br>(swap `langfuse` → `helicone` — same flags) | One-time import of your existing traces into the local DB |
| **Any OTel-emitting agent** | Point your OTLP exporter at `tj serve` (`http://127.0.0.1:7391/v1/traces`) | Zero-code ingestion — no SDK, no patch |

LlamaIndex and the OpenAI Agents SDK ship their own native OTel support — point their exporter at `tj serve` rather than installing an extra. Full matrix: [docs/framework-support.md](docs/framework-support.md).

Prefer a single page walking every path, each ending with a verify step? See
[docs/getting-started.md](docs/getting-started.md).

---

## Full setup — Claude Code

```bash
pipx install tokenjam
tj onboard --claude-code
tj optimize          # cost-saving candidates from your actual usage
tj serve             # open the dashboard at http://127.0.0.1:7391/
```

Onboarding also wires a **zero-token statusline** into Claude Code — `tj statusline` runs out-of-band each turn (no model quota) and shows this session's re-read share with a `/compact` nudge: `◆ Opus 4.8  2.4M tok  🕳️ re-read 95%  → /compact to reclaim quota`. It does **not** add an in-loop MCP server (that's an SDK / API surface — an MCP would tax every turn).

That's it. Run bare `tj` any time and it points you to the next best action (`tj status`, `tj tokenmaxx`, `tj optimize`, or `tj serve`).

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

→ [Python SDK](docs/python-sdk.md) · [TypeScript SDK](docs/typescript-sdk.md) · [Codex onboarding](docs/claude-code-integration.md#codex) · [OTel-compatible agents](docs/framework-support.md)

---

## Six analyzers + Lens. One install.

TokenJam reads telemetry from every major agent runtime, framework, provider, and observability tool and surfaces savings across six areas — then brings them together in a local browser dashboard.

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

Shows your current caching ratio per (provider, model) and suggests Anthropic prompt-cache breakpoints from stable prefixes in your real usage. Two related CLI names under one product — `cache` measures the ratio, `cache-recommend` suggests the breakpoints.

<pre><code>tj optimize cache
tj optimize cache-recommend</code></pre>

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
<tr>
<td width="50%" valign="top">

### 🔁 Reuse

Detects clusters of sessions where your agent re-plans the same work and exports reviewable skeleton templates you can drop into a slash command or script.

<pre><code>tj optimize reuse</code></pre>

[Details →](docs/optimize/reuse.md)

</td>
<td width="50%" valign="top">

### 🧩 Subagent right-sizing

Breaks a session's cost down per subagent (Claude Code `Task` calls) and flags ones that ran on a premium model or were handed more context than the work needed — sometimes a large share of a session's spend, hidden inside the parent total.

<pre><code>tj optimize subagent</code></pre>

[Details →](docs/optimize/subagent.md)

</td>
</tr>
</table>

`tj optimize` (no args) runs every analyzer — the six above, plus `budget-projection` (projects your monthly run-rate against a configured `[budget.<provider>]` ceiling; powers Lens's Budget screen) and `cache-recommend` (the Cache card's breakpoint-suggestion half, above). Run a subset with `tj optimize downsize cache reuse`.

### 🔭 Lens

`tj serve` brings every analyzer's findings, your real spend, and your alerts together in one local browser dashboard. No cloud, no signup, fully offline.

<pre><code>tj serve</code></pre>

[Details →](https://tokenjam.dev/products/lens)

---

## Lens — the local dashboard

`tj serve` runs Lens at `http://127.0.0.1:7391/`: a **Dashboard** that lands you on recoverable waste and health at a glance, with an embedded explorer to slice your usage any way (metric × dimension × chart); plus Status, Traces, Cost, Analytics, Alerts, Drift, Optimize, and Budget screens. Plan-tier-aware, fully offline, no signup.

<table>
<tr>
<td width="50%"><img src="docs/screenshots/tj-dashboard.png" alt="Dashboard — recoverable waste, health at a glance, and the embedded pivot explorer" /></td>
<td width="50%"><img src="docs/screenshots/tj-cost.png" alt="Cost — spend over time + cache savings" /></td>
</tr>
<tr>
<td width="50%"><img src="docs/screenshots/tj-traces.png" alt="Trace waterfall — session-level spans with cost annotations" /></td>
<td width="50%"><img src="docs/screenshots/tj-status.png" alt="Status — per-agent cards" /></td>
</tr>
<tr>
<td width="50%"><img src="docs/screenshots/tj-dashboard-tools.png" alt="Analytics explorer — tool-usage leaderboard" /></td>
<td width="50%"><img src="docs/screenshots/tj-dashboard-leaderboard.png" alt="Analytics explorer — cost-by-model leaderboard" /></td>
</tr>
</table>

→ [tokenjam.dev/products/lens](https://tokenjam.dev/products/lens) for the visual walkthrough.

---

## Beyond optimization

TokenJam is also a full observability stack. The six analyzers and Lens ride on top.

- **Real-time cost tracking** — every LLM call priced as it happens
- **Safety alerts** — 13 alert types, 6 channels (ntfy, Discord, Telegram, webhook, file, stdout)
- **Behavioral drift detection** — Z-score baselines, no LLM required
- **Schema validation** — declare or infer JSON Schema for tool outputs
- **Context & quota audits** — `tj context` (re-read vs. net-new split) and `tj quota-audit` (retroactive Opus usage check) over your Claude Code sessions
- **Close the loop** — `tj loop` annotates a run with a verdict, promotes a bad run into a stored expectation, and tracks whether later runs pass or regress against it
- **Prompt summarization (advisory)** — `tj summarize` finds prompt files worth condensing and estimates the per-call saving
- **Enforcement-plane proxy (suggest mode)** — `tj proxy` surfaces routing suggestions locally, without rewriting requests
- **OTel-native** — point any OTLP exporter at `tj serve` and you're done
- **Statusline** — a zero-token Claude Code status line (`tj statusline`, wired by `tj onboard --claude-code`) showing this session's re-read share + a `/compact` nudge
- **MCP server** — in-request-path tools for **SDK / API** users (not Claude Code / Codex subscription users — an in-loop MCP is a per-turn quota burden there; they get the out-of-band statusline instead)

---

## Prove a swap holds — TokenJam Bench

`tj optimize downsize` flags *candidates*: cheaper models worth a look. It never claims the cheaper model would have produced the same answer. **[TokenJam Bench](https://github.com/Metabuilder-Labs/tokenjam-bench)** is the companion that checks. It runs your original and candidate models against real task suites and reports the pass-rate difference with statistics (Wilson CI + McNemar), so you get a hedged verdict ("holds" or "regressed") instead of a guess.

```bash
pip install tokenjam-bench
tjb run --original anthropic:claude-opus-4-7 --candidate anthropic:claude-haiku-4-5
```

Bench reports measured pass-rate on a suite, never "certified" or "quality preserved." Open source and local, like TokenJam. [Learn more →](https://github.com/Metabuilder-Labs/tokenjam-bench)

---

## CLI

```bash
tj optimize            # every analyzer (the six above, plus budget-projection + cache-recommend)
tj optimize downsize   # one analyzer (positional args)
tj tokenmaxx           # shareable spend-tier callout
tj status              # current cost, tokens, active alerts
tj cost --since 7d     # spend by agent / model / day / tool
tj alerts              # everything that fired while you were away
tj drift               # behavioral drift Z-scores
tj report --reuse      # HTML + Markdown skeleton export for the Reuse analyzer
tj backfill claude-code # ingest historical ~/.claude/projects/ sessions
tj serve               # start Lens + REST API
```

[Full CLI reference →](docs/cli-reference.md)

---

## Documentation

| Topic | Where |
|---|---|
| 🚦 Getting started — every entry path, by persona | [docs/getting-started.md](docs/getting-started.md) |
| ⏱️ The first hour — what to do once data flows | [docs/first-hour.md](docs/first-hour.md) |
| 🪶 Downsize / Cache / Script / Trim deep-dives | [docs/optimize/](docs/optimize/) |
| 🔁 Reuse analyzer deep-dive | [docs/optimize/reuse.md](docs/optimize/reuse.md) |
| 🧪 Prove a downsize candidate holds (TokenJam Bench) | [tokenjam-bench](https://github.com/Metabuilder-Labs/tokenjam-bench) |
| Claude Code & Codex integration | [docs/claude-code-integration.md](docs/claude-code-integration.md) |
| Claude Code vs. Codex vs. SDK vs. OTLP — capability matrix | [docs/agent-capability-matrix.md](docs/agent-capability-matrix.md) |
| Harness run grouping (governors / fan-out launchers) | [docs/harness-integration.md](docs/harness-integration.md) |
| Python SDK reference | [docs/python-sdk.md](docs/python-sdk.md) |
| TypeScript SDK reference | [docs/typescript-sdk.md](docs/typescript-sdk.md) |
| Framework support (LangChain / CrewAI / etc.), including the full OTel provider/framework matrix | [docs/framework-support.md](docs/framework-support.md) |
| Alert channels & rule reference | [docs/alerts.md](docs/alerts.md) |
| Backfill from Langfuse / Helicone / OTLP | [docs/backfill/](docs/backfill/) |
| Enforcement-plane proxy (suggest mode) | [docs/proxy/overview.md](docs/proxy/overview.md) |
| Policy rules | [docs/policy/overview.md](docs/policy/overview.md) |
| Configuration | [docs/configuration.md](docs/configuration.md) |
| Architecture deep-dive | [docs/architecture.md](docs/architecture.md) |
| Installation extras (Trim, framework patches) | [docs/installation.md](docs/installation.md) |
| Export to Grafana / Datadog / NDJSON | [docs/export.md](docs/export.md) |
| NemoClaw sandbox observer | [docs/nemoclaw-integration.md](docs/nemoclaw-integration.md) |
| Release notes | [GitHub Releases](https://github.com/Metabuilder-Labs/tokenjam/releases) |

---

## Roadmap

**Shipped:** Downsize · Cache · Script · Trim · Reuse · Subagent right-sizing · Claude Code + Codex onboarding · MCP server · Lens web UI · Backfill adapters (Langfuse, Helicone, OTLP) · Period comparison · Routing-config export · Read-only policy preview · Context & quota audits · Close-the-loop annotations/expectations · Prompt summarization (advisory) · Enforcement-plane proxy (suggest mode)

**Up next** (roughly):
- [ ] Continued Lens polish + per-product visual branding
- [ ] `tj policy add | edit | apply` — unified rule surface (today: `tj policy list` / `tj policy decisions`)
- [ ] `tj replay` — replay captured sessions against new model versions
- [ ] TypeScript framework patches (LangChain JS, OpenAI Agents SDK)
- [ ] Vercel AI SDK & Mastra integrations
- [ ] Published Docker image
- [ ] GitHub Actions for CI drift/cost checks

Full version-by-version history: [GitHub Releases](https://github.com/Metabuilder-Labs/tokenjam/releases).

---

## Contributing

TokenJam is MIT, and contributions are welcome — from a one-line pricing fix to a whole new framework integration. A few easy on-ramps:

- 🟢 **[Good first issues →](https://github.com/Metabuilder-Labs/tokenjam/labels/good%20first%20issue)** — scoped, newcomer-friendly tasks, ready to pick up.
- **Bugs** — notice something off? File a bug.
- **Documentation** — struggled with something while getting started? Help the next person by writing or updating documentation.
- 💸 **Model pricing** — `tokenjam/pricing/models.toml` is community-maintained. Fix a rate or add a model in a single PR — no issue needed.
- 🔌 **Framework integrations** — provider/framework patches follow one clear pattern (`tokenjam/sdk/integrations/anthropic.py` is the reference). Open an issue first to align on approach.
- 🤖 **Coding Agents are first-class citizens** — TokenJam is built by Humans AND AI coding agents, and contributing with one is first-class. **Claude Code:** read [CLAUDE.md](CLAUDE.md) and run `/init` to bring your agent up to speed. **Codex / other agents:** [AGENTS.md](AGENTS.md) has the critical rules.

Setup and the full dev workflow are in **[CONTRIBUTING.md](CONTRIBUTING.md)**.

If TokenJam saves you tokens, **⭐ star it** and **👁 watch for releases** — we ship often.

---

<div align="center">

**[tokenjam.dev](https://tokenjam.dev)** · [PyPI](https://pypi.org/project/tokenjam/) · [npm](https://www.npmjs.com/package/@tokenjam/sdk) · [TokenJam Bench](https://github.com/Metabuilder-Labs/tokenjam-bench) · [Issues](https://github.com/Metabuilder-Labs/tokenjam/issues)

MIT License · Built by [Metabuilder Labs](https://github.com/Metabuilder-Labs)

TokenJam was created by [Anil Murty](https://github.com/anilmurty) — reach him at [anil@metabldr.com](mailto:anil@metabldr.com).

</div>
