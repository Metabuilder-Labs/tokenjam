# Getting started

There's more than one way onto TokenJam, depending on how much you want to commit up front. This
page stacks every entry path from **least** to **most** commitment — start at the top, and stop as
soon as you've seen enough to decide the next step is worth it. Each rung ends with a way to verify
it actually worked before you move on.

## 1. Zero-config evaluator — no install commitment

```bash
npx tokenjam                          # or: uvx tokenjam quickstart
```

Reads the Claude Code session logs you already have on disk (`~/.claude/projects/*.jsonl`) into a
throwaway in-memory database and prints your quota composition (re-read vs. net-new work) and a
session timeline. Nothing is installed, nothing is written to disk, no daemon runs.

**Verify it worked:** you should see a "Where your quota goes" panel with a percentage breakdown. If
you instead see "No Claude Code logs found," you haven't run Claude Code on this machine yet — that's
expected, and there's nothing further to verify at this rung.

See [docs/installation.md](installation.md) for the runner requirements (`uv` or `pipx`) behind
`npx`/`uvx`.

## 2. Claude Code / Codex onboarding wizards

```bash
pipx install tokenjam
tj onboard --claude-code   # or: tj onboard --codex
```

Installs the CLI for real, generates a config, backfills your recent history, and wires up
CLI-specific telemetry (statusline for Claude Code; session-log ingestion for both). See
[docs/agent-capability-matrix.md](agent-capability-matrix.md) for exactly what each persona gets —
the two wizards aren't equivalent, since they're built against different upstream hooks.

**Verify it worked:** run `tj onboard --verify` (or answer "yes" to the interactive verify prompt at
the end of onboarding). It polls for the first real span and reports confirmed / not-confirmed with
a persona-specific cause if something's off. You can also run `tj ping` any time afterward — it emits
one labeled test span through the real capture path and tells you where it landed (HTTP daemon or
local DB), independent of onboarding.

## 3. Framework integrations

If your agent runs on a framework rather than raw API calls, a one-line patch gets you framework-level
spans with no manual instrumentation:

| Framework | Install | Patch call |
|---|---|---|
| LangChain | `pip install tokenjam[langchain]` | `patch_langchain()` |
| LangGraph | `pip install tokenjam[langchain]` | `patch_langgraph()` |
| CrewAI | `pip install tokenjam[crewai]` | `patch_crewai()` |
| AutoGen | `pip install tokenjam[autogen]` | `patch_autogen()` |
| LlamaIndex | *(native OTel — no patch)* | point its exporter at `tj serve` |
| OpenAI Agents SDK | *(native OTel — no patch)* | point its exporter at `tj serve` |

Full matrix, import paths, and the zero-code OTLP table (LlamaIndex, OpenAI Agents SDK, Google ADK,
Strands, Haystack, Pydantic AI, Semantic Kernel) live in
[docs/framework-support.md](framework-support.md) — this table only summarizes the entry point, not
the depth.

**Verify it worked:** `tj status` should show your `agent_id` with a non-zero token count after one
run of your agent. `tj doctor` also flags a silent-onboarding case (onboarded but zero spans yet) as
an info-level check.

## 4. Raw Python SDK

For any Python agent that isn't covered by a framework patch above — direct API calls, a custom
loop, or an in-house framework:

```bash
pipx install tokenjam
tj onboard
```

```python
from tokenjam.sdk import watch
from tokenjam.sdk.integrations.anthropic import patch_anthropic

patch_anthropic()

@watch(agent_id="my-agent")
def run(task: str) -> str:
    ...
```

`tj onboard` also runs stack auto-detection against the current directory and tailors the printed
instrument snippet to the frameworks/providers it finds already imported in your project — so the
snippet above may show up pre-filled with the right provider patch instead of the generic Anthropic
example. Full reference: [docs/python-sdk.md](python-sdk.md) (see
[docs/typescript-sdk.md](typescript-sdk.md) for the Node/TypeScript equivalent).

**Verify it worked:** `tj ping` emits a self-contained test span through the same `record_llm_call()`
path your instrumented code uses, without needing a real agent run — it confirms interception even
if the daemon is down. For your real code, `tj status --agent my-agent` after one run should show
non-zero tokens.

## 5. Already have telemetry somewhere else

If you're already running Langfuse, Helicone, or emitting OTel spans from anything else, you don't
need to instrument anything new — point TokenJam at what you've already got:

```bash
tj backfill langfuse --source-url <url> --api-key <key>
tj backfill helicone --source-url <url> --api-key <key>
tj backfill otlp --source-file <dump.json>
```

Or, for a live OTel emitter with no batch export step, point its OTLP exporter directly at
`tj serve`'s ingest endpoint — no backfill needed, no code change beyond the exporter config. Details
per source, including field-mapping tables and idempotency guarantees: [docs/backfill/](backfill/).

**Verify it worked:** each `tj backfill` run reports `spans_written` vs. `spans_skipped` — a
non-zero `spans_written` on first run confirms the import landed. `tj doctor` confirms DB
connectivity and ingest-secret validity for the live-OTLP path.

---

Once spans are flowing by any of the paths above, you're past "does this work" and into "what do I do
with it" — see [docs/first-hour.md](first-hour.md) for the next step.
