# Agent capability matrix

`tj` supports four ways of getting telemetry in: **Claude Code**, **Codex CLI**, the **Python SDK**,
and **any OTLP-compliant agent** wired directly at the ingest endpoint. These are not equivalent —
each persona gets a different subset of tj's surface, mostly because the upstream tool exposes a
different set of hooks (or none). This page is the honest, per-capability breakdown, in the shape of
[OpenTelemetry's language implementation matrix](https://opentelemetry.io/docs/languages/): rows are
capabilities, columns are personas, cells are Yes / Partial / No with a one-line reason.

Onboarding entry points: `tj onboard --claude-code` · `tj onboard --codex` · `tj onboard` (Python SDK /
generic — see [python-sdk.md](python-sdk.md)) · no onboarding needed for generic OTLP (point your
exporter at `tj serve`'s `/api/v1/spans`, see [framework-support.md](framework-support.md)).

| Capability | Claude Code | Codex CLI | Python SDK | Generic OTLP |
|---|---|---|---|---|
| **Live ingest** | Yes — OTLP log events converted to spans by `api/routes/logs.py` | Yes — same converter, dedicated Codex event parsers (`api_request`, `sse_event`, `user_prompt`, `tool_decision`, `tool_result`) | Yes — in-process `@watch()` + `patch_*()` spans via `TjSpanExporter` | Yes — `POST /api/v1/spans`, any OTel GenAI-semconv-shaped payload |
| **Historical backfill** | Yes — `tj backfill claude-code` (auto-run by `tj onboard --claude-code`) parses `~/.claude/projects/*.jsonl` | Yes — `tj backfill codex` (offered by `tj onboard --codex`) parses `~/.codex/sessions/YYYY/MM/DD/rollout-*.jsonl`; deterministic span ids make re-runs idempotent, cost recomputed from `pricing/models.toml` | N/A — SDK telemetry is live-only, there's no local session log to backfill from | Partial — `tj backfill otlp --source-file <dump>` replays an OTLP JSON dump if you have one; nothing to backfill for a live-only exporter |
| **Statusline (zero-token in-loop nudge)** | Yes — `tj onboard --claude-code` wires `tj statusline` into `~/.claude/settings.json` non-destructively | No — Codex's TUI has its own built-in `[tui].status_line` (fixed items: model, tokens, cost, git-branch, etc.) but no custom-command hook, so tj cannot inject a line into it | N/A — no TUI, the SDK is a library | N/A — depends entirely on the host agent's own UI, tj has nothing to wire |
| **Hooks (SessionStart resume-brief)** | Yes: resume-brief hook installed by default. The PostToolUse output-cap hook has been removed: A/B testing measured it **+5.6%** whole-session on Claude Code, which already truncates Bash output to ~30KB before a PostToolUse hook ever sees it. There is nothing to opt into, and `tj onboard` / `tj uninstall` unwire any copy an older install left behind | No — Codex CLI has no hook system tj integrates with | N/A — no hook concept; equivalent is manual instrumentation via `record_llm_call()` / `record_tool_call()` | N/A |
| **Per-terminal / per-instance identity** | Yes — the installed `claude` shell wrapper sets a distinct `service.instance.id` per terminal so concurrent sessions render as separate dashboard tiles | No — Codex hardcodes `service.name="codex_exec"` in the binary regardless of `[otel.resource]`, so **all** Codex activity across every terminal collapses into one agent tile | N/A — caller sets `agent_id` explicitly per `@watch()` call, so this is under direct code control | Partial — possible if the agent sets distinct resource attributes itself; tj does not automate this for a tool it doesn't control |
| **Dashboard (Lens web UI)** | Yes | Yes | Yes | Yes — dashboard and analyzers are ingestion-source-agnostic; they read from the DB, not from the persona |
| **Analyzers (twelve: downsize / cache / cache-recommend / script / trim / reuse / subagent / summarize / verbosity / relearn / budget-projection / deadweight)** | Yes: all twelve | Partial, ten of twelve: `deadweight` reads Claude Code transcripts on disk, and `relearn`'s transcript lane is Claude Code only while its span lane deliberately skips coding agents so the same failure is never counted twice | Partial, eleven of twelve: no transcripts on disk, so no `deadweight`. `relearn` runs its span lane, which detects and advises but never applies (there is no workspace to write a fix into) | Partial, eleven of twelve, same as the SDK column |
| **MCP server (in-request-path tools)** | No, by design — an in-loop MCP measured **+36%** model-weighted quota overhead on CC subscription users (ticket #59); the statusline is the zero-cost substitute | No, by design — same reasoning; Codex has no zero-cost substitute today (see Statusline row) | Yes — the primary intended use case, tj sits in the SDK's request path already | Yes, if the host agent supports MCP tool-calling — otherwise not applicable |

## Parity investigation: Codex backfill & statusline

Filed as part of ticket #81 to decide whether the two Codex gaps above (backfill, statusline) are
worth a follow-up or are structurally blocked.

**Backfill — feasible, and now shipped.** Codex CLI writes local session transcripts to
`~/.codex/sessions/YYYY/MM/DD/rollout-*.jsonl` (confirmed via Codex CLI docs and third-party parsers —
e.g. `codex-trace`, "Agent Sessions" — that already read this format for session viewing/resuming).
This is structurally the same situation as `core/backfill.py`'s existing Claude Code adapter: an
undocumented-but-stable-enough, reverse-engineerable per-session JSONL format. The parity follow-up
landed as `tj backfill codex` (`core/ingest_adapters/codex.py`), mirroring `ingest_claude_code()`: a
rollout-JSONL parser → `NormalizedSpan` adapter with deterministic idempotent span ids, cost
recomputed from `pricing/models.toml`, plan tier stamped from config, and the `attributes.source =
"backfill.codex"` tag convention. `tj onboard --codex` offers it once Codex logs are present, so
historical Codex sessions are now visible in the dashboard and analyzers just like live ones.

**Statusline — infeasible today, revisit later.** Codex CLI does have a built-in TUI status line
(`[tui].status_line` in `~/.codex/config.toml`, configurable via `/statusline` or the config file,
with presets), but it is restricted to a fixed list of built-in item IDs (model, tokens, cost,
git-branch, context usage, etc.) sourced from Codex's own internal state. There is no custom-command
mechanism to inject an externally-computed line the way Claude Code's `statusLine: {type: "command"}`
works — the upstream feature requests for one (`openai/codex#17827`, `#20244`) are still open/unresolved
as of this writing. Until Codex ships a command-backed status line item, tj has no hook to wire its
zero-token re-read/quota nudge into, and stays fully out-of-band (`tj tokenmaxx` / `tj traces` reads).
