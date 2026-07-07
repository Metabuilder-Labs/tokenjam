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
| **Historical backfill** | Yes — `tj backfill claude-code` (auto-run by `tj onboard --claude-code`) parses `~/.claude/projects/*.jsonl` | No — no `tj backfill codex` adapter exists yet; see investigation below | N/A — SDK telemetry is live-only, there's no local session log to backfill from | Partial — `tj backfill otlp --source-file <dump>` replays an OTLP JSON dump if you have one; nothing to backfill for a live-only exporter |
| **Statusline (zero-token in-loop nudge)** | Yes — `tj onboard --claude-code` wires `tj statusline` into `~/.claude/settings.json` non-destructively | No — Codex's TUI has its own built-in `[tui].status_line` (fixed items: model, tokens, cost, git-branch, etc.) but no custom-command hook, so tj cannot inject a line into it | N/A — no TUI, the SDK is a library | N/A — depends entirely on the host agent's own UI, tj has nothing to wire |
| **Hooks (SessionStart resume-brief / PostToolUse output-cap)** | Yes — resume-brief hook installed by default; output-cap hook is opt-in | No — Codex CLI has no hook system tj integrates with | N/A — no hook concept; equivalent is manual instrumentation via `record_llm_call()` / `record_tool_call()` | N/A |
| **Per-terminal / per-instance identity** | Yes — the installed `claude` shell wrapper sets a distinct `service.instance.id` per terminal so concurrent sessions render as separate dashboard tiles | No — Codex hardcodes `service.name="codex_exec"` in the binary regardless of `[otel.resource]`, so **all** Codex activity across every terminal collapses into one agent tile | N/A — caller sets `agent_id` explicitly per `@watch()` call, so this is under direct code control | Partial — possible if the agent sets distinct resource attributes itself; tj does not automate this for a tool it doesn't control |
| **Dashboard (Lens web UI)** | Yes | Yes | Yes | Yes — dashboard and analyzers are ingestion-source-agnostic; they read from the DB, not from the persona |
| **Analyzers (downsize / cache / script / trim / reuse / budget-projection)** | Yes | Yes | Yes | Yes |
| **MCP server (in-request-path tools)** | No, by design — an in-loop MCP measured **+36%** model-weighted quota overhead on CC subscription users (ticket #59); the statusline is the zero-cost substitute | No, by design — same reasoning; Codex has no zero-cost substitute today (see Statusline row) | Yes — the primary intended use case, tj sits in the SDK's request path already | Yes, if the host agent supports MCP tool-calling — otherwise not applicable |

## Parity investigation: Codex backfill & statusline

Filed as part of ticket #81 to decide whether the two Codex gaps above (backfill, statusline) are
worth a follow-up or are structurally blocked.

**Backfill — feasible, follow-up recommended.** Codex CLI writes local session transcripts to
`~/.codex/sessions/YYYY/MM/DD/rollout-*.jsonl` (confirmed via Codex CLI docs and third-party parsers —
e.g. `codex-trace`, "Agent Sessions" — that already read this format for session viewing/resuming).
This is structurally the same situation as `core/backfill.py`'s existing Claude Code adapter: an
undocumented-but-stable-enough, reverse-engineerable per-session JSONL format. A `tj backfill codex`
adapter mirroring `ingest_claude_code()` looks buildable without any missing capability on Codex's
side — it just hasn't been written yet. **Not implemented in this PR** (out of scope — this ticket
only requires the decision, not the adapter); worth filing as its own ticket if wanted.

**Statusline — infeasible today, revisit later.** Codex CLI does have a built-in TUI status line
(`[tui].status_line` in `~/.codex/config.toml`, configurable via `/statusline` or the config file,
with presets), but it is restricted to a fixed list of built-in item IDs (model, tokens, cost,
git-branch, context usage, etc.) sourced from Codex's own internal state. There is no custom-command
mechanism to inject an externally-computed line the way Claude Code's `statusLine: {type: "command"}`
works — the upstream feature requests for one (`openai/codex#17827`, `#20244`) are still open/unresolved
as of this writing. Until Codex ships a command-backed status line item, tj has no hook to wire its
zero-token re-read/quota nudge into, and stays fully out-of-band (`tj tokenmaxx` / `tj traces` reads).
