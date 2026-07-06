# Claude Code Integration

Monitor every Claude Code session — costs, tool calls, API requests, errors — with two commands:

```bash
pipx install tokenjam
tj onboard --claude-code
# Restart Claude Code, then:
tj status --agent claude-code-<project>
```

`tj onboard --claude-code` does everything in one shot:
- Creates a shared config at `~/.config/tj/config.toml` (one config for all your projects)
- Writes OTLP exporter vars to `~/.claude/settings.json` so Claude Code sends telemetry automatically
- Tags this project's sessions by writing `OTEL_RESOURCE_ATTRIBUTES=service.name=claude-code-<project>` to `.claude/settings.json`
- Wires the **zero-token statusline** into `~/.claude/settings.json` (`"statusLine": {"type": "command", "command": "tj statusline"}`) — non-destructively; an existing statusLine you (or another tool) authored is left untouched
- Installs a background daemon (launchd on macOS, systemd on Linux) to keep `tj serve` alive across restarts
- Adds Docker harness-compatible OTLP env vars to `~/.zshrc`

**Claude Code must be restarted** after running `tj onboard --claude-code` for the new `settings.json` env vars to take effect.

## Adding a second (or third) project

Run once per project directory, no reinstall needed:

```bash
cd /path/to/other-project
tj onboard --claude-code   # adds agent to shared config, tags this project
# Restart Claude Code
```

Each project gets its own agent ID (`claude-code-<repo-name>`), all sharing one running server and one ingest secret. Running `tj onboard --claude-code` in a new project never rotates the secret or breaks other projects.

Claude Code emits OTLP log events which `tj serve` converts into spans — every API request, tool result, tool decision, and error becomes a first-class span with cost tracking, alert evaluation, and drift detection. Works in both interactive and autonomous (headless) mode.

## Statusline (zero token cost)

`tj onboard --claude-code` wires a **statusline** into `~/.claude/settings.json`:

```json
"statusLine": {"type": "command", "command": "tj statusline"}
```

Claude Code runs `tj statusline` out-of-band after each turn (it never enters the model's context, so it costs **zero quota**) and prints one line — your model, this session's total tokens, and its **re-read share** (cache-read tokens ÷ total), with an actionable `/compact` nudge once re-reading starts eating your budget:

```text
◆ Opus 4.8  2.4M tok  🕳️ re-read 95%  → /compact to reclaim quota
```

The wiring is non-destructive: if you already have a `statusLine` (hand-authored, or from a tool like ccstatusline), tj leaves it untouched and tells you to set it to `tj statusline` yourself. For the deep dive, run `tj tokenmaxx` / `tj optimize`.

## MCP server — for SDK / API users, not Claude Code

The MCP is tj's **in-request-path** surface, meant for **SDK / API** integrations where tj already sits in the loop (real-time enforcement, policy, budgets). It is **not** wired for Claude Code by `tj onboard --claude-code`, and you shouldn't add it as a Claude Code subscription user: an in-loop MCP is a **per-turn quota burden** (a measured A/B showed **+36%** model-weighted quota vs a no-tj control) — the exact tax the out-of-band statusline above avoids. SDK / API users who want the in-loop tools can wire it manually with `claude mcp add tj --scope user -- tj mcp`. After doing so you have 13 tools available in every session:

| Tool | What it does |
|---|---|
| `get_status` | Current agent state — tokens, cost, active alerts |
| `get_budget_headroom` | Budget limit vs spend for an agent |
| `list_active_sessions` | All running sessions across agents |
| `list_agents` | All known agents with lifetime cost |
| `get_cost_summary` | Cost breakdown by day / agent / model |
| `list_alerts` | Alert history with severity and unread filtering |
| `list_traces` | Recent traces with cost and duration |
| `get_trace` | Full span waterfall for a single trace |
| `get_tool_stats` | Tool call counts and average duration |
| `get_drift_report` | Behavioral drift baseline vs latest session |
| `acknowledge_alert` | Mark an alert as acknowledged |
| `setup_project` | Configure a project to send telemetry to TokenJam |
| `open_dashboard` | Open the web UI — starts `tj serve` on demand if needed |

The MCP server opens the DuckDB file read-only — no lock conflicts with `tj serve` if both are running. The single write operation (`acknowledge_alert`) opens a short-lived read-write connection only for its UPDATE.

**Per-project telemetry tagging** — after installing the MCP server globally, ask Claude Code to set up each project:

> "Set up TokenJam for this project"

Claude calls `setup_project`, which writes `.claude/settings.json` with `OTEL_RESOURCE_ATTRIBUTES=service.name=<project>` so spans from that project are tagged with the right agent ID.

## Codex

`tj onboard --codex` is a **smaller** integration than `tj onboard --claude-code` — Codex CLI exposes
less to hook into, not because tj treats it as second-class. It wires OTel telemetry only:

```bash
tj onboard --codex
# Restart Codex, then:
tj tokenmaxx   # or: tj traces
```

This writes an `[otel]` block to `~/.codex/config.toml` pointing at `tj serve`'s OTLP endpoint. Codex
CLI hardcodes `service.name="codex_exec"` in its binary regardless of `[otel.resource]`, so every Codex
terminal's activity lands under one `codex_exec` agent tile — there is no per-project or per-terminal
split like Claude Code gets.

What `--codex` does **not** do, and why:
- **No historical backfill** — no `tj backfill codex` adapter exists yet (Codex does write local
  `~/.codex/sessions/*.jsonl` transcripts that look backfillable; see the investigation below).
- **No statusline** — Codex's own TUI status line has no custom-command hook for tj to inject into.
- **No MCP registration** — same reasoning as Claude Code: an in-loop MCP is a per-turn quota tax,
  and Codex has no zero-cost statusline substitute to fall back on either.

tj stays fully out-of-band for Codex: telemetry flows in automatically once you restart, and you read
it with `tj tokenmaxx` / `tj traces` / the dashboard — no in-loop cost either way.

See **[docs/agent-capability-matrix.md](agent-capability-matrix.md)** for the full Claude Code vs.
Codex vs. Python SDK vs. OTLP capability breakdown, including the backfill/statusline parity
investigation.

## Uninstalling

```bash
# Remove all TokenJam data, config, daemon, MCP registration, and env vars from every onboarded project:
tj uninstall --yes

# Then remove the package itself (tj uninstall intentionally skips this):
pip uninstall tokenjam -y
```

`tj uninstall` cleans up everything set by `tj onboard --claude-code`:
- Stops and removes the background daemon (launchd/systemd)
- Deregisters the MCP server from Claude Code
- Deletes `~/.tj/` (telemetry database)
- Deletes `~/.config/tj/` (global config and projects index)
- Removes OTLP env vars from `~/.claude/settings.json`
- Removes `OTEL_RESOURCE_ATTRIBUTES` from `.claude/settings.json` in every onboarded project
- Removes the harness env block from `~/.zshrc`
