# Claude Code Integration

Monitor every Claude Code session — costs, tool calls, API requests, errors — with two commands:

```bash
pip install "tokenjam[mcp]"
tj onboard --claude-code
# Restart Claude Code, then:
tj status --agent claude-code-<project>
```

`tj onboard --claude-code` does everything in one shot:
- Creates a shared config at `~/.config/ocw/config.toml` (one config for all your projects)
- Writes OTLP exporter vars to `~/.claude/settings.json` so Claude Code sends telemetry automatically
- Tags this project's sessions by writing `OTEL_RESOURCE_ATTRIBUTES=service.name=claude-code-<project>` to `.claude/settings.json`
- Registers the MCP server globally (`claude mcp add --scope user tj -- tj mcp`)
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

## MCP server

The MCP server is included in the `[mcp]` extra and registered automatically by `tj onboard --claude-code`. It gives Claude Code direct access to your observability data inside the session itself. After restarting Claude Code you have 13 tools available in every session:

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
| `setup_project` | Configure a project to send telemetry to OCW |
| `open_dashboard` | Open the web UI — starts `tj serve` on demand if needed |

The MCP server opens the DuckDB file read-only — no lock conflicts with `tj serve` if both are running. The single write operation (`acknowledge_alert`) opens a short-lived read-write connection only for its UPDATE.

**Per-project telemetry tagging** — after installing the MCP server globally, ask Claude Code to set up each project:

> "Set up OCW for this project"

Claude calls `setup_project`, which writes `.claude/settings.json` with `OTEL_RESOURCE_ATTRIBUTES=service.name=<project>` so spans from that project are tagged with the right agent ID.

## Uninstalling

```bash
# Remove all OCW data, config, daemon, MCP registration, and env vars from every onboarded project:
tj uninstall --yes

# Then remove the package itself (ocw uninstall intentionally skips this):
pip uninstall tokenjam -y
```

`tj uninstall` cleans up everything set by `tj onboard --claude-code`:
- Stops and removes the background daemon (launchd/systemd)
- Deregisters the MCP server from Claude Code
- Deletes `~/.ocw/` (telemetry database)
- Deletes `~/.config/ocw/` (global config and projects index)
- Removes OTLP env vars from `~/.claude/settings.json`
- Removes `OTEL_RESOURCE_ATTRIBUTES` from `.claude/settings.json` in every onboarded project
- Removes the harness env block from `~/.zshrc`
