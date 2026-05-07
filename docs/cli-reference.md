# CLI Reference

All commands support `--json` for machine-readable output. Commands that query alerts use exit code 1 if active (unacknowledged, unsuppressed) alerts exist.

## Global options

```
--config PATH    Override config file path
--db PATH        Override database path
--agent ID       Filter to a specific agent
--json           Output in JSON format
--no-color       Disable color output
-v, --verbose    Verbose output
```

## Commands

### `tj onboard`

Guided setup wizard. Creates config file, generates ingest secret, optionally installs background daemon.

```bash
tj onboard                  # interactive setup
tj onboard --claude-code    # configure Claude Code telemetry
tj onboard --no-daemon      # skip daemon installation
tj onboard --budget 5.00    # set daily budget during setup
tj onboard --force          # overwrite existing config
```

### `tj doctor`

Health check — validates config, database connectivity, ingest secret, and alert channel reachability.

```bash
tj doctor
```

Exit codes: 0 = healthy, 1 = warnings, 2 = errors.

### `tj status`

Current agent state: session info, cost, token counts, active alerts.

```bash
tj status
tj status --agent my-agent
```

### `tj traces`

Trace listing with span waterfall view.

```bash
tj traces
tj traces --since 1h
tj trace <trace-id>         # full span waterfall for a single trace
```

### `tj cost`

Cost breakdown by agent, model, day, or tool.

```bash
tj cost
tj cost --since 7d
tj cost --group-by model    # group by model
tj cost --group-by day      # group by day
tj cost --group-by agent    # group by agent
tj cost --group-by tool     # group by tool
```

### `tj alerts`

Alert history with severity and type filtering.

```bash
tj alerts
tj alerts --severity critical
tj alerts --type sensitive_action
tj alerts --since 1h
tj alerts --unread           # only unacknowledged alerts
```

### `tj budget`

View and set daily/session cost limits.

```bash
tj budget                                    # view all budgets
tj budget --agent my-agent --daily 5.00      # set daily limit
tj budget --agent my-agent --session 1.00    # set session limit
```

### `tj drift`

Behavioral drift report: baseline vs latest session Z-scores.

```bash
tj drift
tj drift --agent my-agent
```

Exit code 1 if any agent has drifted (useful for CI gating).

### `tj tools`

Tool call summary: call counts, average duration, error rates.

```bash
tj tools
tj tools --since 1h
```

### `tj export`

Export spans in multiple formats.

```bash
tj export --format json
tj export --format csv --output spans.csv
tj export --format otlp
tj export --format openevals --output traces.json
```

### `tj mcp`

Start the MCP server (stdio transport for Claude Code). Registered automatically by `tj onboard --claude-code`.

```bash
tj mcp
```

### `tj serve`

Start the local REST API server with web UI and Prometheus metrics.

```bash
tj serve                    # foreground
tj serve &                  # background
tj serve --host 0.0.0.0    # bind to all interfaces
tj serve --port 8080        # custom port
tj serve --reload           # auto-reload for development
```

Web UI: `http://127.0.0.1:7391/`
API docs: `http://127.0.0.1:7391/docs`
Metrics: `http://127.0.0.1:7391/metrics`

### `tj stop`

Stop the background daemon or `tj serve` process.

```bash
tj stop
```

### `tj uninstall`

Remove all OCW data, config, daemon, MCP registration, and env vars.

```bash
tj uninstall          # interactive confirmation
tj uninstall --yes    # skip confirmation
```
