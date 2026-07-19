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
tj onboard --verify         # poll for the first span after setup and report confirmed/not-confirmed
tj onboard --verify-only    # skip setup; just re-poll an existing install (post-restart re-check)
```

Key flags for non-interactive setup: `--plan`, `--budget`, `--no-daemon` (plus `--project` for `--claude-code`) skip every prompt — use these to run onboarding unattended (CI, Docker, a script). `--verify` is separate: it opts *into* the post-setup telemetry poll instead of the interactive "verify now?" confirm.

**`--verify-only`** is the lightweight post-restart re-check: it skips the whole wizard (no config rewrite, no summary, no restart banner) and only polls an already-configured install for its first *live* span. Use it after you've restarted Claude Code / Codex — `tj onboard --claude-code --verify-only` (or `--codex`, or bare for an SDK install) reads that persona's existing config and reports confirmed / not-confirmed. Backfilled history doesn't count here; the poll waits for a new live span.

**`--verify` and `--no-daemon`:** verification polls for the first span through whichever read path is available. With the daemon running it reads over HTTP; with `--no-daemon`, the poller opens the DuckDB file directly — the same file the SDK would need to write to. If nothing else is writing yet (the pre-first-run case) this works; if something else holds the write lock, verification reports "start `tj serve`" rather than confirming, even though onboarding itself succeeded. Run `tj ping` instead to prove interception without touching the DB lock at all, or start `tj serve` temporarily to get a live confirmation.

### `tj ping`

Emit one clearly-labeled test span through the real SDK export path to prove instrumentation is wired up, without needing a whole agent. Reports whether the span was intercepted and where it was delivered (running daemon over HTTP, or local DuckDB directly).

```bash
tj ping
tj ping --agent my-agent
tj ping --json
```

Exit codes: 0 = intercepted and delivered, 1 = interception or delivery failed.

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

### `tj context`

Diagnose where your Claude Code quota goes: what share of tokens is re-reading prior context (conversation history, CLAUDE.md, tool output) vs. net-new work, plus recurring inclusions (capture-gated) and `/compact` candidates. Subscription plans see a token-share/quota headline; API plans see dollars as a secondary line. Needs a direct DB connection or a running `tj serve` (computed server-side when the daemon holds the write lock).

```bash
tj context
tj context --since 7d --agent my-agent
tj context --json
```

Key flags: `--since`, `--agent`, `--json`.

### `tj session-story`

Turn-by-turn reconstruction of *how* a Claude Code session attempted its task — its ordered moves (`delegate` / `dead_end` / `verify` / `act`), and for every subagent delegation, that subagent's mandate plus a factual tool-category tally (reads/edits/searches/commands) and its own recursively-rendered method spine. With no `--session`, auto-selects the most recent session with real activity. Reads the on-disk transcript, falling back to a persisted snapshot when the transcript was pruned; needs a direct DB connection or a running `tj serve` (the daemon does the reconstruction+fallback server-side when it holds the write lock).

```bash
tj session-story
tj session-story --session <session-id>
tj session-story --json
```

Key flags: `--session`, `--last` (default when `--session` is omitted), `--json`.

### `tj quota-audit`

Retroactive audit of your premium (Opus/Fable) quota: which past premium-tier sessions were structurally Sonnet-shaped (small input/output, few tool calls)? Covers Fable (the tier above Opus) as well as Opus. Reports the percent of premium quota that went to Sonnet-shaped sessions (a retrospective behaviour mirror — those tokens are already spent, so it is misallocated, not "reclaimable"), example sessions to spot-check, and an optional tuned routing-config export. Subscription users see a habit nudge; API-billed users additionally see the already-billed dollar counterfactual. Quota-share framing, never a dollar "saving" claim. The JSON field is `percent_quota_misallocated` (the pre-0.6 `percent_quota_reclaimable` key is still emitted as a deprecated alias for one release). Needs a direct DB connection (can't run against a live `tj serve`).

```bash
tj quota-audit
tj quota-audit --since 30d --agent my-agent
tj quota-audit --export-config claude-code
tj quota-audit --json
```

Key flags: `--since`, `--agent`, `--export-config`, `--json`.

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

### `tj loop`

Close the loop on a run: annotate it, promote it into an expectation, and track
whether later runs pass or regress. Local-first — pass/regress is your recorded
verdict, not an automated score. Also available as the Lens "Loop" tab on a
session's detail page.

```bash
# Annotate a run with a human note + optional verdict (good/bad/mixed/unknown)
tj loop annotate <session_id> --verdict bad --note "retried the same tool 5x"
tj loop annotations <session_id>

# Promote a run into a stored expectation, then record reruns against it
tj loop expect <session_id> --name "no retry loop" --desc "must not retry >3x"
tj loop expectations
tj loop record <expectation_id> <session_id> --outcome pass --note "fixed it"
tj loop history <expectation_id>
```

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

### `tj optimize`

Analyze recent usage for cost-saving candidates, cache opportunities, prompt trimming, workflow reuse, and budget exposure.

```bash
tj optimize                                # run all analyzers
tj optimize downsize cache reuse           # run selected analyzers
tj optimize --since 7d --agent my-agent    # scope the analysis window
tj optimize --compare last-7d              # compare against a prior window
tj optimize --export-config claude-code    # write advisory routing recommendations
tj optimize --json                         # machine-readable report
```

Key flags: `--since`, `--agent`, `--budget`, `--budget-usd`, `--compare`, `--export-config`, `--export-templates`, `--json`.

### `tj route`

Compile advisory router configs from downsize findings. Exports are written under the TokenJam config directory for manual review and are not applied automatically.

```bash
tj route export --target ccr
tj route export --target litellm --since 7d
tj route export --check
tj route export --target ccr --json
```

Key flags on `tj route export`: `--target ccr|litellm`, `--check`, `--agent`, `--since`, `--json`.

### `tj tokenmaxx`

Show a shareable spend-tier summary for the selected usage window, paired with the downsize savings figure.

```bash
tj tokenmaxx
tj tokenmaxx --since 7d
tj tokenmaxx --json
```

Key flags: `--since`, `--json`.

### `tj pricing`

Read-only inspection of the resolved model pricing table — one row per `(provider, model)` with input/output/cache-read/cache-write rates in USD per million tokens, plus a `source` column (`override` vs. `packaged`).

```bash
tj pricing list
tj pricing list --model claude-opus
tj pricing list --json
```

Key flags: `--model`, `--json`.

### `tj backfill`

Ingest historical telemetry from local Claude Code logs or external observability exports.

```bash
tj backfill claude-code
tj backfill claude-code --since 30d --quiet
tj backfill langfuse --source-file observations.json
tj backfill helicone --source-url https://api.helicone.ai --api-key <key>
tj backfill otlp --source-file spans.ndjson
```

Subcommands: `claude-code`, `langfuse`, `helicone`, `otlp`.

Key flags: `--since` on all sources; `--root`, `--since-days`, and `--quiet` for `claude-code`; `--source-url`, `--source-file`, and `--api-key` for Langfuse and Helicone; `--source-url` and `--source-file` for OTLP.

### `tj report`

Generate standalone HTML reports for analyzer findings. Reuse reports can also write Markdown skeleton sidecars.

```bash
tj report --trim
tj report --trim my-agent --since 7d
tj report --reuse
tj report --reuse my-agent --no-open
```

Key flags: `--trim [agent_id]`, `--reuse [agent_id]`, `--since`, `--no-open`.

### `tj summarize`

Structure-aware prompt summarization (advisory). `list` scans for prompt files worth summarizing and estimates the per-call token saving (read-only). `prep` wraps a prompt's structure behind verbatim markers and emits it for a model to rewrite — `--via claude-p` or `--via api` runs the rewrite for you in one shot. `check` verifies a rewrite preserved every structure block (a hard gate) and stages it. `apply` writes a staged rewrite back to the file (default dry-run; `--go` writes, with a backup); `undo` restores from that backup.

```bash
tj summarize list
tj summarize list --recursive --json
tj summarize prep path/to/prompt.md
tj summarize prep path/to/prompt.md --via claude-p
tj summarize check path/to/prompt.md --summary rewrite.md --prepped-hash <hash>
tj summarize apply path/to/prompt.md
tj summarize apply --go
tj summarize undo path/to/prompt.md --go
```

Subcommands: `list`, `prep`, `check`, `apply`, `undo`.

Key flags: `list` — `--recursive`, `--repo`, `--no-global`, `--ext`, `--min-prose`, `--json`; `prep` — `--via claude-p|api`, `--ratio`, `--json`; `check` — `--summary`, `--prepped-hash`, `--json`; `apply`/`undo` — `--go`, `--dry-run`, `--json`.

### `tj policy`

Inspect policy-adjacent configuration and recent suggest-mode policy decisions.

```bash
tj policy list
tj policy list --json
tj policy decisions
tj policy decisions --since 7d --limit 50
tj policy decisions --json
```

Subcommands: `list`, `decisions`.

Key flags: `--json` for both subcommands; `--limit` and `--since` for `decisions`.

### `tj proxy`

Manage the optional suggest-mode proxy and its provider base-URL wiring.

```bash
tj proxy status
tj proxy enable
tj proxy disable
tj proxy killswitch
tj proxy killswitch --off
```

Subcommands: `enable`, `disable`, `status`, `killswitch`.

Key flag: `--off` on `killswitch` releases pass-through mode.

## Integration entrypoints

These are wired by `tj onboard --claude-code` and invoked by Claude Code itself (via hooks / the statusline / the shell wrapper), not typically run by hand — documented here for completeness and troubleshooting.

### `tj statusline`

Zero-model-token status line for Claude Code. Reads the session payload JSON Claude Code pipes on stdin and prints one line: model, session token total, and the re-read share (cache-read ÷ total tokens), with a `/compact` nudge once re-reading dominates. Runs out-of-band after each turn — never enters the model's context, so it costs no quota. Wired into `~/.claude/settings.json`'s `statusLine` by `tj onboard --claude-code`.

```bash
tj statusline   # reads payload JSON on stdin; not meant to be typed interactively
```

### `tj resume-brief`

Hands a resuming (or post-compaction) session a compact brief of its prior method — task, progress, dead ends, working files — instead of re-investigating. Deterministic, no LLM, zero in-loop token cost. `tj onboard --claude-code` wires `--from-hook` into a `SessionStart` hook automatically.

```bash
tj resume-brief --from-hook       # SessionStart-hook mode: reads session_id/transcript_path from stdin
tj resume-brief --session <id>
tj resume-brief --transcript <path>
tj resume-brief --last            # manual: most recently active session by mtime
```

Key flags: `--from-hook`, `--session`, `--transcript`, `--last` (exactly one is expected).

### `tj otel-resource-attrs`

Prints this project's OTel resource attributes (`service.name=claude-code-<repo>[,service.namespace=<project>]`) on one bare line. Called by the `claude` shell wrapper (installed by `tj onboard --claude-code`) to build each terminal's `OTEL_RESOURCE_ATTRIBUTES`, appending a per-terminal `service.instance.id`.

```bash
tj otel-resource-attrs
```

### `tj session-end`

Reports a terminal's Claude Code session(s) as closed, so the dashboard archives that tile immediately (Claude Code emits no close event of its own). Called best-effort by the `claude` shell wrapper on exit/interrupt; talks to the running daemon over HTTP and never touches the DB directly. Always exits 0 — a failure here must never break the user's shell.

```bash
tj session-end --instance <terminal-id>
tj session-end --session <session_id>
tj session-end -v --instance <terminal-id>   # -v surfaces what happened on failure
```

Key flags: `--instance`, `--session` (at least one required).

### `tj demo`

Run reproducible Agent Incident Library scenarios without API keys or external services.

```bash
tj demo                         # list available scenarios
tj demo retry-loop              # run one scenario
tj demo retry-loop --json       # machine-readable scenario output
```

Key flag: `--json`.

### `tj mcp`

Start the MCP server (stdio transport, for SDK / API integrations). `tj onboard --claude-code` / `--codex` do **not** register it — an in-loop MCP is a per-turn quota tax on subscription users (ticket #59); wire it manually with `claude mcp add tj --scope user -- tj mcp` only if you're building an SDK / API integration.

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

Full removal: all TokenJam data, config, daemon, MCP registration, and env vars — AND the `tokenjam`
package itself (pipx/uv-tool installs are removed automatically; a plain pip/venv install gets the
exact `pip uninstall` command printed instead of a guess). The symmetric counterpart to `tj onboard`.

```bash
tj uninstall          # interactive confirmation
tj uninstall --yes    # skip confirmation
```

### `tj reset`

Config-only teardown — the same wiring/config cleanup as `tj uninstall` above, but leaves the
`tokenjam` package installed so `tj onboard` works again without reinstalling. Use this to reconfigure
or pause TokenJam.

```bash
tj reset          # interactive confirmation
tj reset --yes    # skip confirmation
```
