# Manual Release Testing

Run through this sequence after a new release is published to PyPI to verify the release works end-to-end.

## Prerequisites

- `ANTHROPIC_API_KEY` set (for Anthropic examples)
- `OPENAI_API_KEY` set (for LiteLLM/OpenAI examples)
- Both should be in `~/tokenjam/.env.local` and sourced before running

## Test sequence

```bash
# 1. Clean slate
tj uninstall --yes 2>/dev/null
rm -rf ~/.tj ~/.config/tj .tj

# 2. Install latest
pip3 install --upgrade tokenjam
tj --version

# 3. Onboard
# Note: daemon auto-installs by default (use --no-daemon to skip).
# Budget prompt appears — enter a value or press enter for default.
tj onboard

# 4. Stop daemon before manual testing (daemon auto-started by onboard)
tj stop

# 5. Run an example (no server — tests direct DuckDB write)
cd ~/tokenjam
source .env.local
python3 examples/single_provider/anthropic_agent.py

# 6. Verify CLI (direct DuckDB, no server)
tj status       # should show agent with cost > $0, tokens, completed status
tj traces       # should show traces with span waterfall
tj cost --since 1h   # should show cost breakdown by model (not $0.000000)
tj budget       # should show budget table with configured limits
tj alerts       # should show alert history (may be empty)
tj doctor       # exit 0 or 1 (warnings ok), no errors

# 7. Start server (tests web UI + HTTP exporter)
tj serve &
sleep 2

# 8. Run another example (tests SDK HTTP fallback)
python3 examples/single_provider/litellm_agent.py

# 9. Verify web UI
open http://127.0.0.1:7391/
# Check: Status page shows agent cards with cost, tokens
# Check: Traces page shows span waterfall
# Check: Cost page shows non-zero USD values
# Check: Sidebar has TJ jar-mark SVG + "TokenJam" wordmark, monochrome black/white
#        styling matching tokenjam.dev (no blue UI chrome — only the LLM workflow bar
#        uses brand blue as a categorical fill)
# Check: Light/Dark/System theme toggle in sidebar footer cycles through all three
#        states and persists across reloads

# 10. Verify both agents show up
tj status       # should show both agents
tj traces       # should show traces from both runs
tj cost --since 1h   # model names should be clean (gpt-4o-mini, not openai/gpt-4o-mini)

# 11. Clean up
tj stop
```

## Claude Code integration (if applicable)

```bash
# After step 3 above, also test:
tj onboard --claude-code
# Should: not prompt for daemon, write config to ~/.config/tj/config.toml,
#         write settings to ~/.claude/settings.json,
#         register MCP server with Claude Code (if claude CLI is installed)

# Verify settings written
cat ~/.claude/settings.json | python3 -m json.tool
# Should contain OTEL_LOGS_EXPORTER, OTEL_EXPORTER_OTLP_ENDPOINT, etc.

# Re-run to test secret resync (should not crash)
tj onboard --claude-code --budget 5
# Output should include: "Daemon: already running (skipped reinstall)"
# macOS should NOT show a second "Background Items Added" prompt

# Verify projects index exists
cat ~/.config/tj/projects.json   # should list current cwd

# Multi-project onboard — daemon should NOT reinstall, secret should NOT rotate
ORIG_SECRET=$(python3 -c "import json,os; print(json.load(open(os.path.expanduser('~/.claude/settings.json')))['env']['OTEL_EXPORTER_OTLP_HEADERS'])")
mkdir -p /tmp/tj-test-project-2 && cd /tmp/tj-test-project-2
git init -q
tj onboard --claude-code
NEW_SECRET=$(python3 -c "import json,os; print(json.load(open(os.path.expanduser('~/.claude/settings.json')))['env']['OTEL_EXPORTER_OTLP_HEADERS'])")
[ "$ORIG_SECRET" = "$NEW_SECRET" ] && echo "ok: secret unchanged" || echo "FAIL: secret rotated"
cd ~/tokenjam

# Verify global config fallback: CLI works from a dir with no local config
cd /tmp && tj status && cd ~/tokenjam
```

## Codex CLI integration (if applicable)

Codex onboarding is **one-time global** (Codex hardcodes `service.name=codex_exec`); all Codex traces land under the `codex_exec` agent ID.

```bash
# Onboard Codex (no prereqs — writes to global config; does NOT read server.state).
tj onboard --codex
# Should: write [otel] + [mcp_servers.tj] to ~/.codex/config.toml,
#         use ingest secret from ~/.config/tj/config.toml (creating it if absent),
#         NOT write [otel.resource] block (Codex ignores it).

# Start tj serve so the codex exec test below can ingest.
tj serve &
sleep 2

# Verify secret synced between server and Codex config.
# ~/.codex/config.toml uses TOML format `Authorization = "Bearer <secret>"`,
# so the grep must allow the spaces around `=` and the surrounding quotes.
SERVER_SECRET=$(grep ingest_secret ~/.config/tj/config.toml | sed 's/.*= "//' | tr -d '"')
CODEX_SECRET=$(grep -oE 'Bearer [^"]+' ~/.codex/config.toml | sed 's/Bearer //')
[ "$SERVER_SECRET" = "$CODEX_SECRET" ] && echo "ok: secret synced"

# Re-run is a no-op when both [otel] and [mcp_servers.tj] already present
tj onboard --codex

# If codex CLI is installed, drive a session and verify ingest
codex exec "say hello"
tj status --agent codex_exec   # should show codex_exec (NOT codex-<project>)
tj traces --agent codex_exec

# Stop the background server started in the prereq
tj stop
```

## Incident library demos

```bash
# Zero-config scenarios — no API keys, no live agents needed.
tj demo                # lists available scenarios (no flag)
tj demo retry-loop
tj demo surprise-cost
tj demo hallucination-drift
# Each writes synthetic spans; verify visible in `tj traces` and `tj alerts`.
```

## What to look for

| Step | Pass criteria |
|------|--------------|
| 2 | Version matches the release being tested |
| 3 | Config created at `.tj/config.toml`, ingest secret generated, daemon installed |
| 5 | Agent runs without errors, no DuckDB lock warnings |
| 6 | `tj status` shows non-zero cost and tokens; `tj cost` shows real USD values (not $0.000000) |
| 7 | Server starts on `:7391`, prints correct metrics URL |
| 8 | Agent runs without "Could not set lock on file" error (HTTP fallback works) |
| 9 | Web UI loads, shows data; sidebar has TJ jar SVG + "TokenJam" wordmark in monochrome; theme toggle cycles System/Light/Dark |
| 10 | CLI queries work while server is running (API fallback); model names are clean |
| 11 | `tj stop` stops the server cleanly |
