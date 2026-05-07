# Manual Release Testing

Run through this sequence after a new release is published to PyPI to verify the release works end-to-end.

## Prerequisites

- `ANTHROPIC_API_KEY` set (for Anthropic examples)
- `OPENAI_API_KEY` set (for LiteLLM/OpenAI examples)
- Both should be in `~/tokenjam/.env.local` and sourced before running

## Test sequence

```bash
# 1. Clean slate
ocw uninstall --yes 2>/dev/null
rm -rf ~/.ocw ~/.config/ocw .ocw

# 2. Install latest
pip3 install --upgrade tokenjam
ocw --version

# 3. Onboard
# Note: daemon auto-installs by default (use --no-daemon to skip).
# Budget prompt appears — enter a value or press enter for default.
ocw onboard

# 4. Stop daemon before manual testing (daemon auto-started by onboard)
ocw stop

# 5. Run an example (no server — tests direct DuckDB write)
cd ~/tokenjam
source .env.local
python3 examples/single_provider/anthropic_agent.py

# 6. Verify CLI (direct DuckDB, no server)
ocw status       # should show agent with cost > $0, tokens, completed status
ocw traces       # should show traces with span waterfall
ocw cost --since 1h   # should show cost breakdown by model (not $0.000000)
ocw budget       # should show budget table with configured limits
ocw alerts       # should show alert history (may be empty)
ocw doctor       # exit 0 or 1 (warnings ok), no errors

# 7. Start server (tests web UI + HTTP exporter)
ocw serve &
sleep 2

# 8. Run another example (tests SDK HTTP fallback)
python3 examples/single_provider/litellm_agent.py

# 9. Verify web UI
open http://127.0.0.1:7391/
# Check: Status page shows agent cards with cost, tokens
# Check: Traces page shows span waterfall
# Check: Cost page shows non-zero USD values
# Check: Sidebar has SVG logo + opencla.watch styling (deep navy + blue accent)

# 10. Verify both agents show up
ocw status       # should show both agents
ocw traces       # should show traces from both runs
ocw cost --since 1h   # model names should be clean (gpt-4o-mini, not openai/gpt-4o-mini)

# 11. Clean up
ocw stop
```

## Claude Code integration (if applicable)

```bash
# After step 3 above, also test:
ocw onboard --claude-code
# Should: not prompt for daemon, write config to ~/.config/ocw/config.toml,
#         write settings to ~/.claude/settings.json,
#         register MCP server with Claude Code (if claude CLI is installed)

# Verify settings written
cat ~/.claude/settings.json | python3 -m json.tool
# Should contain OTEL_LOGS_EXPORTER, OTEL_EXPORTER_OTLP_ENDPOINT, etc.

# Re-run to test secret resync (should not crash)
ocw onboard --claude-code --budget 5
# Output should include: "Daemon: already running (skipped reinstall)"
# macOS should NOT show a second "Background Items Added" prompt

# Verify projects index exists
cat ~/.config/ocw/projects.json   # should list current cwd

# Multi-project onboard — daemon should NOT reinstall, secret should NOT rotate
ORIG_SECRET=$(python3 -c "import json,os; print(json.load(open(os.path.expanduser('~/.claude/settings.json')))['env']['OTEL_EXPORTER_OTLP_HEADERS'])")
mkdir -p /tmp/ocw-test-project-2 && cd /tmp/ocw-test-project-2
git init -q
ocw onboard --claude-code
NEW_SECRET=$(python3 -c "import json,os; print(json.load(open(os.path.expanduser('~/.claude/settings.json')))['env']['OTEL_EXPORTER_OTLP_HEADERS'])")
[ "$ORIG_SECRET" = "$NEW_SECRET" ] && echo "ok: secret unchanged" || echo "FAIL: secret rotated"
cd ~/tokenjam

# Verify global config fallback: CLI works from a dir with no local config
cd /tmp && tj status && cd ~/tokenjam
```

## Codex CLI integration (if applicable)

Codex onboarding is **one-time global** (Codex hardcodes `service.name=codex_exec`); all Codex traces land under the `codex_exec` agent ID.

```bash
# Prereq: tj serve running — onboard reads ~/.local/share/ocw/server.state
# to find the running server's config and sync the ingest secret.
ocw serve &
sleep 2
test -f ~/.local/share/ocw/server.state && echo "ok: server.state exists"

ocw onboard --codex
# Should: write [otel] + [mcp_servers.ocw] to ~/.codex/config.toml,
#         use ingest secret from running server,
#         NOT write [otel.resource] block (Codex ignores it).

# Verify secret synced between server and Codex config.
# ~/.codex/config.toml uses TOML format `Authorization = "Bearer <secret>"`,
# so the grep must allow the spaces around `=` and the surrounding quotes.
SERVER_SECRET=$(grep ingest_secret ~/.config/ocw/config.toml | sed 's/.*= "//' | tr -d '"')
CODEX_SECRET=$(grep -oE 'Bearer [^"]+' ~/.codex/config.toml | sed 's/Bearer //')
[ "$SERVER_SECRET" = "$CODEX_SECRET" ] && echo "ok: secret synced"

# Re-run is a no-op when both [otel] and [mcp_servers.ocw] already present
ocw onboard --codex

# If codex CLI is installed, drive a session and verify ingest
codex exec "say hello"
ocw status --agent codex_exec   # should show codex_exec (NOT codex-<project>)
ocw traces --agent codex_exec

# Stop the background server started in the prereq
ocw stop
```

## Incident library demos

```bash
# Zero-config scenarios — no API keys, no live agents needed.
ocw demo                # lists available scenarios (no flag)
ocw demo retry-loop
ocw demo surprise-cost
ocw demo hallucination-drift
# Each writes synthetic spans; verify visible in `ocw traces` and `ocw alerts`.
```

## What to look for

| Step | Pass criteria |
|------|--------------|
| 2 | Version matches the release being tested |
| 3 | Config created at `.ocw/config.toml`, ingest secret generated, daemon installed |
| 5 | Agent runs without errors, no DuckDB lock warnings |
| 6 | `ocw status` shows non-zero cost and tokens; `ocw cost` shows real USD values (not $0.000000) |
| 7 | Server starts on `:7391`, prints correct metrics URL |
| 8 | Agent runs without "Could not set lock on file" error (HTTP fallback works) |
| 9 | Web UI loads, shows data, sidebar has SVG logo |
| 10 | CLI queries work while server is running (API fallback); model names are clean |
| 11 | `ocw stop` stops the server cleanly |
