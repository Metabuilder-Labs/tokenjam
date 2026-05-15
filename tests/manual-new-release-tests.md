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

# 11. tj optimize + tj backfill smoke check (new in 0.3.x)
# Stop the server so the DB is unlocked (optimize uses a read-only fallback
# but backfill needs the write lock).
tj stop

# If this machine has Claude Code history, backfill is idempotent and surfaces
# real spend numbers. Otherwise this step is a no-op.
ls ~/.claude/projects/ >/dev/null 2>&1 && tj backfill claude-code

tj optimize                                   # both analyzers
tj optimize --budget anthropic --budget-usd 5 # force an over-budget finding to see the renderer
tj optimize --json | python3 -c "import json,sys; r=json.load(sys.stdin); d=r.get('downgrade'); assert d is None or 'Candidate-flagging heuristic' in d['caveat']; print('ok: caveat enforced')"
# [ ] Empty-DB case prints "No usage data found." (run after `tj uninstall` if curious)
# [ ] Over-budget projection shows exhaustion date
# [ ] Spend totals reconcile between `tj optimize` and `tj cost --since 30d`

# 12. Clean up
```

## Claude Code integration (if applicable)

Smoke check only — multi-project, secret-rotation, and global-config-fallback details live in `manual-pre-release-testing.md` and are exercised before release.

```bash
tj onboard --claude-code
# Verify expected files written
cat ~/.claude/settings.json | python3 -m json.tool | grep -E "OTEL_LOGS_EXPORTER|OTEL_EXPORTER_OTLP_ENDPOINT"
cat ~/.config/tj/projects.json   # should list current cwd

# Re-run should be a quiet no-op (no second "Background Items Added" prompt on macOS)
tj onboard --claude-code --budget 5

# Backfill ran automatically during onboard — verify history is present
tj cost --since 30d --agent claude-code-tokenjam
```

## Codex CLI integration (if applicable)

Smoke check only — the full multi-step Codex test (cross-sync, no-op re-runs, secret rotation) lives in `manual-pre-release-testing.md` and doesn't need to repeat for a published-release verification.

```bash
tj onboard --codex
tj serve &
sleep 2
# Spot-check the secret was synced
SERVER_SECRET=$(grep ingest_secret ~/.config/tj/config.toml | sed 's/.*= "//' | tr -d '"')
CODEX_SECRET=$(grep -oE 'Bearer [^"]+' ~/.codex/config.toml | sed 's/Bearer //')
[ "$SERVER_SECRET" = "$CODEX_SECRET" ] && echo "ok: secret synced"

# If codex CLI installed, drive a session and confirm ingest
codex exec "say hello" 2>/dev/null && tj traces --agent codex_exec
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
| 11 | `tj backfill claude-code` is idempotent on re-run; `tj optimize` prints both analyzers (or a friendly empty-DB message); JSON output includes the caveat string; spend totals reconcile with `tj cost` |
| 12 | `tj stop` stops the server cleanly |
