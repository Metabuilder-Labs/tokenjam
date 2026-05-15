# Pre-Release Testing

Run through this sequence to test a branch before merging and cutting a release. This uses a local editable install so changes take effect immediately without publishing to PyPI.

## Prerequisites

- `ANTHROPIC_API_KEY` set (for Anthropic examples)
- `OPENAI_API_KEY` set (for LiteLLM/OpenAI examples)
- Both should be in `~/tokenjam/.env.local` and sourced before running

## Test sequence

```bash
# 1. Clean slate
tj uninstall --yes 2>/dev/null
rm -rf ~/.tj ~/.config/tj .tj

# 2. Check out the branch to test
cd ~/tokenjam
git fetch origin
git checkout <branch-name>

# 3. Install locally (editable — uses local files, no pip publish needed)
# Uninstall any prior PyPI install of `tokenjam` first so the editable
# install isn't shadowed. Targeted — we don't `--force-reinstall` the whole
# dep tree (that bumps shared deps and breaks unrelated packages like litellm).
pip3 uninstall -y tokenjam
pip3 install -e ".[dev,mcp]"
tj --version
# Verify the installed `tj` actually imports from the repo, not site-packages
python3 -c "import tokenjam; print(tokenjam.__file__)"   # must point inside ~/tokenjam

# 4. Run automated tests first
pytest tests/unit/ tests/synthetic/ tests/agents/ tests/integration/
ruff check tokenjam/

# 5. Onboard fresh
# Note: daemon auto-installs by default (use --no-daemon to skip).
tj onboard

# 6. Stop daemon before manual testing (daemon auto-started by onboard)
tj stop

# 7. Populate test data — simulated (free, no API keys)
python3 examples/alerts_and_drift/sensitive_actions_demo.py
python3 examples/alerts_and_drift/budget_breach_demo.py
python3 examples/alerts_and_drift/drift_demo.py

# 7b. Run incident library demos (zero-config, no API keys)
tj demo                # lists available scenarios (no flag)
tj demo retry-loop
tj demo surprise-cost
tj demo hallucination-drift
# [ ] Each runs without errors
# [ ] Each writes spans to the DB (verify in step 9 with `tj traces`)

# 8. Populate test data — real API calls
source .env.local
python3 examples/single_provider/anthropic_agent.py
python3 examples/single_provider/litellm_agent.py

# 9. Verify CLI (direct DuckDB, no server)
tj status        # agents visible with cost > $0, tokens counted
tj traces        # spans from all runs
tj cost --since 1h   # real USD values, not $0.000000
tj alerts        # alerts from sensitive_actions and budget_breach demos
tj drift         # baseline built from drift_demo sessions
tj budget        # budget table with configured limits
tj doctor        # exit 0 (or 1 with warnings); no errors. Checks config, DB, secret, drift readiness

# 9b. tj optimize + tj backfill (new in 0.3.x)
# Empty-DB short-circuit before backfill:
tj uninstall --yes 2>/dev/null && rm -rf ~/.tj && tj onboard --no-daemon  # fresh DB
tj optimize       # should print "No usage data found." and exit 0

# If a Claude Code log history exists on this machine, backfill is idempotent.
ls ~/.claude/projects/ >/dev/null 2>&1 && tj backfill claude-code
# [ ] Prints "Backfilled N of N sessions" with span over multiple days
# [ ] Re-running prints "Skipped … spans already present (idempotent re-run)"
tj backfill claude-code

# Re-populate the DB for the rest of step 9:
source .env.local
python3 examples/single_provider/anthropic_agent.py

# Optimize after data is present:
tj optimize                                   # both analyzers (no [budget.*] yet — only model finding may appear)
tj optimize --budget anthropic --budget-usd 5 # force an over-budget projection
# [ ] Model-downgrade finding (if any) ends with the caveat line "Candidate-flagging heuristic, not a quality judgment."
# [ ] Budget projection shows exhaustion date when over budget
# [ ] Spend total from `tj optimize` matches `tj cost --since 30d` for the same scope
tj optimize --json                            # caveat present in JSON payload
tj optimize --json | python3 -c "import json,sys; r=json.load(sys.stdin); d=r.get('downgrade'); assert d is None or 'Candidate-flagging heuristic' in d['caveat']; print('ok: caveat enforced')"

# 10. Start server
# Note: must stop daemon first (step 6) or this will fail with "Address already in use"
tj serve &
sleep 2

# 11. Run one more example (tests SDK HTTP fallback while server holds DB lock)
python3 examples/single_provider/anthropic_agent.py

# 12. Verify web UI — Status
open http://127.0.0.1:7391/
# [ ] Multiple agent cards visible
# [ ] Each card shows cost, tokens, tool calls, duration
# [ ] "Last seen" time shown
# [ ] Cards are clickable (navigate to filtered traces)
# [ ] Sidebar: TJ jar-mark SVG + "TokenJam" wordmark, all single-color
#     (monochrome — SVG uses currentColor, so it follows text color)
# [ ] Sidebar footer: API docs, GitHub, Light/Dark/System theme toggle, and version
# [ ] Theme toggle (sidebar footer): clicking cycles System ⊙ → Light ☀ → Dark ☾,
#     preference persists across reloads, no flash of wrong theme on reload
# [ ] Light mode renders correctly: white bg, black text, status colors still legible

# 13. Verify web UI — Traces
# [ ] Agent name is first column, no Trace ID column
# [ ] Type shows friendly names (LLM Call, Tool Call, Agent Run)
# [ ] Click chevron (→) visible in last column
# [ ] Click a trace — waterfall renders with correct nesting
# [ ] Span bars have hover glow effect
# [ ] Fast tool calls (0ms) still have visible bars
# [ ] "Click a span for details" hint appears
# [ ] Click a span — detail panel shows provider, model, tokens, cost
# [ ] Friendly span name as heading, raw name in dim text below

# 14. Verify web UI — Cost
# [ ] Summary row shows total cost, input tokens, output tokens
# [ ] Group-by selector works (day / agent / model / tool)
# [ ] Redundant columns hidden based on group-by selection
# [ ] Costs show real USD values (not $0.000000)

# 15. Verify web UI — Alerts
# [ ] Alerts table populated from sensitive_actions and budget_breach demos
# [ ] Friendly type names (Sensitive Action, Daily Budget, etc.)
# [ ] Severity badges with correct colors (critical=red, warning=yellow, info=blue)
# [ ] ▸/▾ expand toggle on rows, click shows detail JSON

# 16. Verify web UI — Drift
# [ ] At least one agent shows baseline data
# [ ] Metric table shows baseline mean ± stddev, latest value, Z-score
# [ ] Pass badges are green, drift badges are red
# [ ] Threshold shown in header (2.0σ)

# 17. Verify CLI works while server is running (API fallback)
tj status
tj traces
tj cost --since 1h

# 18. Clean up
tj stop
```

## Claude Code integration (if applicable)

```bash
# Test after step 5:
tj onboard --claude-code
# Should: write config to ~/.config/tj/config.toml (global, not project-local),
#         write settings to ~/.claude/settings.json,
#         register MCP server if claude CLI available,
#         auto-install daemon
#         create ~/.config/tj/projects.json with current cwd

# Verify global config exists.
# Note: a project-local `.tj/config.toml` likely already exists from step 5's
# `tj onboard` — `--claude-code` does NOT delete or overwrite it; it only
# writes to the global config. The "no project-local" check belongs in the
# multi-project block below where cwd is genuinely fresh.
test -f ~/.config/tj/config.toml && echo "ok: global config"

# Verify projects.json tracks the cwd
cat ~/.config/tj/projects.json   # should contain current working directory

# Verify no crash on re-run (secret resync, same project)
tj onboard --claude-code --budget 5
# Output should include: "Daemon: already running (skipped reinstall)"
# macOS should NOT show another "Background Items Added" notification.

# Verify multi-project onboard (the 490ad8e fix)
mkdir -p /tmp/tj-test-project-2 && cd /tmp/tj-test-project-2
git init -q
tj onboard --claude-code
# [ ] Output shows "Daemon: already running (skipped reinstall)" — daemon NOT reinstalled
# [ ] No new "Background Items Added" prompt on macOS
# [ ] ~/.config/tj/projects.json now lists BOTH project paths
# [ ] ingest_secret in ~/.claude/settings.json unchanged from first onboard
#     (so the original project's auth still works)
cat ~/.config/tj/projects.json
# Confirm --claude-code did NOT create a project-local config in this fresh
# directory (this dir has no preexisting .tj/, so the check is meaningful here).
test ! -f .tj/config.toml && echo "ok: --claude-code did not create project-local config"
cd ~/tokenjam

# Verify global config fallback: CLI works from a directory with no local config
cd /tmp && tj status   # should resolve to global config, not error out
cd ~/tokenjam

# Verify --force does reinstall the daemon
tj onboard --claude-code --force
# Output should include: "Daemon: installing..."

# Verify MCP server starts
tj mcp --help
```

## Codex CLI integration (if applicable)

Codex hardcodes `service.name=codex_exec` in its binary, so this is a **one-time global** setup, not per-project. All Codex traces land under the `codex_exec` agent ID regardless of which project directory you onboard from.

```bash
# Onboard Codex (no prereqs — onboarding writes to the global config and syncs
# the secret into ~/.codex/config.toml; it does NOT read server.state).
tj onboard --codex
# Should: write [otel] block + [mcp_servers.tj] to ~/.codex/config.toml,
#         use ingest secret from ~/.config/tj/config.toml (creating it if absent),
#         NOT write [otel.resource] (Codex ignores it — would cause stale agent IDs)

# Start tj serve so the codex exec test below can ingest. This also writes
# ~/.local/share/tj/server.state for diagnostics (informational only).
tj serve &
sleep 2
test -f ~/.local/share/tj/server.state && echo "ok: server.state exists (informational)"

# Verify Codex config
cat ~/.codex/config.toml
# [ ] Contains [otel] block with otlp_endpoint, otlp_headers (Authorization=Bearer ...)
# [ ] Contains [mcp_servers.tj] block
# [ ] Does NOT contain [otel.resource] block

# Verify secret matches the running server.
# Note: ~/.codex/config.toml uses TOML format `Authorization = "Bearer <secret>"`
# (with spaces around `=` and surrounding quotes), so the grep must allow for
# that — a literal `Authorization=Bearer ` pattern silently misses every match.
SERVER_SECRET=$(grep ingest_secret ~/.config/tj/config.toml | sed 's/.*= "//' | tr -d '"')
CODEX_SECRET=$(grep -oE 'Bearer [^"]+' ~/.codex/config.toml | sed 's/Bearer //')
[ "$SERVER_SECRET" = "$CODEX_SECRET" ] && echo "ok: secret synced" || echo "FAIL: secret mismatch"

# Verify skip-on-rerun (must have BOTH [otel] and [mcp_servers.tj])
tj onboard --codex   # should print "already configured" / no-op

# Stop the foreground `tj serve` from the prereq before any --force flow,
# so the daemon reinstall doesn't collide with the running server on port 7391.
tj stop

# Verify cross-sync: re-onboarding Claude Code updates Codex config too.
# `--force` reinstalls the launchd daemon which auto-starts a new `tj serve`,
# so do NOT manually run `tj serve &` after this — that would collide on
# port 7391. The just-reinstalled daemon is already serving.
tj onboard --claude-code --force
# After: ingest secret in ~/.claude/settings.json, ~/.codex/config.toml,
#        and ~/.config/tj/config.toml should all match.
sleep 2  # give the auto-started daemon a moment to bind the port

# Drive a Codex session (if codex CLI installed) and verify ingestion
codex exec "say hello"   # or any short codex command
tj status --agent codex_exec   # should show codex_exec agent with spans/cost
tj traces --agent codex_exec
# [ ] Spans land under agent_id=codex_exec (NOT codex-<project-name>)
# [ ] /v1/logs endpoint accepted the OTLP log records (no 400 in `tj serve` output)

tj stop
```

## Quick test (skip web UI — just verify core)

For smaller changes that don't touch the UI:

```bash
tj uninstall --yes 2>/dev/null
rm -rf ~/.tj ~/.config/tj .tj
cd ~/tokenjam
git checkout <branch-name>
pip3 install -e ".[dev,mcp]"
tj onboard --no-daemon
source .env.local
python3 examples/single_provider/anthropic_agent.py
tj status && tj traces && tj cost --since 1h && tj optimize
# Verify: cost > $0, tokens counted, traces visible, optimize renders both analyzers
pytest tests/unit/ tests/synthetic/ tests/agents/ tests/integration/
ruff check tokenjam/
```

## What to look for

| Step | Pass criteria |
|------|--------------|
| 3 | Installs without errors, version shows expected value |
| 4 | All tests pass, no lint errors |
| 5 | Config created, ingest secret generated, daemon installed |
| 7 | Simulated demos run without errors, no API keys needed |
| 8 | Real examples run, no DuckDB lock errors |
| 9 | CLI shows data: cost > $0, tokens counted, traces visible, alerts fired, drift baseline built |
| 9b | `tj backfill claude-code` is idempotent (skip count > 0 on re-run); `tj optimize` JSON output includes a `caveat` field on any downgrade finding; `tj optimize` and `tj cost` agree on window totals |
| 10 | Server starts on `:7391`, prints correct metrics URL |
| 11 | No "Could not set lock on file" error — HTTP fallback works |
| 12-16 | Web UI views render correctly with real data |
| 17 | CLI queries work while server is running (no lock errors) |

## Switching back to main after testing

```bash
tj stop
tj uninstall --yes 2>/dev/null
git checkout main
pip3 install -e ".[dev,mcp]"
```
