# Pre-Release Testing

Run through this sequence to test a branch before merging and cutting a release. Uses a local editable install so changes take effect immediately without publishing to PyPI.

## Prerequisites

- `ANTHROPIC_API_KEY` and `OPENAI_API_KEY` set (used by example agents)
- Both in `~/tokenjam/.env.local`, sourced before running
- A clean shell (`tj uninstall --yes && rm -rf ~/.tj ~/.config/tj .tj` if anything from a prior test lingers)
- **No lingering daemon from a prior install.** Run `tj stop` early — if a previous `tj onboard` installed the launchd / systemd unit, the daemon may still be live with stale config. Verify with `launchctl list | grep tokenjam` (macOS) or `systemctl --user is-active tokenjam` (Linux).

## 1. Install and verify the build

```bash
cd ~/tokenjam
git fetch origin
git checkout <branch-name>

# Uninstall any prior PyPI install so the editable install isn't shadowed.
# Targeted (don't --force-reinstall the whole tree — that breaks shared deps
# in unrelated packages like litellm).
pip3 uninstall -y tokenjam
pip3 install -e ".[dev,mcp]"

tj --version
# Confirm the installed `tj` imports from the repo, not site-packages.
python3 -c "import tokenjam; print(tokenjam.__file__)"   # must point inside ~/tokenjam
```

**Pass criteria:** version prints, import path is the repo, no install errors.

## 2. Automated tests + lint

```bash
pytest tests/unit/ tests/synthetic/ tests/agents/ tests/integration/
ruff check tokenjam/
```

**Pass criteria:** all tests green, ruff clean (no errors). If ruff reports anything, only proceed if the errors are pre-existing on `main` — confirm by running `ruff check` on `main` first.

## 3. Onboard

```bash
tj onboard --no-daemon   # daemon auto-starts otherwise; stop afterward
```

The bare `tj onboard` doesn't prompt for plan tier (plan tier is per-provider — it's set by the integration-specific flows). It only asks for a daily-budget number.

Plan-tier prompting happens in `--claude-code` and `--codex`:

```bash
# Use whichever plan you actually have on the test machine.
# The examples below use max_5x; substitute max_20x / pro / plus / etc.
# as appropriate. The format of the expected output is what matters,
# not the specific plan label.
tj onboard --claude-code --plan max_5x --no-daemon
# [ ] prompts for plan tier if --plan not provided
# [ ] writes plan = "max_5x" under [budget.anthropic] in ~/.config/tj/config.toml

# Confirm the plan landed.
grep -A3 "^\[budget.anthropic\]" ~/.config/tj/config.toml
# [ ] plan = "max_5x" (or whichever you picked)
# [ ] does NOT auto-write usd = 200 (that default is gone in v0.3.x)

# --reconfigure on the integration paths actually re-prompts.
tj onboard --claude-code --reconfigure --plan api
# [ ] config updated; plan field flipped

# Bare `tj onboard --reconfigure` is an error now (#68 §1) — points at
# the integration-specific flows. Verify the explicit error renders.
tj onboard --reconfigure --plan max_5x; echo "expected exit 1, got $?"
# [ ] prints "--reconfigure has no effect without --claude-code or --codex"
# [ ] exit code 1
```

## 4. Populate test data

```bash
# 4a. Zero-config demos (no API keys)
python3 examples/alerts_and_drift/sensitive_actions_demo.py
python3 examples/alerts_and_drift/budget_breach_demo.py
python3 examples/alerts_and_drift/drift_demo.py
tj demo retry-loop
tj demo surprise-cost
tj demo hallucination-drift

# 4b. Real API calls
source .env.local
python3 examples/single_provider/anthropic_agent.py
python3 examples/single_provider/litellm_agent.py
```

**Pass criteria:** every example runs without errors. `tj traces` shows spans from all of them in step 5.

## 5. Core CLI smoke (no server)

```bash
tj status     # agents visible, cost > $0, tokens counted
tj traces     # spans from all runs
tj cost --since 1h
tj alerts     # sensitive_actions + budget_breach alerts present
tj drift      # baseline built from drift_demo
tj budget     # configured limits
tj doctor     # exit 0 (or 1 with warnings); no errors
```

## 6. Cost-optimization analyzers (the four products)

### 6a. Downsize

```bash
tj optimize downsize
# [ ] If candidates found: caveat "Candidate-flagging heuristic, not a quality judgment." is present
# [ ] If no candidates: clean "No candidates flagged" message — not a crash
tj optimize downsize --json | python3 -c \
  "import json,sys;r=json.load(sys.stdin);d=r.get('downgrade');assert d is None or 'Candidate-flagging heuristic' in d['caveat'];print('ok: caveat enforced')"
```

### 6b. Cache

```bash
# cache works without content capture
tj optimize cache
# [ ] Per (provider, model) rows. Anthropic shows numerically-accurate ratios.
# [ ] OpenAI / Gemini rows (if present) carry the best-effort caveat.
# [ ] Rows for Bedrock / LiteLLM / Cohere (if present) show unsupported, not flagged.

# cache-recommend needs capture.prompts. Without it, the analyzer returns a hint.
tj optimize cache-recommend
# [ ] Without capture.prompts: surfaces the "enable capture.prompts" hint, doesn't crash.
# [ ] With capture.prompts and ≥3 calls sharing a long prefix: surfaces breakpoint candidates.
```

To exercise the content-needed branch, set `capture.prompts = true` in `.tj/config.toml`, re-run an example, then re-run `cache-recommend`.

### 6c. Script

```bash
tj optimize script
# [ ] If ≥20 sessions match a single (tool_name, arg_shape) signature: cluster surfaces with
#     "review carefully" caveat. With v1's conservative thresholds, most fresh test DBs see no candidates.
# [ ] No crash; "no clusters found" message is acceptable.
```

### 6d. Trim

Trim requires the optional `tokenjam[bloat]` extra (LLMLingua-2 + torch + transformers, ~2GB).

```bash
# Without the extra installed: self-registers, errors gracefully with install hint.
tj optimize trim
# [ ] Output points the user at: pip install "tokenjam[bloat]"

# Install the extra and re-run — only do this if you actually want to test Trim end-to-end
# (the 2GB download is real). Skip this on quick passes.
pip3 install -e ".[dev,mcp,bloat]"
# Enable capture.prompts in .tj/config.toml, re-run examples to populate content, then:
tj optimize trim
# [ ] First run downloads the ~110MB BERT model under ~/.cache/tokenjam/models/.
# [ ] Subsequent runs are offline.

# HTML report renderer
tj report --trim --no-open
# [ ] Writes to ~/.cache/tokenjam/reports/trim-<timestamp>.html
# [ ] HTML contains a caveat block + per-prompt sections
```

### 6e. Run all analyzers + verify defaults

```bash
tj optimize   # runs every registered analyzer in ANALYZER_ORDER
tj optimize --budget anthropic --budget-usd 5   # force an over-budget finding
# [ ] Budget projection shows exhaustion date when over budget
# [ ] Spend total reconciles with tj cost --since 30d for the same scope
tj optimize --json | python3 -c \
  "import json,sys;r=json.load(sys.stdin);assert 'plan' in r and 'pricing_mode' in r;print('ok: plan-tier metadata present')"
```

### 6f. TokenMaxx (v0.3.4 — six-tier ladder)

```bash
tj tokenmaxx
# [ ] Bordered "TokenJam TokenMaxxing Report" panel renders
# [ ] On api plan: shows absolute spend; no multiplier line
# [ ] `tj optimize` reference in the action line renders in green bold
# [ ] Share-line is teal, mentions @tokenjamdev (NOT #tokenmaxx)

# Tier label must be one of the v0.3.4 six.
tj tokenmaxx --json | python3 -c \
  "import json,sys;d=json.load(sys.stdin);ok={'TokenSipper','TokenModerator','TokenMaxxer','TokenSuperMaxxer','TokenMegaMaxxer','TokenGigaMaxxer'};assert d['tier'] in ok,d['tier'];print('ok:',d['tier'])"

# Force a different tier by exercising a subscription plan (use whatever
# plan matches your test data — heavy usage on a Pro plan will land you
# higher up the ladder than the same usage on a Max-20x plan).
tj onboard --claude-code --reconfigure --plan max_5x
tj tokenmaxx
# [ ] Output now shows "That's N× your Max 5x plan cost ($100/mo flat)."
# [ ] Tier name follows the v0.3.4 ladder (Sipper / Moderator / Maxxer /
#     SuperMaxxer / MegaMaxxer / GigaMaxxer)
# [ ] At thresholds: 1× / 4× / 10× / 20× / 50× crosses tier boundaries

# Reset to api before continuing
tj onboard --claude-code --reconfigure --plan api
```

## 7. Plan-tier-aware rendering

Reconfigure to a subscription plan and re-run `tj optimize` — output should reframe.

```bash
# Use the plan you actually have on this machine. Expected output shape:
#   pro     → "Pro plan, $20/mo flat"
#   max_5x  → "Max 5x plan, $100/mo flat"
#   max_20x → "Max 20x plan, $200/mo flat"
#   plus    → "ChatGPT Plus, $20/mo flat"
tj onboard --claude-code --reconfigure --plan max_5x
tj optimize
# [ ] Header reads "(<Plan label>, $<fee>/mo flat)" + "Implied API value: $X — about Y× your plan cost"
# [ ] NO line that uses the word "spend" against a dollar figure
# [ ] Downgrade body (if any) uses token-share framing, not "$X/mo savings"

tj onboard --claude-code --reconfigure --plan api
tj optimize
# [ ] Back to "$X spend (last 30d)..." header and dollar-denominated downgrade savings

# Unknown plan: drop the plan field, restart, verify suppression.
# (Easiest way: manually remove plan = "..." from [budget.anthropic] in config.)
tj status   # [ ] one-line note: "N session(s) have unknown plan tier..."
tj optimize # [ ] header note + dollar figures suppressed for unknown-tier sessions
```

## 8. Backfill adapters

```bash
# 8a. Claude Code (existing source — covered briefly)
ls ~/.claude/projects/ >/dev/null 2>&1 && tj backfill claude-code
# [ ] "Backfilled N of N sessions" message
# [ ] Re-running prints "Skipped … spans already present" (idempotent)

# 8b. Langfuse — file mode against the committed fixture
tj backfill langfuse --source-file tests/fixtures/langfuse_real_response.json
# [ ] "Read N observations, wrote N new spans" — first run
# Re-run:
tj backfill langfuse --source-file tests/fixtures/langfuse_real_response.json
# [ ] "skipped N already present" — idempotent

# 8c. Helicone — same shape, against its fixture
tj backfill helicone --source-file tests/fixtures/helicone_real_response.json
tj backfill helicone --source-file tests/fixtures/helicone_real_response.json   # idempotent
# [ ] Same "wrote N / skipped N" pattern

# 8d. Raw OTLP
tj backfill otlp --source-file tests/fixtures/otlp_sample.json
tj backfill otlp --source-file tests/fixtures/otlp_sample.json   # idempotent
# [ ] spans_seen / spans_written / spans_skipped / spans_rejected counters
# [ ] spans_rejected = 0 on the committed fixture

# 8e. Cross-source coherence
tj cost --since 30d
# [ ] Backfilled spans show up alongside native-captured spans, indistinguishable
```

## 9. Period comparison

```bash
tj cost --since 7d --compare previous
# [ ] Current + Previous summary lines
# [ ] Cost delta + token delta with ▲ / ▼ indicators
# [ ] Top per-agent / per-model shifts (if any)
tj cost --since 7d --compare last-month   # 30d prior window
tj cost --since 7d --compare 2026-04-01:2026-04-30   # explicit range

tj cost --compare previous --json | python3 -c \
  "import json,sys;d=json.load(sys.stdin);assert {'current','previous','cost_delta_usd','tokens_delta'} <= d.keys();print('ok')"

tj optimize --compare previous
# [ ] Optimize report renders normally, with a "Window comparison" section appended at the end
```

## 10. Config export

```bash
tj optimize --export-config claude-code
# [ ] Writes to ~/.config/tokenjam/exports/claude-code-<date>.json
# [ ] Prints next-step instructions ("Open the file and copy the routing block...")
# [ ] No file outside ~/.config/tokenjam/exports/ is touched
# [ ] No --apply flag exists (verify: tj optimize --export-config claude-code --apply errors)

cat ~/.config/tokenjam/exports/claude-code-*.json
# [ ] Contains a "tokenjam.routing_recommendations" block
# [ ] Honest-framing caveat appears as JSON // comments
# [ ] On api plan: rules carry "estimated_savings_usd_month"
# [ ] On subscription plan: rules carry "estimated_tokens_freed" instead
```

## 11. Policy list (read-only preview)

```bash
tj policy list
# [ ] Table with POLICY / SETTING / SOURCE columns
# [ ] Rows for [alerts], [defaults.budget] (if set), [budget.<provider>], per-agent overrides, [capture]
# [ ] Each row's SOURCE points at the TOML section it was read from
# [ ] Footer note: "read-only preview" + "add|edit|apply lands next sprint"

tj policy list --json | python3 -m json.tool | head
# [ ] policies array + note field

# Confirm write/edit subcommands are intentionally absent
tj policy add foo 2>&1 | grep -E "No such command|Usage" >/dev/null && echo "ok: add absent"
```

## 12. Start the server + verify HTTP fallback

```bash
tj stop   # ensure no daemon competing for the port
tj serve &
sleep 2

# Run an example while the server holds the DB write lock — exercises SDK HTTP fallback.
python3 examples/single_provider/anthropic_agent.py

# CLI commands should work via API fallback while server is up
tj status
tj traces
tj cost --since 1h
```

**Pass criteria:** no "Could not set lock on file" errors. CLI works during `tj serve`.

## 13. Web UI smoke

```bash
open http://127.0.0.1:7391/
```

Spot-check (don't repeat every theme/typography detail — those are one-time UI work):

- [ ] Status page: agent cards render with cost / tokens / tool calls
- [ ] Traces page: waterfall renders, span detail panel shows provider / model / tokens / cost
- [ ] Cost page: real USD values (not $0.000000); group-by selector works
- [ ] Alerts page: rows populated from step 4 demos, severity badges colored
- [ ] Drift page: baseline + Z-score table
- [ ] Sidebar: theme toggle cycles System / Light / Dark and persists

If any UI element regresses *visibly broken* relative to main, dig in. Otherwise move on.

### Offline-UI verification (v0.3.4 — issue #87 / PR #88)

The dashboard must work fully offline. The "local-first, no data egress" pitch breaks the moment a render-time external load happens.

Open Chrome DevTools (or your browser's equivalent) → **Network tab** → reload `http://127.0.0.1:7391/`.

- [ ] **Zero failed requests** to `fonts.googleapis.com`, `fonts.gstatic.com`, `esm.sh`, or `tokenjam.dev`
- [ ] Dashboard interactivity works (sidebar nav, tab switches) — proves the vendored Preact / htm under `/ui/vendor/` is serving correctly
- [ ] Favicon renders (data: URL, no external fetch)

The `tests/unit/test_ui_offline.py` regression test pins this contract in CI, but a manual eyeball catches anything the regex assertions miss (e.g. background-image URLs in CSS, inline `fetch()` calls in JS).

### Cache cost-correctness (v0.3.4 — PRs #90 + #92)

```bash
# Spans table has cache_write_tokens (migration 5).
duckdb ~/.tj/telemetry.duckdb "PRAGMA table_info(spans)" 2>/dev/null \
  | grep cache_write_tokens \
  && echo "ok: cache_write_tokens column present"

# Any Anthropic cache-hit span (cache_read>0, no input/output) is now
# costed. Pre-0.3.4 these were dropped as no-ops and silently $0.
duckdb ~/.tj/telemetry.duckdb "
  SELECT COUNT(*) AS hits, MIN(cost_usd) AS min_cost
  FROM spans
  WHERE cache_tokens > 0
    AND (input_tokens = 0 OR input_tokens IS NULL)
    AND (output_tokens = 0 OR output_tokens IS NULL)
" 2>/dev/null
# [ ] If hits > 0: min_cost > 0
# [ ] If hits = 0: this run didn't trigger a pure cache-only span — fine
```

If `duckdb` CLI isn't installed, skip — the unit + synthetic tests covering these paths run in CI.

## 14. Clean up

```bash
tj stop
```

---

## Claude Code integration (if the change touches onboard / settings.json / daemon)

```bash
tj onboard --claude-code --plan max_5x   # non-interactive
# [ ] writes ~/.config/tj/config.toml (global, not project-local)
# [ ] writes ~/.claude/settings.json with OTEL_EXPORTER_OTLP_ENDPOINT and Bearer header
# [ ] registers MCP server if `claude` CLI on PATH
# [ ] auto-installs daemon
# [ ] adds cwd to ~/.config/tj/projects.json

# Re-run is a quiet no-op (no duplicate "Background Items Added" notification on macOS)
tj onboard --claude-code --plan max_5x

# Multi-project: secret must NOT rotate on second project
mkdir -p /tmp/tj-test-project-2 && cd /tmp/tj-test-project-2 && git init -q
tj onboard --claude-code --plan max_5x
# [ ] "Daemon: already running (skipped reinstall)"
# [ ] ~/.config/tj/projects.json lists BOTH paths
# [ ] ingest_secret in ~/.claude/settings.json unchanged from the first onboard
test ! -f .tj/config.toml && echo "ok: --claude-code did not create project-local config"
cd ~/tokenjam

# --force does reinstall the daemon
tj onboard --claude-code --plan max_5x --force
# [ ] "Daemon: installing..."

tj mcp --help   # MCP server CLI exists
```

## Codex CLI integration (if the change touches Codex onboard / /v1/logs)

```bash
tj onboard --codex --plan plus   # one-time global, not per-project
cat ~/.codex/config.toml
# [ ] Contains [otel] block with otlp_endpoint + Authorization Bearer header
# [ ] Contains [mcp_servers.tj] block
# [ ] Does NOT contain [otel.resource] (Codex would ignore it anyway and cause stale agent IDs)

# Secret synced between Codex config and global tj config
SERVER_SECRET=$(grep ingest_secret ~/.config/tj/config.toml | sed 's/.*= "//' | tr -d '"')
CODEX_SECRET=$(grep -oE 'Bearer [^"]+' ~/.codex/config.toml | sed 's/Bearer //')
[ "$SERVER_SECRET" = "$CODEX_SECRET" ] && echo "ok: secret synced" || echo "FAIL: secret mismatch"

# Re-running is a no-op
tj onboard --codex --plan plus   # "already configured"

# Drive a Codex session (if codex CLI installed)
tj serve & sleep 2
codex exec "say hello"
tj traces --agent codex_exec
# [ ] Spans land under agent_id=codex_exec (NOT codex-<project-name>)
tj stop
```

---

## Quick test (skip web UI — verify core + analyzers only)

For smaller changes that don't touch the UI:

```bash
tj uninstall --yes 2>/dev/null
rm -rf ~/.tj ~/.config/tj .tj
cd ~/tokenjam
git checkout <branch-name>
pip3 install -e ".[dev,mcp]"
pytest tests/unit/ tests/synthetic/ tests/agents/ tests/integration/
ruff check tokenjam/

tj onboard --no-daemon --plan api
source .env.local
python3 examples/single_provider/anthropic_agent.py

tj status && tj traces && tj cost --since 1h
tj optimize           # all analyzers
tj optimize downsize
tj optimize cache
tj tokenmaxx          # six-tier ladder
tj backfill langfuse --source-file tests/fixtures/langfuse_real_response.json
tj backfill helicone --source-file tests/fixtures/helicone_real_response.json
tj backfill otlp --source-file tests/fixtures/otlp_sample.json
tj policy list
```

## Switching back to main after testing

```bash
tj stop
tj uninstall --yes 2>/dev/null
git checkout main
pip3 install -e ".[dev,mcp]"
```
