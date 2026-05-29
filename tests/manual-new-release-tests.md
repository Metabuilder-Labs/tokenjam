# Post-Release Smoke Test

Run this after a new release publishes to PyPI to verify it works end-to-end. This is intentionally lighter than `manual-pre-release-testing.md` — the multi-project / secret-rotation / theme-toggle detail lives there and was exercised on the branch before merge. Here we're confirming the published artifact actually installs and the core surfaces are alive.

## Prerequisites

- `ANTHROPIC_API_KEY` and `OPENAI_API_KEY` in `~/tokenjam/.env.local`
- Sourced before running

## 1. Install the published release

```bash
tj uninstall --yes 2>/dev/null
rm -rf ~/.tj ~/.config/tj .tj

pip3 install --upgrade tokenjam
tj --version
```

**Pass criteria:** version matches the release being tested.

## 2. Onboard

```bash
tj onboard --no-daemon   # daemon auto-installs by default; --no-daemon for this smoke pass
```

The prompt should ask for plan tier (api / pro / max_5x / max_20x). Pick `api` so dollar rendering kicks in for the rest of the smoke.

```bash
grep "^plan = " ~/.config/tj/config.toml || grep "^plan = " .tj/config.toml
# [ ] plan field is set; no auto-written usd = 200 in [budget.anthropic]
```

## 3. Drive an example + verify CLI

```bash
cd ~/tokenjam
source .env.local
python3 examples/single_provider/anthropic_agent.py

tj status        # agent with cost > $0, tokens > 0
tj traces        # waterfall renders
tj cost --since 1h    # real USD values (not $0.000000)
tj doctor             # exit 0 or 1 (warnings ok)
```

## 4. Cost-optimization analyzers (smoke — verify each runs)

```bash
tj optimize                                # all analyzers
tj optimize downsize      # Downsize
tj optimize cache       # Cache (efficacy)
tj optimize cache-recommend      # Cache (recommend) — surfaces "enable capture.prompts" if not set
tj optimize script # Script — likely no candidates on a fresh DB
tj optimize trim         # Trim — should print install hint without [bloat] extra

# Caveat enforcement on the downgrade finding
tj optimize --json | python3 -c \
  "import json,sys;r=json.load(sys.stdin);d=r.get('downgrade');assert d is None or 'Candidate-flagging heuristic' in d['caveat'];print('ok: caveat enforced')"

# Plan-tier metadata in JSON
tj optimize --json | python3 -c \
  "import json,sys;d=json.load(sys.stdin);assert 'plan' in d and 'pricing_mode' in d;print('ok')"
```

**Pass criteria:** every positional analyzer name runs without crashing. Optional analyzers (`cache-recommend`, `trim`) surface clear hints when their prereqs aren't met instead of erroring.

## 5. Backfill adapters (smoke against committed fixtures)

```bash
tj backfill langfuse --source-file tests/fixtures/langfuse_real_response.json
tj backfill helicone --source-file tests/fixtures/helicone_real_response.json
tj backfill otlp --source-file tests/fixtures/otlp_sample.json
# Re-run any of them — must be idempotent
tj backfill langfuse --source-file tests/fixtures/langfuse_real_response.json
# [ ] "skipped N already present" on the re-run
```

If this machine has Claude Code history:

```bash
ls ~/.claude/projects/ >/dev/null 2>&1 && tj backfill claude-code
```

## 6. Period comparison + config export

```bash
tj cost --since 7d --compare previous
# [ ] Current + Previous summary lines, cost/token delta lines

tj optimize --export-config claude-code
ls ~/.config/tokenjam/exports/
# [ ] claude-code-<date>.json file present
# [ ] Contains "STRUCTURAL HEURISTIC ONLY" caveat comments
```

## 7. Policy list

```bash
tj policy list
# [ ] Table renders with POLICY / SETTING / SOURCE columns
# [ ] "read-only preview" footer note
```

## 8. Server + Web UI

```bash
tj serve &
sleep 2

# HTTP fallback while server holds the lock
python3 examples/single_provider/litellm_agent.py
tj cost --since 1h   # works via API fallback

open http://127.0.0.1:7391/
```

Spot-check:

- [ ] Status page shows both agents with cost / tokens
- [ ] Traces page renders the waterfall
- [ ] Cost page shows non-zero USD values
- [ ] Sidebar theme toggle works

```bash
tj stop
```

---

## Claude Code integration (smoke)

```bash
tj onboard --claude-code --plan max_5x   # substitute your actual plan
cat ~/.claude/settings.json | python3 -m json.tool | grep -E "OTEL_LOGS_EXPORTER|OTEL_EXPORTER_OTLP_ENDPOINT"
cat ~/.config/tj/projects.json   # current cwd present

# Re-run is a quiet no-op
tj onboard --claude-code --plan max_5x   # substitute your actual plan

# Backfill ran automatically during onboard — verify history is present
tj cost --since 30d --agent claude-code-tokenjam || true
```

## Codex integration (smoke)

```bash
tj onboard --codex --plan plus
tj serve &
sleep 2

SERVER_SECRET=$(grep ingest_secret ~/.config/tj/config.toml | sed 's/.*= "//' | tr -d '"')
CODEX_SECRET=$(grep -oE 'Bearer [^"]+' ~/.codex/config.toml | sed 's/Bearer //')
[ "$SERVER_SECRET" = "$CODEX_SECRET" ] && echo "ok: secret synced"

# If codex CLI is installed
codex exec "say hello" 2>/dev/null && tj traces --agent codex_exec

tj stop
```

---

## Pass criteria summary

| Step | Pass criteria |
|------|--------------|
| 1 | `pip install --upgrade tokenjam` succeeds, version matches release |
| 2 | Onboard prompts for plan tier; config records it; no auto `usd = 200` written |
| 3 | Example runs without DB-lock errors; CLI shows real USD values |
| 4 | All four optimize analyzers run; caveat appears in downgrade JSON; `plan` + `pricing_mode` in JSON output |
| 5 | All three backfill adapters ingest from fixtures; re-runs are idempotent |
| 6 | `--compare previous` produces a diff report; `--export-config` writes a snippet with caveat comments |
| 7 | `tj policy list` renders the unified table |
| 8 | `tj serve` starts, web UI loads, HTTP fallback works while server holds lock |
| Claude Code | Onboard writes settings.json + projects.json; re-run is a no-op |
| Codex | Onboard writes `[otel]` + `[mcp_servers.tj]` to codex config; secret synced |

If any of these break, **don't ship from this release tag** — file a follow-up fix and re-cut.
