# Post-Release Smoke Test

Run this after a new release publishes to PyPI to verify it works end-to-end. This is intentionally lighter than `manual-pre-release-testing.md` — the multi-project / secret-rotation / theme-toggle detail lives there and was exercised on the branch before merge. Here we're confirming the published artifact actually installs and the core surfaces are alive.

## Prerequisites

- `ANTHROPIC_API_KEY` and `OPENAI_API_KEY` in `~/tokenjam/.env.local`
- Sourced before running

## 1. Install the published release

```bash
tj uninstall --yes 2>/dev/null
rm -rf ~/.tj ~/.config/tj .tj

# Recommended install path (PEP 668-safe on Homebrew Python and
# Debian 12+/Ubuntu 24+). `--force` so we reinstall even if a prior
# version is present.
pipx install --force tokenjam
tj --version

# Older `pip3 install --upgrade tokenjam` path still works inside
# a clean venv but fails on system Python — that's the bug pipx
# solves, and verifying pipx is what we ship docs telling users to do.
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

## 4b. TokenMaxx tier classification

```bash
tj tokenmaxx
# [ ] Bordered "TokenJam TokenMaxxing Report" panel renders
# [ ] On api plan: shows absolute spend; no multiplier line
# [ ] Action line surfaces either downsize savings or "no obvious
#     savings flagged yet" (both are valid)

# Verify the JSON tier label is one of the six valid v0.3.4 tiers.
tj tokenmaxx --json | python3 -c \
  "import json,sys;d=json.load(sys.stdin);ok={'TokenSipper','TokenModerator','TokenMaxxer','TokenSuperMaxxer','TokenMegaMaxxer','TokenGigaMaxxer'};assert d['tier'] in ok,d['tier'];print('ok:',d['tier'])"

# Reconfigure to a subscription plan and re-run — the multiplier line
# should appear. Pick whichever plan matches your test config.
tj onboard --claude-code --reconfigure --plan max_5x
tj tokenmaxx
# [ ] Multiplier line "That's N× your Max 5x plan cost ($100/mo flat)."
# [ ] Tier may shift if the multiplier crosses a boundary

# Flip back to api so subsequent steps render dollar figures.
tj onboard --claude-code --reconfigure --plan api
```

**Pass criteria:** the report renders without crashing, the JSON `tier` field carries one of the 6 v0.3.4 tier names, and the multiplier line appears under a subscription plan.

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

### Offline-UI verification (v0.3.4 — PR #88)

Open Chrome DevTools (or your browser's equivalent) → **Network tab** → reload `http://127.0.0.1:7391/`.

- [ ] **Zero failed requests** to `fonts.googleapis.com`, `fonts.gstatic.com`, `esm.sh`, or `tokenjam.dev`
- [ ] Dashboard interactivity works (sidebar nav, tab switches) — proves the vendored Preact / htm under `/ui/vendor/` is being served, not loading from the CDN
- [ ] Favicon renders (data: URL, no external fetch)

Bonus: turn off wifi entirely, hard-refresh, and confirm the page still renders + the JS still hydrates. The whole dashboard must work air-gapped.

```bash
tj stop
```

## 9. Cache cost-correctness (v0.3.4 — PRs #90 + #92)

Cache-only spans (cache_read > 0, input/output = 0) used to be costed at $0. Cache-creation tokens on the live OTLP path used to be silently dropped. Both fixed in v0.3.4.

```bash
# Spans table now has cache_write_tokens (migration 5).
duckdb ~/.tj/telemetry.duckdb "PRAGMA table_info(spans)" 2>/dev/null \
  | grep cache_write_tokens \
  && echo "ok: cache_write_tokens column present"

# Any captured Anthropic cache-hit span should have non-zero cost_usd.
duckdb ~/.tj/telemetry.duckdb "
  SELECT COUNT(*) AS hits,
         MIN(cost_usd) AS min_cost
  FROM spans
  WHERE cache_tokens > 0
    AND (input_tokens = 0 OR input_tokens IS NULL)
    AND (output_tokens = 0 OR output_tokens IS NULL)
" 2>/dev/null
# [ ] If hits > 0: min_cost > 0 (cache hits ARE being costed; was $0 pre-0.3.4)
# [ ] If hits = 0: this release's runs didn't trigger a pure cache-only span — fine, unit tests cover the path
```

If you don't have `duckdb` CLI installed, skip the SQL checks — the unit + synthetic tests covering these paths run in CI and are the canonical verification.

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
| 1 | `pipx install --force tokenjam` succeeds, version matches release |
| 2 | Onboard prompts for plan tier; config records it; no auto `usd = 200` written |
| 3 | Example runs without DB-lock errors; CLI shows real USD values |
| 4 | All optimize analyzers run; caveat appears in downgrade JSON; `plan` + `pricing_mode` in JSON output |
| 4b | `tj tokenmaxx` renders the bordered report panel; JSON `tier` is one of the 6 v0.3.4 tier names; subscription plan shows multiplier line |
| 5 | All three backfill adapters ingest from fixtures; re-runs are idempotent |
| 6 | `--compare previous` produces a diff report; `--export-config` writes a snippet with caveat comments |
| 7 | `tj policy list` renders the unified table |
| 8 | `tj serve` starts, web UI loads, HTTP fallback works while server holds lock; **zero external requests in DevTools Network tab** (offline-UI fix shipped in v0.3.4) |
| 9 | `cache_write_tokens` column present on the spans table (migration 5); cache-hit spans show non-zero cost_usd |
| Claude Code | Onboard writes settings.json + projects.json; re-run is a no-op |
| Codex | Onboard writes `[otel]` + `[mcp_servers.tj]` to codex config; secret synced |

If any of these break, **don't ship from this release tag** — file a follow-up fix and re-cut.
