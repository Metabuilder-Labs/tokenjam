# Post-Release Smoke Test

Run this after a new release publishes to PyPI to verify it works end-to-end. This is intentionally lighter than `manual-pre-release-testing.md` — the multi-project / secret-rotation / theme-toggle detail lives there and was exercised on the branch before merge. Here we're confirming the published artifact actually installs and the core surfaces are alive.

> **Code blocks are command-only on purpose.** Explanations and checklists live in
> prose around each block so the markdown "copy" button copies a runnable command,
> not a comment (smoke-test finding #3).

## Prerequisites

- `ANTHROPIC_API_KEY` and `OPENAI_API_KEY` in `~/tokenjam/.env.local`, sourced before running.
- The example scripts import provider SDKs (`anthropic`, `openai`, `litellm`) that are **not** tokenjam dependencies. A `pipx install` isolates tokenjam in its own venv, so those SDKs must be injected into that venv and the examples run with its interpreter (Steps 3 and 8 do this).

## 1. Install the published release

`tj uninstall` removes data/config/daemon but **not** the package; `pipx uninstall` then removes the venv so `pipx install --force` rebuilds it pristine. The bare `pip3 install --upgrade tokenjam` path fails on system Python (PEP 668) — pipx is the path we ship docs telling users to use, so it's what we verify.

```bash
tj uninstall --yes 2>/dev/null
pipx uninstall tokenjam 2>/dev/null
rm -rf ~/.tj ~/.config/tj .tj
pipx install --force tokenjam
tj --version
```

**Pass criteria:** version matches the release being tested. `pipx install` reports a fresh venv (not "Installing to existing venv 'tokenjam'") thanks to the `pipx uninstall` step.

## 2. Onboard

`tj onboard` prompts for daily budget and plan tier; `--plan api` sets the plan non-interactively (so plan-aware dollar rendering kicks in for the rest of the smoke), and `--no-daemon` skips the launchd/systemd install for this pass. We onboard against Claude Code here so the plan lands in the global config.

```bash
tj onboard --claude-code --plan api --no-daemon
grep '^plan = ' ~/.config/tj/config.toml
```

**Pass criteria:** the `plan` field is set under `[budget.anthropic]`; no auto-written `usd = 200`.

## 3. Drive an example + verify CLI

Inject the provider SDK into the pipx venv and run the example with that interpreter (see Prerequisites).

```bash
pipx inject tokenjam anthropic openai
PIPX_PY="$(pipx environment --value PIPX_LOCAL_VENVS)/tokenjam/bin/python"

cd ~/tokenjam
source .env.local
"$PIPX_PY" examples/single_provider/anthropic_agent.py

tj status
tj traces
tj cost --since 1h
tj doctor
```

**Pass criteria:**

- [ ] `tj status` shows the agent with cost > $0 and tokens > 0
- [ ] `tj traces` renders the waterfall
- [ ] `tj cost --since 1h` shows real USD values (not `$0.000000`)
- [ ] `tj doctor` exits 0 or 1 (warnings ok)

## 4. Cost-optimization analyzers (smoke — verify each runs)

```bash
tj optimize
tj optimize downsize
tj optimize cache
tj optimize cache-recommend
tj optimize script
tj optimize trim

tj optimize --json | python3 -c \
  "import json,sys;r=json.load(sys.stdin);d=r.get('downgrade');assert d is None or 'Candidate-flagging heuristic' in d['caveat'];print('ok: caveat enforced')"

tj optimize --json | python3 -c \
  "import json,sys;d=json.load(sys.stdin);assert 'plan' in d and 'pricing_mode' in d;print('ok')"
```

**Pass criteria:** every positional analyzer name runs without crashing, and the two JSON checks print `ok`. Optional analyzers surface a clear hint rather than erroring when a prereq isn't met:

- `cache-recommend` and `trim` both require `[capture] prompts = true`. With capture off (the default), each checks that prereq **first**, so `trim` prints the capture hint — **not** the `tokenjam[bloat]` install hint (you only reach the `[bloat]` hint once capture is on but the extra is missing). `script` likely reports no candidates on a fresh DB.

## 4b. TokenMaxx quota/efficiency card (#7)

The card is a quota/efficiency artifact (NOT a spend-tier flex): it leads with
the overhead-vs-real-work composition from the `tj context` diagnostic and is
quota-native for subscription plans.

```bash
tj tokenmaxx

tj tokenmaxx --json | python3 -c \
  "import json,sys;d=json.load(sys.stdin);ok={'TokenMinimizer','LeanOperator','SteadyState','ContextHeavy','QuotaSink'};assert d['tier'] in ok,d['tier'];assert 'overhead_share' in d and 'work_share' in d;print('ok:',d['tier'],'overhead=%.2f'%d['overhead_share'])"
```

Check the rendered card:

- [ ] Bordered "TokenJam Quota / Efficiency Card" panel renders
- [ ] Headline is the composition ("X% overhead" vs "Y% real work"), NOT a dollar spend figure
- [ ] On a subscription plan: figures read as "% of cycle tokens", NO `$` anywhere
- [ ] Action line surfaces what's reclaimable (points at `tj context`)

Then reconfigure to an `api` plan and re-run to confirm the secondary dollar
line appears (demoted, labeled "Implied API value"), then flip back so later
steps behave consistently:

```bash
tj onboard --claude-code --reconfigure --plan api
tj tokenmaxx     # api plan: "Implied API value: $… (calibration only)" appears
tj tokenmaxx --weekly   # "Weekly Recap" title, "this week" window
tj onboard --claude-code --reconfigure --plan max_5x
```

- [ ] `api` plan shows the secondary "Implied API value" line (never the headline)
- [ ] `--weekly` renders the "Quota Wrapped — Weekly Recap" title

**Pass criteria:** the card renders without crashing, the JSON `tier` field
carries one of the 5 efficiency-tier names, `overhead_share`/`work_share` are
present, and a subscription plan renders no dollar figure.

## 5. Backfill adapters (smoke against committed fixtures)

```bash
tj backfill langfuse --source-file tests/fixtures/langfuse_real_response.json
tj backfill helicone --source-file tests/fixtures/helicone_real_response.json
tj backfill otlp --source-file tests/fixtures/otlp_sample.json
tj backfill langfuse --source-file tests/fixtures/langfuse_real_response.json
```

The final line re-runs langfuse to check idempotency:

- [ ] The re-run reports "skipped N already present"

If this machine has Claude Code history:

```bash
ls ~/.claude/projects/ >/dev/null 2>&1 && tj backfill claude-code
```

## 6. Period comparison + config export

```bash
tj cost --since 7d --compare previous

tj optimize --export-config claude-code
ls ~/.config/tokenjam/exports/
```

- [ ] `--compare` prints Current + Previous summary lines and cost/token delta lines
- [ ] A `claude-code-<date>.jsonc` file is present (it's JSONC — it carries comments)
- [ ] It contains the "STRUCTURAL HEURISTIC ONLY" caveat comments

## 7. Policy list

```bash
tj policy list
```

- [ ] Table renders with POLICY / SETTING / SOURCE columns
- [ ] "read-only preview" footer note

## 8. Server + Web UI

Start the server, inject `litellm` into the pipx venv (one-time per smoke run), drive the litellm example over the HTTP transport, and confirm `tj cost` works via the API fallback while the server holds the DB lock:

```bash
tj serve &
sleep 2

pipx inject tokenjam litellm
"$PIPX_PY" examples/single_provider/litellm_agent.py
tj cost --since 1h

open http://127.0.0.1:7391/
```

Spot-check:

- [ ] Status page shows both agents with cost / tokens
- [ ] Traces page renders the waterfall
- [ ] Cost page shows non-zero USD values
- [ ] Sidebar theme toggle works

### Offline-UI verification (PR #88)

Open Chrome DevTools (or your browser's equivalent) → **Network tab** → reload `http://127.0.0.1:7391/`.

- [ ] **Zero failed requests** to `fonts.googleapis.com`, `fonts.gstatic.com`, `esm.sh`, or `tokenjam.dev`
- [ ] Dashboard interactivity works (sidebar nav, tab switches) — proves the vendored Preact / htm / uPlot under `/ui/vendor/` is being served, not loading from a CDN
- [ ] Favicon renders (data: URL, no external fetch)

Bonus: turn off wifi entirely, hard-refresh, and confirm the page still renders + the JS still hydrates. The whole dashboard must work air-gapped.

```bash
tj stop
```

## 9. Cache cost-correctness (PRs #90 + #92)

Cache-only spans (cache_read > 0, input/output = 0) used to be costed at $0, and cache-creation tokens on the live OTLP path used to be silently dropped — both fixed. The spans table gained `cache_write_tokens` in migration 5.

```bash
duckdb ~/.tj/telemetry.duckdb "PRAGMA table_info(spans)" 2>/dev/null \
  | grep cache_write_tokens \
  && echo "ok: cache_write_tokens column present"

duckdb ~/.tj/telemetry.duckdb "
  SELECT COUNT(*) AS hits, MIN(cost_usd) AS min_cost
  FROM spans
  WHERE cache_tokens > 0
    AND (input_tokens = 0 OR input_tokens IS NULL)
    AND (output_tokens = 0 OR output_tokens IS NULL)
" 2>/dev/null
```

- [ ] `cache_write_tokens` column is present on the spans table
- [ ] If `hits > 0`: `min_cost > 0` (cache hits ARE costed; was $0 pre-fix)
- [ ] If `hits = 0`: this release's runs didn't trigger a pure cache-only span — fine; unit tests cover the path

If you don't have the `duckdb` CLI installed, skip the SQL checks — the unit + synthetic tests covering these paths run in CI and are the canonical verification.

---

## Claude Code integration (smoke)

Onboard against Claude Code (substitute your actual plan), confirm the OTEL env + project tracking landed, and that a re-run is a quiet no-op. Backfill runs automatically during onboard, so history should be queryable afterward.

```bash
tj onboard --claude-code --plan max_5x
cat ~/.claude/settings.json | python3 -m json.tool | grep -E "OTEL_LOGS_EXPORTER|OTEL_EXPORTER_OTLP_ENDPOINT"
cat ~/.config/tj/projects.json

tj onboard --claude-code --plan max_5x

tj cost --since 30d --agent claude-code-tokenjam || true
```

- [ ] `settings.json` carries the OTEL exporter env; `projects.json` lists the current cwd
- [ ] The second onboard is a quiet no-op

## Codex integration (smoke)

```bash
tj onboard --codex --plan plus
tj serve &
sleep 2

SERVER_SECRET=$(grep ingest_secret ~/.config/tj/config.toml | sed 's/.*= "//' | tr -d '"')
CODEX_SECRET=$(grep -oE 'Bearer [^"]+' ~/.codex/config.toml | sed 's/Bearer //')
[ "$SERVER_SECRET" = "$CODEX_SECRET" ] && echo "ok: secret synced"

codex exec "say hello" 2>/dev/null && tj traces --agent codex_exec

tj stop
```

- [ ] `ok: secret synced` prints (the Codex Bearer token matches the server's ingest secret)
- [ ] If the `codex` CLI is installed, a `codex exec` run shows up under `tj traces --agent codex_exec`

---

## Pass criteria summary

| Step | Pass criteria |
|------|--------------|
| 1 | `pipx install --force tokenjam` succeeds, version matches release |
| 2 | Onboard records the plan tier; no auto `usd = 200` written |
| 3 | Example runs without DB-lock errors; CLI shows real USD values |
| 4 | All optimize analyzers run; caveat appears in downgrade JSON; `plan` + `pricing_mode` in JSON output |
| 4b | `tj tokenmaxx` renders the bordered report panel; JSON `tier` is one of the 6 tier names; subscription plan shows multiplier line |
| 5 | All three backfill adapters ingest from fixtures; re-runs are idempotent |
| 6 | `--compare previous` produces a diff report; `--export-config` writes a `.jsonc` snippet with caveat comments |
| 7 | `tj policy list` renders the unified table |
| 8 | `tj serve` starts, web UI loads, HTTP fallback works while server holds lock; **zero external requests in DevTools Network tab** |
| 9 | `cache_write_tokens` column present on the spans table (migration 5); cache-hit spans show non-zero cost_usd |
| Claude Code | Onboard writes settings.json + projects.json; re-run is a no-op |
| Codex | Onboard writes `[otel]` + `[mcp_servers.tj]` to codex config; secret synced |

If any of these break, **don't ship from this release tag** — file a follow-up fix and re-cut.
