# Pre-release checklist — v0.4.0

Focused on the delta since v0.3.5. The runner (`tests/agent-pre-release-runner.md`) walks each step in order.

**What's new in v0.4.0** (so you know what we're validating):
- TokenJam Lens UI rebrand (Overview triage screen + Optimize tab + real charts)
- Reuse — 5th analyzer + `tj report --reuse` artifact export
- `core/framing.py` — single source of truth for plan-tier rendering
- `estimated_recoverable_usd` contract on every savings analyzer
- `cache_write_tokens` surfaced through DB → API → CLI → UI
- Security: `.tj/config.toml` untracked + CI guard
- Onboard: plain `tj onboard --plan` honored; tool_inputs capture toggle added; stale URLs removed

Stable surfaces (ingest, alerts, drift, schema validation, MCP server) are **not** in scope — they're covered by the standing pre-release playbook in `tests/agent-pre-release-testing.md`.

---

## Step 1: Baseline — daemon is current

**What:** confirm the daemon is running the code under test, not an old version.

**Test:**
```bash
tj --version
curl -s http://127.0.0.1:7391/api/v1/version
```

**Expected:**
- Both commands report the same version
- Version matches what's in `pyproject.toml` (`grep '^version' pyproject.toml`)

---

## Step 2: TokenMaxx — api plan (no multiplier line)

**What:** verify `tj tokenmaxx` renders the bordered panel on api-pricing mode without the multiplier line.

**Setup:**
```bash
tj onboard --claude-code --reconfigure --plan api
```

**Test:**
```bash
tj tokenmaxx
```

**Expected:**
- A bordered "TokenJam TokenMaxxing Report" panel renders
- An absolute USD/mo spend figure appears (e.g., "Your spend: $X/mo")
- **No** multiplier line ("That's N× your plan cost") — api users don't get one
- An action line mentioning either downsize savings or "no obvious savings flagged yet"

---

## Step 3: TokenMaxx — subscription plan (multiplier line appears)

**What:** verify subscription rendering activates the multiplier line and tier classification works across plan boundaries.

**Setup:**
```bash
tj onboard --claude-code --reconfigure --plan max_5x
```

**Test:**
```bash
tj tokenmaxx
```

**Expected:**
- Bordered panel renders
- A line like "That's N× your Max 5x plan cost ($100/mo flat)" appears
- The tier label is one of: TokenSipper / TokenModerator / TokenMaxxer / TokenSuperMaxxer / TokenMegaMaxxer / TokenGigaMaxxer

**Assertions:**
```bash
tj tokenmaxx --json | python3 -c "import json,sys;d=json.load(sys.stdin);ok={'TokenSipper','TokenModerator','TokenMaxxer','TokenSuperMaxxer','TokenMegaMaxxer','TokenGigaMaxxer'};assert d['tier'] in ok,d['tier'];assert d.get('plan_multiplier') is not None;print('ok:',d['tier'],'mult=',d['plan_multiplier'])"
```

---

## Step 4: Restore api plan for the rest of the run

**What:** flip back to api so subsequent dollar-bearing checks render literal dollars.

**Test:**
```bash
tj onboard --claude-code --reconfigure --plan api
```

**Expected:**
- Command exits 0
- `grep '^plan = ' ~/.config/tj/config.toml` shows `plan = "api"`

---

## Step 5: Run all five analyzers — no crashes

**What:** verify every registered analyzer (downsize / cache / cache-recommend / script / trim / reuse) executes without errors. Optional analyzers should print clear hints when prereqs aren't met, not crash.

**Test:**
```bash
tj optimize --since 30d
```

**Expected:**
- Output contains "Downsize", "Cache", "Cache recommend", "Script", "Trim", **"Reuse"** sections (or honest "not ready" / "no candidates" hints for any)
- No Python tracebacks
- No "ERROR" lines
- Plan-tier banner appears if applicable ("Plan tier unknown" qualifier or similar)

**Assertions:**
```bash
tj optimize --json | python3 -c "import json,sys;d=json.load(sys.stdin);f=d['findings'];assert 'reuse' in f,'reuse missing';assert 'cache' in f,'cache missing';assert 'script' in f,'script missing';print('ok: all wave-2 analyzers in findings')"
```

---

## Step 6: Reuse — the brand new analyzer

**What:** verify `tj optimize reuse` runs and produces honest output. `tj report --reuse` is tested with a fallback because it doesn't currently have an HTTP-backed path (#154 tracks the v0.4.1 fix) — when the daemon holds the DB lock, the command exits with a friendly hint pointing at `tj stop`.

**Test:**
```bash
tj optimize reuse --since 30d
tj report --reuse --no-open || true
```

**Expected:**
- `tj optimize reuse` either lists clusters with cache-reuse + script-replacement numbers, OR prints "No clusters above threshold" — both are valid outcomes
- `tj report --reuse --no-open` behavior depends on daemon state:
  - **Daemon stopped:** writes `reuse-*.html` + `reuse-*-*.md` under `~/.cache/tokenjam/reports/` and exits 0, OR prints "No repeated planning detected" + exits 0 if no clusters
  - **Daemon running:** exits non-zero with the explicit hint *"needs direct database access. Stop the daemon with `tj stop` …"* — this is current limitation #154, not a regression
- No Python tracebacks either way

**Assertions:**
```bash
tj optimize --json | python3 -c "import json,sys;d=json.load(sys.stdin);r=d['findings']['reuse'];assert 'estimate_basis' in r and 'review' in r['estimate_basis'].lower();print('ok: reuse contract intact')"
```

---

## Step 7: `tj cost` shows cache R + cache W columns

**What:** verify the cost-transparency fix from #149 — cache-read and cache-write tokens appear in the table.

**Test:**
```bash
tj cost --since 30d --group-by model
```

**Expected:**
- Header row includes `CACHE R` and `CACHE W` columns (alongside `TOKENS IN` / `TOKENS OUT`)
- At least one data row OR a "no data" message; if data exists, the cache columns are populated (not all zeros)
- A totals row at the bottom

**Assertions:**
```bash
curl -s "http://127.0.0.1:7391/api/v1/cost?since=30d&group_by=model" | python3 -c "import json,sys;d=json.load(sys.stdin);assert 'total_cache_write_tokens' in d,'total missing';assert all('cache_write_tokens' in r for r in d.get('rows',[])),'row field missing';print('ok: cache_write surfaced in API')"
```

---

## Step 8: `tj onboard --plan` on plain path

**What:** verify #148 — bare `tj onboard --plan` actually writes the plan field (was silent no-op pre-v0.4.0). Needs care because plain `tj onboard` refuses to overwrite an existing global config (correct safety behavior), so we run the test against a temporarily isolated `XDG_CONFIG_HOME` instead of touching the real user config.

**Setup:**
```bash
TMP=$(mktemp -d)
HOME_BAK="$HOME"
# Isolate config + data into the tmp dir so the user's real global config
# isn't seen by `tj onboard`. find_config_file walks $HOME/.config/tj/ —
# pointing HOME at a fresh dir bypasses any existing config cleanly.
export HOME="$TMP"
cd "$TMP"
```

**Test:**
```bash
tj onboard --no-daemon --budget 0 --plan max_5x </dev/null
grep '^plan = ' .tj/config.toml
```

**Expected:**
- Command exits 0
- `.tj/config.toml` contains `plan = "max_5x"` under `[budget.anthropic]`
- File also contains all 4 capture toggles: `prompts = false`, `completions = false`, `tool_inputs = false`, `tool_outputs = false`
- No `openclawwatch` substring anywhere in the file

**Assertions:**
```bash
python3 -c "import pathlib,re;t=pathlib.Path('.tj/config.toml').read_text();assert 'plan = \"max_5x\"' in t;assert all(f'{k} = false' in t for k in ('prompts','completions','tool_inputs','tool_outputs'));assert 'openclawwatch' not in t;print('ok: onboard writes plan + all 4 capture toggles + no stale URL')"
export HOME="$HOME_BAK"
cd - >/dev/null
```

---

## Step 9: Lens UI — default route + footer version

**What:** verify the dashboard lands on Overview (not Status) and the footer fetches the live version.

**Test:**
```bash
curl -s http://127.0.0.1:7391/ | grep -E '<title>|window.location.hash|tj-version' | head -5
curl -s http://127.0.0.1:7391/api/v1/version
```

**Expected:**
- `<title>` contains "TokenJam Lens"
- The HTML references `#/overview` as the default route (and **not** `location.hash = '#/status'` as a render-time redirect — that was issue #132)
- `/api/v1/version` returns the same version as Step 1

**Assertions:**
```bash
python3 -c "import urllib.request;html=urllib.request.urlopen('http://127.0.0.1:7391/').read().decode();assert 'TokenJam Lens' in html;assert 'overview' in html.lower();assert 'class=\"chart-card band-hero\"' not in html,'chart-as-button regression';print('ok: brand + default route + drill-link guards intact')"
```

---

## Step 10: `/api/v1/optimize` carries the framing block + reuse

**What:** verify the API contract — every dollar-bearing endpoint includes a `framing` block, and `findings.reuse` is present.

**Test:**
```bash
curl -s "http://127.0.0.1:7391/api/v1/optimize?since=30d" | python3 -m json.tool | head -40
```

**Expected:**
- The response includes a top-level `framing` key with `pricing_mode`, `plan_tier`, `display_rule` fields
- The `findings` object includes a `reuse` key
- `downgrade` is either null or a typed object with `estimated_recoverable_usd`, `monthly_savings_usd`, and a non-empty caveat

**Assertions:**
```bash
curl -s "http://127.0.0.1:7391/api/v1/optimize?since=30d" | python3 -c "import json,sys;d=json.load(sys.stdin);assert 'framing' in d and 'pricing_mode' in d['framing'];assert 'reuse' in d['findings'];g=d.get('downgrade');assert g is None or g.get('estimated_recoverable_usd') is not None;print('ok: framing + reuse + downgrade contract')"
```

---

## Step 11: Backfill smoke — sanity check, no regression

**What:** quick check that the three fixture-based backfill adapters haven't regressed under the v0.4.0 changes.

**Test:**
```bash
tj backfill langfuse --source-file tests/fixtures/langfuse_real_response.json
tj backfill helicone --source-file tests/fixtures/helicone_real_response.json
tj backfill otlp --source-file tests/fixtures/otlp_sample.json
```

**Expected:**
- Each command reports either "wrote N new spans" or "skipped N already present" — both are valid (depends on prior runs)
- No tracebacks
- Exit code 0 for all three

---

## Step 12: CLAUDE.md tracked-secret guard test (the meta one)

**What:** confirm the CI test from #150 actually runs and passes — meta-check that the guardrail we added is effective.

**Test:**
```bash
pytest tests/unit/test_no_tracked_dev_secrets.py -v
```

**Expected:**
- One test (`test_tj_config_not_tracked`) runs and passes
- No warnings about `.tj/config.toml` being tracked

---

## Final summary the runner produces

After all 12 steps, the result log should end with:

```
**Recommendation:** Ready to release v0.4.0 / Hold — <N> failures / Investigate — <K> UNCLEAR steps

**Steps PASSED:**  1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12  (or whichever subset)
**Steps FAILED:**  <numbers>
**Steps UNCLEAR:** <numbers>

**Notable observations:**
- <any one-liners worth surfacing>
```

If everything passes, the release-cut PR can be opened directly. If anything FAILs or is UNCLEAR, the parent agent investigates before cutting.
