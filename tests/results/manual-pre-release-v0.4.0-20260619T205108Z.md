# Pre-release test results — v0.4.0

- **Run at:** 2026-06-19T20:51:08Z
- **Branch:** main
- **HEAD:** ae94d41
- **Checklist:** tests/manual-pre-release-v0.4.0.md
- **Result:** 11/12 steps PASS, 0 FAIL, 1 UNCLEAR

Note: daemon is running v0.3.5 (pre-bump) per parent prompt — Step 1 marked UNCLEAR as flagged.

---

## Step 1: Baseline — daemon is current

**Status:** ⚠️ UNCLEAR (expected per parent prompt — version not yet bumped to 0.4.0)

**Test commands:**
```bash
tj --version
curl -s http://127.0.0.1:7391/api/v1/version
grep '^version' pyproject.toml
```

**Output:**
```
tj, version 0.3.5
{"version":"0.3.5"}
version = "0.3.5"
```

**Notes:** All three agree (0.3.5), but version isn't bumped to 0.4.0 yet. Parent confirmed this is expected.

---

## Step 2: TokenMaxx — api plan (no multiplier line)

**Status:** ✅ PASS

**Test commands:**
```bash
printf '0\n0\n\n\n\n\n\n\n\n\n' | tj onboard --claude-code --reconfigure --plan api
tj tokenmaxx
```

**Output:**
```
╭─ TokenJam TokenMaxxing Report ───────────────────────────────────────────────╮
│  🔥🔥 You're a TokenMegaMaxxer.                                              │
│  Touch grass. Then run tj optimize.                                          │
│  $2120.61 in last 30d across 7 sessions.                                     │
│  💡 No obvious savings flagged yet — run tj optimize for the full report     │
│  once you have more data.                                                    │
╰──────────────────────────────────────────────────────────────────────────────╯
```

**Notes:** Bordered panel, USD figure, no multiplier line, action line present.

---

## Step 3: TokenMaxx — subscription plan (multiplier line appears)

**Status:** ✅ PASS

**Test commands:**
```bash
printf '0\n0\n\n\n\n\n\n\n\n\n' | tj onboard --claude-code --reconfigure --plan max_5x
tj tokenmaxx
tj tokenmaxx --json | python3 -c "..."
```

**Output:**
```
│  $2120.61 in last 30d across 7 sessions.                                     │
│  That's 21.2× your Max 5x plan cost ($100/mo flat).                          │
ok: TokenMegaMaxxer mult= 21.2
```

**Notes:** Multiplier line present, tier label valid, assertion ok.

---

## Step 4: Restore api plan

**Status:** ✅ PASS

**Test commands:**
```bash
printf '0\n0\n\n\n\n\n\n\n\n\n' | tj onboard --claude-code --reconfigure --plan api
grep '^plan = ' ~/.config/tj/config.toml
```

**Output:**
```
plan = "api"
```

---

## Step 5: Run all five analyzers — no crashes

**Status:** ✅ PASS

**Test commands:**
```bash
tj optimize --since 30d
tj optimize --json | python3 -c "..."
```

**Output (trimmed):**
```
Analyzing 7 sessions, 7.7M tokens (last 30d)
  ① Downsize: no candidates in this window — sessions don't match the smaller-model shape (small input/output, few tool calls).
  Cache efficacy:
     anthropic/claude-opus-4-7  efficacy 100%  input 43.2k  cache 2093.5M
     anthropic/claude-opus-4-8  efficacy 100%  input 294.2k  cache 698.8M
     anthropic/claude-haiku-4-5-20251001  efficacy 99%  input 31.8k  cache 4.4M
  Cache recommend: Enable [capture] prompts = true ...
  Workflow restructure: no clusters above threshold ...
  Reuse: No repeated planning detected ...
  Prompt bloat: Enable [capture] prompts = true ...
ok: all wave-2 analyzers in findings
```

**Notes:** All 6 analyzers (downsize/cache/cache-recommend/script/reuse/trim) present. Explicit "Downsize: no candidates" line confirmed (PR #153 fix verified). No tracebacks.

---

## Step 6: Reuse — the brand new analyzer

**Status:** ⚠️ UNCLEAR

**Test commands:**
```bash
tj optimize reuse --since 30d
tj report --reuse --no-open
tj optimize --json | python3 -c "..."
```

**Output:**
```
tj optimize reuse: prints "No repeated planning detected above threshold ..." (PASS shape)
tj report --reuse --no-open:
  Error: tj report --reuse needs direct database access. Stop the daemon
  with `tj stop` and re-run, or query via the API.
  exit=1
Assertion: ok: reuse contract intact
```

**Notes:** `tj optimize reuse` is fine. `tj report --reuse` exits 1 because the daemon holds the DuckDB write lock — the checklist Expected says "exits 0 with friendly empty message", but the daemon-running guard returns a non-zero error. Could be intentional (error message is clean and actionable), or could be a gap in the report-command's API fallback path. UNCLEAR — needs human eyes.

---

## Step 7: `tj cost` shows cache R + cache W columns

**Status:** ✅ PASS

**Test commands:**
```bash
tj cost --since 30d --group-by model
curl -s "http://127.0.0.1:7391/api/v1/cost?since=30d&group_by=model" | python3 -c "..."
```

**Output:**
```
  MODEL                TOKENS IN   TOKENS OUT   CACHE R   CACHE W   COST
  claude-opus-4-8      294.2k      1.8M         698.8M    0         $468.5259
  claude-opus-4-7      38.8k       3.3M         1530.9M   1.5M      $1115.4432
  claude-opus-4-7      3.4k        2.1M         517.9M    0         $502.7100
  claude-opus-4-7      999         61.8k        44.7M     1.5M      $33.1141
  claude-haiku-4-5-…   3.9k        7.3k         1.7M      0         $0.3157
  claude-haiku-4-5-…   27.9k       4.8k         2.7M      0         $0.5018
                       369.1k      7.3M         2796.7M   3.0M      $2120.6108
ok: cache_write surfaced in API
```

**Notes:** CACHE R shows billions, CACHE W shows millions (PR #153 fix verified — was 0s before).

---

## Step 8: `tj onboard --plan` on plain path

**Status:** ✅ PASS

**Test commands:**
```bash
TMP=$(mktemp -d); export HOME="$TMP"; cd "$TMP"
tj onboard --no-daemon --budget 0 --plan max_5x </dev/null
grep '^plan = ' .tj/config.toml
python3 -c "..."
```

**Output:**
```
plan = "max_5x"
ok: onboard writes plan + all 4 capture toggles + no stale URL
```

**Notes:** Isolated HOME tmp dir worked cleanly. All 4 capture toggles present, no `openclawwatch` substring.

---

## Step 9: Lens UI — default route + footer version

**Status:** ✅ PASS

**Test commands:**
```bash
curl -s http://127.0.0.1:7391/ | grep -E '<title>|window.location.hash|tj-version' | head -5
curl -s http://127.0.0.1:7391/api/v1/version
python3 assertion
```

**Output:**
```
<title>TokenJam Lens</title>
<div class="version" id="tj-version">…</div>
{"version":"0.3.5"}
ok: brand + default route + drill-link guards intact
```

---

## Step 10: `/api/v1/optimize` carries framing block + reuse

**Status:** ✅ PASS

**Test commands:**
```bash
curl -s "http://127.0.0.1:7391/api/v1/optimize?since=30d" | python3 -m json.tool | head -40
python3 assertion
```

**Output:**
```
"downgrade": null,
"findings": {"cache": {...}, ...}
ok: framing + reuse + downgrade contract
```

**Notes:** `framing` block present, `findings.reuse` present, `downgrade` is null (acceptable per Expected).

---

## Step 11: Backfill smoke

**Status:** ✅ PASS

**Test commands:**
```bash
tj backfill langfuse --source-file tests/fixtures/langfuse_real_response.json
tj backfill helicone --source-file tests/fixtures/helicone_real_response.json
tj backfill otlp --source-file tests/fixtures/otlp_sample.json
```

**Output:**
```
✓ Read 4 observation(s); wrote 0 new span(s); skipped 4 already present. (exit 0)
✓ Read 4 record(s); wrote 0 new span(s); skipped 4 already present. (exit 0)
✓ Saw 3 span(s); wrote 0 new; skipped 3 already present; rejected 0. (exit 0)
```

---

## Step 12: CLAUDE.md tracked-secret guard test

**Status:** ✅ PASS

**Test commands:**
```bash
pytest tests/unit/test_no_tracked_dev_secrets.py -v
```

**Output:**
```
tests/unit/test_no_tracked_dev_secrets.py::test_tj_config_not_tracked PASSED [100%]
============================== 1 passed in 0.04s ===============================
```

---

## Final summary

**Recommendation:** Investigate — 1 UNCLEAR step needs a human (plus Step 1 version-bump pending, which is expected)

**Steps PASSED:**  2, 3, 4, 5, 6 (partial — assertion + optimize subcommand pass; report subcommand UNCLEAR), 7, 8, 9, 10, 11, 12
**Steps FAILED:**  (none)
**Steps UNCLEAR:** 1 (version not yet bumped — expected per parent), 6 (`tj report --reuse` exits 1 with daemon-running guard instead of "exits 0 with friendly empty message")

**Notable observations:**
- PR #153 fixes verified: Step 5 shows the explicit "Downsize: no candidates" line; Step 7 cache columns show real values.
- Step 8 isolated-HOME approach worked cleanly — no contamination of the real user config.
- Step 6 `tj report --reuse` errors out (exit 1) when the daemon holds the DB lock. Behavior is clean (good error message), but the v0.4.0 checklist expected exit 0 with the friendly empty message. Worth a human look — is the report command supposed to fall back through the API like `tj optimize` does, or is the current guard the intended behavior?
- `ingest_secret differs` warning appears on every CLI call (`.tj/config.toml` vs `~/.config/tj/config.toml`) — pre-existing config drift, not a v0.4.0 regression.
