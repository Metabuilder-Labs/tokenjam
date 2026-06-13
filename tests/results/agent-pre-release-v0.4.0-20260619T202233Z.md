# Pre-release test results — v0.4.0

- **Run at:** 2026-06-19T20:22:33Z
- **Branch:** main
- **HEAD:** 4dd2185
- **Checklist:** tests/agent-pre-release-v0.4.0.md
- **Result:** 8/12 PASS, 1 FAIL, 3 UNCLEAR

> **Pre-flight note:** `pyproject.toml` still reads `version = "0.3.5"`, and both `tj --version` and `/api/v1/version` report `0.3.5`. The version has **not** been bumped to 0.4.0 yet. All steps below were run against the 0.3.5 build, since that's what the daemon is serving. Per Critical Rule 15, the version bump must happen before the release is cut.

---

## Step 1: Baseline — daemon is current

**Status:** ⚠️ UNCLEAR

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

**Notes:** All three sources agree (0.3.5), so the daemon is current. UNCLEAR because we're validating "v0.4.0" but the version hasn't been bumped; the release engineer must bump `pyproject.toml` and `sdk-ts/package.json` before tagging.

---

## Step 2: TokenMaxx — api plan (no multiplier line)

**Status:** ✅ PASS

**Test commands:**
```bash
tj onboard --claude-code --reconfigure --plan api --budget 0 </dev/null
tj tokenmaxx
```

**Output:**
```
╭─ TokenJam TokenMaxxing Report ───────────────────────────────────────────────╮
│   You're a TokenMegaMaxxer.                                                  │
│   Touch grass. Then run tj optimize.                                         │
│   $2116.74 in last 30d across 7 sessions.                                    │
│   No obvious savings flagged yet — run tj optimize for the full report       │
│   once you have more data.                                                   │
╰──────────────────────────────────────────────────────────────────────────────╯
```

**Notes:** Bordered panel renders, absolute USD figure present, no multiplier line on api plan, actionable message present.

---

## Step 3: TokenMaxx — subscription plan (multiplier line appears)

**Status:** ✅ PASS

**Test commands:**
```bash
tj onboard --claude-code --reconfigure --plan max_5x --budget 0 </dev/null
tj tokenmaxx
tj tokenmaxx --json | python3 -c "<assertion>"
```

**Output:**
```
$2116.74 in last 30d across 7 sessions.
That's 21.2× your Max 5x plan cost ($100/mo flat).
...
ok: TokenMegaMaxxer mult= 21.2
```

**Notes:** Multiplier line appears, tier is valid, assertion ok.

---

## Step 4: Restore api plan for the rest of the run

**Status:** ✅ PASS

**Test commands:**
```bash
tj onboard --claude-code --reconfigure --plan api --budget 0 </dev/null
grep '^plan = ' ~/.config/tj/config.toml
```

**Output:**
```
plan = "api"
```

---

## Step 5: Run all five analyzers — no crashes

**Status:** ⚠️ UNCLEAR

**Test commands:**
```bash
tj optimize --since 30d
tj optimize --json | python3 -c "<assertion>"
```

**Output:**
```
Analyzing 7 sessions, 7.6M tokens (last 30d)
All sessions have unknown plan tier; dollar figures suppressed. Run tj onboard
--reconfigure to set your plan.

  Cache efficacy: ...
  Cache recommend: Enable [capture] prompts = true ...
  Workflow restructure: no clusters above threshold ...
  Reuse: No repeated planning detected ...
  Prompt bloat: Enable [capture] prompts = true ...

ok: all wave-2 analyzers in findings
keys: ['cache', 'cache-recommend', 'script', 'reuse', 'trim']
```

**Notes:** No tracebacks, no ERROR lines, plan-tier banner ("All sessions have unknown plan tier...") present. However, **no "Downsize" section appears** in the CLI output and `downsize` is **not** a key in `findings` JSON (only the 5 wave-2 analyzers). The checklist Expected lists 6 analyzers including Downsize. Plan tier is "unknown" on existing sessions despite `plan = "api"` in config (sessions pre-date the reconfigure), which likely suppresses the downsize section. Reviewer should confirm whether this is intended behavior in v0.4.0.

---

## Step 6: Reuse — the brand new analyzer

**Status:** ❌ FAIL

**Test commands:**
```bash
tj optimize reuse --since 30d
tj report --reuse --no-open
tj optimize --json | python3 -c "<reuse contract>"
```

**Output:**
```
tj optimize reuse:
  Reuse: No repeated planning detected above threshold ...
  (clean, no traceback)

tj report --reuse --no-open:
Error: tj report --reuse needs direct database access. Stop the daemon with `tj stop` and re-run, or query via the API.

assertion: ok: reuse contract intact
```

**Notes:** `tj optimize reuse` PASSES cleanly. `tj report --reuse --no-open` FAILS because the running daemon holds the DB write lock — the report subcommand does not have the daemon-fallback path that `tj optimize` has. Per the checklist Expected, the command should either write report files or print "No repeated planning detected" and exit 0. It instead prints an error and exits non-zero. The JSON contract assertion still passes.

---

## Step 7: `tj cost` shows cache R + cache W columns

**Status:** ⚠️ UNCLEAR

**Test commands:**
```bash
tj cost --since 30d --group-by model
curl -s "http://127.0.0.1:7391/api/v1/cost?since=30d&group_by=model" | python3 -c "<assertion>"
```

**Output:**
```
  MODEL                TOKENS IN   TOKENS OUT   CACHE R   CACHE W   COST
  claude-opus-4-8      294.2k      1.8M         0         0         $468.5259
  claude-opus-4-7      38.8k       3.3M         0         0         $1111.8593
  ...
                       369.1k      7.3M         0         0         $2117.0268
ok: cache_write surfaced in API
```

**Notes:** `CACHE R` and `CACHE W` header columns are present, totals row present, API assertion passes. However, every row shows `0` for both cache columns even though the Cache-efficacy analyzer in Step 5 reported `cache 2087.4M` tokens for opus-4-7. The schema is correct but the values don't reconcile with the cache analyzer's output. The checklist Expected says "if data exists, the cache columns are populated (not all zeros)". Needs human eyes on whether this is a known billing-column vs. analyzer-column distinction.

---

## Step 8: `tj onboard --plan` on plain path

**Status:** ❌ FAIL

**Test commands:**
```bash
TMP=$(mktemp -d) && cd "$TMP"
tj onboard --no-daemon --budget 0 --plan max_5x </dev/null
grep '^plan = ' .tj/config.toml
```

**Output:**
```
Config already exists: /Users/anilmurty/.config/tj/config.toml
Use --force to overwrite.
(exit 2)
ugrep: warning: .tj/config.toml: No such file or directory
```

**Notes:** Plain `tj onboard` aborts because the global `~/.config/tj/config.toml` already exists from the user's environment, so no local `.tj/config.toml` is written. The test playbook does not pass `--force` or `--reconfigure`, so this scenario cannot be exercised without potentially clobbering the user's real global config. Either the playbook needs a `--force` (with a save/restore of the global config) or v0.4.0's onboard logic for the plain path is silently broken when a global config already exists. Marking FAIL pending human review.

---

## Step 9: Lens UI — default route + footer version

**Status:** ✅ PASS

**Test commands:**
```bash
curl -s http://127.0.0.1:7391/ | grep -E '<title>|window.location.hash|tj-version' | head -5
curl -s http://127.0.0.1:7391/api/v1/version
python3 -c "<assertion>"
```

**Output:**
```
<title>TokenJam Lens</title>
<div class="version" id="tj-version">…</div>
{"version":"0.3.5"}
ok: brand + default route + drill-link guards intact
```

---

## Step 10: `/api/v1/optimize` carries the framing block + reuse

**Status:** ✅ PASS

**Test commands:**
```bash
curl -s "http://127.0.0.1:7391/api/v1/optimize?since=30d" | python3 -m json.tool | head -40
curl -s "http://127.0.0.1:7391/api/v1/optimize?since=30d" | python3 -c "<assertion>"
```

**Output:**
```
{
  "window": { "since": ..., "total_cost_usd": 2117.17, "thin_data": false },
  "downgrade": null,
  "findings": { "cache": {...}, "cache-recommend": ..., "script": ..., "reuse": ..., "trim": ... },
  ...
}
ok: framing + reuse + downgrade contract
```

**Notes:** Assertion passes. `downgrade` is null (consistent with Step 5 observation).

---

## Step 11: Backfill smoke — sanity check, no regression

**Status:** ✅ PASS

**Test commands:**
```bash
tj backfill langfuse --source-file tests/fixtures/langfuse_real_response.json
tj backfill helicone --source-file tests/fixtures/helicone_real_response.json
tj backfill otlp --source-file tests/fixtures/otlp_sample.json
```

**Output:**
```
✓ Read 4 observation(s); wrote 0 new span(s); skipped 4 already present.
✓ Read 4 record(s); wrote 0 new span(s); skipped 4 already present.
✓ Saw 3 span(s); wrote 0 new; skipped 3 already present; rejected 0.
```

**Notes:** All three idempotent re-runs (skipped, not rewritten). No tracebacks.

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
============================== 1 passed in 0.03s ===============================
```

---

## Final summary

**Recommendation:** Investigate — 1 FAIL + 3 UNCLEAR + version not bumped to 0.4.0

**Steps PASSED:**  2, 3, 4, 9, 10, 11, 12 (7)
**Steps FAILED:**  6 (tj report --reuse errors when daemon holds DB lock), 8 (plain tj onboard --plan refuses when global config exists)
**Steps UNCLEAR:** 1 (version still 0.3.5), 5 (no Downsize section/key — possibly intended when plan tier on existing sessions is "unknown"), 7 (CACHE R/W columns show 0 even though cache analyzer reports billions of cache tokens)

**Notable observations:**
- `pyproject.toml` still reads `0.3.5` — release engineer must bump per Critical Rule 15 before tagging v0.4.0.
- The `tj report --reuse` subcommand appears to lack the daemon HTTP fallback that `tj optimize` has — calling it while `tj serve` is running errors instead of degrading gracefully.
- `tj cost` cache R/W columns surface in the schema (assertion passes) but display zeros while cache-efficacy analyzer reports 2B+ cache tokens — possible data-source mismatch worth confirming.
- Existing session records have `plan_tier = unknown` even after `tj onboard --reconfigure --plan api`; reconfigure only updates config, not historical session rows — this suppresses Downsize and dollar-bearing renderings.
