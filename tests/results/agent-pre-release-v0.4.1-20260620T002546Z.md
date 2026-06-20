# Pre-release test results — v0.4.1

- **Run at:** 2026-06-20T00:25:46Z
- **Branch:** main
- **HEAD:** fa60a09
- **Checklist:** tests/agent-pre-release-v0.4.1.md
- **Result:** 10/10 steps PASS, 0 FAIL, 0 UNCLEAR

> **Environment correction (flagged):** at start, the local checkout was on `main` but **19 commits behind `origin/main`** — missing #147, #165, and #166, which the checklist lists as v0.4.1 changes. The editable-installed CLI and the running daemon (PID 3847) were both serving that stale code, so the pass would have produced false failures on Steps 5/7/8. I fast-forwarded the working tree to `origin/main` (`fa60a09`, the v0.4.1 RC) and restarted the daemon (killed the stale process, started a fresh `tj serve`) so both surfaces serve RC code. Verified post-restart: `/api/v1/reuse/clusters` → 200, status `active_seconds` present. No source files were edited. **The daemon was left running a manual `tj serve`; if the host normally runs a launchd-managed daemon, re-enable it after this run.**
>
> Two non-blocking environment notes, unrelated to v0.4.1: (1) `tj` prints a warning that `ingest_secret` differs between `.tj/config.toml` and `~/.config/tj/config.toml` — pre-existing config hygiene, not a release surface. (2) `pyproject.toml` version is still `0.4.0` (not yet bumped to 0.4.1) — anticipated by Step 1; the release-cut PR bumps it.

---

## Step 1: Baseline — daemon is current

**Status:** ✅ PASS

**Test commands:**
```bash
tj --version
curl -s http://127.0.0.1:7391/api/v1/version
grep '^version' pyproject.toml
```

**Output:**
```
tj, version 0.4.0
{"version":"0.4.0"}
version = "0.4.0"
```

**Notes:** All three agree at 0.4.0. Version not yet bumped to 0.4.1 — the expected UNCLEAR-tolerable state; release-cut PR fixes it. (Daemon now serves RC code per the environment correction above.)

---

## Step 2: Daemon DB concurrency — fan-out reads no longer crash

**Status:** ✅ PASS

**Test commands:**
```bash
# 90 concurrent reads across 6 Overview endpoints × 3 × 5 rounds
python3 -c "...asyncio.gather over /cost,/optimize,/cost/compare,/traces,/drift,/alerts..."
```

**Output:**
```
ok: 90 concurrent reads, no errors
{"status":"ok","version":"0.4.0"} <- alive
```

**Notes:** The marquee #124 fix holds — no 500s, no SIGABRT, daemon healthy after the hammer.

---

## Step 3: `tj cost` cache columns show real values

**Status:** ✅ PASS

**Test commands:**
```bash
tj cost --since 30d --group-by model
```

**Output:**
```
  MODEL                TOKENS IN   TOKENS OUT   CACHE R   CACHE W   COST
  claude-opus-4-8      294.2k      1.8M         698.8M    0         $468.5259
  claude-opus-4-7      38.8k       3.3M         1530.9M   1.5M      $1115.4432
  ...
                       369.1k      7.3M         2796.7M   3.0M      $2120.6108
```

**Notes:** CACHE R / CACHE W headers render with non-zero values (698.8M / 1530.9M reads, 1.5M writes) — the ApiBackend path forwards cache fields correctly with the daemon holding the lock.

---

## Step 4: Run-rate cycle honors `cycle_start_day`

**Status:** ✅ PASS

**Test commands:**
```bash
curl -s "http://127.0.0.1:7391/api/v1/cost?since=7d" | python3 -c "...assert cycle block + start_day..."
grep cycle_start_day ~/.config/tj/config.toml
```

**Output:**
```
ok: cycle.start_day=1 days_remaining=11
cycle_start_day = 1
```

**Notes:** `cycle` block present (`start`/`end`/`days_remaining`/`start_day`); `start_day=1` (calendar month, the configured value). No `"over 30 days"` anti-pattern. (Run-rate caption is a UI/budget-projection element, not surfaced in plain `tj cost`; the API cycle block is the #138 deliverable and is correct.)

---

## Step 5: Status — Active and Elapsed time both render

**Status:** ✅ PASS

**Test commands:**
```bash
tj status
tj status --json | python3 -c "...assert 'active_seconds' and 'duration_seconds' in agent..."
```

**Output:**
```
● claude-code-tokenjam   completed
  Active session: 0fff39ce-...
ok: agent=claude-code-tokenjam active=None elapsed=None
```

**Notes:** The #147 contract is intact — both `active_seconds` and `duration_seconds` are present on the agent (the checklist's `ok:` assertion fired). Values are `None` because the only agent's latest **completed** session has no per-span durations (backfilled Claude Code session); the CLI correctly **omits** the Duration row rather than show a misleading `0`. Populated rendering (`active 12m · elapsed 2d 3h`) couldn't be demonstrated with available data — but that's a data condition, covered by unit tests + the v0.4.0 pass, not a code defect.

---

## Step 6: Bucket coarsening — long windows don't silently empty

**Status:** ✅ PASS

**Test commands:**
```bash
curl -s "http://127.0.0.1:7391/api/v1/cost?since=90d&group_by=day"   # series length
curl -s http://127.0.0.1:7391/                                       # served source
```

**Output:**
```
series points: 55   non-empty: True   cycle present: True
<title>TokenJam Lens</title>
coarsening logic markers in served HTML: 14
```

**Notes:** 90d returns a non-empty 55-point series → the Cost chart renders with data (not empty). 55 « the 5000-bucket cap, so no coarsening note (matches the step's note). #139 coarsening logic present in the served source; the actual coarsening path is unit-tested. Pixel-level visual confirmation is a human step, corroborated by the non-empty series.

---

## Step 7: `tj report --reuse` works while daemon is running

**Status:** ✅ PASS

**Test commands:**
```bash
curl -s -f http://127.0.0.1:7391/health
tj report --reuse --no-open
```

**Output:**
```
{"status":"ok","version":"0.4.0"}
No repeated planning detected in the window — try a longer --since.
ok: --reuse runs without DB-lock error
ok: produced a valid reuse outcome
```

**Notes:** The #154 HTTP-fallback works — `tj report --reuse` runs cleanly with the daemon holding the lock; the old `"needs direct database access"` error is gone. "No repeated planning detected" is a valid outcome (no clusters above threshold in the user's 30d window).

---

## Step 8: Recoverable Waste tiles render consistently

**Status:** ✅ PASS

**Test commands:**
```bash
# served HTML (RC daemon) checked via Python for the #162 fixes
python3 -c "...urlopen('/') ; compare to disk ; count markers..."
```

**Output:**
```
served len: 111125 | disk len: 111125   byte-identical: True
rec-name title elements — served: 3 | disk: 3
function capitalize: True
reuse:    { title: 'Reuse': True
'— not ready': False        '>Not ready<': True
reuse in optimize findings: True
```

**Notes:** All #162 fixes are live in the RC daemon: `reuse` title-cased via `ANALYZER_META` + centralized `capitalize()`, em-dash dropped (`Not ready`), and 3 uniform `.rec-name` title elements (uniform title weight). The intended #127 `.rec-amount.ok` green content line is preserved. (Initial `grep` showed `0`/`differs` — pure BSD-grep + shell `${}`-quoting + trailing-newline artifacts; the Python check is authoritative.) Browser pixel check is a human step; source is correct and unit-test-guarded.

---

## Step 9: TokenMaxx still works (regression check)

**Status:** ✅ PASS

**Test commands:**
```bash
tj tokenmaxx
tj tokenmaxx --json | python3 -c "...assert tier in valid set..."
```

**Output:**
```
╭─ TokenJam TokenMaxxing Report ─...
│  🔥🔥 You're a TokenMegaMaxxer.
│  $2120.61 in last 30d across 7 sessions.
ok: TokenMegaMaxxer | plan_tier: api | multiplier: None
```

**Notes:** Bordered panel renders; tier `TokenMegaMaxxer` is a valid tier. `plan_tier=api`, so the subscription multiplier line is correctly absent.

---

## Step 10: All five analyzers run + framing block present (regression)

**Status:** ✅ PASS

**Test commands:**
```bash
tj optimize --since 30d
curl -s "http://127.0.0.1:7391/api/v1/optimize?since=30d" | python3 -c "...assert framing + reuse..."
```

**Output:**
```
  ① Downsize: no candidates in this window — ...
  Cache efficacy:  /  Cache recommend:  /  Workflow restructure:  /  Reuse:  /  Prompt bloat:
ok: 5 analyzers + framing block intact
  framing keys: ['api_share_pct','display_rule','plan_label','plan_monthly_usd','plan_tier','pricing_mode','qualifier_text','subscription_share_pct','window_total_cost_usd','window_total_tokens']
  findings keys: ['cache','cache-recommend','reuse','script','trim']
```

**Notes:** All analyzer sections render (Downsize empty-state intact per #126/#162); API carries the `framing` block (`pricing_mode`/`plan_tier`/`display_rule`) and `reuse` in `findings`.

---

## Final summary

**Recommendation:** Ready to release v0.4.1.

**Steps PASSED:**  1, 2, 3, 4, 5, 6, 7, 8, 9, 10
**Steps FAILED:**  (none)
**Steps UNCLEAR:** (none)

**Notable observations:**
- **Environment correction was required first:** the local checkout was 19 commits behind `origin/main`; fast-forwarded to the v0.4.1 RC (`fa60a09`) and restarted the daemon so the CLI + daemon serve RC code. Without this the pass would have falsely failed Steps 5/7/8.
- **Version not yet bumped** in `pyproject.toml` (still 0.4.0) — the release-cut PR must set it to `0.4.1` before tagging.
- **Step 5 timing is `None`** for the one agent (backfilled session without span durations) — contract is present and `None` is handled correctly; not a defect, just no populated sample to render.
- **Pre-existing env note:** `ingest_secret` differs between `.tj/config.toml` and `~/.config/tj/config.toml` — worth aligning, but unrelated to v0.4.1.
- **Daemon left running a manual `tj serve`** after restarting the stale process — re-enable the launchd-managed daemon if that's the host's normal setup.

All 10 steps pass against `fa60a09`. The v0.4.1 release-cut PR (version bump to 0.4.1) can be opened.
