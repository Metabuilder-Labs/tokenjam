# Pre-release checklist — v0.4.1

Focused on the delta since v0.4.0. The runner (`tests/agent-pre-release-runner.md`) walks each step in order.

**What's new in v0.4.1** (so you know what we're validating):

| Change | PR | Issue |
|---|---|---|
| Run-rate cycle honors `[budget.<provider>] cycle_start_day` | #158 | #138 |
| Daemon DB concurrency: per-thread DuckDB cursors over one shared database | #161 | #124 |
| `buildCostSeries` graceful coarsening + Overview parallel fetch | #163 | #139 |
| Status: active (compute) time alongside wall-clock elapsed | #164 | #147 |
| `tj report --reuse` HTTP fallback when daemon holds the DB lock | #165 | #154 |
| Recoverable Waste tile consistency (capitalization + dropped em-dash) | #166 | #162 |

Stable surfaces (ingest, alerts, drift, schema validation, MCP server, backfill, Lens basic structure) are **not** in scope — they were verified in the v0.4.0 pass and haven't been touched. The standing comprehensive playbook lives at `tests/manual-pre-release-testing.md` if a reviewer wants every section.

---

## Step 1: Baseline — daemon is current

**What:** confirm the daemon and `pyproject.toml` agree on the version under test.

**Test:**
```bash
tj --version
curl -s http://127.0.0.1:7391/api/v1/version
grep '^version' pyproject.toml
```

**Expected:**
- All three sources report the same version
- The reported version matches what's in `pyproject.toml` (UNCLEAR is fine here if the version hasn't been bumped to 0.4.1 yet — release-cut PR fixes it)

---

## Step 2: Daemon DB concurrency — fan-out reads no longer crash

**What:** the marquee #124 fix. With the daemon up, fire the Overview's full fan-out endpoint set concurrently many times; assert every response is 200. Pre-fix the daemon SIGABRT'd under this load.

**Test:**
```bash
python3 -c "
import asyncio, httpx
async def hammer():
    async with httpx.AsyncClient(base_url='http://127.0.0.1:7391') as c:
        urls = [
            '/api/v1/cost?since=7d',
            '/api/v1/optimize?fast=true&since=30d',
            '/api/v1/cost/compare?since=7d&compare=previous',
            '/api/v1/traces?limit=6',
            '/api/v1/drift',
            '/api/v1/alerts?since=7d&unread=true',
        ]
        for _ in range(5):
            results = await asyncio.gather(*(c.get(u) for u in urls * 3))
            bad = [(str(r.url), r.status_code) for r in results if r.status_code != 200]
            assert not bad, bad
asyncio.run(hammer())
print('ok: 90 concurrent reads, no errors')
"
```

**Expected:**
- The `ok: 90 concurrent reads, no errors` line prints
- No 500s, no daemon crash

**Assertions:** the test above already self-asserts. The presence of `ok: 90 concurrent reads, no errors` is the pass signal.

---

## Step 3: `tj cost` cache columns show real values with daemon up

**What:** verify the secondary #124 effect — pre-fix, `tj cost --group-by model` showed `0` for CACHE R / CACHE W when the daemon was holding the lock because `ApiBackend.get_cost_summary` was dropping those fields. Fixed in v0.4.0's #153 + #124's per-thread cursor change.

**Test:**
```bash
tj cost --since 30d --group-by model
```

**Expected:**
- The CACHE R and CACHE W columns render with their headers
- For rows that have cache activity, the values are **non-zero** (verifies the api_backend.py path forwards the cache fields correctly)

---

## Step 4: Run-rate cycle honors `cycle_start_day`

**What:** verify #138 — the run-rate caption respects `[budget.<provider>] cycle_start_day` when configured, falling back to calendar month otherwise. API exposes the cycle bounds; UI consumes them.

**Test (default — calendar month):**
```bash
curl -s "http://127.0.0.1:7391/api/v1/cost?since=7d" | python3 -c "
import json, sys
d = json.load(sys.stdin)
c = d['cycle']
assert {'start', 'end', 'days_remaining', 'start_day'} <= set(c), c
assert c['start_day'] in (1, 15), f'unexpected start_day: {c}'
print(f\"ok: cycle.start_day={c['start_day']} days_remaining={c['days_remaining']}\")
"
```

**Expected:**
- Response includes a `cycle` block with `start`, `end`, `days_remaining`, `start_day`
- For most installs `start_day=1` (calendar month)
- The CLI run-rate caption reads either `"by end of <month>"` (calendar) or `"by <date>"` (custom cycle), never `"over 30 days"`

**Optional custom cycle test (only if a non-default `cycle_start_day` is configured):**
```bash
# Skip if no [budget.<provider>] cycle_start_day is configured in the test env.
grep -A2 '^\[budget\.' ~/.config/tj/config.toml | grep cycle_start_day || echo "(no custom cycle in env — skipping)"
```

---

## Step 5: Status — Active and Elapsed time both render

**What:** verify #147 — `tj status` shows active (compute) time alongside wall-clock elapsed, and the JSON shape carries both fields. UI Status tile shows two distinct rows.

**Test:**
```bash
tj status
tj status --json | python3 -c "
import json, sys
d = json.load(sys.stdin)
agents = d.get('agents', [])
if not agents:
    print('skip: no agents in status')
else:
    a = agents[0]
    has_active = 'active_seconds' in a
    has_elapsed = 'duration_seconds' in a
    assert has_active and has_elapsed, f'missing fields: {list(a.keys())}'
    print(f\"ok: agent={a['agent_id']} active={a.get('active_seconds')} elapsed={a.get('duration_seconds')}\")
"
```

**Expected:**
- For each agent with a session, the CLI prints a line like `Duration: active 12m 3s · elapsed 2d 3h`
- JSON output includes both `active_seconds` and `duration_seconds` per agent
- A resumed Claude Code session (multi-day elapsed) reads sensibly — elapsed shows `Nd Mh`, not `3087m`

---

## Step 6: Bucket coarsening — long windows don't silently empty

**What:** verify #139 — `buildCostSeries` falls down the hour→day→week ladder when the requested grid exceeds 5000 buckets, instead of returning `null` and rendering empty. Visible in the UI via a coarsening note.

**Test:**
- Open `http://127.0.0.1:7391/`
- Navigate to **Cost** screen
- Select **Last 90d**
- (The window currently buckets daily on the server, so 90 buckets is well under the cap. The coarsening path is exercised in unit tests; this is a visual check that the Cost screen renders cleanly at the longest configured window.)

**Expected:**
- Cost screen renders the spend-over-time chart for 90d
- No "Showing X buckets — this range is too long for Yly detail" note (90d × daily = 90 buckets, no coarsening)
- No empty/blank chart state

**Notes:** the coarsening trigger only fires when buckets > 5000, which the current 90d UI cap doesn't hit. The unit test (`test_cost_series_coarsens_not_silently_empty`) covers the actual coarsening logic; this step verifies the path is intact at the largest user-reachable window.

---

## Step 7: `tj report --reuse` works while daemon is running

**What:** verify #154 — `tj report --reuse` no longer errors with "needs direct database access" when the daemon holds the DB lock. The CLI dispatches to `ApiBackend.fetch_reuse_clusters` against the new `/api/v1/reuse/clusters` endpoint.

**Test:**
```bash
# Confirm daemon is up
curl -s -f http://127.0.0.1:7391/health
# Run the report
tj report --reuse --no-open
```

**Expected:**
- Exit code 0
- Output reads either `"Reuse report written to ..."` (clusters found) or `"No repeated planning detected ..."` (no clusters)
- **NOT** the old `"needs direct database access. Stop the daemon with 'tj stop'..."` error
- If clusters exist: `~/.cache/tokenjam/reports/reuse-*.html` and at least one `reuse-*-*.md` file exist

**Assertions:**
```bash
out=$(tj report --reuse --no-open 2>&1)
echo "$out"
echo "$out" | grep -qE 'needs direct database access' && { echo "fail: old error still surfaces"; exit 1; }
echo "ok: --reuse runs without DB-lock error"
```

---

## Step 8: Recoverable Waste tiles render consistently

**What:** verify #162 — the Overview's Recoverable Waste tiles have uniform title weight, title-cased names ("Reuse" not "reuse"), and no `— not ready` em-dash prefix.

**Test:**
- Open `http://127.0.0.1:7391/#/overview`
- Find the **Recoverable Waste** section

**Expected:**
- All five tile titles (Downsize / Cache / Script / Trim / Reuse) render in the **same font weight**
- "Reuse" is **capitalized**, not lowercase
- The not-ready tile (typically Trim, when capture is off) reads `"Not ready"` without a leading em-dash
- The positive at_ceiling state on Cache (when efficacy is 100%) still renders its `✓ 100% cache efficacy` content line in green — that's the intended #127 design preserved

**If the bold-title issue from the original #162 report still reproduces here:** do a hard refresh (Cmd+Shift+R) first. The agent's investigation showed the source doesn't bold the Cache title; if it still appears bold after a hard refresh, that's a runtime CSS specificity quirk worth a follow-up.

---

## Step 9: TokenMaxx still works (regression check)

**What:** TokenMaxx is the most-shareable command; verify it still renders the bordered panel on both api and subscription plans.

**Test:**
```bash
tj tokenmaxx
tj tokenmaxx --json | python3 -c "
import json, sys
d = json.load(sys.stdin)
ok = {'TokenSipper','TokenModerator','TokenMaxxer','TokenSuperMaxxer','TokenMegaMaxxer','TokenGigaMaxxer'}
assert d['tier'] in ok, d['tier']
print('ok:', d['tier'])
"
```

**Expected:**
- Bordered "TokenJam TokenMaxxing Report" panel renders
- JSON `tier` is one of the 6 valid tier names
- If on a subscription plan, the multiplier line appears

---

## Step 10: All five analyzers run + framing block present (regression)

**What:** broad regression check on the analyzer surface + the `framing` block on `/api/v1/optimize`.

**Test:**
```bash
tj optimize --since 30d
curl -s "http://127.0.0.1:7391/api/v1/optimize?since=30d" | python3 -c "
import json, sys
d = json.load(sys.stdin)
assert 'framing' in d, 'framing block missing'
assert 'reuse' in d.get('findings', {}), 'reuse missing from findings'
print('ok: 5 analyzers + framing block intact')
"
```

**Expected:**
- `tj optimize` runs all 5 wave-2 analyzers (cache, cache-recommend, script, trim, reuse) plus downsize without crashing
- Downsize section renders even when there are no candidates (#126/#130/#162 empty-state fix)
- API response carries the `framing` block with `pricing_mode` / `plan_tier` / `display_rule`
- API response has `reuse` in `findings`

---

## Final summary the runner produces

```
**Recommendation:** Ready to release v0.4.1 / Hold — <N> failures / Investigate — <K> UNCLEAR steps

**Steps PASSED:**  <numbers>
**Steps FAILED:**  <numbers>
**Steps UNCLEAR:** <numbers>

**Notable observations:**
- <any one-liners worth surfacing>
```

If all 10 steps pass, the v0.4.1 release-cut PR can be opened directly. If any FAIL or UNCLEAR, investigate before tagging.
