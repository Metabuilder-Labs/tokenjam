# Post-release checklist — v0.4.0

Focused on verifying the **published artifact** (PyPI) actually installs and the v0.4.0-new surfaces work end-to-end against a real user-style install. The runner (`tests/agent-post-release-runner.md`) walks each step in order with an isolated `$HOME`, so this won't touch the developer's real install.

The comprehensive human playbook is at `tests/manual-new-release-tests.md` if a reviewer wants every section. This file is the agent-runnable subset.

**What's new in v0.4.0** (so you know what we're validating):
- TokenJam Lens UI rebrand (Overview + Optimize + real charts)
- Reuse — 5th analyzer + `tj report --reuse` artifact export
- `cache_write_tokens` surfaced through DB → API → CLI → UI
- `core/framing.py` — plan-tier-aware rendering across CLI + API
- Onboard `--plan` honored on the plain path (#148)
- `tj mcp` works without the `[mcp]` extra (fastmcp in base deps, #131)

---

## Step 1: Install from PyPI

**What:** the published v0.4.0 artifact installs cleanly via the recommended path.

**Setup:**
```bash
# Runner has already set HOME to an isolated tmp dir
pipx uninstall tokenjam 2>/dev/null || true   # safe: isolated HOME has no prior install
```

**Test:**
```bash
pipx install tokenjam
"$HOME/.local/bin/tj" --version
```

**Expected:**
- `pipx install` succeeds (writes `installed package tokenjam 0.4.0`)
- `tj --version` reports `tj, version 0.4.0`
- No "Installing to existing venv" message — the isolated HOME guarantees a fresh venv

**Assertions:**
```bash
"$HOME/.local/bin/tj" --version | grep -q "0.4.0" && echo "ok: 0.4.0 installed"
```

---

## Step 2: First-run onboard

**What:** plain `tj onboard --plan` works on a fresh install — covers #148 (the plain-path plan tier fix).

**Test:**
```bash
"$HOME/.local/bin/tj" onboard --no-daemon --budget 0 --plan max_5x </dev/null
grep '^plan = ' "$HOME/.config/tj/config.toml" || \
  grep '^plan = ' .tj/config.toml
```

**Expected:**
- Command exits 0
- Either `~/.config/tj/config.toml` or `.tj/config.toml` exists and contains `plan = "max_5x"` under `[budget.anthropic]`
- All 4 capture toggles are present (`prompts`, `completions`, `tool_inputs`, `tool_outputs`)
- No `openclawwatch` substring anywhere in the written config

**Assertions:**
```bash
python3 -c "
import pathlib, os
candidates = [pathlib.Path.home() / '.config/tj/config.toml', pathlib.Path('.tj/config.toml')]
cfg = next((p for p in candidates if p.exists()), None)
assert cfg, 'no config file found'
t = cfg.read_text()
assert 'plan = \"max_5x\"' in t
assert all(f'{k} = false' in t for k in ('prompts','completions','tool_inputs','tool_outputs'))
assert 'openclawwatch' not in t
print('ok: onboard writes plan + 4 capture toggles + no stale URL')
"
```

---

## Step 3: `tj mcp` is importable out of the box (no extra needed)

**What:** verifies the #101 fix (fastmcp moved into base deps) still holds. Pre-fix, `tj mcp` errored on fresh installs unless the user remembered `pipx install 'tokenjam[mcp]'`.

**Test:**
```bash
"$HOME/.local/bin/tj" mcp --help
```

**Expected:**
- Exit code 0
- Help text describes the MCP stdio server
- No `ImportError`, no "fastmcp not installed" message

**Assertions:**
```bash
"$HOME/.local/bin/tj" mcp --help 2>&1 | grep -qi "mcp\|claude code\|stdio" && echo "ok: tj mcp loadable"
```

---

## Step 4: `tj optimize` runs all 5 analyzers + carries the framing block

**What:** even with no data, `tj optimize` should exit cleanly and surface a friendly "no data" message; the JSON output should carry the v0.4.0 contract (framing block, all 5 analyzer slots).

**Test:**
```bash
"$HOME/.local/bin/tj" optimize --since 30d --json
```

**Expected:**
- Exit code 0
- Output is valid JSON with either:
  - an `error: "no_data"` shape (fresh install, no spans yet), OR
  - a full report with `findings` containing `reuse`, `cache`, `script`, `trim`, `cache-recommend` keys + a top-level `framing` object
- No Python tracebacks

**Assertions:**
```bash
"$HOME/.local/bin/tj" optimize --since 30d --json | python3 -c "
import json, sys
d = json.load(sys.stdin)
if d.get('error') == 'no_data':
    print('ok: no_data shape (expected on fresh install)')
else:
    f = d.get('findings', {})
    assert 'reuse' in f, 'reuse missing'
    assert 'framing' in d, 'framing block missing'
    print('ok: contract intact (findings + framing)')
"
```

---

## Step 5: `tj cost` shows the CACHE R / CACHE W column headers

**What:** verify the cost-transparency fix from #149 is in the published artifact. Even with no data, the table header should list the cache columns.

**Test:**
```bash
"$HOME/.local/bin/tj" cost --since 30d --group-by model
```

**Expected:**
- Either:
  - a message indicating no cost data (fresh install), OR
  - a table whose header row includes `CACHE R` and `CACHE W` alongside `TOKENS IN` / `TOKENS OUT`
- No tracebacks

**Notes:** populated CACHE values can't be verified here (no real data), but the column header presence is the published-artifact check. The pre-release pass already verified populated values.

---

## Step 6: `tj tokenmaxx` works on a subscription plan tier

**What:** the marquee shareable command — verify it renders the bordered panel and the multiplier line.

**Test:**
```bash
"$HOME/.local/bin/tj" tokenmaxx
"$HOME/.local/bin/tj" tokenmaxx --json
```

**Expected:**
- Either a bordered "TokenJam TokenMaxxing Report" panel, OR a friendly "not enough data" message
- JSON output is valid; if data exists, `tier` is one of the 6 valid tier names

**Assertions:**
```bash
"$HOME/.local/bin/tj" tokenmaxx --json | python3 -c "
import json, sys
d = json.load(sys.stdin)
ok = {'TokenSipper','TokenModerator','TokenMaxxer','TokenSuperMaxxer','TokenMegaMaxxer','TokenGigaMaxxer'}
tier = d.get('tier')
# Fresh install may report 'unknown' or omit tier — that's OK
assert tier is None or tier in ok or 'spend' in d, f'unexpected tokenmaxx shape: {d}'
print('ok: tokenmaxx shape valid')
"
```

---

## Step 7: `tj serve` starts and the Lens UI renders "TokenJam Lens"

**What:** the marquee UI rebrand — verify `/` returns HTML titled "TokenJam Lens" and `/api/v1/version` reports 0.4.0.

**Setup:**
```bash
"$HOME/.local/bin/tj" serve > "$HOME/tj-serve.log" 2>&1 &
TJ_SERVE_PID=$!
sleep 4
```

**Test:**
```bash
curl -s -f http://127.0.0.1:7391/health
curl -s http://127.0.0.1:7391/api/v1/version
curl -s http://127.0.0.1:7391/ | grep -oE '<title>[^<]+</title>'
```

**Expected:**
- `/health` returns `{"status":"ok","version":"0.4.0"}`
- `/api/v1/version` returns `{"version":"0.4.0"}`
- The HTML `<title>` is exactly `<title>TokenJam Lens</title>`
- No connection refused / no 500s

**Assertions:**
```bash
curl -s "http://127.0.0.1:7391/api/v1/version" | python3 -c "
import json, sys
d = json.load(sys.stdin)
assert d['version'] == '0.4.0', f'wrong version: {d}'
print('ok: published 0.4.0 daemon responding')
"
curl -s http://127.0.0.1:7391/ | python3 -c "
import sys
html = sys.stdin.read()
assert '<title>TokenJam Lens</title>' in html, 'brand title missing'
assert '#/overview' in html or 'overview' in html.lower(), 'overview default route missing'
print('ok: Lens brand + overview default route')
"
```

**Teardown:**
```bash
kill $TJ_SERVE_PID 2>/dev/null || true
"$HOME/.local/bin/tj" stop 2>/dev/null || true
```

---

## Step 8: `tj uninstall` prints the correct pipx hint

**What:** verifies the #131 fix — `tj uninstall` detects pipx installs and prints `pipx uninstall tokenjam` (not `pip uninstall`).

**Test:**
```bash
"$HOME/.local/bin/tj" uninstall --yes 2>&1 | tail -5
```

**Expected:**
- The command exits 0 (or 1; both are OK if it had nothing to clean)
- The final hint reads "**To remove the package itself, run: pipx uninstall tokenjam**" (or similar with `pipx`)
- NOT `pip uninstall tokenjam` — that was the pre-#131 wording and won't work on a pipx install

**Assertions:**
```bash
out=$("$HOME/.local/bin/tj" uninstall --yes 2>&1 || true)
echo "$out" | grep -qE "pipx uninstall tokenjam" && echo "ok: pipx-aware uninstall hint"
```

---

## Step 9: pipx-injected provider SDK is importable through the pipx venv interpreter

**What:** verifies the #131 documented pattern — provider SDKs (e.g. `anthropic`) injected via `pipx inject tokenjam` are reachable from the pipx venv's Python. This is how users run the example scripts.

**Test:**
```bash
pipx install --force tokenjam   # re-install since Step 8 may have removed it
pipx inject tokenjam anthropic
PIPX_PY="$(pipx environment --value PIPX_LOCAL_VENVS)/tokenjam/bin/python"
"$PIPX_PY" -c "import anthropic; print('ok: anthropic importable via pipx venv')"
```

**Expected:**
- `pipx inject` succeeds without errors
- The pipx venv interpreter can `import anthropic` cleanly
- The `ok:` print appears

**Notes:** This step hits PyPI again (for `anthropic`), so it may take 10–30 seconds.

---

## Step 10: artifact files ship in the wheel

**What:** verify the v0.4.0 artifact actually ships the Lens UI files + pricing TOML + vendored uPlot. Catches the class of bugs where `[tool.hatch.build.targets.wheel]` somehow excludes them.

**Test:**
```bash
PIPX_PKG_ROOT="$(pipx environment --value PIPX_LOCAL_VENVS)/tokenjam/lib"
find "$PIPX_PKG_ROOT" -path "*/tokenjam/ui/index.html" -o -path "*/tokenjam/ui/vendor/uplot.js" -o -path "*/tokenjam/pricing/models.toml" | head
```

**Expected:**
- All three paths exist under the installed package
- `uplot.js` is non-trivially sized (>10 KB — confirms it's the real vendored library, not a placeholder)

**Assertions:**
```bash
PIPX_PKG_ROOT="$(pipx environment --value PIPX_LOCAL_VENVS)/tokenjam/lib"
INDEX=$(find "$PIPX_PKG_ROOT" -path "*/tokenjam/ui/index.html" | head -1)
UPLOT=$(find "$PIPX_PKG_ROOT" -path "*/tokenjam/ui/vendor/uplot.js" | head -1)
PRICING=$(find "$PIPX_PKG_ROOT" -path "*/tokenjam/pricing/models.toml" | head -1)
[ -f "$INDEX" ] && [ -f "$UPLOT" ] && [ -f "$PRICING" ] && \
  [ "$(wc -c < "$UPLOT")" -gt 10000 ] && \
  echo "ok: UI + uPlot + pricing.toml ship in the wheel"
```

---

## Final summary the runner produces

```
**Recommendation:** Release v0.4.0 is healthy / Hold — <N> failures / Investigate — <K> UNCLEAR steps

**Steps PASSED:**  <numbers>
**Steps FAILED:**  <numbers>
**Steps UNCLEAR:** <numbers>

**Notable observations:**
- <any one-liners worth surfacing>

HOME restored. Isolated dir removed: <path>
```

If all 10 steps PASS, the published artifact is verified end-to-end and the release is healthy. If anything FAILs, file an issue, decide whether to yank the release from PyPI/npm, and either patch or roll forward.
