# Fresh-install release smoke checklist

A lightweight pre-release gate so **plan-tier / framing regressions can't ship
silently**. The 0.5.0 first-run review burned real time chasing a "Max 20x"
framing that turned out to be stale data, not a code bug — there was no
automated *or* manual signal either way. Run this once against a clean machine
(or a throwaway VM / container) after publishing a release candidate and before
announcing the release.

This is distinct from [`tests/manual-pre-release-testing.md`](../../tests/manual-pre-release-testing.md),
which tests a **branch** via an editable install. This checklist tests the
**published wheel** the way a brand-new user installs it: `pipx install tokenjam`.

The automated counterpart to this checklist is
`tests/integration/test_first_run_roundtrip_239.py`, which pins the
config → backfill → framing contract in CI. This checklist covers the parts CI
can't: a real `pipx` install, a real `tj onboard --claude-code`, and the
rendered Lens badge.

## 0. Clean environment

Start from a machine (or container / VM) with **no prior TokenJam state**. On a
machine you've used before, wipe it first:

```bash
tj stop 2>/dev/null            # halt any daemon holding the DB / port 7391
tj uninstall --yes 2>/dev/null # remove launchd / systemd unit files
pipx uninstall tokenjam 2>/dev/null
pip3 uninstall -y tokenjam 2>/dev/null
rm -rf ~/.tj ~/.config/tj .tj  # default DB (~/.tj/telemetry.duckdb), global + project config
```

Verify nothing lingers:

```bash
launchctl list | grep tokenjam      # macOS — expect no output
systemctl --user is-active tokenjam # Linux — expect "inactive"/"unknown"
which tj                            # expect "not found"
```

## 1. Fresh install

```bash
pipx install tokenjam
tj --version          # matches the release tag (vX.Y.Z)
```

`pipx`, not `pip` — that's the recommended path (sidesteps PEP 668 on Homebrew
Python and Debian 12+/Ubuntu 24+).

## 2. Onboard with a known plan

```bash
tj onboard --claude-code --plan max_5x
```

Use `--plan max_5x` (non-interactive) so the expected framing is unambiguous.
This writes `[budget.anthropic] plan = "max_5x"` to the **global** config
(`~/.config/tj/config.toml`) and auto-runs the Claude Code backfill at the end.

Confirm the config landed:

```bash
grep -A2 'budget.anthropic' ~/.config/tj/config.toml   # plan = "max_5x"
```

> If this machine has no `~/.claude/projects/*.jsonl` history, seed a little by
> running Claude Code a few times first — otherwise the backfill finds nothing
> and the framing falls back to the config-declared plan (still `max_5x`, but
> you won't exercise the data path).

## 3. Verify the backfill session count is sane

The onboard backfill prints a summary. Sanity-check it against the table — the
two must reconcile (the #238 contract):

```bash
tj status                      # session count, span count, window
# Compare against the printed "N new / M existing · T total" summary.
```

Red flags: a "1 session" summary on a machine with months of history (new-only
reporting — #238), or a session count wildly off from the spans/cost shown.

## 4. Verify the Lens plan badge matches the config

```bash
tj serve &                     # or rely on the daemon installed by onboard
open http://localhost:7391/    # macOS; otherwise browse to it
```

On the **Overview** screen:

- The plan badge / framing reads **Max 5x plan** (subscription), **not** Max 20x
  or an API-dollar framing.
- Cost figures render as token-share / "% of cycle", **not** raw API dollars
  (subscription users never see "spend").

Optionally confirm the raw framing block:

```bash
curl -s 'http://localhost:7391/api/v1/cost?since=90d' | python3 -m json.tool | grep -A12 '"framing"'
# expect: "pricing_mode": "subscription", "plan_tier": "max_5x", "plan_monthly_usd": 100.0
```

## 5. Verify `tj optimize` agrees with Lens

```bash
tj optimize --since 90d
```

- The framing must match the Lens badge: subscription / Max 5x, **implied API
  value** language, token-share savings — never dollar "spend" or a different
  plan than step 4 showed.
- Savings lines say "estimated recoverable", never "saves you" (Critical Rule 14).

If Lens and `tj optimize` disagree on the plan, **stop** — that's the exact
class of regression this gate exists to catch. File a blocker before releasing.

## Pass criteria

- [ ] Clean machine → `pipx install tokenjam` succeeds; `tj --version` == release tag
- [ ] `tj onboard --claude-code --plan max_5x` writes `plan = "max_5x"` to global config
- [ ] Backfill summary reconciles with `tj status` (new/existing/total is sane, not new-only)
- [ ] Lens Overview badge reads **Max 5x** (subscription framing, no API dollars)
- [ ] `tj optimize` framing agrees with Lens (same plan, implied-value language)
