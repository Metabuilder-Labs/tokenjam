#!/usr/bin/env node
/*
 * Zero-install launcher for TokenJam (`npx tokenjam`), issue #6.
 *
 * TokenJam's CLI is a Python package (`tokenjam`, command `tj`). This thin npm
 * wrapper exists for one reason: the Claude Code / ccusage crowd reaches for
 * `npx <tool>` first. `npx tokenjam` here resolves a Python launcher with NO pip env,
 * NO daemon, NO onboarding — it shells out to the Python CLI via the first
 * available runner and hands every argument straight through. Bare `npx tokenjam`
 * (no subcommand) routes to `tj quickstart`: where your Claude Code quota goes,
 * from the same ~/.claude/projects/*.jsonl files ccusage reads, in one command.
 *
 * Runner preference (first that exists wins):
 *   1. `uvx --from tokenjam tj …`  — fully ephemeral, downloads nothing global
 *   2. `pipx run --spec tokenjam tj …`
 *   3. `tj …`                      — an already-installed CLI on PATH
 *
 * If none are present we print actionable install guidance and exit non-zero.
 *
 * Note: tokenjam >=0.5.4 also ships a `tokenjam` console-script alias
 * alongside `tj`, so bare `uvx tokenjam` / `pipx run tokenjam` work too. This
 * wrapper keeps the explicit `--from tokenjam tj` / `--spec tokenjam tj` form
 * below for back-compat with the 0.5.3 and earlier releases it also targets.
 *
 * Freshness (issue #111): `uv` reuses its cached tool environment and never
 * re-resolves on its own, so a machine that first ran this wrapper on an old
 * release keeps getting that release forever, even after newer ones hit
 * PyPI. To avoid pinning stale versions indefinitely, the `uvx` branch passes
 * `--refresh` at most once per 24h (tracked via a timestamp file — see
 * `shouldRefresh`/`markRefreshed` below). `pipx run` isn't touched: its own
 * cache already expires after ~14 days on its own. The installed-`tj` branch
 * has no cache to go stale.
 */
"use strict";

const { spawnSync } = require("child_process");
const fs = require("fs");
const os = require("os");
const path = require("path");

// PyPI package name vs. command name differ (`tokenjam` ships the `tj` script),
// so ephemeral runners must be told the source package explicitly.
const PACKAGE = "tokenjam";
const COMMAND = "tj";

function has(bin) {
  // `<bin> --version` is a cheap, side-effect-free presence probe.
  const probe = spawnSync(bin, ["--version"], { stdio: "ignore" });
  return probe.status === 0 || probe.status === 1; // 1 = exists but no --version
}

function runners() {
  return [
    { bin: "uvx", prefix: ["--from", PACKAGE, COMMAND] },
    { bin: "pipx", prefix: ["run", "--spec", PACKAGE, COMMAND] },
    { bin: COMMAND, prefix: [] }, // already installed on PATH
  ];
}

// --- uvx cache freshness (issue #111) -------------------------------------
//
// Only the `uvx` runner needs this: `uv` caches a resolved tool environment
// and reuses it forever unless told to `--refresh`, so a returning user
// silently keeps whatever version they first resolved. Tracked with a plain
// timestamp file rather than anything fancier — this wrapper is intentionally
// dependency-free (stdlib `fs`/`os`/`path` only).

const REFRESH_INTERVAL_MS = 24 * 60 * 60 * 1000; // 24h

function refreshCacheDir() {
  const xdgCacheHome = process.env.XDG_CACHE_HOME;
  const base =
    xdgCacheHome && xdgCacheHome.trim()
      ? xdgCacheHome
      : path.join(os.homedir(), ".cache");
  return path.join(base, "tokenjam-npx");
}

function refreshTimestampPath() {
  return path.join(refreshCacheDir(), "last-refresh");
}

// True if it's been >24h (or we've never refreshed / can't tell). Fails
// open on any fs error — an unwritable/unreadable cache dir must never
// break the wrapper, it just means we skip the freshness nudge this run.
function shouldRefresh() {
  try {
    const stat = fs.statSync(refreshTimestampPath());
    return Date.now() - stat.mtimeMs > REFRESH_INTERVAL_MS;
  } catch {
    return true; // no timestamp yet (or unreadable) => treat as stale
  }
}

// Called only AFTER `uvx --refresh` has actually returned with a zero exit
// status (not before we spawn it, and not on failure). If the refresh's
// download is interrupted partway through (network drop, Ctrl-C, OOM kill),
// spawnSync never returns normally, this never runs. If it returns but uv
// exits non-zero (PyPI unreachable, partial download), the caller also
// skips this call. Either way the *next* invocation still sees a
// stale/missing timestamp and retries `--refresh`. Writing the timestamp up
// front, or unconditionally on return, would mark the cache "fresh" even
// though that refresh never completed, silently pinning a broken/partial
// environment for a full 24h. Best-effort/fail-open: swallow fs errors.
function markRefreshed() {
  try {
    fs.mkdirSync(refreshCacheDir(), { recursive: true });
    fs.writeFileSync(refreshTimestampPath(), String(Date.now()));
  } catch {
    // fail open — worst case we just try to refresh again next run
  }
}

function main() {
  // Bare `npx tokenjam` IS the zero-install first run — the quota report the
  // docs promise. The branded home screen that bare LOCAL `tj` prints assumes
  // an installed CLI and would dead-end an npx user ("You're set up",
  // suggesting commands they don't have). Any explicit args pass through
  // untouched; a bare invocation stays bare (no synthetic subcommand — there
  // is no public/typeable command for this) and instead sets an env var that
  // the Python CLI's own no-subcommand branch reads to pick the report over
  // the home screen.
  const argv = process.argv.slice(2);
  const passthrough = argv;
  const childEnv = argv.length
    ? process.env
    : { ...process.env, TJ_NPX_ZERO_INSTALL_REPORT: "1" };

  for (const { bin, prefix } of runners()) {
    if (!has(bin)) continue;
    const isUvx = bin === "uvx";
    const doRefresh = isUvx && shouldRefresh();
    const args = doRefresh ? ["--refresh", ...prefix] : prefix;
    const result = spawnSync(bin, [...args, ...passthrough], {
      stdio: "inherit",
      env: childEnv,
    });
    if (result.error) continue; // try the next runner on spawn failure
    if (doRefresh && result.status === 0) markRefreshed();
    process.exit(result.status === null ? 1 : result.status);
  }

  process.stderr.write(
    "\n" +
      "tj (TokenJam) needs a Python runner to launch its CLI.\n" +
      "Install one of these, then re-run `npx tokenjam`:\n" +
      "  • uv     →  https://docs.astral.sh/uv/  (then `uvx --from tokenjam tj`)\n" +
      "  • pipx   →  `brew install pipx` / `apt install pipx`\n" +
      "Or install TokenJam directly:  pipx install tokenjam\n\n"
  );
  process.exit(1);
}

main();
