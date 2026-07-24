#!/usr/bin/env node
/*
 * Zero-install launcher for TokenJam (`npx tokenjam`), issue #6.
 *
 * TokenJam's CLI is a Python package (`tokenjam`, command `tj`). This thin npm
 * wrapper exists for one reason: the Claude Code / ccusage crowd reaches for
 * `npx <tool>` first. `npx tokenjam` here resolves a Python launcher with NO pip env,
 * NO daemon, NO onboarding — it shells out to the Python CLI via the first
 * available runner and hands every argument straight through. Bare `npx tokenjam`
 * (no subcommand) prints the zero-install report: where your Claude Code quota
 * goes, from the same ~/.claude/projects/*.jsonl files ccusage reads, in one
 * command.
 *
 * Runner preference (first that exists wins):
 *   1. `uvx --from tokenjam==<own version> tj …`  — fully ephemeral, downloads nothing global
 *   2. `pipx run --spec tokenjam==<own version> tj …`
 *   3. `tj …`                      — an already-installed CLI on PATH
 *
 * If none are present we print actionable install guidance and exit non-zero.
 *
 * Note: tokenjam >=0.5.4 also ships a `tokenjam` console-script alias
 * alongside `tj`, so bare `uvx tokenjam` / `pipx run tokenjam` work too. This
 * wrapper keeps the explicit `--from tokenjam tj` / `--spec tokenjam tj` form
 * below for back-compat with the 0.5.3 and earlier releases it also targets.
 *
 * Version pinning: `uv`/`pipx` cache a resolved tool environment and reuse it
 * forever unless the requested spec changes — an unpinned `--from tokenjam`
 * silently keeps reusing whatever was resolved first (e.g. a prior `uv tool
 * install tokenjam` at an old version), never re-resolving on its own, no
 * matter how many newer releases hit PyPI since. Pinning `--from
 * tokenjam==<version>` / `--spec tokenjam==<version>` to this wrapper's OWN
 * version (kept in sync with the release tag by publish-npm.yml's `npm
 * version ${GITHUB_REF_NAME#v}` step) forces the resolver past that shortcut,
 * so `npx tokenjam` always runs the release it shipped with. If the pinned
 * spec can't be resolved yet (this wrapper published slightly ahead of PyPI
 * propagation), we fall back to the unpinned form rather than fail outright.
 *
 * Staleness note: pinning only fixes what THIS wrapper runs. A bare `tj`
 * invoked directly (no `npx`) still runs whatever was separately installed
 * via `uv tool install` / `pipx install` / `pip install` / Homebrew, which can
 * sit on an old version indefinitely. See `warnIfShadowedByStaleInstall`
 * below: detect-and-tell only, never mutates, never auto-upgrades.
 *
 * Two on-disk markers (under the cache dir from `cacheDir()`) keep both of
 * the above cheap on repeat runs, since `npx tokenjam` is meant to be run
 * often, not once:
 *   - `pin-ok-<bin>-<version>`: once the pinned spec has been CONFIRMED to
 *     resolve for a given runner+version (see `resolves()`), skip the
 *     confirmation probe on later runs for that same exact version — see the
 *     "second process" note above `resolves()` for why this exists.
 *   - `last-stale-check`: throttles `warnIfShadowedByStaleInstall`'s four
 *     subprocess probes (uv/pipx/pip/brew) to at most once per interval,
 *     since it's a purely advisory nudge and doesn't need to run every time.
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

// This wrapper's own version. `publish-npm.yml`'s wrapper-publish job runs
// `npm version ${GITHUB_REF_NAME#v}` against npm-wrapper/package.json before
// `npm publish`, so whatever version this file ships inside always matches
// the tokenjam release it was cut alongside — safe to read at runtime as the
// version to pin the Python side to.
function ownVersion() {
  try {
    return require(path.join(__dirname, "..", "package.json")).version;
  } catch {
    return null;
  }
}

function runners(version) {
  const pinnedSpec = version ? `${PACKAGE}==${version}` : null;
  return [
    {
      bin: "uvx",
      pinnedPrefix: pinnedSpec ? ["--from", pinnedSpec, COMMAND] : null,
      prefix: ["--from", PACKAGE, COMMAND],
    },
    {
      bin: "pipx",
      pinnedPrefix: pinnedSpec ? ["run", "--spec", pinnedSpec, COMMAND] : null,
      prefix: ["run", "--spec", PACKAGE, COMMAND],
    },
    { bin: COMMAND, pinnedPrefix: null, prefix: [] }, // already installed on PATH, nothing to pin
  ];
}

// Cheap(ish), side-effect-free resolution probe: does `<bin> <args>
// --version` succeed? Used to decide, before the real invocation, whether
// the pinned package spec actually resolves on this runner — falls back to
// the unpinned prefix when it doesn't (wrapper published ahead of PyPI
// propagation). Unlike `has()`, this requires an exact status 0: `uv`
// exits 1 both for "resolution failed" AND for "binary ran fine but
// doesn't understand --version", so treating 1 as success here would
// silently paper over real resolution failures (verified: `uvx --from
// tokenjam==<bogus> tj --version` also exits 1, indistinguishable from the
// unsupported-flag case `has()` is built around).
//
// Cost tradeoff: this is a genuine second full `uvx`/`pipx` process, on top
// of the real invocation below — for `uvx` specifically, an UNCACHED pinned
// version means this probe pays the actual network resolve + ephemeral venv
// build, and the real invocation right after it just reuses that now-warm
// local cache (fast). We can't fold the probe into the real invocation
// without either (a) risking running the user's real command twice — once
// pinned, once unpinned on fallback, with real side effects/output printed
// twice — or (b) buffering the real command's stdout/stderr to inspect it
// for a resolution-failure signature first, which breaks live streaming for
// anything interactive (e.g. `tj onboard`'s prompts). Given the pin's whole
// point is correctness (never silently run a stale cached version), a
// probe-first design that never double-runs the real command is kept
// deliberately, and its cost is bounded to a one-time-per-version hit via
// the `pin-ok-<bin>-<version>` cache marker in `main()` below — this
// function itself is only ever called once per (runner, version) between
// cache expiries, not on every invocation.
function resolves(bin, args) {
  const probe = spawnSync(bin, [...args, "--version"], { stdio: "ignore" });
  return probe.status === 0;
}

// --- on-disk cache markers -------------------------------------------------

function cacheDir() {
  const xdgCacheHome = process.env.XDG_CACHE_HOME;
  const base =
    xdgCacheHome && xdgCacheHome.trim()
      ? xdgCacheHome
      : path.join(os.homedir(), ".cache");
  return path.join(base, "tokenjam-npx");
}

// True if `name`'s marker was touched within the last `ttlMs`. Fails closed
// (treats as NOT fresh) on any fs error — an unwritable/unreadable cache dir
// must never break the wrapper, it just means we redo the check this run.
function isFresh(name, ttlMs) {
  try {
    const stat = fs.statSync(path.join(cacheDir(), name));
    return Date.now() - stat.mtimeMs < ttlMs;
  } catch {
    return false;
  }
}

function touchMarker(name) {
  try {
    fs.mkdirSync(cacheDir(), { recursive: true });
    fs.writeFileSync(path.join(cacheDir(), name), String(Date.now()));
  } catch {
    // best-effort — worst case we just redo the check next run
  }
}

// How long a confirmed pinned-resolve is trusted before re-probing. Bounds
// the (rare) risk of a locally evicted uv/pipx cache silently going stale
// between confirmation and use — 7 days is long enough that a `npx
// tokenjam` run pays the double-invocation cost roughly once per version
// per machine, not on every run, while still re-verifying periodically.
const PIN_CONFIRM_TTL_MS = 7 * 24 * 60 * 60 * 1000;

// How often the (purely advisory) stale-shadowing-install check runs. Unlike
// the pin above, a stale window here can't cause incorrect behavior — worst
// case is a delayed nudge — so the wider interval buys back the four
// subprocess probes (uv/pipx/pip/brew) on every single invocation.
const STALE_CHECK_TTL_MS = 24 * 60 * 60 * 1000;

// --- stale shadowing install detection ------------------------------------
//
// Detect-and-tell only, per design: this never mutates anything and never
// upgrades on the user's behalf, in interactive or non-interactive/CI
// contexts alike. It's a best-effort nudge — any detection failure (missing
// binary, unexpected output, timeout) is swallowed and skipped silently; it
// must never break or slow down the primary command above by much.

const DETECT_TIMEOUT_MS = 2000;

function safeSpawn(bin, args) {
  try {
    return spawnSync(bin, args, {
      stdio: ["ignore", "pipe", "pipe"],
      timeout: DETECT_TIMEOUT_MS,
      encoding: "utf8",
    });
  } catch {
    return null;
  }
}

function versionParts(v) {
  return String(v)
    .trim()
    .split(".")
    .map((n) => parseInt(n, 10) || 0);
}

function isOlder(a, b) {
  const pa = versionParts(a);
  const pb = versionParts(b);
  const len = Math.max(pa.length, pb.length);
  for (let i = 0; i < len; i++) {
    const x = pa[i] || 0;
    const y = pb[i] || 0;
    if (x !== y) return x < y;
  }
  return false;
}

// Each detector below is skipped up front via `has()` when its own binary
// isn't even on PATH, so a machine without e.g. Homebrew never pays for a
// `brew list` spawn.

function detectUvTool() {
  if (!has("uv")) return null;
  const result = safeSpawn("uv", ["tool", "list"]);
  if (!result || result.status !== 0 || !result.stdout) return null;
  const match = result.stdout.match(/^tokenjam\s+v?(\S+)/m);
  if (!match) return null;
  return {
    method: "uv tool",
    version: match[1],
    upgradeCmd: "uv tool upgrade tokenjam",
  };
}

function detectPipx() {
  if (!has("pipx")) return null;
  const result = safeSpawn("pipx", ["list", "--json"]);
  if (!result || result.status !== 0 || !result.stdout) return null;
  try {
    const data = JSON.parse(result.stdout);
    const venv = data.venvs && data.venvs[PACKAGE];
    const version =
      venv &&
      venv.metadata &&
      venv.metadata.main_package &&
      venv.metadata.main_package.package_version;
    if (!version) return null;
    return { method: "pipx", version, upgradeCmd: "pipx upgrade tokenjam" };
  } catch {
    return null;
  }
}

function detectPip() {
  for (const pipBin of ["pip3", "pip"]) {
    if (!has(pipBin)) continue;
    const result = safeSpawn(pipBin, ["show", PACKAGE]);
    if (!result || result.status !== 0 || !result.stdout) continue;
    const match = result.stdout.match(/^Version:\s*(\S+)/m);
    if (!match) continue;
    // Covers both a plain `pip install` and `pip install --user` — pip
    // doesn't distinguish the two in `pip show` output, and either way the
    // fix command is the same.
    return {
      method: "pip",
      version: match[1],
      upgradeCmd: `${pipBin} install --upgrade ${PACKAGE}`,
    };
  }
  return null;
}

function detectHomebrew() {
  if (!has("brew")) return null;
  const result = safeSpawn("brew", ["list", "--versions", PACKAGE]);
  if (!result || result.status !== 0 || !result.stdout) return null;
  const match = result.stdout.trim().match(/^tokenjam\s+(\S+)/);
  if (!match) return null;
  return {
    method: "Homebrew",
    version: match[1],
    upgradeCmd: "brew upgrade tokenjam",
  };
}

function warnIfShadowedByStaleInstall(wrapperVersion) {
  if (!wrapperVersion) return;
  const detectors = [detectUvTool, detectPipx, detectPip, detectHomebrew];
  for (const detect of detectors) {
    let found = null;
    try {
      found = detect();
    } catch {
      found = null;
    }
    if (!found || !isOlder(found.version, wrapperVersion)) continue;
    process.stderr.write(
      "\n" +
        `Note: a ${found.method} install of tokenjam is at v${found.version}, older than v${wrapperVersion} run here.\n` +
        `Upgrade it with: ${found.upgradeCmd}\n`
    );
    return; // one line is enough — first stale install found wins
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

  const version = ownVersion();

  for (const { bin, pinnedPrefix, prefix } of runners(version)) {
    if (!has(bin)) continue;

    let args = prefix;
    if (pinnedPrefix) {
      const pinMarker = `pin-ok-${bin}-${version}`;
      if (isFresh(pinMarker, PIN_CONFIRM_TTL_MS)) {
        args = pinnedPrefix; // previously confirmed — skip the probe
      } else if (resolves(bin, pinnedPrefix)) {
        touchMarker(pinMarker);
        args = pinnedPrefix;
      } // else: falls back to the unpinned `prefix` already assigned above
    }

    const result = spawnSync(bin, [...args, ...passthrough], {
      stdio: "inherit",
      env: childEnv,
    });
    if (result.error) continue; // try the next runner on spawn failure

    // Only nudge about a stale shadowing install when the real command
    // actually succeeded — a non-zero exit means the underlying CLI itself
    // failed (bad args, etc.), which isn't the moment to pile on an
    // unrelated advisory note.
    if (result.status === 0 && !isFresh("last-stale-check", STALE_CHECK_TTL_MS)) {
      warnIfShadowedByStaleInstall(version);
      touchMarker("last-stale-check");
    }

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
