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
 * so `npx tokenjam` always runs the release it shipped with.
 *
 * Fail-closed on an unresolved pin: publish-npm.yml's wrapper job now blocks
 * on the matching PyPI release actually being visible before it publishes
 * this wrapper, so "pinned spec doesn't resolve" should no longer happen in
 * the ordinary run. Silently falling back to the unpinned form here traded a
 * loud, correct, RETRYABLE failure for a silent PERMANENT one: uv/pipx cache
 * whatever the unpinned spec resolves to and never re-resolve on their own,
 * so one unlucky run during any remaining propagation gap left a stale
 * version cached forever — surfacing as "0.6.2 shipped a palette rework but
 * npx still shows the old colors" with no obvious cause. A resolution
 * failure is now reported and the next runner (or an already-installed PATH
 * `tj`) is tried instead — never a silent unpinned run.
 *
 * Staleness note: pinning only fixes what THIS wrapper runs. A bare `tj`
 * invoked directly (no `npx`) still runs whatever was separately installed
 * via `uv tool install` / `pipx install` / `pip install` / Homebrew, which can
 * sit on an old version indefinitely. See `warnIfShadowedByStaleInstall`
 * below: detect-and-tell only, never mutates, never auto-upgrades.
 */
"use strict";

const { spawnSync } = require("child_process");
const fs = require("fs");
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

// Cheap, side-effect-free resolution probe: does `<bin> <args> --version`
// succeed? Used to decide, before the real invocation, whether the pinned
// package spec actually resolves on this runner — falls back to the
// unpinned prefix when it doesn't (wrapper published ahead of PyPI
// propagation). Unlike `has()`, this requires an exact status 0: `uv`
// exits 1 both for "resolution failed" AND for "binary ran fine but
// doesn't understand --version", so treating 1 as success here would
// silently paper over real resolution failures (verified: `uvx --from
// tokenjam==<bogus> tj --version` also exits 1, indistinguishable from the
// unsupported-flag case `has()` is built around).
function resolves(bin, args) {
  const probe = spawnSync(bin, [...args, "--version"], { stdio: "ignore" });
  return probe.status === 0;
}

// Resolve `bin` to an absolute path the way the OS would launch it, without
// spawning it — a plain PATH walk, no shell involved.
function resolveOnPath(bin) {
  const dirs = (process.env.PATH || "").split(path.delimiter);
  const exts =
    process.platform === "win32"
      ? (process.env.PATHEXT || ".EXE;.CMD;.BAT").split(";")
      : [""];
  for (const dir of dirs) {
    if (!dir) continue;
    for (const ext of exts) {
      const candidate = path.join(dir, bin + ext);
      try {
        fs.accessSync(candidate, fs.constants.X_OK);
        return candidate;
      } catch {
        // not here, keep looking
      }
    }
  }
  return null;
}

// True when `candidatePath` resolves (through symlinks) to this very file.
// `npx` symlinks this package's own `tj`/`tokenjam` bin into a temp bin dir
// and puts that dir on PATH, so a naive "is tj on PATH" probe finds
// ourselves — re-spawning that would recurse into this same wrapper forever
// instead of ever reaching a real, separately-installed CLI.
function isOwnShim(candidatePath) {
  if (!candidatePath) return false;
  try {
    return fs.realpathSync(candidatePath) === fs.realpathSync(__filename);
  } catch {
    return false; // couldn't stat either side — don't claim a match
  }
}

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
  let sawUnresolvedPin = false;

  for (const { bin, pinnedPrefix, prefix } of runners(version)) {
    if (bin === COMMAND) {
      // The installed-CLI fallback: only real if it resolves to something
      // OTHER than this wrapper (see isOwnShim above). Under npx, `has(bin)`
      // alone would find our own re-exec'd shim and spawn it recursively.
      const resolved = resolveOnPath(bin);
      if (!resolved || isOwnShim(resolved)) continue;
    } else if (!has(bin)) {
      continue;
    }

    let args;
    if (pinnedPrefix) {
      if (!resolves(bin, pinnedPrefix)) {
        // Fail closed: never silently drop to the unpinned spec (see the
        // version-pinning comment at the top of this file for why). Report
        // and move on to the next runner instead.
        sawUnresolvedPin = true;
        process.stderr.write(
          `Note: ${bin} can't resolve tokenjam==${version} yet — skipping it rather than running an unpinned (potentially stale-cached) version.\n`
        );
        continue;
      }
      args = pinnedPrefix;
    } else {
      args = prefix;
    }

    const result = spawnSync(bin, [...args, ...passthrough], {
      stdio: "inherit",
      env: childEnv,
    });
    if (result.error) continue; // try the next runner on spawn failure
    warnIfShadowedByStaleInstall(version);
    process.exit(result.status === null ? 1 : result.status);
  }

  if (sawUnresolvedPin) {
    process.stderr.write(
      "\n" +
        `tj (TokenJam): couldn't resolve tokenjam==${version} on PyPI through uv/pipx, and no already-installed 'tj' was found on PATH.\n` +
        "This should be transient — PyPI's CDN can lag briefly right after a release. Wait a minute and re-run `npx tokenjam`.\n" +
        "If it persists, check https://pypi.org/project/tokenjam/#history for the release, or install an unpinned copy yourself:\n" +
        "  uvx --from tokenjam tj   /   pipx run --spec tokenjam tj\n\n"
    );
    process.exit(1);
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
