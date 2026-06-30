#!/usr/bin/env node
/*
 * Zero-install launcher for TokenJam (`npx tj`), issue #6.
 *
 * TokenJam's CLI is a Python package (`tokenjam`, command `tj`). This thin npm
 * wrapper exists for one reason: the Claude Code / ccusage crowd reaches for
 * `npx <tool>` first. `npx tj` here resolves a Python launcher with NO pip env,
 * NO daemon, NO onboarding — it shells out to the Python CLI via the first
 * available runner and hands every argument straight through. Bare `npx tj`
 * (no subcommand) routes to `tj quickstart`: where your Claude Code quota goes,
 * from the same ~/.claude/projects/*.jsonl files ccusage reads, in one command.
 *
 * Runner preference (first that exists wins):
 *   1. `uvx --from tokenjam tj …`  — fully ephemeral, downloads nothing global
 *   2. `pipx run --spec tokenjam tj …`
 *   3. `tj …`                      — an already-installed CLI on PATH
 *
 * If none are present we print actionable install guidance and exit non-zero.
 */
"use strict";

const { spawnSync } = require("child_process");

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

function main() {
  const passthrough = process.argv.slice(2);

  for (const { bin, prefix } of runners()) {
    if (!has(bin)) continue;
    const result = spawnSync(bin, [...prefix, ...passthrough], {
      stdio: "inherit",
    });
    if (result.error) continue; // try the next runner on spawn failure
    process.exit(result.status === null ? 1 : result.status);
  }

  process.stderr.write(
    "\n" +
      "tj (TokenJam) needs a Python runner to launch its CLI.\n" +
      "Install one of these, then re-run `npx tj`:\n" +
      "  • uv     →  https://docs.astral.sh/uv/  (then `uvx --from tokenjam tj`)\n" +
      "  • pipx   →  `brew install pipx` / `apt install pipx`\n" +
      "Or install TokenJam directly:  pipx install tokenjam\n\n"
  );
  process.exit(1);
}

main();
