# TokenJam — Claude Code plugin

This directory is the Claude Code plugin source (`.claude-plugin/plugin.json` at its root). It wraps the [`tj` CLI](https://github.com/Metabuilder-Labs/tokenjam) — a local-first, OTel-native cost-saving utility for AI agents — as slash commands. It installs nothing on its own: `tj` still needs to be on your `PATH` (`npx tokenjam@latest` or `pipx install tokenjam`).

## Install

The plugin lives in a subdirectory of the `tokenjam` repo, so install it via a `git-subdir` marketplace source pointing at `plugin` (see the [marketplace docs](https://code.claude.com/docs/en/plugin-marketplaces#git-subdirectories)):

```json
{
  "name": "tokenjam",
  "source": {
    "source": "git-subdir",
    "url": "https://github.com/Metabuilder-Labs/tokenjam.git",
    "path": "plugin"
  }
}
```

Or, once listed in a marketplace: `/plugin install tokenjam`.

## Commands

| Command      | Runs                       | Does |
|--------------|-----------------------------|------|
| `/onboard`   | `tj onboard --claude-code`  | Wires the zero-token statusline, the resume-brief `SessionStart` hook, and local OTel telemetry ingest. Idempotent — safe to re-run. |
| `/status`    | `tj status`                 | Token usage, cost today, active alerts, per agent. |
| `/optimize`  | `tj optimize`                | Savings report: where quota is going and the concrete fix for each finding. |
| `/doctor`    | `tj doctor`                  | Health check on config, ingest endpoint, and storage. |
| `/uninstall` | `tj uninstall --yes`         | Unwires the statusline, hooks, and OTel env vars `/onboard` set up. |

## Why no `hooks.json` or `.mcp.json`

This plugin ships neither, on purpose:

- **Statusline.** A plugin's own `settings.json` only supports the `agent` and `subagentStatusLine` keys — there is no plugin-manifest path to the main terminal `statusLine` ([plugins-reference](https://code.claude.com/docs/en/plugins-reference), "File locations reference" table; [statusline docs](https://code.claude.com/docs/en/statusline#subagent-status-lines)). `/onboard` gets there by calling `tj onboard --claude-code` instead, which writes `statusLine` directly into the user's `~/.claude/settings.json`.
- **Hooks.** `tj onboard --claude-code` already writes a `SessionStart` hook (`tj resume-brief --from-hook`) into the user's global settings, matched/deduped by a marker in the command string. A second, independent `hooks/hooks.json` in this plugin would not recognize that marker and would double-fire the hook on every session start. Routing through `/onboard` keeps `~/.claude/settings.json` to one writer.
- **MCP.** `tj mcp` is tokenjam's in-request-path surface, built for SDK/API integrations that already sit in the loop. A measured A/B showed an in-loop MCP costs Claude Code / subscription users +36% model-weighted tokens versus the zero-token statusline — auto-wiring it here would regress the exact thing `tj onboard --claude-code` deliberately avoids. Run `tj mcp` yourself (or `claude mcp add tj -- tj mcp`) if you want it for an SDK-style use case.
