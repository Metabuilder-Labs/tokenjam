---
description: Remove TokenJam's Claude Code integration — unwires the statusline, hooks, and OTel env vars that `/onboard` set up. Equivalent to running `tj uninstall --yes`. Does not remove the plugin itself.
allowed-tools: Bash(tj uninstall *)
---

## tj uninstall
!`tj uninstall --yes 2>&1 || echo "tj is not installed or already removed."`

## Instructions
Confirm to the user what was torn down (statusline, hooks, OTel env vars) and note that this only removes the CLI's Claude Code wiring, not the tokenjam plugin itself — that's removed separately via `/plugin uninstall tokenjam` or the plugin manager.
