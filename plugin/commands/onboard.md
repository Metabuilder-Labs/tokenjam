---
description: Set up TokenJam for this Claude Code install — wires the zero-token statusline, the resume-brief SessionStart hook, and local OTel telemetry ingest via the existing `tj onboard` command. Runs 100% locally, no signup.
argument-hint: "[plan: api|pro|max_5x|max_20x|plus|team|enterprise]"
allowed-tools: Bash(tj *), Bash(command -v tj)
---

## Check for the tj CLI
!`command -v tj >/dev/null 2>&1 && echo "found: $(tj --version 2>&1)" || echo "NOT_FOUND"`

## Run onboarding (only if tj is on PATH)
!`command -v tj >/dev/null 2>&1 && tj onboard --claude-code --reconfigure --plan "${ARGUMENTS:-api}" --budget 0 --backfill-days 30 --analysis-span 30d 2>&1 || echo "Skipped: tj is not installed."`

## Instructions

If the check above printed `NOT_FOUND` or the run was skipped, tell the user to install TokenJam first — `npx tokenjam@latest` or `pipx install tokenjam` — then re-run `/onboard`. Nothing here requires an account or network service; TokenJam reads Claude Code's own OTel telemetry locally.

Otherwise, summarize what onboarding wired into `~/.claude/settings.json`:
- the zero-token statusline (`tj statusline`) — costs no model quota, shows context re-read % after every turn
- the resume-brief `SessionStart` hook
- the local OTel ingest endpoint TokenJam listens on

It used plan `${ARGUMENTS:-api}` (the generic default if no argument was given) to frame dollar figures. If the user's actual Claude plan is a subscription tier, tell them they can re-run `/onboard <plan>` (e.g. `/onboard max_20x`) at any time — it's idempotent and safe to repeat.
