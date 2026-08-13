---
description: Show TokenJam's current status for every tracked agent — token usage, cost today, and active alerts. Equivalent to running `tj status`.
allowed-tools: Bash(tj status *)
---

## tj status
!`tj status --verbose 2>&1 || echo "tj is not installed. Run /onboard first, or install with npx tokenjam@latest / pipx install tokenjam."`

## Instructions
Summarize the status output above for the user in a few sentences: which agents are active, today's cost, and any active alerts worth flagging. If it reports no agents found, tell the user that's expected until they've run an onboarded, instrumented agent at least once.
