---
description: Show TokenJam's savings/optimize report — where quota is going and what's recoverable. Equivalent to running `tj optimize`.
argument-hint: "[since, e.g. 7d/30d/90d]"
allowed-tools: Bash(tj optimize *)
---

## tj optimize
!`tj optimize --since "${ARGUMENTS:-30d}" 2>&1 || echo "tj is not installed. Run /onboard first, or install with npx tokenjam@latest / pipx install tokenjam."`

## Instructions
Summarize the optimize report above: the biggest recoverable-cost findings and the concrete fix each one suggests. If it reports no usage data yet, tell the user that's expected until TokenJam has ingested at least one session — run `/onboard` first if they haven't.
