---
description: Managed-block discipline for dotfiles onboard/uninstall write (zshrc, Codex config, Claude settings).
paths:
  - "tokenjam/cli/cmd_onboard.py"
  - "tokenjam/cli/cmd_uninstall.py"
  - "tokenjam/cli/cmd_stop.py"
  - "tokenjam/cli/cmd_statusline.py"
---

# Onboarding / managed dotfile rules

### Critical Rule 21 — Dotfile-managed blocks (onboard/uninstall) must never key off the current marker string

Onboard writes managed blocks into `~/.zshrc` (OTEL exports) and the `claude()` wrapper; match them by
a STABLE sentinel pair and strip **every** legacy marker before writing exactly one fresh block, never
"append if current-marker absent." Precedent: the zshrc OTEL marker drifted once already
(`# ocw harness observability` → `# tj harness observability` at the openclawwatch→Token Juice rename,
commit `281275f`, shipped with no migration), which orphaned already-onboarded users' `.zshrc` —
re-onboard appended a stale-secret duplicate instead of replacing it, and `tj uninstall` left the old
block behind (fixed via a shared `_strip_zshrc_otel_blocks()` in `cmd_onboard.py`). Codex's
`[otel]` config had the analogous issue, handled by `_codex_purge_legacy_ocw`. Any new
managed-dotfile-block feature needs an onboard→uninstall round-trip test asserting zero residue,
including from seeded legacy markers.
