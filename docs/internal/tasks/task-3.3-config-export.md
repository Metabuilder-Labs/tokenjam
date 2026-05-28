# Task 3.3 — Config export for Claude Code (snippet-only)

**Wave 3. Dispatch after Wave 2 fully merges. Runs in parallel with Tasks 3.1, 3.2, and (optionally) Task 4.**

## Prerequisites

- Read `CLAUDE.md`, `docs/internal/tasks/decisions-locked.md`.

## Summary

`tj optimize export-config --target claude-code` writes a routing config snippet to a file. The user reads the file and copies the snippet into their own configuration manually.

**No `--apply` flag.** Locked decision. Claude Code does not honor TokenJam-defined routing keys in its `settings.json`. Writing the block automatically would change nothing in Claude Code's behavior, breaking trust. The strategy doc's position is that TokenJam doesn't sit in the call path; this command stays consistent with that.

## Honest framing baked into the snippet

Since Pro-tier validation is deferred, exports include Level-1 (structural) recommendations only, with explicit caveats baked into the file as JSON comments:

```jsonc
{
  "tokenjam": {
    "routing_recommendations": {
      // ⚠ STRUCTURAL HEURISTIC ONLY: review before applying.
      //
      // These recommendations come from session structural patterns,
      // not validated quality equivalence. Each rule names a class of
      // sessions and suggests a cheaper model. Test on a few real
      // sessions before trusting broadly.
      //
      // TokenJam does not enforce these rules. You apply them by
      // configuring your routing layer (Claude Code settings, LiteLLM
      // config, framework code) according to your judgment.
      //
      // See https://tokenjam.dev/products/downsize for details.
      "rules": [
        {
          "match": { "task_class": "small-edit-with-test" },
          "model": "claude-haiku-4-5",
          "confidence": "structural",
          "estimated_savings_usd_month": 1632
        }
      ]
    }
  }
}
```

## Scope

- New CLI subcommand under existing `cmd_optimize` group: `tj optimize export-config --target claude-code`.
- New module: `tokenjam/core/export/claude_code.py` (the format generator).
- Writes snippet file to `~/.config/tokenjam/exports/claude-code-<YYYY-MM-DD>.json`.
- CLI prints clear next-step instructions on stdout:
  ```
  Snippet written to ~/.config/tokenjam/exports/claude-code-2026-05-27.json.
  Open the file and copy the routing block into your .claude/settings.json
  or your routing layer of choice (LiteLLM router config, framework code, etc.).
  ```
- The snippet file includes the honest-framing comments verbatim.
- Plan-tier-aware: subscription users get `estimated_tokens_freed` in place of `estimated_savings_usd_month`. Unknown-tier users get neither, with a comment noting "configure plan tier with `tj onboard --reconfigure` to see savings projections."
- Tests for export-format generation.
- Documentation in `docs/optimize/export-configs.md`.

## Files touched

- New: `tokenjam/cli/cmd_optimize_export.py` (subcommand of `cmd_optimize`, or extend `cmd_optimize.py` directly per repo conventions)
- New: `tokenjam/core/export/__init__.py`, `tokenjam/core/export/claude_code.py`
- New: `tests/unit/test_export_claude_code.py`
- `docs/optimize/export-configs.md`
- `CHANGELOG.md`

## Coordination

- Adds a subcommand to `cmd_optimize`. Wave 2 analyzer tasks self-registered via the registry; they don't touch the command structure. Trivial conflict at most.
- No file outside the TokenJam config directory is touched. **Do not write to `~/.claude/settings.json` under any circumstance.**

## Done-when

- A user with a populated optimize report runs `tj optimize export-config --target claude-code` and receives a snippet file at the expected path.
- The snippet contains the honest-framing comments.
- The snippet contains the Level-1 routing rules from the user's current optimize report.
- CLI prints clear next-step instructions on stdout.
- Subscription users see `estimated_tokens_freed` instead of dollar figures in the snippet.
- Unknown-tier users see a comment guiding them to `tj onboard --reconfigure`.
- No file outside `~/.config/tokenjam/exports/` is created or modified.
