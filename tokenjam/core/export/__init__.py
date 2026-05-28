"""
Export-format generators for `tj optimize export-config`.

Each module under this package emits a target-specific JSON snippet that
encodes TokenJam's current routing recommendations. The user copies the
snippet into their routing layer of choice — TokenJam does not write to
external configs, run as a proxy, or otherwise sit in the call path.

Targets:
  - claude-code: TokenJam routing block formatted for `.claude/settings.json`
"""
