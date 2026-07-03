"""Tests for the onboarding wiring of the cap-output PostToolUse hook.

Covers the idempotent, non-destructive merge into ~/.claude/settings.json.
"""
from __future__ import annotations

from tokenjam.cli.cmd_onboard import (
    _CAP_OUTPUT_MATCHER,
    _is_tj_cap_output_entry,
    _unwire_claude_output_cap_hook,
    _wire_claude_output_cap_hook,
)


def _tj_entry(settings):
    return settings["hooks"]["PostToolUse"]


def test_wire_into_empty_settings_writes():
    s: dict = {}
    assert _wire_claude_output_cap_hook(s) == "written"
    post = _tj_entry(s)
    assert len(post) == 1
    assert post[0]["matcher"] == _CAP_OUTPUT_MATCHER
    assert _is_tj_cap_output_entry(post[0])


def test_wire_is_idempotent():
    s: dict = {}
    _wire_claude_output_cap_hook(s)
    # second run is a no-op ("kept")
    assert _wire_claude_output_cap_hook(s) == "kept"
    assert len(_tj_entry(s)) == 1


def test_wire_preserves_foreign_posttooluse_hooks():
    foreign = {"matcher": "Edit", "hooks": [{"type": "command", "command": "my-linter"}]}
    s = {"hooks": {"PostToolUse": [foreign]}}
    assert _wire_claude_output_cap_hook(s) == "written"
    post = _tj_entry(s)
    assert foreign in post          # foreign hook untouched
    assert any(_is_tj_cap_output_entry(e) for e in post)
    assert len(post) == 2


def test_wire_updates_stale_tj_entry_in_place():
    s = {"hooks": {"PostToolUse": [
        {"matcher": "Bash", "hooks": [{"type": "command", "command": "/old/tj hook cap-output"}]},
    ]}}
    assert _wire_claude_output_cap_hook(s) == "updated"
    post = _tj_entry(s)
    assert len(post) == 1           # updated in place, not duplicated
    assert post[0]["matcher"] == _CAP_OUTPUT_MATCHER


def test_wire_preserves_other_settings_keys():
    s = {"env": {"FOO": "bar"}, "statusLine": {"type": "command", "command": "x"}}
    _wire_claude_output_cap_hook(s)
    assert s["env"] == {"FOO": "bar"}
    assert s["statusLine"] == {"type": "command", "command": "x"}


def test_unwire_removes_only_tj_entry():
    foreign = {"matcher": "Edit", "hooks": [{"type": "command", "command": "my-linter"}]}
    s = {"hooks": {"PostToolUse": [foreign]}}
    _wire_claude_output_cap_hook(s)
    assert _unwire_claude_output_cap_hook(s) is True
    assert _tj_entry(s) == [foreign]     # foreign survives, tj entry gone


def test_unwire_cleans_up_empty_hooks_block():
    s: dict = {}
    _wire_claude_output_cap_hook(s)
    assert _unwire_claude_output_cap_hook(s) is True
    assert "hooks" not in s               # empty block removed entirely


def test_unwire_noop_when_absent():
    assert _unwire_claude_output_cap_hook({}) is False
    assert _unwire_claude_output_cap_hook({"hooks": {"PostToolUse": []}}) is False
