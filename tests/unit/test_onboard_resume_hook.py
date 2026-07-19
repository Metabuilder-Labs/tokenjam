"""Tests for the onboarding wiring of the resume-brief SessionStart hook.

Covers the idempotent, non-destructive merge into ~/.claude/settings.json and
the uninstall path.
"""
from __future__ import annotations

from tokenjam.cli.cmd_onboard import (
    _RESUME_BRIEF_MATCHER,
    _is_tj_resume_brief_entry,
    _unwire_claude_resume_brief_hook,
    _wire_claude_resume_brief_hook,
)


def _entries(settings):
    return settings["hooks"]["SessionStart"]


def test_wire_into_empty_settings_writes():
    s: dict = {}
    assert _wire_claude_resume_brief_hook(s) == "written"
    start = _entries(s)
    assert len(start) == 1
    assert start[0]["matcher"] == _RESUME_BRIEF_MATCHER
    assert _is_tj_resume_brief_entry(start[0])
    assert "resume-brief --from-hook" in start[0]["hooks"][0]["command"]


def test_wire_is_idempotent():
    s: dict = {}
    _wire_claude_resume_brief_hook(s)
    assert _wire_claude_resume_brief_hook(s) == "kept"
    assert len(_entries(s)) == 1


def test_wire_preserves_foreign_sessionstart_hooks():
    foreign = {"matcher": "startup", "hooks": [{"type": "command", "command": "my-greeter"}]}
    s = {"hooks": {"SessionStart": [foreign]}}
    assert _wire_claude_resume_brief_hook(s) == "written"
    start = _entries(s)
    assert foreign in start
    assert any(_is_tj_resume_brief_entry(e) for e in start)
    assert len(start) == 2


def test_wire_updates_stale_tj_entry_in_place():
    s = {"hooks": {"SessionStart": [
        {"matcher": "resume", "hooks": [{"type": "command", "command": "/old/tj resume-brief --last"}]},
    ]}}
    assert _wire_claude_resume_brief_hook(s) == "updated"
    start = _entries(s)
    assert len(start) == 1  # updated in place, not duplicated
    assert start[0]["matcher"] == _RESUME_BRIEF_MATCHER


def test_wire_preserves_other_settings_keys():
    s = {"env": {"FOO": "bar"}, "statusLine": {"type": "command", "command": "x"}}
    _wire_claude_resume_brief_hook(s)
    assert s["env"] == {"FOO": "bar"}
    assert s["statusLine"] == {"type": "command", "command": "x"}


def test_wire_coexists_with_other_hook_events():
    s = {"hooks": {"PostToolUse": [{"matcher": "Bash", "hooks": [
        {"type": "command", "command": "some-other-tool hook"}]}]}}
    _wire_claude_resume_brief_hook(s)
    assert "PostToolUse" in s["hooks"]  # untouched
    assert "SessionStart" in s["hooks"]


def test_unwire_removes_only_tj_entry():
    foreign = {"matcher": "startup", "hooks": [{"type": "command", "command": "my-greeter"}]}
    s = {"hooks": {"SessionStart": [foreign]}}
    _wire_claude_resume_brief_hook(s)
    assert _unwire_claude_resume_brief_hook(s) is True
    assert _entries(s) == [foreign]


def test_unwire_cleans_up_empty_hooks_block():
    s: dict = {}
    _wire_claude_resume_brief_hook(s)
    assert _unwire_claude_resume_brief_hook(s) is True
    assert "hooks" not in s


def test_unwire_noop_when_absent():
    assert _unwire_claude_resume_brief_hook({}) is False
    assert _unwire_claude_resume_brief_hook({"hooks": {"SessionStart": []}}) is False
