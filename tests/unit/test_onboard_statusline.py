"""Onboarding wires tj out-of-band, not in-loop (#59).

Claude Code gets the zero-token statusline (idempotent + non-destructive);
Codex's previously-registered tj MCP block is retired. Neither path pushes an
in-loop MCP quota burden on subscription users.
"""
from __future__ import annotations

from tokenjam.cli.cmd_onboard import (
    _codex_strip_tj_mcp_from_content,
    _is_tj_statusline,
    _wire_claude_statusline,
)


# --- statusline wiring (Claude Code) ---------------------------------------


def test_wire_writes_statusline_when_absent():
    settings: dict = {"env": {"X": "1"}}
    status = _wire_claude_statusline(settings)
    assert status == "written"
    sl = settings["statusLine"]
    assert sl["type"] == "command"
    assert "tj statusline" in sl["command"]
    assert settings["env"] == {"X": "1"}  # untouched


def test_wire_is_idempotent_when_already_ours(monkeypatch):
    # Force a stable command string so "already ours" compares equal.
    monkeypatch.setattr(
        "tokenjam.cli.cmd_onboard._tj_statusline_command", lambda: "tj statusline"
    )
    settings = {"statusLine": {"type": "command", "command": "tj statusline"}}
    status = _wire_claude_statusline(settings)
    assert status == "kept"
    assert settings["statusLine"]["command"] == "tj statusline"


def test_wire_refreshes_our_own_stale_command(monkeypatch):
    monkeypatch.setattr(
        "tokenjam.cli.cmd_onboard._tj_statusline_command",
        lambda: "/usr/local/bin/tj statusline",
    )
    settings = {"statusLine": {"type": "command", "command": "tj statusline"}}
    status = _wire_claude_statusline(settings)
    assert status == "updated"
    assert settings["statusLine"]["command"] == "/usr/local/bin/tj statusline"


def test_wire_never_clobbers_foreign_statusline():
    foreign = {"type": "command", "command": "ccstatusline"}
    settings = {"statusLine": dict(foreign)}
    status = _wire_claude_statusline(settings)
    assert status == "skipped"
    assert settings["statusLine"] == foreign  # left exactly intact


def test_is_tj_statusline_recognizes_only_ours():
    assert _is_tj_statusline({"command": "/opt/tj statusline"})
    assert not _is_tj_statusline({"command": "ccstatusline"})
    assert not _is_tj_statusline("ccstatusline")
    assert not _is_tj_statusline(None)


# --- Codex MCP retirement ---------------------------------------------------


def test_retire_strips_tj_managed_mcp_block():
    content = (
        '[otel]\n'
        'log_user_prompt = false\n'
        '\n'
        '[mcp_servers.tj]\n'
        '# Managed by tj — gives Codex access to TokenJam observability tools\n'
        'command = "tj"\n'
        'args = ["mcp"]\n'
    )
    new_content, removed = _codex_strip_tj_mcp_from_content(content)
    assert removed is True
    assert "[mcp_servers.tj]" not in new_content
    assert "[otel]" in new_content  # OTel (out-of-band) stays


def test_retire_leaves_foreign_same_named_server_untouched():
    # A user's unrelated server that happens to be named tj but points elsewhere.
    content = (
        '[mcp_servers.tj]\n'
        'command = "my-own-thing"\n'
        'args = ["serve"]\n'
    )
    new_content, removed = _codex_strip_tj_mcp_from_content(content)
    assert removed is False
    assert new_content == content


def test_retire_noop_when_no_mcp_block():
    content = '[otel]\nlog_user_prompt = false\n'
    new_content, removed = _codex_strip_tj_mcp_from_content(content)
    assert removed is False
    assert "[otel]" in new_content
