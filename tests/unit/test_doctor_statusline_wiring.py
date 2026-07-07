"""Doctor statusline-wiring check (#59, #105): warns a Claude Code user who's
missing the zero-token statusline, but stays informational (exit 0) on a
pure-SDK install."""
from __future__ import annotations

import json
from types import SimpleNamespace

from tokenjam.cli.cmd_doctor import _check_statusline_wiring, _claude_code_context


def _config(agent_ids=()):
    return SimpleNamespace(agents={aid: SimpleNamespace() for aid in agent_ids})


# --- _claude_code_context ---------------------------------------------------


def test_context_true_when_claude_home_exists(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    (tmp_path / ".claude").mkdir()
    assert _claude_code_context(_config([])) is True


def test_context_true_from_claude_code_agent(monkeypatch, tmp_path):
    # No ~/.claude, but a claude-code-* agent in config is positive evidence.
    monkeypatch.setenv("HOME", str(tmp_path))
    assert _claude_code_context(_config(["claude-code-myproj"])) is True


def test_context_false_for_pure_sdk(monkeypatch, tmp_path):
    # No ~/.claude and only a non-Claude-Code agent — the SDK-only case. A
    # `claude` binary possibly on PATH is deliberately NOT consulted (#105).
    monkeypatch.setenv("HOME", str(tmp_path))
    assert _claude_code_context(_config(["my-sdk-agent"])) is False


# --- _check_statusline_wiring ----------------------------------------------


def _write_statusline(home, command):
    settings = home / ".claude" / "settings.json"
    settings.parent.mkdir(parents=True, exist_ok=True)
    settings.write_text(json.dumps({"statusLine": {"type": "command", "command": command}}))


def test_ok_when_tj_statusline_wired(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    _write_statusline(tmp_path, "tj statusline")
    check = _check_statusline_wiring(_config(["claude-code-x"]))
    assert check["level"] == "ok"


def test_info_when_foreign_statusline(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    _write_statusline(tmp_path, "ccstatusline")
    check = _check_statusline_wiring(_config(["claude-code-x"]))
    assert check["level"] == "info"
    assert "untouched" in check["message"]


def test_warning_when_cc_context_but_not_wired(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    (tmp_path / ".claude").mkdir()  # CC context, but no statusLine set
    check = _check_statusline_wiring(_config([]))
    assert check["level"] == "warning"
    assert "tj onboard --claude-code" in check["message"]


def test_info_exit0_for_pure_sdk_install(monkeypatch, tmp_path):
    # The #105 regression: a correct SDK-only setup must not warn (which would
    # drag `tj doctor` to exit 1). No ~/.claude, only an SDK agent -> info.
    monkeypatch.setenv("HOME", str(tmp_path))
    check = _check_statusline_wiring(_config(["my-sdk-agent"]))
    assert check["level"] == "info"
    assert "SDK" in check["message"]
