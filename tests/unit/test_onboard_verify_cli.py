"""Post-onboard verification wiring (#80): the prompt gate and the runner that
both onboarding paths call at their end."""
from __future__ import annotations

from types import SimpleNamespace

import click
import pytest

from tokenjam.cli import cmd_onboard
from tokenjam.core.onboard_verify import VerifyResult


def _ctx() -> click.Context:
    return click.Context(cmd_onboard.cmd_onboard)


def _config():
    return SimpleNamespace(
        api=SimpleNamespace(
            host="127.0.0.1", port=7391,
            auth=SimpleNamespace(enabled=False, api_key=None),
        ),
        storage=SimpleNamespace(path="/tmp/x.db"),
    )


def test_runner_prints_success_when_confirmed(monkeypatch, capsys):
    closed = {"n": 0}
    backend = SimpleNamespace(close=lambda: closed.__setitem__("n", closed["n"] + 1))
    monkeypatch.setattr(
        "tokenjam.core.onboard_verify.open_read_backend",
        lambda config: (backend, "api", None),
    )
    monkeypatch.setattr(
        "tokenjam.core.onboard_verify.poll_for_first_span",
        lambda *a, **k: VerifyResult(True, 3.0, first_trace_id="t1"),
    )
    cmd_onboard._run_onboard_verification(_config(), "sdk", timeout_s=1.0)
    out = capsys.readouterr().out
    assert "Receiving telemetry" in out
    assert closed["n"] == 1  # backend closed even on the happy path


def test_runner_prints_persona_cause_on_timeout(monkeypatch, capsys):
    monkeypatch.setattr(
        "tokenjam.core.onboard_verify.open_read_backend",
        lambda config: (SimpleNamespace(close=lambda: None), "api", None),
    )
    monkeypatch.setattr(
        "tokenjam.core.onboard_verify.poll_for_first_span",
        lambda *a, **k: VerifyResult(False, 60.0),
    )
    cmd_onboard._run_onboard_verification(_config(), "claude_code", timeout_s=60.0)
    out = capsys.readouterr().out
    assert "No telemetry yet" in out
    assert "restart" in out.lower()  # persona-specific cause


def test_runner_reports_when_backend_unavailable(monkeypatch, capsys):
    monkeypatch.setattr(
        "tokenjam.core.onboard_verify.open_read_backend",
        lambda config: (None, None, "the database is locked"),
    )
    cmd_onboard._run_onboard_verification(_config(), "sdk")
    out = capsys.readouterr().out
    assert "Can't verify" in out
    assert "tj serve" in out


def test_maybe_verify_runs_directly_with_flag(monkeypatch):
    calls = []
    monkeypatch.setattr(
        cmd_onboard, "_run_onboard_verification",
        lambda config, persona: calls.append(persona),
    )
    cmd_onboard._maybe_verify_onboarding(_config(), persona="sdk", verify=True)
    assert calls == ["sdk"]


def test_maybe_verify_skips_noninteractive_without_flag(monkeypatch):
    calls = []
    monkeypatch.setattr(
        cmd_onboard, "_run_onboard_verification",
        lambda config, persona: calls.append(persona),
    )
    monkeypatch.setattr(cmd_onboard.sys.stdin, "isatty", lambda: False)
    cmd_onboard._maybe_verify_onboarding(_config(), persona="sdk", verify=False)
    assert calls == []


# --- --verify-only (the lean post-restart re-check, #102) -------------------


def test_verify_only_sdk_loads_config_and_polls(monkeypatch, tmp_path):
    cfg = tmp_path / ".tj" / "config.toml"
    cfg.parent.mkdir(parents=True)
    cfg.write_text('version = "1"\n')
    calls = []
    monkeypatch.setattr(cmd_onboard, "find_config_file", lambda: cfg)
    monkeypatch.setattr(cmd_onboard, "load_config", lambda _p: _config(), raising=False)
    monkeypatch.setattr(
        cmd_onboard, "_run_onboard_verification",
        lambda config, persona: calls.append(persona),
    )
    # `load_config` is imported lazily inside the helper.
    monkeypatch.setattr("tokenjam.core.config.load_config", lambda _p: _config())

    cmd_onboard._run_verify_only(_ctx(), claude_code=False, codex=False)
    assert calls == ["sdk"]


def test_verify_only_claude_code_reads_global_config(monkeypatch, tmp_path):
    home = tmp_path / "home"
    global_cfg = home / ".config" / "tj" / "config.toml"
    global_cfg.parent.mkdir(parents=True)
    global_cfg.write_text('version = "1"\n')
    monkeypatch.setenv("HOME", str(home))
    calls = []
    monkeypatch.setattr("tokenjam.core.config.load_config", lambda _p: _config())
    monkeypatch.setattr(
        cmd_onboard, "_run_onboard_verification",
        lambda config, persona: calls.append(persona),
    )
    cmd_onboard._run_verify_only(_ctx(), claude_code=True, codex=False)
    assert calls == ["claude_code"]


def test_verify_only_errors_cleanly_when_no_config(monkeypatch, capsys):
    monkeypatch.setattr(cmd_onboard, "find_config_file", lambda: None)
    called = []
    monkeypatch.setattr(
        cmd_onboard, "_run_onboard_verification",
        lambda config, persona: called.append(persona),
    )
    with pytest.raises(click.exceptions.Exit) as exc:
        cmd_onboard._run_verify_only(_ctx(), claude_code=False, codex=False)
    assert exc.value.exit_code == 1
    out = capsys.readouterr().out
    assert "No tj config found" in out
    assert called == []  # never polled
