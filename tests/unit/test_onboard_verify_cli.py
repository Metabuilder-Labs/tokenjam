"""Post-onboard verification wiring (#80): the prompt gate and the runner that
both onboarding paths call at their end."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from tokenjam.cli import cmd_onboard
from tokenjam.core.onboard_verify import VerifyResult


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
