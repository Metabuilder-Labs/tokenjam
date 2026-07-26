"""OPENAI_API_KEY handling. Zero API calls: these tests either unset the key
entirely (proving the workload refuses to run rather than silently trying an
unauthenticated call) or exercise `--dry-run`, which never imports `openai`.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from _shared import require_api_key

WORKLOADS_DIR = Path(__file__).resolve().parents[1]


def test_require_api_key_exits_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(SystemExit) as excinfo:
        require_api_key()
    assert excinfo.value.code == 1


def test_require_api_key_returns_the_key_when_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-a-real-key")
    assert require_api_key() == "sk-test-not-a-real-key"


def test_missing_key_error_message_never_echoes_a_key(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    """Even when a (fake) key IS set elsewhere in the environment under a
    different name, the missing-key error path for OPENAI_API_KEY must never
    print a key value; only ever the fixed instructional message."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("SOME_OTHER_KEY", "sk-should-never-appear-in-output")
    with pytest.raises(SystemExit):
        require_api_key()
    captured = capsys.readouterr()
    assert "sk-should-never-appear-in-output" not in captured.err
    assert "sk-should-never-appear-in-output" not in captured.out
    assert "OPENAI_API_KEY" in captured.err


def test_workload_script_fails_fast_without_key_in_live_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Running a workload WITHOUT --dry-run and WITHOUT the key set must fail
    immediately with a clear message and make no network attempt (a hang or
    a long delay here would itself indicate a call slipped through)."""
    env = {k: v for k, v in __import__("os").environ.items() if k != "OPENAI_API_KEY"}
    script = WORKLOADS_DIR / "oversized_model.py"
    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=str(WORKLOADS_DIR), capture_output=True, text=True, timeout=15, env=env,
    )
    assert result.returncode == 1
    assert "OPENAI_API_KEY" in result.stderr
    assert "--dry-run" in result.stderr
