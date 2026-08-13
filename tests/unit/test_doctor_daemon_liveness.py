"""`tj doctor` must FAIL when a background daemon is installed but not
actually running -- observed on a real machine: the launchd plist was
present (`RunAtLoad`/`KeepAlive` both true, no `Disabled` key) yet
`launchctl list` showed nothing loaded, with no crash and no error trace.
Nothing anywhere said ingestion had silently stopped.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from tokenjam.cli.cmd_doctor import _check_daemon_liveness


def test_ok_when_daemon_is_alive() -> None:
    check = _check_daemon_liveness(daemon_alive=True)
    assert check["level"] == "ok"


def test_info_when_nothing_installed_and_nothing_running(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No install and no running process is a legitimate pre-onboard / manual
    `tj serve` state -- not an error."""
    monkeypatch.setattr(
        "tokenjam.cli.cmd_doctor._daemon_install_path", lambda: None,
    )
    check = _check_daemon_liveness(daemon_alive=False)
    assert check["level"] == "info"


def test_errors_when_installed_but_not_running(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The exact production failure mode: a registration exists on disk but
    the process isn't actually up. Must FAIL, not warn."""
    plist = tmp_path / "com.tokenjam.serve.plist"
    plist.write_text("<plist/>")
    monkeypatch.setattr(
        "tokenjam.cli.cmd_doctor._daemon_install_path", lambda: plist,
    )
    check = _check_daemon_liveness(daemon_alive=False)
    assert check["level"] == "error"
    assert "not running" in check["message"]
    assert str(plist) in check["message"]
