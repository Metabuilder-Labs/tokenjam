"""Doctor coverage for the daemon lifecycle gaps reported in #614."""
from __future__ import annotations

import json

from tokenjam.cli.cmd_doctor import _check_daemon_lifecycle
from tokenjam.core.config import ApiConfig, TjConfig
from tokenjam.core.server_state import DaemonUnitState, ServeProcess


def _config() -> TjConfig:
    return TjConfig(version="1", api=ApiConfig(port=7391))


def _patch_state_path(monkeypatch, path) -> None:
    monkeypatch.setattr("tokenjam.core.server_state.server_state_path", lambda: path)


def test_doctor_warns_when_launchd_plist_exists_but_is_not_loaded(
    tmp_path, monkeypatch,
):
    plist = tmp_path / "Library/LaunchAgents/com.tokenjam.serve.plist"
    plist.parent.mkdir(parents=True)
    plist.write_text("<plist/>")
    monkeypatch.setattr(
        "tokenjam.core.server_state.inspect_daemon_unit",
        lambda: DaemonUnitState("launchd", plist, True, False, None),
    )
    monkeypatch.setattr("tokenjam.core.server_state.list_serve_processes", lambda: [])
    _patch_state_path(monkeypatch, tmp_path / "missing.state")

    checks = _check_daemon_lifecycle(_config())
    service = next(check for check in checks if check["name"] == "Daemon service")

    assert service["level"] == "warning"
    assert "not loaded" in service["message"]
    assert "tj onboard" in service["message"]


def test_doctor_reports_multiple_live_serve_instances(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "tokenjam.core.server_state.inspect_daemon_unit",
        lambda: DaemonUnitState(None, None, False, None, None),
    )
    monkeypatch.setattr(
        "tokenjam.core.server_state.list_serve_processes",
        lambda: [
            ServeProcess(111, "/usr/local/bin/tj serve"),
            ServeProcess(222, "/opt/venv/bin/tj serve --port 9341"),
        ],
    )
    _patch_state_path(monkeypatch, tmp_path / "missing.state")

    checks = _check_daemon_lifecycle(_config())
    instances = next(check for check in checks if check["name"] == "Daemon instances")

    assert instances["level"] == "warning"
    assert "111" in instances["message"]
    assert "222" in instances["message"]


def test_doctor_reports_server_state_pointing_at_a_dead_pid(tmp_path, monkeypatch):
    state_path = tmp_path / "server.state"
    state_path.write_text(json.dumps({"pid": 333, "port": 8123, "config_path": None}))
    monkeypatch.setattr(
        "tokenjam.core.server_state.inspect_daemon_unit",
        lambda: DaemonUnitState(None, None, False, None, None),
    )
    monkeypatch.setattr("tokenjam.core.server_state.list_serve_processes", lambda: [])
    monkeypatch.setattr("tokenjam.core.server_state.is_pid_alive", lambda pid: False)
    _patch_state_path(monkeypatch, state_path)

    checks = _check_daemon_lifecycle(_config())
    state = next(check for check in checks if check["name"] == "Server state")

    assert state["level"] == "warning"
    assert "dead PID 333" in state["message"]
    assert "port 8123" in state["message"]


def test_doctor_reports_live_state_on_the_wrong_port(tmp_path, monkeypatch):
    state_path = tmp_path / "server.state"
    state_path.write_text(json.dumps({"pid": 444, "port": 9341, "config_path": None}))
    monkeypatch.setattr(
        "tokenjam.core.server_state.inspect_daemon_unit",
        lambda: DaemonUnitState(None, None, False, None, None),
    )
    monkeypatch.setattr(
        "tokenjam.core.server_state.list_serve_processes",
        lambda: [ServeProcess(444, "/opt/venv/bin/tj serve --port 9341")],
    )
    monkeypatch.setattr("tokenjam.core.server_state.is_pid_alive", lambda pid: True)
    monkeypatch.setattr("tokenjam.core.server_state.is_serve_process", lambda pid: True)
    _patch_state_path(monkeypatch, state_path)

    checks = _check_daemon_lifecycle(_config())
    state = next(check for check in checks if check["name"] == "Server state")

    assert state["level"] == "warning"
    assert "port 9341" in state["message"]
    assert "expects 7391" in state["message"]
