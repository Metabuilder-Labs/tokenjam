"""Doctor path display (#104): long paths that wrap mid-filename are uncopyable,
so the home dir is collapsed to ``~`` in the paths doctor prints."""
from __future__ import annotations

from types import SimpleNamespace

from tokenjam.cli.cmd_doctor import _check_config, _check_db


def test_config_path_message_collapses_home(monkeypatch, tmp_path):
    home = tmp_path / "corp" / "very-long-username" / "home"
    cfg = home / ".config" / "tj" / "config.toml"
    cfg.parent.mkdir(parents=True)
    cfg.write_text('version = "1"\n')
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr("tokenjam.cli.cmd_doctor.find_config_file", lambda: cfg)
    monkeypatch.setattr("tokenjam.cli.cmd_doctor.load_config", lambda _p: None)

    check = _check_config()
    assert check["level"] == "ok"
    assert "~/.config/tj/config.toml" in check["message"]
    assert str(home) not in check["message"]


def test_db_path_message_collapses_home(monkeypatch, tmp_path):
    home = tmp_path / "corp" / "very-long-username" / "home"
    db_path = home / ".tj" / "telemetry.duckdb"
    db_path.parent.mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    config = SimpleNamespace(storage=SimpleNamespace(path=str(db_path)))

    check = _check_db(config)
    assert check["level"] == "ok"
    assert "~/.tj/telemetry.duckdb" in check["message"]
    assert str(home) not in check["message"]
