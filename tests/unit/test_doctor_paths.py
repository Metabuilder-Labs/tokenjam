"""Doctor path display (#104): long paths that wrap mid-filename are uncopyable,
so the home dir is collapsed to ``~`` in the paths doctor prints."""
from __future__ import annotations

import json
from types import SimpleNamespace

from tokenjam.cli.cmd_doctor import _check_config, _check_db


def test_config_path_message_collapses_home(monkeypatch, tmp_path):
    home = tmp_path / "corp" / "very-long-username" / "home"
    cfg = home / ".config" / "tj" / "config.toml"
    cfg.parent.mkdir(parents=True)
    cfg.write_text('version = "1"\n')
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr("tokenjam.cli.cmd_doctor.resolve_config_path", lambda *_a: cfg)
    monkeypatch.setattr("tokenjam.cli.cmd_doctor.load_config", lambda _p: None)

    check = _check_config()
    assert check["level"] == "ok"
    assert "~/.config/tj/config.toml" in check["message"]
    assert str(home) not in check["message"]


def test_check_config_honors_tj_config(monkeypatch, tmp_path):
    """`_check_config` must resolve the same file `load_config` reads —
    otherwise doctor reports "No config file found" while every other check
    (which reads `ctx.obj["config"]`, loaded via `load_config`/TJ_CONFIG) is
    passing fine against a real, TJ_CONFIG-pointed config."""
    cfg = tmp_path / "custom-location.toml"
    cfg.write_text('version = "1"\n')
    monkeypatch.setenv("TJ_CONFIG", str(cfg))
    # No SEARCH_PATHS patching here — the point is that TJ_CONFIG alone,
    # with nothing discoverable at the default search paths, is enough.
    import tokenjam.core.config as cfg_mod
    monkeypatch.setattr(cfg_mod, "SEARCH_PATHS", [])

    check = _check_config()
    assert check["level"] == "ok"
    assert "custom-location.toml" in check["message"]


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


# -- #775: the writability probe beside the process's own bounded connection --

def _storage(db_path, **overrides):
    from tokenjam.core.config import StorageConfig
    return StorageConfig(path=str(db_path), **overrides)


def test_check_db_is_ok_beside_the_backend_connection(tmp_path):
    """`main` opens the backend (bounded settings) before doctor runs. A probe
    that opened a second, differently-configured connection to the same file
    got DuckDB's "different configuration" error and reported a healthy
    database as unwritable."""
    from tokenjam.core.db import open_db

    storage = _storage(tmp_path / "t.duckdb")
    backend = open_db(storage)
    try:
        check = _check_db(SimpleNamespace(storage=storage), backend)
    finally:
        backend.close()
    assert check["level"] == "ok", check["message"]
    assert "different configuration" not in check["message"]


def test_check_db_is_ok_beside_a_live_connection_it_was_not_handed(tmp_path):
    """Even without the backend object, the probe opens the file with the
    backend's own settings, so DuckDB's per-file instance accepts it."""
    from tokenjam.core.db import open_db

    storage = _storage(tmp_path / "t.duckdb")
    backend = open_db(storage)
    try:
        check = _check_db(SimpleNamespace(storage=storage))
    finally:
        backend.close()
    assert check["level"] == "ok", check["message"]


def test_check_db_still_reports_an_unopenable_database_as_an_error(tmp_path):
    blocker = tmp_path / "not-a-dir"
    blocker.write_text("")
    storage = _storage(blocker / "t.duckdb")

    check = _check_db(SimpleNamespace(storage=storage))
    assert check["level"] == "error"
    assert check["message"].startswith("Cannot open database:")


def test_check_db_still_downgrades_a_lock_held_elsewhere(tmp_path, monkeypatch):
    import duckdb

    def held(*_a, **_k):
        raise duckdb.IOException(
            'IO Error: Could not set lock on file "t.duckdb": Conflicting lock '
            "is held in /usr/bin/python (PID 4242)"
        )

    monkeypatch.setattr("tokenjam.core.db.duckdb.connect", held)
    check = _check_db(SimpleNamespace(storage=_storage(tmp_path / "t.duckdb")))
    assert check["level"] == "info"
    assert "write lock held by another process" in check["message"]


def test_doctor_command_reports_the_database_writable(tmp_path):
    """End to end through `main`, which opens the backend before dispatch."""
    from unittest.mock import patch

    from click.testing import CliRunner

    from tokenjam.cli.main import cli
    from tokenjam.core.config import TjConfig

    config = TjConfig(version="1")
    config.storage.path = str(tmp_path / "t.duckdb")
    with patch("tokenjam.cli.main.load_config", return_value=config):
        result = CliRunner().invoke(cli, ["--json", "doctor"])
    checks = json.loads(result.output)
    db_check = next(c for c in checks if c["name"] == "DuckDB writable")
    assert db_check["level"] == "ok", db_check["message"]


# -- #775 item 2: the settings fallback must not misreport other errors --

def test_connect_bounded_does_not_blame_the_settings_for_a_config_mismatch(tmp_path, caplog):
    import duckdb
    import pytest

    from tokenjam.core.db import connect_bounded

    path = str(tmp_path / "t.duckdb")
    first = duckdb.connect(path)  # default settings, so the bounded open mismatches
    try:
        with caplog.at_level("WARNING", logger="tokenjam.db"):
            with pytest.raises(duckdb.ConnectionException):
                connect_bounded(path, "2GB", 4)
    finally:
        first.close()
    assert "rejected by DuckDB" not in caplog.text


def test_connect_bounded_does_not_blame_the_settings_for_a_held_lock(tmp_path, caplog, monkeypatch):
    import duckdb
    import pytest

    from tokenjam.core.db import connect_bounded

    def held(*_a, **_k):
        raise duckdb.IOException("IO Error: Could not set lock on file: Conflicting lock is held")

    monkeypatch.setattr("tokenjam.core.db.duckdb.connect", held)
    with caplog.at_level("WARNING", logger="tokenjam.db"):
        with pytest.raises(duckdb.IOException):
            connect_bounded(str(tmp_path / "t.duckdb"), "2GB", 4)
    assert "rejected by DuckDB" not in caplog.text


def test_connect_bounded_still_degrades_a_rejected_override(tmp_path, caplog):
    from tokenjam.core.db import connect_bounded

    with caplog.at_level("WARNING", logger="tokenjam.db"):
        conn = connect_bounded(str(tmp_path / "t.duckdb"), "banana", 4)
    try:
        limit = conn.execute("SELECT current_setting('memory_limit')").fetchone()[0]
    finally:
        conn.close()
    assert "rejected by DuckDB" in caplog.text
    assert "using the defaults 2GB / 4" in caplog.text
    assert limit  # opened with the defaults rather than failing
