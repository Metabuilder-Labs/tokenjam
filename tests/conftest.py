"""
Shared test configuration.

Note: Many test files define their own db/config fixtures locally.
Local fixtures take precedence over conftest fixtures.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

# The developer's REAL home, captured at collection time (before anything in
# this file patches HOME). `tokenjam.core.config.SEARCH_PATHS` bakes
# `Path.home()` into a module-level constant at import time, so a test can end
# up resolving the real global config/store even after HOME is patched later
# — this is the fixed point the guard below checks against.
_REAL_HOME = Path(os.path.expanduser("~")).resolve()
_REAL_GUARDED_ROOTS = (_REAL_HOME / ".tj", _REAL_HOME / ".config" / "tj")


def _assert_not_real_tj_home(path: Path) -> None:
    resolved = path.resolve()
    for root in _REAL_GUARDED_ROOTS:
        if resolved == root or root in resolved.parents:
            raise RuntimeError(
                f"tests/conftest.py: refusing to open {resolved} — it is "
                f"under the real {root}. Tests must never touch the "
                f"developer's live ~/.tj / ~/.config/tj (lock-contends with a "
                f"running `tj serve` / CLI). Isolate the db/config path under "
                f"tmp_path instead."
            )


@pytest.fixture(autouse=True, scope="session")
def _tj_isolated_home(tmp_path_factory):
    """
    Session-wide backstop: redirect every '~'-relative config/db
    lookup away from the developer's real ~/.tj / ~/.config/tj so the test
    suite never contends for the DuckDB lock with a concurrently running
    `tj serve` / CLI against the real store (and vice versa).

    This does not replace a test's own explicit tmp_path isolation — it's a
    safety net for any test/fixture that forgets to set one up. `HOME` covers
    runtime `Path(...).expanduser()` calls (e.g. StorageConfig's default
    "~/.tj/telemetry.duckdb"); `SEARCH_PATHS` is patched separately because
    it is a module-level constant computed from `Path.home()` at import time,
    so re-pointing HOME alone can't change it after the fact.
    """
    fake_home = tmp_path_factory.mktemp("tj-home")
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("HOME", str(fake_home))
        mp.setenv("USERPROFILE", str(fake_home))  # Windows Path.home()

        from tokenjam.core import config as cfg_mod

        mp.setattr(
            cfg_mod,
            "SEARCH_PATHS",
            [
                Path("tokenjam.toml"),
                Path(".tj/config.toml"),
                fake_home / ".config" / "tj" / "config.toml",
            ],
        )
        yield fake_home


@pytest.fixture(autouse=True)
def _tj_guard_real_home_db(monkeypatch):
    """Fail loudly if any test opens a DuckDB file under the real $HOME/.tj."""
    from tokenjam.core import db as db_mod

    original_init = db_mod.DuckDBBackend.__init__

    def _guarded_init(self, config):
        _assert_not_real_tj_home(Path(config.path).expanduser())
        original_init(self, config)

    monkeypatch.setattr(db_mod.DuckDBBackend, "__init__", _guarded_init)
