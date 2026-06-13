"""SDK config discovery honors TJ_CONFIG (#196).

`ensure_initialised()` (reached by `@watch()` / any `patch_*()`) bootstrapped
against a bare `load_config()`, so a process that set `TJ_CONFIG` to point at a
project/custom config was ignored and the SDK wrote spans into the GLOBAL
DuckDB. These tests pin the fix: `load_config()` honors `TJ_CONFIG` (and the
existing search-path order) when no explicit path is passed, so SDK-bootstrapped
processes write to the intended DB — never the global one.
"""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import duckdb
import pytest

from tokenjam.core.config import load_config

# Worktree root (tokenjam-sdk196/), so the subprocess imports THIS checkout —
# the editable install's .pth points at a different tree.
_REPO_ROOT = Path(__file__).resolve().parents[2]


def _write_config(path: Path, db_path: Path, port: int) -> None:
    path.write_text(textwrap.dedent(f"""
        version = "1"
        [storage]
        path = "{db_path}"
        [api]
        port = {port}
        [security]
        ingest_secret = "test-secret-196"
    """))


# --------------------------------------------------------------------------- #
# load_config() — the single discovery function the SDK + CLI share (#196 AC1)
# --------------------------------------------------------------------------- #
def test_load_config_honors_tj_config(monkeypatch, tmp_path):
    cfg = tmp_path / "proj.toml"
    db = tmp_path / "intended.duckdb"
    _write_config(cfg, db, 59991)
    monkeypatch.setenv("TJ_CONFIG", str(cfg))
    assert load_config().storage.path == str(db)


def test_load_config_explicit_path_beats_tj_config(monkeypatch, tmp_path):
    env_cfg = tmp_path / "env.toml"
    _write_config(env_cfg, tmp_path / "env.duckdb", 59992)
    explicit = tmp_path / "explicit.toml"
    _write_config(explicit, tmp_path / "explicit.duckdb", 59993)
    monkeypatch.setenv("TJ_CONFIG", str(env_cfg))
    # An explicit path argument still wins over the env var (CLI --config beats
    # TJ_CONFIG via Click's precedence; load_config must preserve that).
    assert load_config(str(explicit)).storage.path == str(tmp_path / "explicit.duckdb")


def test_load_config_tj_config_missing_file_raises(monkeypatch, tmp_path):
    # A TJ_CONFIG pointing at a missing file fails loudly rather than silently
    # falling back to the global config — matching the CLI, and avoiding the
    # exact silent-global-fallback hazard #196 is about.
    monkeypatch.setenv("TJ_CONFIG", str(tmp_path / "nope.toml"))
    with pytest.raises(FileNotFoundError):
        load_config()


# --------------------------------------------------------------------------- #
# End-to-end: a process that sets TJ_CONFIG + uses @watch writes spans to the
# TJ_CONFIG DB, not the global one (#196 AC2). Runs in a SUBPROCESS so the OTel
# global TracerProvider + the bootstrap singleton start clean — exactly the
# real-world "a process that sets TJ_CONFIG" scenario.
# --------------------------------------------------------------------------- #
def _count_spans(db_path: Path) -> int:
    conn = duckdb.connect(str(db_path), read_only=True)
    try:
        return conn.execute("SELECT COUNT(*) FROM spans").fetchone()[0]
    finally:
        conn.close()


def test_sdk_bootstrap_writes_to_tj_config_db_not_global(tmp_path):
    # Intended config (pointed at by TJ_CONFIG)
    intended_cfg = tmp_path / "proj" / "config.toml"
    intended_cfg.parent.mkdir()
    intended_db = tmp_path / "intended.duckdb"
    _write_config(intended_cfg, intended_db, 59994)

    # Decoy GLOBAL config under a fake HOME — what the SDK would have picked if
    # it ignored TJ_CONFIG. Its DB must stay empty for the test to pass.
    fake_home = tmp_path / "home"
    global_cfg = fake_home / ".config" / "tj" / "config.toml"
    global_cfg.parent.mkdir(parents=True)
    global_db = tmp_path / "global.duckdb"
    _write_config(global_cfg, global_db, 59995)

    workdir = tmp_path / "cwd"  # empty cwd → no tokenjam.toml / .tj/config.toml
    workdir.mkdir()

    script = tmp_path / "run.py"
    script.write_text(textwrap.dedent("""
        from tokenjam.sdk.agent import watch

        @watch(agent_id="probe-196")
        def go():
            return "ok"

        go()
    """))

    env = dict(os.environ)
    env["TJ_CONFIG"] = str(intended_cfg)
    env["HOME"] = str(fake_home)
    # Prepend the worktree so the subprocess imports the fixed code (the editable
    # .pth resolves a different tree).
    env["PYTHONPATH"] = str(_REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")

    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=str(workdir), env=env,
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, f"subprocess failed:\n{result.stderr}"

    # Spans landed in the TJ_CONFIG-pointed DB ...
    assert intended_db.exists(), f"intended DB never created.\nstderr={result.stderr}"
    assert _count_spans(intended_db) >= 1, "no spans in the TJ_CONFIG DB"

    # ... and NOT in the global one (ideally never even created).
    if global_db.exists():
        assert _count_spans(global_db) == 0, "spans leaked into the global DB"
