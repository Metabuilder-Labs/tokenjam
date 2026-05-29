"""
Test the diverged-secret warning at config-load time (#68 §5).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from tokenjam.core.config import (
    SEARCH_PATHS,
    _reset_secret_divergence_warning,
    load_config,
)


@pytest.fixture(autouse=True)
def _reset_warning_state():
    """Reset the once-per-process warning guard before every test."""
    _reset_secret_divergence_warning()


def _write_config(path: Path, *, secret: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        'version = "1"\n'
        '[security]\n'
        f'ingest_secret = "{secret}"\n'
    )


def test_warns_when_project_and_global_diverge(tmp_path, monkeypatch, capsys):
    """Project-local + global with different secrets → stderr warning."""
    project_local = tmp_path / "project" / ".tj" / "config.toml"
    global_cfg = tmp_path / "home" / ".config" / "tj" / "config.toml"
    _write_config(project_local, secret="A" * 64)
    _write_config(global_cfg, secret="B" * 64)

    # Point the discovery + the helper at our temp paths.
    monkeypatch.setattr(
        "tokenjam.core.config.SEARCH_PATHS",
        [project_local, global_cfg],
    )

    cfg = load_config(str(project_local))
    captured = capsys.readouterr()

    assert cfg.security.ingest_secret == "A" * 64
    assert "warning: ingest_secret differs" in captured.err
    assert ".tj/config.toml" in captured.err or str(project_local) in captured.err


def test_no_warning_when_only_one_config_exists(tmp_path, monkeypatch, capsys):
    """No global config → nothing to compare against → no warning."""
    project_local = tmp_path / ".tj" / "config.toml"
    _write_config(project_local, secret="A" * 64)

    monkeypatch.setattr(
        "tokenjam.core.config.SEARCH_PATHS",
        [project_local, Path("/nonexistent/global/config.toml")],
    )

    load_config(str(project_local))
    captured = capsys.readouterr()

    assert "warning: ingest_secret" not in captured.err


def test_no_warning_when_secrets_match(tmp_path, monkeypatch, capsys):
    """Same secret in both configs → no divergence → no warning."""
    project_local = tmp_path / "project" / ".tj" / "config.toml"
    global_cfg = tmp_path / "home" / ".config" / "tj" / "config.toml"
    _write_config(project_local, secret="SAME" + "x" * 60)
    _write_config(global_cfg, secret="SAME" + "x" * 60)

    monkeypatch.setattr(
        "tokenjam.core.config.SEARCH_PATHS",
        [project_local, global_cfg],
    )

    load_config(str(project_local))
    captured = capsys.readouterr()
    assert "warning: ingest_secret" not in captured.err


def test_warning_fires_at_most_once_per_process(tmp_path, monkeypatch, capsys):
    """Repeated load_config calls in one process emit the warning once."""
    project_local = tmp_path / "project" / ".tj" / "config.toml"
    global_cfg = tmp_path / "home" / ".config" / "tj" / "config.toml"
    _write_config(project_local, secret="A" * 64)
    _write_config(global_cfg, secret="B" * 64)

    monkeypatch.setattr(
        "tokenjam.core.config.SEARCH_PATHS",
        [project_local, global_cfg],
    )

    load_config(str(project_local))
    load_config(str(project_local))
    load_config(str(project_local))

    captured = capsys.readouterr()
    assert captured.err.count("warning: ingest_secret differs") == 1
