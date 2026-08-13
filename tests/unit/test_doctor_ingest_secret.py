"""`tj doctor`'s ingest-secret check: beyond "is a secret set" — do every
config on this machine's search path agree on it? A project-local
`.tj/config.toml` and the global `~/.config/tj/config.toml` can carry
different secrets for one store; span pushes 401 silently with nothing else
surfacing the mismatch (see `core/config.find_diverged_secret_config`). Doctor
is the ONLY place this is reported: the per-invocation load-time warning that
used to duplicate it is gone (see tests/unit/test_config_secret_divergence.py)."""
from __future__ import annotations

from pathlib import Path

from tokenjam.core.config import load_config


def _write_config(path: Path, *, secret: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        'version = "1"\n'
        '[security]\n'
        f'ingest_secret = "{secret}"\n'
    )


def test_check_flags_a_diverged_shadow_config(tmp_path, monkeypatch):
    from tokenjam.cli.cmd_doctor import _check_ingest_secret

    project_local = tmp_path / "project" / ".tj" / "config.toml"
    global_cfg = tmp_path / "home" / ".config" / "tj" / "config.toml"
    _write_config(project_local, secret="A" * 64)
    _write_config(global_cfg, secret="B" * 64)
    monkeypatch.setattr(
        "tokenjam.core.config.SEARCH_PATHS", [project_local, global_cfg],
    )

    config = load_config(str(project_local))
    check = _check_ingest_secret(config)
    assert check["level"] == "warning"
    assert "differs between" in check["message"]
    assert str(project_local) in check["message"]
    assert str(global_cfg) in check["message"]


def test_check_is_ok_when_no_shadow_config_diverges(tmp_path, monkeypatch):
    from tokenjam.cli.cmd_doctor import _check_ingest_secret

    project_local = tmp_path / "project" / ".tj" / "config.toml"
    _write_config(project_local, secret="A" * 64)
    monkeypatch.setattr(
        "tokenjam.core.config.SEARCH_PATHS",
        [project_local, Path("/nonexistent/global/config.toml")],
    )

    config = load_config(str(project_local))
    check = _check_ingest_secret(config)
    assert check["level"] == "ok"


def test_check_still_warns_on_no_secret_configured():
    from tokenjam.cli.cmd_doctor import _check_ingest_secret
    from tokenjam.core.config import SecurityConfig, TjConfig

    config = TjConfig(version="1", security=SecurityConfig(ingest_secret=None))
    check = _check_ingest_secret(config)
    assert check["level"] == "warning"
    assert "No ingest secret set" in check["message"]
