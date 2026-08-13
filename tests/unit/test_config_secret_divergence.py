"""Diverged-secret detection at config-load time — and, more importantly, the
absence of a warning there.

`load_config` used to print a warning to stderr whenever a project-local config
shadowed the global one with a different `ingest_secret`. That fired on EVERY
invocation of every command, in front of unrelated output, on a fault the user
cannot act on from where they are standing, and (in the case that motivated the
change) while nothing was listening on the endpoint at all. A warning that is
always on is a warning people learn to read past.

The detection itself is kept and still exercised here: `find_diverged_secret_config`
is the shared helper, now consumed by `tj doctor`'s ingest-secret check, where
the verdict is queryable and repeatable rather than ambient. These assertions are
therefore inverted rather than deleted — the pairing that used to warn must now
be silent at load time AND still be detectable on demand.
"""
from __future__ import annotations

from pathlib import Path

from tokenjam.core.config import (
    find_diverged_secret_config,
    load_config,
)


def _write_config(path: Path, *, secret: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        'version = "1"\n'
        '[security]\n'
        f'ingest_secret = "{secret}"\n'
    )


def _raw(path: Path) -> dict:
    import sys

    if sys.version_info >= (3, 11):
        import tomllib
    else:
        import tomli as tomllib  # type: ignore[no-redef]
    with open(path, "rb") as f:
        return tomllib.load(f)


def test_load_config_is_silent_when_project_and_global_diverge(tmp_path, monkeypatch, capsys):
    """The divergence exists, and `load_config` says nothing about it."""
    project_local = tmp_path / "project" / ".tj" / "config.toml"
    global_cfg = tmp_path / "home" / ".config" / "tj" / "config.toml"
    _write_config(project_local, secret="A" * 64)
    _write_config(global_cfg, secret="B" * 64)

    monkeypatch.setattr(
        "tokenjam.core.config.SEARCH_PATHS",
        [project_local, global_cfg],
    )

    cfg = load_config(str(project_local))
    captured = capsys.readouterr()

    assert cfg.security.ingest_secret == "A" * 64
    assert "ingest_secret" not in captured.err
    assert "ingest_secret" not in captured.out

    # ...and the same state is still detectable on demand, which is what
    # `tj doctor`'s ingest-secret check reports.
    diverged = find_diverged_secret_config(project_local, _raw(project_local))
    assert diverged is not None
    assert diverged[0] == global_cfg


def test_no_divergence_when_only_one_config_exists(tmp_path, monkeypatch):
    """No global config → nothing to compare against."""
    project_local = tmp_path / ".tj" / "config.toml"
    _write_config(project_local, secret="A" * 64)

    monkeypatch.setattr(
        "tokenjam.core.config.SEARCH_PATHS",
        [project_local, Path("/nonexistent/global/config.toml")],
    )

    assert find_diverged_secret_config(project_local, _raw(project_local)) is None


def test_no_divergence_when_secrets_match(tmp_path, monkeypatch):
    """Same secret in both configs → the aligned state onboarding now produces."""
    project_local = tmp_path / "project" / ".tj" / "config.toml"
    global_cfg = tmp_path / "home" / ".config" / "tj" / "config.toml"
    _write_config(project_local, secret="SAME" + "x" * 60)
    _write_config(global_cfg, secret="SAME" + "x" * 60)

    monkeypatch.setattr(
        "tokenjam.core.config.SEARCH_PATHS",
        [project_local, global_cfg],
    )

    assert find_diverged_secret_config(project_local, _raw(project_local)) is None


def test_repeated_loads_stay_silent(tmp_path, monkeypatch, capsys):
    """No warning on the first load, and none on any later one either — the
    old guard suppressed repeats of a warning that should not fire at all."""
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
    assert "ingest_secret" not in captured.err
