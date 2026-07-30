"""A config WRITE must land in the file the invocation actually read.

`resolve_config_path()` answers "which file would this process discover",
which stops being the same question the moment a per-invocation
`tj --config PATH` override is in play: the override never reaches the
environment, so a rediscovery falls through to TJ_CONFIG or the search path
and names a different file. A writer that rediscovers therefore mutates an
unrelated config while leaving the one it read untouched.
"""
from __future__ import annotations

from pathlib import Path

from click.testing import CliRunner

from tokenjam.cli.main import cli
from tokenjam.core.config import active_config_path, load_config


def _write_config(path: Path, db_path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        'version = "1"\n'
        "[storage]\n"
        f'path = "{db_path}"\n'
    )


def test_active_config_path_reports_the_file_a_config_was_loaded_from(
    tmp_path, monkeypatch,
):
    decoy = tmp_path / "decoy.toml"
    decoy.write_text('version = "1"\n')
    explicit = tmp_path / "explicit.toml"
    explicit.write_text('version = "1"\n')
    monkeypatch.setenv("TJ_CONFIG", str(decoy))

    config = load_config(str(explicit))

    # The API has no click context to forward an override through, so its
    # write target comes from the config object itself.
    assert active_config_path(config) == explicit.resolve()


def test_active_config_path_is_none_for_a_fileless_config():
    """A config that never came from a file has no write target of its own —
    the call site must fall back to discovery rather than inventing one."""
    from tokenjam.core.config import TjConfig

    assert active_config_path(TjConfig(version="1")) is None


def test_budget_write_lands_in_the_explicit_config_not_the_env_one(
    tmp_path, monkeypatch,
):
    """`tj --config X budget --daily N` must update X.

    The decoy is a valid, discoverable config pointed at by TJ_CONFIG. Before
    the fix the budget landed there and X was left unchanged, so the very next
    read (which honors --config) showed no budget at all.
    """
    explicit = tmp_path / "explicit" / "config.toml"
    _write_config(explicit, tmp_path / "explicit.duckdb")
    decoy = tmp_path / "decoy" / "config.toml"
    _write_config(decoy, tmp_path / "decoy.duckdb")
    monkeypatch.setenv("TJ_CONFIG", str(decoy))

    result = CliRunner().invoke(
        cli, ["--config", str(explicit), "budget", "--daily", "8.0"]
    )

    assert result.exit_code == 0, result.output
    assert load_config(str(explicit)).defaults.budget.daily_usd == 8.0
    assert load_config(str(decoy)).defaults.budget.daily_usd is None


def test_doctor_reports_the_explicit_config_not_the_env_one(tmp_path, monkeypatch):
    """`tj --config X doctor` must name X as the live config file — every
    other check reads the config loaded from X, so naming a different file
    describes an install the user is not running."""
    explicit = tmp_path / "explicit" / "config.toml"
    _write_config(explicit, tmp_path / "explicit.duckdb")
    decoy = tmp_path / "decoy" / "config.toml"
    _write_config(decoy, tmp_path / "decoy.duckdb")
    monkeypatch.setenv("TJ_CONFIG", str(decoy))

    result = CliRunner().invoke(
        cli, ["--config", str(explicit), "doctor", "--json"]
    )

    assert "explicit" in result.output
    assert str(decoy) not in result.output
