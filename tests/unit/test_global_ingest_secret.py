"""One ingest secret per machine, and it lives in the global config.

The daemon is launched from a unit file that names `~/.config/tj/config.toml`
explicitly, so it always authenticates with the secret stored there whatever
directory anything else runs from. A secret minted into a project-local
`.tj/config.toml` is therefore one the daemon can only reject, and every span
signed with it 401s silently.

`ensure_global_ingest_secret` is the one place a secret is minted, and
`align_project_secret_to_global` repairs a project file left over from an
older install. The global value is always the survivor: the managed shell
block, the daemon unit file, and every already-working integration carry it.
"""
from __future__ import annotations

import pytest

from tokenjam.core.config import (
    align_project_secret_to_global,
    ensure_global_ingest_secret,
    global_config_path,
    global_ingest_secret,
    load_config,
)


@pytest.fixture
def home(tmp_path, monkeypatch):
    h = tmp_path / "home"
    h.mkdir()
    monkeypatch.setattr("pathlib.Path.home", lambda: h)
    return h


def test_mints_and_stores_when_no_global_config_exists(home):
    assert not global_config_path().exists()
    secret, minted = ensure_global_ingest_secret()

    assert minted is True
    assert secret
    assert global_config_path().exists()
    assert load_config(str(global_config_path())).security.ingest_secret == secret


def test_adopts_the_existing_global_secret_instead_of_rotating(home):
    first, minted_first = ensure_global_ingest_secret()
    second, minted_second = ensure_global_ingest_secret()

    assert minted_first is True
    assert minted_second is False
    assert second == first, "re-onboarding must not rotate a working secret"


def test_alignment_makes_the_global_secret_the_survivor(home, tmp_path, monkeypatch):
    """A project-local config with its own secret is rewritten to the global
    one, never the other way round: the shell block every configured
    integration reads carries the global value."""
    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.chdir(project)
    project_config = project / ".tj" / "config.toml"
    project_config.parent.mkdir()
    project_config.write_text(
        "# a comment onboarding wrote\n"
        'version = "1"\n'
        "[security]\n"
        'ingest_secret = "project-local-secret"\n'
    )

    global_secret, _ = ensure_global_ingest_secret()
    repaired = align_project_secret_to_global(global_secret)

    assert repaired == project_config.resolve()
    assert load_config(str(project_config)).security.ingest_secret == global_secret
    assert global_ingest_secret() == global_secret, "the global config is untouched"
    # A secret repair must not silently delete the rest of the file.
    assert "# a comment onboarding wrote" in project_config.read_text()


def test_alignment_is_a_noop_when_already_aligned(home, tmp_path, monkeypatch):
    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.chdir(project)
    project_config = project / ".tj" / "config.toml"
    project_config.parent.mkdir()
    secret, _ = ensure_global_ingest_secret()
    project_config.write_text(
        'version = "1"\n[security]\ningest_secret = "' + secret + '"\n'
    )
    before = project_config.read_text()

    assert align_project_secret_to_global(secret) is None
    assert project_config.read_text() == before


def test_alignment_is_a_noop_with_no_project_config(home, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert align_project_secret_to_global("some-secret") is None
