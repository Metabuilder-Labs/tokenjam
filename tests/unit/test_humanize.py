"""Pure-stdlib display helpers: token human-sizing and home-dir path abbreviation."""
from __future__ import annotations

from pathlib import Path

from tokenjam.utils.humanize import display_path, format_tokens


# --- format_tokens ----------------------------------------------------------


def test_format_tokens_sub_thousand_is_raw():
    assert format_tokens(0) == "0"
    assert format_tokens(999) == "999"


def test_format_tokens_thousands_use_k():
    assert format_tokens(1_000) == "1.0k"
    assert format_tokens(42_300) == "42.3k"
    assert format_tokens(999_000) == "999.0k"


def test_format_tokens_millions_use_m():
    assert format_tokens(1_000_000) == "1.0M"
    assert format_tokens(2_500_000) == "2.5M"


# --- display_path -----------------------------------------------------------


def test_collapses_home_prefix_to_tilde(monkeypatch, tmp_path):
    home = tmp_path / "very" / "deep" / "corporate" / "home-dir"
    home.mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    assert display_path(home / ".config" / "tj" / "config.toml") == "~/.config/tj/config.toml"


def test_home_itself_is_bare_tilde(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    assert display_path(tmp_path) == "~"


def test_non_home_path_unchanged(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    (tmp_path / "home").mkdir()
    outside = "/private/tmp/workdir/.claude/settings.json"
    assert display_path(outside) == outside


def test_sibling_of_home_not_collapsed(monkeypatch, tmp_path):
    # A path that merely shares the home prefix as a string (no `/` boundary)
    # must not be mangled, e.g. `/home/bob-backup` when HOME is `/home/bob`.
    home = tmp_path / "bob"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    sibling = str(home) + "-backup/file"
    assert display_path(sibling) == sibling


def test_accepts_path_objects_and_never_raises(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    assert display_path(Path(tmp_path) / "x") == "~/x"
