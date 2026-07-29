"""Removal store for the context-audit page.

These are worth their length: every test here guards a way the feature could
lose a user's file. The store's whole promise is that Remove is reversible, so
"the bytes came back" and "we refused rather than guessed" are the properties
under test, not the shape of the JSON.
"""
from __future__ import annotations

import json

import pytest

from tokenjam.core.summarize.context_quarantine import (
    KIND_FILE,
    KIND_HOOK,
    list_removals,
    quarantine_root,
    remove_file,
    remove_hook,
    restore,
)
from tokenjam.core.summarize.session import SummarizeRefused


def test_removed_file_leaves_its_original_path_and_survives_in_quarantine(tmp_path):
    # Arrange
    home = tmp_path / "home"
    home.mkdir()
    target = tmp_path / "rules" / "testing.md"
    target.parent.mkdir()
    target.write_text("mandatory 80% coverage", encoding="utf-8")

    # Act
    rec = remove_file(target, home=home)

    # Assert
    assert not target.exists()
    assert rec.kind == KIND_FILE
    payload = quarantine_root(home) / "files" / rec.payload
    assert payload.read_text(encoding="utf-8") == "mandatory 80% coverage"


def test_restore_puts_the_exact_bytes_back_and_clears_the_record(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    target = tmp_path / "CLAUDE.md"
    target.write_text("# rules\nsome content\n", encoding="utf-8")

    rec = remove_file(target, home=home)
    restore(rec.record_id, home=home)

    assert target.read_text(encoding="utf-8") == "# rules\nsome content\n"
    assert list_removals(home=home) == []


def test_restore_refuses_when_something_new_sits_at_the_original_path(tmp_path):
    """The user recreated the file after removing it. Restoring would clobber
    the newer content, so it is refused and reported rather than silently won."""
    home = tmp_path / "home"
    home.mkdir()
    target = tmp_path / "CLAUDE.md"
    target.write_text("original", encoding="utf-8")
    rec = remove_file(target, home=home)
    target.write_text("i wrote this afterwards", encoding="utf-8")

    listed = [r for r in list_removals(home=home) if r["record_id"] == rec.record_id][0]
    assert listed["restorable"] is False
    assert "already exists" in listed["reason"]
    with pytest.raises(SummarizeRefused):
        restore(rec.record_id, home=home)
    assert target.read_text(encoding="utf-8") == "i wrote this afterwards"


def test_a_symlink_is_refused_rather_than_followed(tmp_path):
    """Moving through a symlink would quarantine the TARGET while recording the
    link's path, so the restore would put the file back in the wrong place."""
    home = tmp_path / "home"
    home.mkdir()
    real = tmp_path / "real.md"
    real.write_text("real content", encoding="utf-8")
    link = tmp_path / "link.md"
    link.symlink_to(real)

    with pytest.raises(SummarizeRefused, match="symlink"):
        remove_file(link, home=home)
    assert real.read_text(encoding="utf-8") == "real content"


def test_a_directory_is_refused(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    d = tmp_path / "rules"
    d.mkdir()
    (d / "keep.md").write_text("x", encoding="utf-8")

    with pytest.raises(SummarizeRefused, match="directory"):
        remove_file(d, home=home)
    assert (d / "keep.md").exists()


def _settings(path, command="echo hi", event="Stop"):
    path.write_text(json.dumps({
        "model": "opus",
        "hooks": {
            event: [{"matcher": "", "hooks": [{"type": "command", "command": command}]}],
            "PreToolUse": [{"matcher": "Bash", "hooks": [{"type": "command", "command": "guard.sh"}]}],
        },
    }), encoding="utf-8")


def test_removing_a_hook_strips_only_that_entry_and_keeps_the_rest_of_settings(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    settings = tmp_path / "settings.json"
    _settings(settings)

    rec = remove_hook(settings, "Stop", "", "echo hi", home=home)

    after = json.loads(settings.read_text(encoding="utf-8"))
    assert rec.kind == KIND_HOOK
    assert "Stop" not in after["hooks"]          # its last hook went, so the event goes
    assert after["hooks"]["PreToolUse"]          # the untouched sibling survives
    assert after["model"] == "opus"              # non-hook config is not disturbed


def test_restoring_a_hook_brings_back_the_whole_prior_settings_file(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    settings = tmp_path / "settings.json"
    _settings(settings)
    before = settings.read_text(encoding="utf-8")

    rec = remove_hook(settings, "Stop", "", "echo hi", home=home)
    restore(rec.record_id, home=home)

    assert settings.read_text(encoding="utf-8") == before


def test_removing_a_hook_that_is_not_there_refuses_instead_of_reporting_success(tmp_path):
    """A Remove that changes nothing but returns OK leaves the user believing a
    hook is gone while it still fires on every turn."""
    home = tmp_path / "home"
    home.mkdir()
    settings = tmp_path / "settings.json"
    _settings(settings)
    before = settings.read_text(encoding="utf-8")

    with pytest.raises(SummarizeRefused, match="nothing was changed"):
        remove_hook(settings, "Stop", "", "a command that was never wired", home=home)
    assert settings.read_text(encoding="utf-8") == before


def test_quarantine_root_is_outside_any_tj_owned_path(tmp_path):
    """The store must survive an uninstall that removes tj's state dir, so it
    must not live under ~/.tj or any name a `.tj*` glob would sweep."""
    root = quarantine_root(tmp_path)
    assert root.parent == tmp_path
    assert not root.name.startswith(".tj")


def test_manifest_is_plain_readable_json_naming_the_original_path(tmp_path):
    """A restore must be doable by hand with no tj installed."""
    home = tmp_path / "home"
    home.mkdir()
    target = tmp_path / "hooks.md"
    target.write_text("body", encoding="utf-8")
    remove_file(target, home=home)

    manifest = json.loads((quarantine_root(home) / "manifest.json").read_text(encoding="utf-8"))
    assert manifest[0]["original_path"] == str(target)
    assert (quarantine_root(home) / "README.txt").exists()


def test_a_corrupt_manifest_reads_as_empty_rather_than_raising(tmp_path):
    """This feeds a GET the page polls; a hand-mangled manifest must not 500 the
    only UI that can fix it."""
    home = tmp_path / "home"
    home.mkdir()
    root = quarantine_root(home)
    root.mkdir(parents=True)
    (root / "manifest.json").write_text("{not json", encoding="utf-8")

    assert list_removals(home=home) == []
