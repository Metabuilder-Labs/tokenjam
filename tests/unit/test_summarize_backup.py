"""Unit tests for backup.list_backups (the Lens 'undo' preflight).

Backups land under a tmp storage dir (never the real ~/.tj). The point: the
``undoable`` flag must mirror exactly what ``undo`` will actually allow —
including the edge cases from the #424 review (missing gzip blob → a dead Undo
button; broken symlink mislabelled as "file no longer exists").
"""
from __future__ import annotations

import pytest

from tokenjam.core.config import StorageConfig, TjConfig
from tokenjam.core.summarize import backup


@pytest.fixture
def cfg(tmp_path):
    return TjConfig(version="1", storage=StorageConfig(path=str(tmp_path / "t.duckdb")))


def _applied(cfg, tmp_path, name="CLAUDE.md", original="old body\n", output="new body\n"):
    """Simulate an applied file: current file == the applied output, with a backup saved."""
    p = tmp_path / name
    p.write_text(output, encoding="utf-8")
    backup.save(cfg, str(p), original=original, output=output, est_tokens_saved=42)
    return str(p), p


def test_empty_when_no_backups(cfg):
    assert backup.list_backups(cfg) == []


def test_undoable_when_file_matches_applied_output(cfg, tmp_path):
    sp, _ = _applied(cfg, tmp_path)
    (rec,) = backup.list_backups(cfg)
    assert rec["source_path"] == sp
    assert rec["undoable"] is True and rec["reason"] == ""
    assert rec["est_tokens_saved"] == 42


def test_not_undoable_when_file_changed_since_apply(cfg, tmp_path):
    _, p = _applied(cfg, tmp_path)
    p.write_text("hand-edited since apply\n", encoding="utf-8")
    (rec,) = backup.list_backups(cfg)
    assert rec["undoable"] is False and "changed since apply" in rec["reason"]


def test_not_undoable_when_backup_blob_missing(cfg, tmp_path):
    # meta.json present but the .orig.gz gone (manual delete / cleanup tooling):
    # undo() would 409, so the preflight must report undoable=False, not a dead button.
    sp, _ = _applied(cfg, tmp_path)
    backup._orig_path(cfg, sp).unlink()
    (rec,) = backup.list_backups(cfg)
    assert rec["undoable"] is False and rec["reason"] == "backup file missing"


def test_symlink_reason_takes_precedence_over_missing(cfg, tmp_path):
    # is_symlink() is checked before exists() (which follows links), so a broken
    # symlink at the source path reads as "symlink…", not "file no longer exists".
    _, p = _applied(cfg, tmp_path)
    p.unlink()
    p.symlink_to(tmp_path / "nowhere.md")   # broken symlink
    (rec,) = backup.list_backups(cfg)
    assert rec["undoable"] is False and "symlink" in rec["reason"]


def test_not_undoable_when_file_gone(cfg, tmp_path):
    _, p = _applied(cfg, tmp_path)
    p.unlink()
    (rec,) = backup.list_backups(cfg)
    assert rec["undoable"] is False and rec["reason"] == "file no longer exists"
