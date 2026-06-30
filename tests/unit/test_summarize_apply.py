"""Unit tests for apply / backup / undo (PR3 — the first surface that writes user files).

Everything is routed at a tmp storage dir, so backups land under tmp/summary/backups,
never the developer's real ~/.tj. The rails (dry-run default, hash-guard, owner-check,
refuse-on-drift, mode preservation) are the point — they get the coverage.
"""
from __future__ import annotations

import os
import re
import stat

import pytest

from tokenjam.core.config import StorageConfig, TjConfig
from tokenjam.core.summarize import apply as apply_mod
from tokenjam.core.summarize import backup
from tokenjam.core.summarize.apply import apply_staged, undo
from tokenjam.core.summarize.session import SummarizeRefused, check, prepare, read_staged

PROSE = "Always act carefully and never drop a required step when you respond. " * 30
_MARKER_RE = re.compile(r'<tj-keep id="\d+"[^>]*?(?:/>|>.*?</tj-keep>)', re.DOTALL)


@pytest.fixture
def cfg(tmp_path):
    return TjConfig(version="1", storage=StorageConfig(path=str(tmp_path / "t.duckdb")))


def _mkfile(tmp_path, name="CLAUDE.md", body=None) -> str:
    body = body if body is not None else PROSE + "\n```\nkeep = 'me'\n```\n"
    f = tmp_path / name
    f.write_text(body, encoding="utf-8")
    return str(f)


def _stage(cfg, path, summary_prose="Be careful; never skip a step.") -> str:
    """prep + check with a structure-preserving summary → stages a result; returns the restored text."""
    res = prepare(path=path)
    summary = summary_prose + " " + " ".join(_MARKER_RE.findall(res.wrapped_prompt))
    verdict = check(cfg, path, summary, res.source_sha256)
    assert verdict.staged
    return verdict.restored


# --- apply: dry-run vs go ---

def test_apply_dry_run_writes_nothing(cfg, tmp_path):
    path = _mkfile(tmp_path)
    before = (tmp_path / "CLAUDE.md").read_text()
    restored = _stage(cfg, path)
    report = apply_staged(cfg, go=False)
    assert report["dry_run"] is True
    assert any(a["path"] == path for a in report["applied"])
    assert (tmp_path / "CLAUDE.md").read_text() == before        # untouched
    assert restored != before                                    # the candidate really differs


def test_apply_go_writes_backs_up_and_clears(cfg, tmp_path):
    path = _mkfile(tmp_path)
    restored = _stage(cfg, path)
    report = apply_staged(cfg, go=True)
    assert report["dry_run"] is False and report["applied"]
    written = (tmp_path / "CLAUDE.md").read_text()
    assert written == restored and "keep = 'me'" in written      # candidate written, structure verbatim
    assert backup.recorded_output(cfg, path) is not None         # backup landed (under tmp, not ~/.tj)
    assert read_staged(cfg, path) is None                        # staged entry cleared


# --- apply guards ---

def test_apply_skips_drifted_file(cfg, tmp_path):
    path = _mkfile(tmp_path)
    _stage(cfg, path)
    (tmp_path / "CLAUDE.md").write_text("edited after check", encoding="utf-8")
    report = apply_staged(cfg, go=True)
    assert not report["applied"]
    assert "changed since check" in report["skipped"][0]["reason"]
    assert (tmp_path / "CLAUDE.md").read_text() == "edited after check"   # never clobbered
    assert backup.recorded_output(cfg, path) is None                      # nothing written → no backup


def test_apply_skips_unowned(cfg, tmp_path, monkeypatch):
    path = _mkfile(tmp_path)
    _stage(cfg, path)
    monkeypatch.setattr(apply_mod, "_owned_by_current_user", lambda p: False)
    report = apply_staged(cfg, go=True)
    assert not report["applied"]
    assert "another user" in report["skipped"][0]["reason"]


def test_apply_preserves_mode(cfg, tmp_path):
    path = _mkfile(tmp_path)
    os.chmod(path, 0o600)
    _stage(cfg, path)
    apply_staged(cfg, go=True)
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o600          # write didn't change perms


def test_apply_take_all_partial(cfg, tmp_path):
    a = _mkfile(tmp_path, "A.md")
    b = _mkfile(tmp_path, "B.md")
    _stage(cfg, a)
    _stage(cfg, b)
    (tmp_path / "B.md").write_text("drifted", encoding="utf-8")   # B drifts after staging
    report = apply_staged(cfg, go=True)
    assert {x["path"] for x in report["applied"]} == {a}          # A applied
    assert {x["path"] for x in report["skipped"]} == {b}          # B skipped, not both refused


def test_apply_unstaged_path_reported(cfg, tmp_path):
    path = _mkfile(tmp_path)                                      # never staged
    report = apply_staged(cfg, path, go=True)
    assert not report["applied"] and report["skipped"][0]["reason"] == "not staged"


# --- undo ---

def test_undo_restores_original(cfg, tmp_path):
    path = _mkfile(tmp_path)
    original = (tmp_path / "CLAUDE.md").read_text()
    _stage(cfg, path)
    apply_staged(cfg, go=True)
    assert (tmp_path / "CLAUDE.md").read_text() != original       # changed by apply
    assert undo(cfg, path, go=True)["restored"] is True
    assert (tmp_path / "CLAUDE.md").read_text() == original        # back to byte-identical original


def test_undo_dry_run_writes_nothing(cfg, tmp_path):
    path = _mkfile(tmp_path)
    _stage(cfg, path)
    apply_staged(cfg, go=True)
    applied_text = (tmp_path / "CLAUDE.md").read_text()
    assert undo(cfg, path, go=False)["dry_run"] is True
    assert (tmp_path / "CLAUDE.md").read_text() == applied_text    # not reverted on a dry-run


def test_undo_refuses_on_post_apply_drift(cfg, tmp_path):
    path = _mkfile(tmp_path)
    _stage(cfg, path)
    apply_staged(cfg, go=True)
    (tmp_path / "CLAUDE.md").write_text("hand-edited after apply", encoding="utf-8")
    with pytest.raises(SummarizeRefused, match="changed since"):
        undo(cfg, path, go=True)


def test_undo_refuses_symlink_path(cfg, tmp_path):
    path = _mkfile(tmp_path)
    _stage(cfg, path)
    apply_staged(cfg, go=True)
    applied = (tmp_path / "CLAUDE.md").read_text()
    target = tmp_path / "target.md"
    target.write_text(applied, encoding="utf-8")
    (tmp_path / "CLAUDE.md").unlink()
    (tmp_path / "CLAUDE.md").symlink_to(target)
    with pytest.raises(SummarizeRefused, match="symlink"):
        undo(cfg, path, go=True)
    assert target.read_text(encoding="utf-8") == applied


def test_undo_without_backup_refuses(cfg, tmp_path):
    path = _mkfile(tmp_path)                                      # never applied
    with pytest.raises(SummarizeRefused, match="no summarize backup"):
        undo(cfg, path, go=True)


# --- backup metadata resilience (the read-only scan must survive a corrupt sidecar) ---

def test_recorded_output_tolerates_corrupt_meta(cfg, tmp_path):
    """A half-written / hand-edited meta sidecar reads as None, never raising — so the
    advisory `tj summarize list` that consumes it can't be crashed by a corrupt backup."""
    path = _mkfile(tmp_path)
    _stage(cfg, path)
    apply_staged(cfg, go=True)
    meta = backup._meta_path(cfg, path)
    assert backup.recorded_output(cfg, path) is not None          # healthy meta reads back
    meta.write_text("{ not valid json", encoding="utf-8")
    assert backup.recorded_output(cfg, path) is None              # corrupt JSON → tolerated
    meta.write_text('{"applied_at": "x"}', encoding="utf-8")      # valid JSON, key missing
    assert backup.recorded_output(cfg, path) is None              # missing key → tolerated
