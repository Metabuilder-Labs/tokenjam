"""Unit tests for the prep/check lifecycle + results staging (no scratch — DEC-024/025/026).

Every test routes the staging dir at a tmp dir via `config.storage.path`, so nothing ever
touches the developer's real ~/.tj.
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pytest

from tokenjam.core.config import StorageConfig, TjConfig
from tokenjam.core.summarize import wrap
from tokenjam.core.summarize.apply import apply_staged, undo
from tokenjam.core.summarize.session import (
    SummarizeRefused, check, clear, list_staged, prepare, read_staged,
)

PROSE = "Always act carefully and never drop a required step when you respond. " * 30
_MARKER_RE = re.compile(r'<tj-keep id="\d+"[^>]*?(?:/>|>.*?</tj-keep>)', re.DOTALL)


@pytest.fixture
def cfg(tmp_path):
    """Config whose summarize anchor is tmp — results land under tmp_path/summary/results."""
    return TjConfig(version="1", storage=StorageConfig(path=str(tmp_path / "t.duckdb")))


def _write(tmp_path, name, text) -> str:
    f = tmp_path / name
    f.write_text(text, encoding="utf-8")
    return str(f)


def _perfect_summary(wrapped: str, new_prose: str) -> str:
    """A summary that shrinks the prose but keeps every marker, in order."""
    return new_prose + " " + " ".join(_MARKER_RE.findall(wrapped))


# --- prepare (stateless: wrap + hash, persists nothing) ---

def test_prepare_wraps_and_hashes(tmp_path):
    text = PROSE + "\n```\ncode = 1\n```\n"
    path = _write(tmp_path, "CLAUDE.md", text)
    res = prepare(path=path)
    assert res.source_sha256 == hashlib.sha256(text.encode("utf-8")).hexdigest()
    assert "<tj-keep" in res.wrapped_prompt
    assert res.target_prose_words == max(8, int(0.5 * res.prose_words))
    assert res.protected_blocks >= 1
    assert "handle" not in res.to_dict()                 # no handle anymore (DEC-024)


def test_prepare_below_gate(tmp_path):
    path = _write(tmp_path, "tiny.md", "short prompt, only a few words here")
    res = prepare(path=path)
    assert res.wrapped_prompt == "" and "gate" in res.note
    assert res.source_sha256                             # hash still computed


def test_prepare_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        prepare(path=str(tmp_path / "nope.md"))


# --- check: roundtrip + staging ---

def test_check_roundtrip_stages(cfg, tmp_path):
    path = _write(tmp_path, "CLAUDE.md", PROSE + "\n```\nkeep = 'me'\n```\n")
    res = prepare(path=path)
    verdict = check(cfg, path, _perfect_summary(res.wrapped_prompt, "Be careful; never skip a step."),
                    res.source_sha256)
    assert verdict.structure_ok and verdict.staged
    assert "keep = 'me'" in verdict.restored             # code restored verbatim
    assert verdict.words_after < verdict.words_before and verdict.est_tokens_saved > 0
    staged = read_staged(cfg, path)                      # landed in tmp staging, not real ~/.tj
    assert staged is not None and staged["source_sha256"] == res.source_sha256
    assert (tmp_path / "summary" / "results").exists()


def test_check_dropped_marker_not_staged(cfg, tmp_path):
    path = _write(tmp_path, "CLAUDE.md", PROSE + "\n```\nkeep = 'me'\n```\n")
    res = prepare(path=path)
    verdict = check(cfg, path, "brand new prose with no markers at all", res.source_sha256)
    assert not verdict.structure_ok and not verdict.staged
    assert "dropped" in verdict.reason
    assert read_staged(cfg, path) is None                # failures are never staged


def test_check_tracks_must_keep(cfg, tmp_path):
    path = _write(tmp_path, "CLAUDE.md", "You must never delete the file. " + PROSE)
    res = prepare(path=path)
    verdict = check(cfg, path, _perfect_summary(res.wrapped_prompt, "Avoid removing the file."),
                    res.source_sha256)
    assert verdict.structure_ok                          # a rephrase doesn't fail the structure gate
    assert "never" in verdict.must_keep_removed           # but the dropped word is tracked


def test_check_records_produced_by_default_manual(cfg, tmp_path):
    """The two-step prep/check path is the human/agent rewriting by hand → produced_by 'manual'."""
    path = _write(tmp_path, "CLAUDE.md", PROSE + "\n```\nx = 1\n```\n")
    res = prepare(path=path)
    verdict = check(cfg, path, _perfect_summary(res.wrapped_prompt, "Short."), res.source_sha256)
    assert verdict.produced_by == "manual"
    assert read_staged(cfg, path)["produced_by"] == "manual"     # persisted on the staged result (DEC-028)


def test_check_records_produced_by_explicit(cfg, tmp_path):
    """Callers stamp the delivery mode (MCP → 'in-session', `--via claude` → 'claude') — the UI seam."""
    path = _write(tmp_path, "CLAUDE.md", PROSE + "\n```\nx = 1\n```\n")
    res = prepare(path=path)
    verdict = check(cfg, path, _perfect_summary(res.wrapped_prompt, "Short."), res.source_sha256,
                    produced_by="in-session")
    assert verdict.produced_by == "in-session"
    assert read_staged(cfg, path)["produced_by"] == "in-session"


# --- check: the hash guard (the safety net DEC-024 added) ---

def test_check_refuses_when_file_changed(cfg, tmp_path):
    path = _write(tmp_path, "CLAUDE.md", PROSE)
    res = prepare(path=path)
    (tmp_path / "CLAUDE.md").write_text(PROSE + " EDITED", encoding="utf-8")   # edit after prep
    with pytest.raises(SummarizeRefused, match="changed since"):
        check(cfg, path, "anything", res.source_sha256)


def test_check_refuses_when_file_missing(cfg, tmp_path):
    path = _write(tmp_path, "CLAUDE.md", PROSE)
    res = prepare(path=path)
    (tmp_path / "CLAUDE.md").unlink()
    with pytest.raises(SummarizeRefused, match="not found"):
        check(cfg, path, "anything", res.source_sha256)


def test_check_no_false_refuse(cfg, tmp_path):
    """prep then an immediate check on the untouched file must NOT trip the hash guard."""
    path = _write(tmp_path, "CLAUDE.md", PROSE + "\n```\nx = 1\n```\n")
    res = prepare(path=path)
    verdict = check(cfg, path, _perfect_summary(res.wrapped_prompt, "Careful prose."), res.source_sha256)
    assert verdict.structure_ok                          # no spurious "changed since prep"


# --- the invariant re-derive depends on: protect() is deterministic ---

def test_protect_is_deterministic():
    text = PROSE + "\n```\nx = 1\n```\nUse `tool` and obey <rules>do X</rules>.\n"
    w1, s1, o1, _ = wrap.protect(text)
    w2, s2, o2, _ = wrap.protect(text)
    assert w1 == w2 and s1 == s2 and o1 == o2            # same input → identical map → re-derive is safe


# --- staging helpers (folded into session.py — DEC-026) ---

def test_staging_roundtrip(cfg, tmp_path):
    path = _write(tmp_path, "CLAUDE.md", PROSE + "\n```\nx = 1\n```\n")
    res = prepare(path=path)
    check(cfg, path, _perfect_summary(res.wrapped_prompt, "Short."), res.source_sha256)
    assert len(list_staged(cfg)) == 1
    staged = read_staged(cfg, path)
    assert staged is not None and staged["path"] == str(tmp_path / "CLAUDE.md")
    assert clear(cfg, path) == 1
    assert list_staged(cfg) == [] and read_staged(cfg, path) is None


# --- review fixes: malformed marker (INC-001), symlinks (DEF-013), path canonicalization ---

def test_check_rejects_malformed_marker(cfg, tmp_path):
    """A malformed marker (id opening, no close) must FAIL the gate — not silently drop the block."""
    path = _write(tmp_path, "CLAUDE.md", PROSE + "\n```\nkeep = 'me'\n```\n")
    res = prepare(path=path)
    verdict = check(cfg, path, 'Short careful prose. <tj-keep id="1">', res.source_sha256)
    assert not verdict.structure_ok and not verdict.staged
    assert verdict.integrity["malformed"]                       # the INC-001 hole, now caught
    assert read_staged(cfg, path) is None


def test_check_rejects_stray_closing_marker(cfg, tmp_path):
    """A stray closing keep-tag must fail the gate, even if all real blocks restored."""
    path = _write(tmp_path, "CLAUDE.md", PROSE + "\n```\nkeep = 'me'\n```\n")
    res = prepare(path=path)
    summary = _perfect_summary(res.wrapped_prompt, "Short.") + " </tj-keep>"
    verdict = check(cfg, path, summary, res.source_sha256)
    assert not verdict.structure_ok and not verdict.staged
    assert verdict.integrity["malformed"]
    assert read_staged(cfg, path) is None


def test_prepare_refuses_symlink(tmp_path):
    real = tmp_path / "real.md"
    real.write_text(PROSE)
    link = tmp_path / "link.md"
    link.symlink_to(real)
    with pytest.raises(SummarizeRefused, match="symlink"):
        prepare(path=str(link))


def test_check_refuses_symlink(cfg, tmp_path):
    real = tmp_path / "real.md"
    real.write_text(PROSE)
    link = tmp_path / "link.md"
    link.symlink_to(real)
    with pytest.raises(SummarizeRefused, match="symlink"):
        check(cfg, str(link), "anything", "deadbeef")


def test_check_stores_resolved_absolute_path(cfg, tmp_path, monkeypatch):
    """A relative path at check time is stored RESOLVED — so apply from another cwd hits the right file."""
    (tmp_path / "CLAUDE.md").write_text(PROSE + "\n```\nx = 1\n```\n")
    monkeypatch.chdir(tmp_path)
    res = prepare(path="CLAUDE.md")
    verdict = check(cfg, "CLAUDE.md", _perfect_summary(res.wrapped_prompt, "Short."), res.source_sha256)
    assert Path(verdict.path).is_absolute()
    assert verdict.path == str((tmp_path / "CLAUDE.md").resolve())


def test_check_rejects_invented_bare_marker(cfg, tmp_path):
    """A bare invented <tj-keep> (no id) that `_KEEP_RE` can't consume must fail the gate (INC-001 round 2)."""
    path = _write(tmp_path, "CLAUDE.md", PROSE + "\n```\nkeep = 'me'\n```\n")
    res = prepare(path=path)
    summary = _perfect_summary(res.wrapped_prompt, "Short.") + " extra <tj-keep>"
    verdict = check(cfg, path, summary, res.source_sha256)
    assert not verdict.structure_ok and not verdict.staged
    assert verdict.integrity["malformed"]                      # caught: an unconsumed `<tj-keep`
    assert read_staged(cfg, path) is None


# --- review round 3: gate-hardening false-positive — prose that legitimately contains the marker ---

def test_check_allows_prose_that_mentions_the_marker(cfg, tmp_path):
    """A prompt documenting the marker syntax: preserving that prose mention is NOT a failure."""
    path = _write(tmp_path, "CLAUDE.md",
                  "The <tj-keep tag is internal; never emit a stray </tj-keep>. " + PROSE
                  + "\n```\nkeep = 'me'\n```\n")
    res = prepare(path=path)
    summary = ("Docs: the <tj-keep tag is internal; never emit a stray </tj-keep>. "
               + " ".join(_MARKER_RE.findall(res.wrapped_prompt)))
    verdict = check(cfg, path, summary, res.source_sha256)
    assert verdict.structure_ok and verdict.staged          # source-baselined → no false fail


def test_check_allows_exact_marker_shaped_prose_mention(cfg, tmp_path):
    """A full marker-looking prose mention is protected before the model sees it, not counted as extra."""
    path = _write(tmp_path, "CLAUDE.md",
                  'This prompt documents <tj-keep id="99"/> as marker syntax. ' + PROSE
                  + "\n```\nkeep = 'me'\n```\n")
    res = prepare(path=path)
    summary = "Docs mention the marker syntax. " + " ".join(_MARKER_RE.findall(res.wrapped_prompt))
    verdict = check(cfg, path, summary, res.source_sha256)
    assert verdict.structure_ok and verdict.staged
    assert '<tj-keep id="99"/>' in verdict.restored


def test_check_fails_when_model_invents_beyond_source(cfg, tmp_path):
    """Even with a marker-mentioning source, residue BEYOND the source baseline still fails."""
    path = _write(tmp_path, "CLAUDE.md", "The <tj-keep tag is internal. " + PROSE + "\n```\nx = 1\n```\n")
    res = prepare(path=path)
    summary = ("Docs: the <tj-keep tag is internal. extra <tj-keep> "
               + " ".join(_MARKER_RE.findall(res.wrapped_prompt)))
    verdict = check(cfg, path, summary, res.source_sha256)
    assert not verdict.structure_ok and verdict.integrity["malformed"]


# --- apply / undo: symlink + TOCTOU safety (DEF-013) ---

def test_apply_skips_symlink_toctou(cfg, tmp_path):
    """If a staged file becomes a symlink before apply, apply skips it — never rewrites through a link."""
    f = _write(tmp_path, "CLAUDE.md", PROSE + "\n```\nx = 1\n```\n")
    res = prepare(path=f)
    check(cfg, f, _perfect_summary(res.wrapped_prompt, "Short."), res.source_sha256)   # staged
    other = tmp_path / "other.md"
    other.write_text("other")
    Path(f).unlink()
    Path(f).symlink_to(other)                                  # TOCTOU: file → symlink after staging
    report = apply_staged(cfg, None, go=True)                  # take-all
    assert not report["applied"]
    assert any("symlink" in s["reason"] for s in report["skipped"])
    assert other.read_text() == "other"                        # the link target was NOT rewritten


def test_undo_refuses_symlink(cfg, tmp_path):
    real = tmp_path / "real.md"
    real.write_text("hello")
    link = tmp_path / "link.md"
    link.symlink_to(real)
    with pytest.raises(SummarizeRefused, match="symlink"):
        undo(cfg, str(link))
