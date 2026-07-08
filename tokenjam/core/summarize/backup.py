"""Backup store for summarize apply / undo (DEC-026 — ``~/.tj/summary/backups/``).

One-level and gzip'd: applying a file stashes its original here plus a meta sidecar
(original + output hashes, timestamp). ``undo`` restores it — refusing if the file
changed since we wrote it. ``recorded_output`` powers the scan's skip-already-done.
This module is the pure store; the file-writing lives in ``apply.py``.
"""
from __future__ import annotations

import gzip
import json
from pathlib import Path

from tokenjam.core.config import TjConfig
from tokenjam.core.summarize.session import SummarizeRefused, sha256, stage_key, summary_root
from tokenjam.utils.time_parse import utcnow


def backups_dir(config: TjConfig) -> Path:
    return summary_root(config) / "backups"


def _orig_path(config: TjConfig, path: str) -> Path:
    return backups_dir(config) / f"{stage_key(path)}.orig.gz"


def _meta_path(config: TjConfig, path: str) -> Path:
    return backups_dir(config) / f"{stage_key(path)}.meta.json"


def save(config: TjConfig, path: str, original: str, output: str, est_tokens_saved: int = 0) -> None:
    """Gzip the original + write the meta (one-level — replaces any prior backup for ``path``).

    ``est_tokens_saved`` is recorded so the Optimize box can report applied savings
    (the read-only scan drops applied files, so their saving is otherwise lost)."""
    d = backups_dir(config)
    d.mkdir(parents=True, exist_ok=True)
    _orig_path(config, path).write_bytes(gzip.compress(original.encode("utf-8")))
    _meta_path(config, path).write_text(
        json.dumps({
            "source_path": path,
            "original_sha256": sha256(original),
            "output_sha256": sha256(output),
            "est_tokens_saved": int(est_tokens_saved),
            "applied_at": utcnow().isoformat(),
        }, ensure_ascii=False),
        encoding="utf-8",
    )


def recorded_output(config: TjConfig, path: str) -> str | None:
    """The sha256 of what we last wrote to ``path`` (for skip-already-done), or None.

    Tolerant of a missing / partial / hand-edited meta sidecar: an unreadable record
    reads as 'no record' (``None``), so the read-only ``tj summarize list`` scan that
    consumes this can never be crashed by a corrupt backup — it just declines to skip.
    """
    f = _meta_path(config, path)
    if not f.exists():
        return None
    try:
        return json.loads(f.read_text(encoding="utf-8"))["output_sha256"]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, KeyError):
        return None


def list_backups(config: TjConfig) -> list[dict]:
    """Every applied file that still has a backup — the Lens 'undo' surface.

    Each record: ``source_path``, ``applied_at``, plus a computed ``undoable`` flag
    (+ ``reason`` when false). A backup is undoable iff the gzip blob is present, the
    file still exists, is not a symlink, and its CURRENT content matches what apply
    wrote — the same conditions ``undo`` enforces, surfaced up front so the UI can
    show why a row can't be undone.
    """
    d = backups_dir(config)
    if not d.exists():
        return []
    out: list[dict] = []
    for meta_f in sorted(d.glob("*.meta.json")):
        try:
            meta = json.loads(meta_f.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        sp = meta.get("source_path")
        if not sp:
            continue
        # Pair the blob to THIS meta file by name — don't recompute stage_key from
        # source_path, which resolves symlinks and would map a now-symlinked source
        # to a different key and spuriously read as "backup file missing".
        orig_f = meta_f.with_name(meta_f.name[: -len(".meta.json")] + ".orig.gz")
        p = Path(sp).expanduser()
        undoable, reason = True, ""
        if not orig_f.exists():
            # meta.json without its gzip blob (deleted/cleaned) — undo() would 409;
            # report it up front rather than offering an Undo that can't run.
            undoable, reason = False, "backup file missing"
        elif p.is_symlink():
            # is_symlink() before exists(): exists() follows the link, so a broken
            # symlink would otherwise be mislabelled "file no longer exists".
            undoable, reason = False, "symlink — refusing to restore through it"
        elif not p.exists():
            undoable, reason = False, "file no longer exists"
        else:
            try:
                current = p.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                # UnicodeDecodeError is a ValueError, not an OSError — catch it too, or
                # one non-UTF-8 applied file 500s the whole /backups response (#424).
                undoable, reason = False, "cannot read the file"
            else:
                if sha256(current) != meta.get("output_sha256"):
                    undoable, reason = False, "changed since apply — undo would lose newer edits"
        out.append({
            "source_path": sp,
            "applied_at": meta.get("applied_at", ""),
            "est_tokens_saved": int(meta.get("est_tokens_saved", 0) or 0),
            "undoable": undoable,
            "reason": reason,
        })
    return out


def load_original(config: TjConfig, path: str, current: str | None) -> str:
    """Return the backed-up original for ``path``.

    Raises ``SummarizeRefused`` if there is no backup, or (when ``current`` is given)
    the file changed since we wrote it — i.e. ``current`` no longer matches the recorded
    output hash, so undoing would clobber newer edits.
    """
    orig_f, meta_f = _orig_path(config, path), _meta_path(config, path)
    if not (orig_f.exists() and meta_f.exists()):
        raise SummarizeRefused(f"no summarize backup for {path} — nothing to undo.")
    try:
        if current is not None:
            meta = json.loads(meta_f.read_text(encoding="utf-8"))
            if sha256(current) != meta["output_sha256"]:
                raise SummarizeRefused(
                    f"{path} has changed since `tj summarize apply` wrote it — "
                    "refusing to undo (newer edits would be lost).")
        return gzip.decompress(orig_f.read_bytes()).decode("utf-8")
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, KeyError) as exc:
        # Corrupt/unreadable sidecar or blob (non-UTF-8, truncated JSON, missing key,
        # bad gzip) → refuse cleanly (undo() → 409), never a 500. (The drift
        # SummarizeRefused above is a different type and propagates untouched.)
        raise SummarizeRefused(
            f"summarize backup for {path} is unreadable or corrupt — cannot undo.") from exc
