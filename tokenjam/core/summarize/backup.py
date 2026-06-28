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


def save(config: TjConfig, path: str, original: str, output: str) -> None:
    """Gzip the original + write the meta (one-level — replaces any prior backup for ``path``)."""
    d = backups_dir(config)
    d.mkdir(parents=True, exist_ok=True)
    _orig_path(config, path).write_bytes(gzip.compress(original.encode("utf-8")))
    _meta_path(config, path).write_text(
        json.dumps({
            "source_path": path,
            "original_sha256": sha256(original),
            "output_sha256": sha256(output),
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
    except (OSError, json.JSONDecodeError, KeyError):
        return None


def load_original(config: TjConfig, path: str, current: str | None) -> str:
    """Return the backed-up original for ``path``.

    Raises ``SummarizeRefused`` if there is no backup, or (when ``current`` is given)
    the file changed since we wrote it — i.e. ``current`` no longer matches the recorded
    output hash, so undoing would clobber newer edits.
    """
    orig_f, meta_f = _orig_path(config, path), _meta_path(config, path)
    if not (orig_f.exists() and meta_f.exists()):
        raise SummarizeRefused(f"no summarize backup for {path} — nothing to undo.")
    if current is not None:
        meta = json.loads(meta_f.read_text(encoding="utf-8"))
        if sha256(current) != meta["output_sha256"]:
            raise SummarizeRefused(
                f"{path} has changed since `tj summarize apply` wrote it — "
                "refusing to undo (newer edits would be lost).")
    return gzip.decompress(orig_f.read_bytes()).decode("utf-8")
