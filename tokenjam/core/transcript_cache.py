"""Persistent per-file cache for ``core.transcript.read_records``.

The relearn and deadweight analyzers (``core.optimize.analyzers.{relearn,
deadweight}``) each re-parse the FULL on-disk Claude Code transcript corpus
on every run — a JSONL file per session, some spanning hundreds of turns.
Profiling a real corpus (5,362 transcripts, 1.7GB) attributed most of
``compute_deadweight_finding``'s wall time to ``transcript.read_records``,
almost all of it in ``json.loads`` + the file read. Most of that corpus is
closed, historic sessions whose content never changes between runs —
re-parsing them on every invocation is pure waste, and both analyzers already
mtime-filter their session list, so a cache trusting the same signal adds no
correctness risk beyond what they already accept.

This module is the cache: one small JSON file per transcript under a cache
directory, keyed on ``(path, size, mtime)`` — the exact staleness signal the
callers already trust. A cache hit skips the original file's read + parse
entirely; a miss (cold entry, or a changed file) recomputes and rewrites
atomically (temp file + rename), so a concurrent reader never observes a
partial write.

Opt-in only: ``core.transcript.read_records`` still defaults to its original
always-reparse behavior. Only a caller that explicitly passes ``cache_dir``
(today: the deadweight/relearn analyzers' registry entry points, resolved via
``default_cache_dir`` below) pays a persistent-cache write — every other
caller (session story rendering, resume-brief, the API's session/status
routes, and every existing test of these functions) is unaffected.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

#: Env override so tests (and anyone else) can point the cache at a temp dir
#: without touching a real local install — mirrors ``TJ_CLAUDE_PROJECTS_ROOT``.
_CACHE_DIR_ENV = "TJ_TRANSCRIPT_CACHE_DIR"


def default_cache_dir(config: Any | None = None) -> Path:
    """Resolve the cache directory a ``run(ctx)`` entry point should use.

    Precedence: the env override (tests), else the config-honoring storage dir
    (``relearn_apply._storage_base_dir`` — never the real ``~/.tj`` for an
    in-memory-configured caller, matching every other on-disk cache this
    codebase keeps under the storage parent), else the legacy ``~/.tj``
    default when no config is available at all.
    """
    env = os.environ.get(_CACHE_DIR_ENV)
    if env:
        return Path(env)
    if config is not None:
        try:
            from tokenjam.core.optimize.relearn_apply import _storage_base_dir

            return _storage_base_dir(config) / "transcript_cache"
        except Exception:
            pass
    return Path.home() / ".tj" / "transcript_cache"


def _cache_key(path: Path) -> str:
    """Filesystem-safe cache filename for ``path`` — a hash of the absolute
    path string (never of the transcript's own content), so lookup is a
    single stat + read with no directory-listing scan needed."""
    resolved = str(path.resolve())
    digest = hashlib.sha256(resolved.encode("utf-8")).hexdigest()[:32]
    return f"{digest}.json"


def cached_read_records(path: Path, cache_dir: Path) -> list[dict[str, Any]]:
    """``read_records(path)``, transparently cached under ``cache_dir``.

    Cache validity is ``(size, mtime)`` matching the live file's current
    stat — the exact pair both analyzers already trust for their own mtime
    filter. A stat failure (the transcript vanished mid-scan) degrades to
    ``[]``, matching ``read_records``'s own tolerant-of-missing-files
    contract rather than raising.
    """
    from tokenjam.core.transcript import _parse_records

    try:
        st = path.stat()
    except OSError:
        return []
    size, mtime = st.st_size, st.st_mtime

    cache_path = cache_dir / _cache_key(path)
    cached = _load(cache_path)
    if (
        cached is not None
        and cached.get("size") == size
        and cached.get("mtime") == mtime
    ):
        records = cached.get("records")
        if isinstance(records, list):
            return records

    records = _parse_records(path)
    _store(cache_path, path, size, mtime, records)
    return records


def _load(cache_path: Path) -> dict[str, Any] | None:
    try:
        raw = json.loads(cache_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return raw if isinstance(raw, dict) else None


def _store(
    cache_path: Path,
    source: Path,
    size: int,
    mtime: float,
    records: list[dict[str, Any]],
) -> None:
    """Atomic temp-file + rename write. Best-effort — a cache write must
    never break the scan it exists to speed up; any OSError is swallowed."""
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"path": str(source), "size": size, "mtime": mtime, "records": records}
        # Per-pid temp name so two processes racing the same cache entry (a
        # CLI run and a live `tj serve`) never collide mid-write — the loser
        # just overwrites the winner's file a moment later with the same
        # (deterministic) content, never a torn read in between.
        tmp = cache_path.with_name(f"{cache_path.name}.tmp{os.getpid()}")
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        tmp.replace(cache_path)
    except OSError:
        pass


def prune_orphaned_entries(cache_dir: Path) -> int:
    """Delete cache entries whose source transcript no longer exists on disk.

    This is the cache's pruning story: entries track 1:1 with distinct
    transcript paths ever seen, so the cache is bounded by corpus size, not
    unbounded — but a retention job elsewhere can delete old transcripts
    without this module knowing, so orphaned entries need an explicit sweep
    rather than growing forever. Opportunistic and safe to call anytime (or
    skip entirely); never raises. Returns the number of entries removed.
    """
    if not cache_dir.exists():
        return 0
    removed = 0
    for entry in cache_dir.glob("*.json"):
        data = _load(entry)
        source = data.get("path") if data else None
        if not isinstance(source, str) or not Path(source).exists():
            try:
                entry.unlink()
                removed += 1
            except OSError:
                pass
    return removed


__all__ = [
    "default_cache_dir",
    "cached_read_records",
    "prune_orphaned_entries",
]
