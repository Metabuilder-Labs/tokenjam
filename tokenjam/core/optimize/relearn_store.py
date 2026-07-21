"""On-disk cache for the relearn aggregator's (expensive, full-corpus) result.

The detector (``core.optimize.analyzers.relearn``) takes tens of seconds over
a real local corpus — far too slow to compute per HTTP request. ``tj serve``
computes it on a background schedule using a FRESH DuckDB connection (mirrors
the retention job's own-connection pattern in ``cli/cmd_serve.py``, so a slow
scan never contends with the live request connection's write lock — see the
DuckDB single-writer relearn this very module exists to help catch more of).
This module is the read/write boundary: a small JSON file at
``~/.tj/relearn_cache.json`` plus an in-process lock so two overlapping
recomputes never race each other's writes.
"""
from __future__ import annotations

import json
import threading
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from tokenjam.core.optimize.analyzers.relearn import RelearnFinding, compute_relearn_finding

if TYPE_CHECKING:
    from tokenjam.core.config import TjConfig

_LOCK = threading.Lock()
_COMPUTING = threading.Event()


def default_cache_path(config: TjConfig | None = None) -> Path:
    """``<storage-parent>/relearn_cache.json`` when ``config`` is given — this
    honors ``--config`` / ``storage.path`` (and falls back to a config-scoped
    TEMP dir, never the real ``~/.tj``, when ``storage.path`` is ``""``/
    ``":memory:"``; see ``relearn_apply._storage_base_dir``). Without a
    ``config`` (legacy callers), the old hardcoded ``~/.tj`` default."""
    if config is not None:
        from tokenjam.core.optimize.relearn_apply import _storage_base_dir

        return _storage_base_dir(config) / "relearn_cache.json"
    return Path.home() / ".tj" / "relearn_cache.json"


def read_cache(
    path: Path | None = None, *, config: TjConfig | None = None,
) -> dict[str, Any] | None:
    """The last-written ``{"computed_at", "finding"}`` payload, or ``None`` if
    no recompute has ever completed (fresh install) or the file is corrupt."""
    p = path or default_cache_path(config)
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return raw if isinstance(raw, dict) else None


def write_cache(
    finding: RelearnFinding, path: Path | None = None, *, config: TjConfig | None = None,
) -> dict[str, Any]:
    """Atomically write the finding (temp file + rename), never a partial file
    a concurrent reader could observe.

    The cache file is shared with the cost-proposal producer (see
    ``write_cost_proposals``): the two write on different cadences (the relearn
    detector job vs the optimize path). To keep this "the same proposal store"
    without one producer clobbering the other, an existing ``cost_proposals``
    block is read back and preserved here rather than dropped.

    Detection time is also when each cluster gets its stable ``proposal_id``
    (``relearn_proposals.stamp_proposal_ids``): the apply paths accept a stored
    proposal ID and nothing else, so the IDs have to exist on the record the
    detector itself wrote.
    """
    from tokenjam.core.optimize.relearn_proposals import stamp_proposal_ids

    p = path or default_cache_path(config)
    existing = read_cache(p, config=config) or {}
    # Explicit annotation: without it mypy infers the dict-literal's value
    # type from the two initial values (str, dict[str, Any]) and joins them
    # down to `Collection[str]`, rejecting the `cost_computed_at` assignment
    # below even though the payload is really `dict[str, Any]`.
    payload: dict[str, Any] = {
        "computed_at": datetime.now(timezone.utc).isoformat(),
        "finding": stamp_proposal_ids(asdict(finding)),
    }
    if "cost_proposals" in existing:
        payload["cost_proposals"] = existing["cost_proposals"]
        payload["cost_computed_at"] = existing.get("cost_computed_at")
    _atomic_write(p, payload)
    return payload


def _atomic_write(p: Path, payload: dict[str, Any]) -> None:
    """Temp-file + rename write; a concurrent reader never sees a partial file.
    Best-effort — an OSError (read-only fs, missing parent that can't be made)
    degrades to a no-op, never raises."""
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        tmp.replace(p)
    except OSError:
        pass


def read_cost_proposals(
    path: Path | None = None, *, config: TjConfig | None = None,
) -> dict[str, Any] | None:
    """The last-written cost-proposal block, or ``None`` if none has ever been
    computed. Shape: ``{"cost_computed_at": iso, "cost_proposals": [dict, ...]}``."""
    raw = read_cache(path, config=config)
    if raw is None or "cost_proposals" not in raw:
        return None
    return {
        "cost_computed_at": raw.get("cost_computed_at"),
        "cost_proposals": raw.get("cost_proposals") or [],
    }


def write_cost_proposals(
    proposals: list[Any], path: Path | None = None, *, config: TjConfig | None = None,
) -> dict[str, Any]:
    """Write the cost proposals into the SAME cache file the relearn finding
    lives in, under a separate ``cost_proposals`` key, preserving the relearn
    ``finding`` block. ``proposals`` is a list of ``CostProposal`` (or plain
    dicts). Atomic; best-effort on I/O error."""
    from dataclasses import is_dataclass

    p = path or default_cache_path(config)
    existing = read_cache(p, config=config) or {}
    # `is_dataclass()` alone narrows to `DataclassInstance | type[DataclassInstance]`
    # (it also accepts a dataclass *class*), but `asdict()` only accepts an
    # instance. Excluding `type` narrows to the instance case for mypy and
    # matches what we actually want here — `proposals` holds instances, never
    # classes.
    serialised = [
        asdict(pr) if is_dataclass(pr) and not isinstance(pr, type) else dict(pr)
        for pr in proposals
    ]
    payload = dict(existing)
    payload["cost_proposals"] = serialised
    payload["cost_computed_at"] = datetime.now(timezone.utc).isoformat()
    _atomic_write(p, payload)
    return payload


def is_computing() -> bool:
    return _COMPUTING.is_set()


def recompute_now(
    conn: Any | None, *, cache_path: Path | None = None, config: Any | None = None,
) -> dict[str, Any] | None:
    """Synchronous compute + cache write on the CALLING thread/connection.

    Returns ``None`` (no-op) if a recompute is already in flight elsewhere —
    never blocks waiting for the other one to finish. Callers that want
    non-blocking HTTP-request behaviour should run this on a background
    thread instead (see ``trigger_background_recompute``).
    """
    if not _LOCK.acquire(blocking=False):
        return None
    _COMPUTING.set()
    try:
        # `[loop].transcript_path` lets a Claude Agent SDK app point the loop at
        # its OWN transcript root instead of ~/.claude/projects. None keeps the
        # historical env/default resolution.
        projects_root = None
        if config is not None:
            try:
                from tokenjam.core.transcript import loop_transcript_root

                projects_root = loop_transcript_root(config)
            except Exception:
                projects_root = None
        finding = compute_relearn_finding(conn, projects_root=projects_root)
        # cache_path, when omitted, resolves via `config` (honors --config /
        # storage.path, and a :memory:/"" storage.path never falls through to
        # the real ~/.tj — see default_cache_path).
        result = write_cache(finding, cache_path, config=config)
        return result
    finally:
        _COMPUTING.clear()
        _LOCK.release()


def trigger_background_recompute(
    backend_factory: Callable[[], Any],
    *,
    cache_path: Path | None = None,
    config: Any | None = None,
) -> bool:
    """Fire-and-forget a recompute on a daemon thread.

    ``backend_factory`` builds a FRESH ``StorageBackend`` (e.g.
    ``lambda: DuckDBBackend(config.storage)``) — never the caller's live
    request connection, so the scan's DuckDB read never contends with a
    concurrent writer. The backend is closed when the job finishes. Returns
    ``False`` (no-op, nothing started) if a recompute is already running.

    ``config`` (optional): passed straight through to ``recompute_now`` so
    its Phase 3 verify pass can locate ``applied_fixes.json``.
    """
    if is_computing():
        return False

    def _job() -> None:
        backend = None
        try:
            backend = backend_factory()
            conn = getattr(backend, "conn", None)
            recompute_now(conn, cache_path=cache_path, config=config)
        except Exception:
            # Best-effort background job — never crash the scheduler/thread pool.
            pass
        finally:
            close = getattr(backend, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass

    threading.Thread(target=_job, name="relearn-recompute", daemon=True).start()
    return True
