"""On-disk cache for the pothole aggregator's (expensive, full-corpus) result.

The detector (``core.optimize.analyzers.pothole``) takes tens of seconds over
a real local corpus — far too slow to compute per HTTP request. ``tj serve``
computes it on a background schedule using a FRESH DuckDB connection (mirrors
the retention job's own-connection pattern in ``cli/cmd_serve.py``, so a slow
scan never contends with the live request connection's write lock — see the
DuckDB single-writer pothole this very module exists to help catch more of).
This module is the read/write boundary: a small JSON file at
``~/.tj/pothole_cache.json`` plus an in-process lock so two overlapping
recomputes never race each other's writes.
"""
from __future__ import annotations

import json
import threading
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from tokenjam.core.optimize.analyzers.pothole import PotholeFinding, compute_pothole_finding

if TYPE_CHECKING:
    from tokenjam.core.config import TjConfig

_LOCK = threading.Lock()
_COMPUTING = threading.Event()


def default_cache_path(config: TjConfig | None = None) -> Path:
    """``<storage-parent>/pothole_cache.json`` when ``config`` is given — this
    honors ``--config`` / ``storage.path`` (and falls back to a config-scoped
    TEMP dir, never the real ``~/.tj``, when ``storage.path`` is ``""``/
    ``":memory:"``; see ``pothole_apply._storage_base_dir``). Without a
    ``config`` (legacy callers), the old hardcoded ``~/.tj`` default."""
    if config is not None:
        from tokenjam.core.optimize.pothole_apply import _storage_base_dir

        return _storage_base_dir(config) / "pothole_cache.json"
    return Path.home() / ".tj" / "pothole_cache.json"


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
    finding: PotholeFinding, path: Path | None = None, *, config: TjConfig | None = None,
) -> dict[str, Any]:
    """Atomically write the finding (temp file + rename), never a partial file
    a concurrent reader could observe."""
    p = path or default_cache_path(config)
    payload = {
        "computed_at": datetime.now(timezone.utc).isoformat(),
        "finding": asdict(finding),
    }
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        tmp.replace(p)
    except OSError:
        pass
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

    Phase 3 (verify, SPEC §4 step 6): "on each rescan" is THIS pass — when
    ``config`` is supplied, every applied (non-reverted) fix's recurrence is
    re-measured against the same fresh ``conn`` right after the detector
    itself recomputes, so Verify always runs on the same cadence as Detect
    with no separate scheduler entry. ``config=None`` (e.g. an older caller)
    just skips the verify pass — degrade, never fail the detector recompute.
    """
    if not _LOCK.acquire(blocking=False):
        return None
    _COMPUTING.set()
    try:
        finding = compute_pothole_finding(conn)
        # cache_path, when omitted, resolves via `config` (honors --config /
        # storage.path, and a :memory:/"" storage.path never falls through to
        # the real ~/.tj — see default_cache_path).
        result = write_cache(finding, cache_path, config=config)
        if config is not None:
            try:
                from tokenjam.core.optimize import pothole_verify

                pothole_verify.rescan_all(config, conn)
            except Exception:
                pass   # best-effort — a verify failure never sinks the detector's own cache write
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

    threading.Thread(target=_job, name="pothole-recompute", daemon=True).start()
    return True
