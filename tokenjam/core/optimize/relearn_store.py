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

from tokenjam.core.analysis_span import retention_days_for
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


def _distill_cache_dir_for(config: TjConfig | None) -> Path | None:
    """The distill cache this run may write to, or None to leave the default.

    Imported lazily and never allowed to raise: a cache-path resolution
    failure must not sink a recompute that would otherwise succeed.
    """
    if config is None:
        return None
    try:
        from tokenjam.core.optimize.analyzers.relearn import _distill_cache_dir

        return _distill_cache_dir(config)
    except Exception:
        return None


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
        # `cost_window_days`/`cost_excluded` are written alongside
        # `cost_proposals` by `write_cost_proposals` and read back through
        # `read_cost_proposals`/`_headline_window_days` (see
        # `api/routes/relearn.py`) to label the Review inbox headline with the
        # window its figures were actually observed over. This relearn-detector
        # write shares the same cache file (see this module's docstring) and
        # used to preserve only the two keys above, silently forgetting a
        # non-default cost window on every relearn recompute. Harmless while
        # every route falls back to the same default, but round-tripping both
        # keys here means a variable window survives regardless of which
        # producer wrote the cache last.
        if "cost_window_days" in existing:
            payload["cost_window_days"] = existing["cost_window_days"]
        if "cost_excluded" in existing:
            payload["cost_excluded"] = existing["cost_excluded"]
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
    computed AND no recompute has ever failed either (a genuinely fresh
    install). Shape: ``{"cost_computed_at": iso, "cost_proposals": [dict,
    ...], "cost_proposals_error": str | None, "cost_proposals_error_at": iso |
    None, "cost_window_days": int, "cost_excluded": dict}``.
    ``cost_proposals_error``
    is the last recompute failure's message (behavioral requirement #5) —
    present alongside a GOOD ``cost_proposals`` list when a later recompute
    failed after an earlier one had already succeeded, so a transient failure
    never hides the last good result. ``cost_window_days`` is the window the
    stored figures were OBSERVED over, so the headline names the window its
    own data came from rather than whatever picker the reader's screen is set
    to. The pace inputs this block used to also carry (``cost_active_days`` /
    ``cost_n_sessions``) are gone with the projection they fed: nothing in the
    cost pipeline paces a figure any more, and a cached pace input is an
    invitation to re-derive one. ``cost_excluded`` is the rollup's cross-reference for
    waste a caller deliberately did NOT sum in — generic infrastructure with
    no current occupant (``summarize`` used to be the one entry here until
    it got a real peer card instead; see ``cost_proposals.
    COST_ANALYZERS``); ``{}`` for a cache written before it existed, and
    ``{}`` going forward until a future analyzer needs it again."""
    raw = read_cache(path, config=config)
    if raw is None:
        return None
    has_proposals = "cost_proposals" in raw
    has_error = "cost_proposals_error" in raw
    if not has_proposals and not has_error:
        return None
    return {
        "cost_computed_at": raw.get("cost_computed_at"),
        "cost_proposals": raw.get("cost_proposals") or [],
        "cost_proposals_error": raw.get("cost_proposals_error"),
        "cost_proposals_error_at": raw.get("cost_proposals_error_at"),
        "cost_window_days": raw.get("cost_window_days") or 0,
        "cost_excluded": raw.get("cost_excluded") or {},
    }


def write_cost_proposals_error(
    message: str, path: Path | None = None, *, config: TjConfig | None = None,
) -> dict[str, Any]:
    """Record a cost-proposals recompute failure (behavioral requirement #5),
    preserving whatever ``cost_proposals``/``finding`` block already exists —
    a failed refresh must never wipe the last GOOD result, only annotate that
    the most recent attempt didn't produce a fresher one. Atomic; best-effort
    on I/O error (mirrors every other write in this module)."""
    p = path or default_cache_path(config)
    existing = read_cache(p, config=config) or {}
    payload = dict(existing)
    payload["cost_proposals_error"] = message
    payload["cost_proposals_error_at"] = datetime.now(timezone.utc).isoformat()
    _atomic_write(p, payload)
    return payload


def clear_cost_proposals_error(
    path: Path | None = None, *, config: TjConfig | None = None,
) -> dict[str, Any]:
    """Clear a previously-recorded cost-proposals error after a SUCCESSFUL
    recompute — called by ``cost_proposals.recompute_cost_proposals`` right
    after it writes a fresh good result, so a one-off transient failure
    doesn't keep flagging the tab as degraded once refreshes recover."""
    p = path or default_cache_path(config)
    existing = read_cache(p, config=config) or {}
    if "cost_proposals_error" not in existing and "cost_proposals_error_at" not in existing:
        return existing
    payload = {
        k: v for k, v in existing.items()
        if k not in ("cost_proposals_error", "cost_proposals_error_at")
    }
    _atomic_write(p, payload)
    return payload


def write_cost_proposals(
    proposals: list[Any], path: Path | None = None, *, config: TjConfig | None = None,
    window_days: int | None = None, excluded: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write the cost proposals into the SAME cache file the relearn finding
    lives in, under a separate ``cost_proposals`` key, preserving the relearn
    ``finding`` block. ``proposals`` is a list of ``CostProposal`` (or plain
    dicts). Atomic; best-effort on I/O error.

    ``window_days`` is the window this recompute ran over, stored alongside
    the proposals so a later reader labels the figures with the window they
    were actually observed over rather than one it assumes. ``None`` (the
    default) leaves any previously-stored value untouched, so a caller that
    doesn't track it yet (a legacy call site) doesn't silently zero out a real
    prior value. It is a LABEL, never a divisor: nothing rescales a stored
    figure by it.

    ``excluded`` is the rollup's cross-reference block for waste a caller
    deliberately did not fold in as a peer card — generic infrastructure with
    no current occupant (see ``read_cost_proposals``) — always written when
    this call succeeds (unlike ``window_days`` above, this one has no "leave
    untouched" case: a fresh recompute always knows the current excluded
    state, even if it's "nothing")."""
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
    if window_days is not None:
        payload["cost_window_days"] = window_days
    payload["cost_excluded"] = excluded or {}
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
        #
        # The analyzer scope is consulted FIRST: this daemon job is a second
        # entry point into the same scan, so a scope honored only by
        # `analyzers/relearn.run` would leak the machine's global transcript
        # tree back in through the cache the served routes read. See
        # `core/optimize/scope.py`.
        from tokenjam.core.optimize.scope import (
            resolve_analyzer_scope,
            resolve_write_scope,
        )

        scope = resolve_analyzer_scope(config)
        if not scope.enabled:
            return None
        projects_root = scope.projects_root if scope.source == "flag" else None
        if projects_root is None and config is not None:
            try:
                from tokenjam.core.transcript import loop_transcript_root

                projects_root = loop_transcript_root(config)
            except Exception:
                projects_root = None
        # The persistent per-file parse cache (core.transcript_cache): this
        # background job re-scans the FULL corpus on every scheduled tick, so
        # warming/reusing the cache here is where the recurring cost actually
        # gets paid down across ticks, not just within one `tj optimize` run.
        transcript_cache_dir = None
        if config is not None:
            try:
                from tokenjam.core.transcript_cache import default_cache_dir

                transcript_cache_dir = default_cache_dir(config)
            except Exception:
                transcript_cache_dir = None
        # Full-corpus persona classification (relearn scans unbounded history
        # like the finding itself, not a window) — same functions
        # `runner.build_report` uses for `AnalyzerContext.persona`/
        # `OptimizeReport.persona`, so the daemon's relearn cache gates its
        # workspace write by the same rule the rest of the product does.
        persona = "unknown"
        if conn is not None:
            try:
                from tokenjam.core.framing import (
                    agent_persona_mix,
                    config_declared_plan,
                    dominant_persona,
                )

                persona = dominant_persona(
                    agent_persona_mix(conn), declared_plan=config_declared_plan(config),
                )
            except Exception:
                persona = "unknown"
        finding = compute_relearn_finding(
            conn, projects_root=projects_root, transcript_cache_dir=transcript_cache_dir,
            persona=persona,
            # The apply target has to agree with the scope the findings came
            # from — a card whose evidence is scoped and whose write target is
            # not describes two different machines. Routed through
            # `resolve_write_scope` rather than reading `scope.claude_home`
            # directly, because the API's write guard authorizes against the
            # OTHER half of that same type; deriving the two independently is
            # what let the suggestion and the guard disagree.
            claude_home=resolve_write_scope(scope=scope).suggest_root,
            # Scoped like every other relearn artifact — the distill cache is a
            # SECOND cache beside this module's own, and it wrote real files
            # under the real ~/.tj even from an isolated config until it was
            # threaded through the same `_storage_base_dir`.
            distill_cache_dir=_distill_cache_dir_for(config),
            # The archive lane's horizon: what tokenjam kept, not what Claude
            # Code left on disk. See `compute_relearn_finding`.
            retention_days=retention_days_for(
                getattr(config, "storage", None)
            ) if getattr(config, "storage", None) is not None else None,
        )
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
