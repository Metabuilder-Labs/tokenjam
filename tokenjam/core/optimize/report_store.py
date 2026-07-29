"""On-disk store for the full analyzer report — the ONLY thing routes may read.

Why this module exists
----------------------
``build_report`` dispatches every registered analyzer over the corpus. Some of
them are full-corpus scans measured in tens of seconds to minutes on a real
local install (``relearn`` alone calls the same ``compute_relearn_finding``
that ``relearn_store`` exists to cache). Running that on an HTTP request thread
made the Dashboard's recoverable-waste panel and Budget-at-risk card hang for
minutes, while the Review inbox — which reads a stored block — painted instantly.

So: **no request path runs an analyzer.** Analyzer runs happen in exactly three
places, all of them off the request path and all of them landing here:

* at daemon boot (``cli/cmd_serve.py``'s lifespan kick),
* on the daemon's scheduled interval (``[optimize] scan_interval_hours``),
* when a user presses Rescan (``POST /api/v1/optimize/rescan``, which starts a
  BACKGROUND recompute and returns immediately — it does not compute inline).

Routes read :func:`read_report` and render the stored result plus its
``computed_at``. Ingestion is untouched and still continuously updates
everything derived from it (traces, spans, sessions, maps, approach, timeline);
only the analyzer layer moved off the request path.

Design is deliberately the same shape as ``relearn_store`` rather than a second
caching mechanism with its own semantics: atomic temp-file+rename writes, a
non-blocking ``_LOCK`` plus a ``_COMPUTING`` flag so overlapping recomputes
no-op instead of stacking, and ``trigger_background_recompute`` spawning a
daemon thread with a FRESH DuckDB connection so a slow scan never contends with
the live request connection.

Cold vs empty
-------------
:func:`report_status` distinguishes ``never_run`` (nothing has ever been
computed) from ``ready`` (a scan completed, and it may legitimately have found
nothing). Callers must render those differently: a cold store is NOT zero and
NOT "nothing found". Rendering it as ``0`` / ``$0.00`` / "no waste" is a
reassurance the data does not support.
"""
from __future__ import annotations

import json
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from tokenjam.core.optimize.build_stamp import tj_build

if TYPE_CHECKING:
    from tokenjam.core.config import TjConfig

_LOCK = threading.Lock()
_COMPUTING = threading.Event()
# Monotonic timestamp of the last recompute that actually STARTED. The rescan
# rate limit is derived from this, not from the stored `computed_at`: a scan
# that fails still consumed the corpus pass, so it still counts against the
# floor. Reset per-process, which is the right scope — the floor exists to stop
# one user hammering a live daemon, not to persist across restarts.
_LAST_RUN_MONOTONIC: float | None = None
_LAST_RUN_LOCK = threading.Lock()

# Status vocabulary, shared with the routes and mirrored by the UI. Kept here so
# a surface can never invent a fourth state that means "cold" but reads as empty.
STATUS_NEVER_RUN = "never_run"
STATUS_COMPUTING = "computing"
STATUS_READY = "ready"
STATUS_ERROR = "error"


def default_report_path(config: TjConfig | None = None) -> Path:
    """``<storage-parent>/optimize_report.json``.

    Resolved through ``relearn_apply._storage_base_dir`` so it honors
    ``--config`` / ``storage.path`` and never falls through to a real ``~/.tj``
    for an in-memory-configured caller (a test must not write into a real
    install). Without a ``config``, the legacy ``~/.tj`` default.
    """
    if config is not None:
        from tokenjam.core.optimize.relearn_apply import _storage_base_dir

        return _storage_base_dir(config) / "optimize_report.json"
    return Path.home() / ".tj" / "optimize_report.json"


def _atomic_write(p: Path, payload: dict[str, Any]) -> None:
    """Temp-file + rename; a concurrent reader never observes a partial file.
    Best-effort — an OSError (read-only fs, un-creatable parent) degrades to a
    no-op rather than raising into a background job or a request."""
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        tmp.replace(p)
    except OSError:
        pass


def read_report(
    path: Path | None = None, *, config: TjConfig | None = None,
) -> dict[str, Any] | None:
    """The stored ``{"computed_at", "report", "window_days", "since", "until",
    "error", "error_at"}`` payload, or ``None`` when no scan has ever completed
    AND none has ever failed (a genuinely cold store) or the file is corrupt.

    ``None`` means COLD, never empty. A caller that renders it as a zero or as
    an absence claim is asserting more than the data supports.
    """
    p = path or default_report_path(config)
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return raw if isinstance(raw, dict) else None


def write_report(
    report_dict: dict[str, Any],
    path: Path | None = None,
    *,
    config: TjConfig | None = None,
    window_days: int | None = None,
    since: str | None = None,
    until: str | None = None,
) -> dict[str, Any]:
    """Store a freshly-computed report, clearing any previously-recorded error.

    ``window_days`` / ``since`` / ``until`` travel with the result so every
    surface can label the figures with the window they were OBSERVED over
    rather than the window its own picker happens to be set to. They are
    labels, never divisors — nothing rescales a stored figure by them.
    """
    p = path or default_report_path(config)
    payload: dict[str, Any] = {
        "computed_at": datetime.now(timezone.utc).isoformat(),
        "report": report_dict,
        "window_days": window_days,
        "since": since,
        "until": until,
        # WHICH BUILD PRODUCED THIS, not just when. `computed_at` answers HOW
        # OLD and readers take it for WHICH VERSION. These stores are caches
        # with no build identity and nothing invalidates them on upgrade, so
        # after upgrading tokenjam the next pass stamps a fresh timestamp over
        # figures the replaced binary may have produced — and the audience for
        # that is precisely the user who upgraded to get a fix and will
        # conclude it did not work. A surface can only qualify the freshness
        # claim if the producing build travels with the result.
        "tj_version": tj_build(),
    }
    _atomic_write(p, payload)
    return payload


def write_report_error(
    message: str, path: Path | None = None, *, config: TjConfig | None = None,
) -> dict[str, Any]:
    """Record a failed scan WITHOUT discarding the last good report.

    A transient failure must never turn a populated surface into an empty one:
    the stored ``report``/``computed_at`` stay put and the surface keeps
    rendering them with a degraded flag beside them. Only a store that has
    never succeeded reports :data:`STATUS_ERROR`.
    """
    p = path or default_report_path(config)
    existing = read_report(p, config=config) or {}
    payload = dict(existing)
    payload["error"] = message
    payload["error_at"] = datetime.now(timezone.utc).isoformat()
    _atomic_write(p, payload)
    return payload


def is_computing() -> bool:
    """True while a scan is in flight (scheduled, boot or user-pressed)."""
    return _COMPUTING.is_set()


def report_status(
    stored: dict[str, Any] | None, *, computing: bool | None = None,
) -> str:
    """Resolve the store into one of the four :data:`STATUS_READY` states.

    The distinction that matters: a store with no ``computed_at`` and no
    recorded error is :data:`STATUS_NEVER_RUN` — COLD. It is not empty, and it
    is not zero. A store whose only attempts failed is :data:`STATUS_ERROR`;
    an empty analyzer result that a scan genuinely produced is
    :data:`STATUS_READY` with empty findings, which is the ONLY state where
    "nothing found" is an honest thing to say.
    """
    in_flight = is_computing() if computing is None else computing
    has_good = bool(stored and stored.get("computed_at"))
    if in_flight and not has_good:
        return STATUS_COMPUTING
    if has_good:
        return STATUS_COMPUTING if in_flight else STATUS_READY
    if stored and stored.get("error"):
        return STATUS_ERROR
    return STATUS_NEVER_RUN


def stored_report_block(
    config: TjConfig | None = None, *, path: Path | None = None,
) -> dict[str, Any]:
    """The envelope every analyzer-consuming route returns verbatim.

    One shape, computed in one place, so two surfaces reading the same store
    can never disagree about whether it is cold, stale or degraded.
    """
    from tokenjam.core.optimize import scan_cycle

    stored = read_report(path, config=config)
    computing = is_computing()
    status = report_status(stored, computing=computing)
    error = (stored or {}).get("error")
    return {
        "status": status,
        "computed_at": (stored or {}).get("computed_at"),
        "window_days": (stored or {}).get("window_days"),
        "scan_since": (stored or {}).get("since"),
        "scan_until": (stored or {}).get("until"),
        "computing": computing,
        # THE CYCLE, not this store. `computing` above goes false the instant
        # the report lands, while the relearn cache and the cost proposals the
        # same pass feeds are still being built — so a surface reading only it
        # asserts freshness over figures that are still the previous pass's.
        # `scan_cycle.is_cycle_computing` covers the whole pass; a surface's
        # "scanning" state is the OR of the two, never either alone.
        "cycle_computing": scan_cycle.is_cycle_computing(),
        # Build provenance. `computed_build` is the build that PRODUCED the
        # stored figures (absent on anything written before this stamp existed);
        # `build` is the one serving them. When they differ, the timestamp is
        # still honest about age and no longer sufficient on its own — see
        # `write_report`.
        "computed_build": (stored or {}).get("tj_version"),
        "build": tj_build(),
        # `degraded` is for the case a LATER scan failed after an earlier one
        # succeeded: the surface still renders the last good result, with the
        # failure disclosed beside it rather than silently pretending the last
        # refresh worked.
        "degraded": bool(error) and status in (STATUS_READY, STATUS_COMPUTING),
        "last_error": error,
        "last_error_at": (stored or {}).get("error_at"),
    }


def stored_report_dict(
    config: TjConfig | None = None, *, path: Path | None = None,
) -> dict[str, Any] | None:
    """The stored report AS A DICT — the wire format, served verbatim.

    **This is what routes should return.** The store holds exactly what
    ``report_to_dict`` produced, and handing that straight to the client means
    no rehydration step can sit between the analyzer and the reader. Even
    though the round trip is lossless now (see ``runner.hydrate_dataclass`` and
    ``tests/unit/test_report_roundtrip.py``), a route that rehydrates only to
    re-serialize is doing work whose sole possible effect is to lose something.

    ``None`` means COLD, never empty.
    """
    stored = read_report(path, config=config)
    body = (stored or {}).get("report")
    return body if isinstance(body, dict) else None


def stored_report(
    config: TjConfig | None = None, *, path: Path | None = None,
) -> Any | None:
    """The stored report rebuilt into an ``OptimizeReport``, or ``None`` when
    the store is cold (or the payload can't be rehydrated).

    For the consumers that genuinely need typed objects — ranking, the
    persona gate, `gather_planning_texts`. A route that only needs to SERVE the
    report wants :func:`stored_report_dict` instead.
    """
    from tokenjam.core.optimize import report_from_dict

    body = stored_report_dict(config, path=path)
    if body is None:
        return None
    try:
        return report_from_dict(body)
    except Exception:
        return None


def stored_finding(
    config: TjConfig | None = None, name: str = "", *, path: Path | None = None,
) -> Any | None:
    """One typed finding out of the stored report, or ``None``.

    For the narrow case where a route needs a real dataclass to hand to a
    helper (``gather_planning_texts`` wants a ``ReuseFinding``) while still
    serving the stored dict verbatim on the wire.
    """
    report = stored_report(config, path=path)
    if report is None:
        return None
    return (report.findings or {}).get(name)


def seconds_since_last_run() -> float | None:
    """Seconds since the last scan STARTED, or ``None`` if none has this process."""
    with _LAST_RUN_LOCK:
        last = _LAST_RUN_MONOTONIC
    return None if last is None else time.monotonic() - last


def rescan_throttled(config: TjConfig | None = None) -> bool:
    """True when a rescan request arrives inside the configured floor.

    The rail this implements: a user cannot hammer rescan into stacking
    full-corpus passes. Requests inside the floor are answered with the stored
    result rather than starting another scan.
    """
    floor = _min_rescan_seconds(config)
    if floor <= 0:
        return False
    elapsed = seconds_since_last_run()
    return elapsed is not None and elapsed < floor


def _min_rescan_seconds(config: TjConfig | None) -> int:
    opt = getattr(config, "optimize", None)
    try:
        return int(getattr(opt, "scan_min_rescan_seconds", 60))
    except (TypeError, ValueError):
        return 60


def _window_days(config: TjConfig | None, conn: Any = None) -> int:
    """The window this scan observes over — the SAME seam the Review inbox's
    cost-proposal recompute resolves through (``core/optimize/report_window``).

    It used to read ``scan_window_days`` alone while the cost side read the
    resolved analysis span, so the two surfaces published one metric under two
    windows and neither said which. See that module's docstring; do not
    reintroduce a local derivation here.
    """
    from tokenjam.core.optimize.report_window import report_window_days

    return report_window_days(config, conn)


def recompute_now(
    db: Any,
    config: TjConfig,
    *,
    path: Path | None = None,
    window_days: int | None = None,
    until: Any | None = None,
) -> dict[str, Any] | None:
    """Run every analyzer and store the result, on the CALLING thread.

    Returns ``None`` (a no-op, nothing computed) when a scan is already in
    flight — this NEVER blocks waiting for the other one, so two overlapping
    triggers (boot kick landing on the interval, or a user pressing rescan
    twice) cost one scan, not two. Callers that must not block a request thread
    use :func:`trigger_background_recompute` instead.

    A failure is recorded via :func:`write_report_error` and re-raised to the
    caller's discretion only as a stored error — the last good report survives.
    """
    global _LAST_RUN_MONOTONIC

    if not _LOCK.acquire(blocking=False):
        return None
    _COMPUTING.set()
    with _LAST_RUN_LOCK:
        _LAST_RUN_MONOTONIC = time.monotonic()
    try:
        from tokenjam.core.optimize import build_report, report_to_dict
        from tokenjam.utils.time_parse import utcnow

        days = (
            window_days if window_days is not None
            else _window_days(config, getattr(db, "conn", None))
        )
        # ONE anchor across a scan cycle. When several stores are refreshed
        # together (`core/optimize/scan_cycle.py`) they subtract their window
        # from the SAME instant, so two surfaces publishing one metric cannot
        # cover windows offset from each other. `None` means a lone refresh,
        # which owns its own anchor. Anything that is not a datetime is
        # discarded rather than raised on: the anchor crosses a thread
        # boundary, and a malformed one must not sink a pass that would
        # otherwise have succeeded.
        until = until if isinstance(until, datetime) else utcnow()
        since = until - timedelta(days=days)
        try:
            report = build_report(db=db, config=config, since=since, until=until)
        except Exception as exc:  # noqa: BLE001 - stored, never propagated
            return write_report_error(f"{type(exc).__name__}: {exc}", path, config=config)
        return write_report(
            report_to_dict(report),
            path,
            config=config,
            window_days=days,
            since=since.isoformat(),
            until=until.isoformat(),
        )
    finally:
        _COMPUTING.clear()
        _LOCK.release()


def trigger_background_recompute(
    backend_factory: Callable[[], Any],
    config: TjConfig,
    *,
    path: Path | None = None,
    window_days: int | None = None,
    until: Any | None = None,
) -> bool:
    """Fire-and-forget a scan on a daemon thread. Returns ``False`` when one is
    already running (the overlap guard — nothing is started, nothing stacks).

    ``backend_factory`` builds a FRESH backend (``lambda: DuckDBBackend(
    config.storage)``), never the caller's live request connection, so the scan
    never contends with a concurrent writer. The backend is closed when done.
    """
    if is_computing():
        return False

    def _job() -> None:
        backend = None
        try:
            backend = backend_factory()
            recompute_now(
                backend, config, path=path, window_days=window_days, until=until,
            )
        except Exception as exc:   # noqa: BLE001
            # Never crash the scheduler thread — but never swallow the failure
            # either. `relearn_store`'s equivalent job discards its exception
            # entirely, so `POST /relearn/refresh` reports "started" whether
            # the scan then succeeded or died; a rescan that failed looks
            # identical to one that worked. Recording it here is what lets the
            # UI say "the last refresh failed" instead of quietly showing stale
            # numbers as though they were fresh. `recompute_now` records
            # failures INSIDE the scan; this covers the ones outside it, i.e.
            # the backend never opening at all.
            try:
                write_report_error(f"{type(exc).__name__}: {exc}", path, config=config)
            except Exception:
                pass
        finally:
            close = getattr(backend, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass

    threading.Thread(target=_job, name="optimize-report-scan", daemon=True).start()
    return True
