"""Cheap on-disk hand-off of the top recurring-inclusion driver.

``tj context`` (``core/context_diagnostic``) already knows WHICH file, search,
prompt, or tool output is re-included most often across sessions — but
answering that needs a live DuckDB connection plus the capture-gated attribute
data. The statusline (``cli/cmd_statusline``) is the opposite: zero-token,
pure-stdlib, invoked after every turn, and must never open the DB or do
anything slower than a linear transcript scan.

``tj backfill claude-code`` (and `tj onboard`, which calls the same
``ingest_claude_code`` function directly) already holds that connection and
the ``[capture]`` flags, so it is one process that computes the window's
top driver and hands it off here as a tiny JSON file. The DAEMON's analyzer
scan cycle (``core/optimize/scan_cycle``) refreshes it too, for the reason
below. The statusline does a plain stat+read of that file — no query, no live
computation.

TWO WRITERS, BECAUSE ONE OF THEM ONLY RAN BY ACCIDENT. The backfill was the
sole caller, so on a machine that never runs ``tj backfill`` the cache aged
past its TTL and the driver suffix vanished for good — a capability that
exists, is correct, and simply never reaches the surface again. Nothing tells
a user to re-run a backfill, and a backfill has no relationship to the surface
it feeds. The daemon's scan cycle already holds a live connection and the
``[capture]`` flags at the end of every pass, so it refreshes the cache as its
last leg and the display stays current on its own.

FRESH / STALE / ABSENT, NOT PRESENT / ABSENT. A reader of this cache has three
situations, not two: no usable cache at all, a usable one within the TTL, and a
usable one PAST it. Only the last is a case where we know the answer and are
choosing not to show it, and collapsing it into "nothing" is a silent
degradation — the surrounding figure stays correct, so nothing looks broken and
the reader cannot tell a richer line exists. :func:`resolve_driver` reports
which of the three it has, so a display can age-mark the stale case instead of
hiding it. :func:`format_driver` keeps the fresh-only contract for callers that
want exactly that.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any

# Window the cached driver is computed over on each refresh.
ATTRIBUTION_WINDOW_DAYS = 30

# A cached driver older than this is no longer presented as CURRENT. It is not
# discarded: `resolve_driver` reports it as stale so a display can show it with
# an age marker (see this module's docstring). Kept at 7 days deliberately —
# the threshold now only decides whether the figure is labelled as dated, and
# disclosing the age is a better answer than widening the window at which we
# silently claim currency.
MAX_CACHE_AGE_SECONDS = 7 * 24 * 60 * 60

#: `resolve_driver` states. ABSENT covers "no file", "malformed" and "cannot be
#: proven fresh" (no / unparseable ``computed_at``) — three situations with one
#: remedy: there is nothing to show. STALE is the one where there IS an answer.
DRIVER_FRESH = "fresh"
DRIVER_STALE = "stale"
DRIVER_ABSENT = "absent"


@dataclass(frozen=True)
class DriverStatus:
    """What the cache holds for display, and whether it is current.

    ``label`` is the ``"<label> ×<count>"`` display string (``None`` only when
    ``state`` is :data:`DRIVER_ABSENT`), ``inclusion_type`` the driver's
    classified kind (or ``None`` for a pre-upgrade cache), and ``age_days`` the
    cache's whole-day age, carried so a stale render can say HOW dated it is
    rather than just that it is.
    """

    state: str
    label: str | None = None
    inclusion_type: str | None = None
    age_days: int | None = None


def _cache_path() -> Path:
    """Resolved lazily (not at import) so it can be redirected in tests.

    THIS function is the seam every test uses — patched directly (the
    statusline tests) or sidestepped by passing an explicit ``path=`` to the
    readers and writers (the cache tests). Patch it rather than ``Path.home``:
    one function to redirect, and it holds even if the location stops being
    derived from the home directory.
    """
    return Path.home() / ".local" / "share" / "tj" / "attribution_cache.json"


def write_attribution_cache(
    label: str,
    occurrences: int,
    sessions: int,
    inclusion_type: str | None = None,
    *,
    path: Path | None = None,
) -> None:
    """Persist the top recurring-inclusion driver. Best-effort; never raises.

    ``inclusion_type`` is the driver's classified kind (``file_read`` / ``search``
    / ``prompt`` / ``tool_output``, from ``core.context_diagnostic``). It's the
    piece the statusline consumer needs to make its remedy driver-conditional —
    ``/compact`` shrinks conversation history only, so it can't reduce statically
    re-injected content (CLAUDE.md, ``@file`` reads, re-run searches). Optional /
    nullable so a pre-upgrade cache (or a caller that doesn't classify) still
    round-trips; the consumer degrades to a driver-agnostic remedy then.
    """
    from tokenjam.utils.time_parse import utcnow

    target = path or _cache_path()
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps({
            "top_label": label,
            "occurrences": occurrences,
            "sessions": sessions,
            "inclusion_type": inclusion_type,
            "computed_at": utcnow().isoformat(),
        }))
    except Exception:  # a cache write must never break ingest
        pass


def _read_usable(path: Path | None) -> tuple[dict[str, Any], float] | None:
    """``(data, age_seconds)`` for a cache entry we can prove the age of.

    ``None`` for every ABSENT case — no file, malformed JSON, a non-dict
    payload, missing label/count, or a missing / non-string / unparseable
    ``computed_at``. The last of those matters: without a usable timestamp we
    cannot prove the entry is fresh OR say how dated it is, so it is nothing to
    show rather than a stale driver that would otherwise display forever.
    """
    target = path or _cache_path()
    if not target.is_file():
        return None
    data = json.loads(target.read_text())
    if not isinstance(data, dict):
        return None
    if not data.get("top_label") or not data.get("occurrences"):
        return None
    computed_at = data.get("computed_at")
    if not isinstance(computed_at, str):
        return None
    age = _age_seconds(computed_at)
    if age is None:
        return None
    return data, age


def read_attribution_cache(
    *, path: Path | None = None, max_age_seconds: int = MAX_CACHE_AGE_SECONDS
) -> dict[str, Any] | None:
    """Read the cached top driver, or ``None`` if missing/stale/corrupt.

    Fail-safe for the statusline hook: any error (missing file, malformed
    JSON, an aged-out entry) degrades to ``None`` rather than raising. A caller
    that wants the entry whatever its age calls :func:`resolve_driver`, which
    reports the age rather than discarding the entry.
    """
    try:
        usable = _read_usable(path)
        if usable is None:
            return None
        data, age = usable
        if age > max_age_seconds:
            return None
        return data
    except Exception:  # fail-safe read for the statusline hook
        return None


def resolve_driver(
    *, path: Path | None = None, max_age_seconds: int = MAX_CACHE_AGE_SECONDS
) -> DriverStatus:
    """The cached top driver AND whether it is current — the display seam.

    Reports :data:`DRIVER_FRESH`, :data:`DRIVER_STALE` or :data:`DRIVER_ABSENT`
    rather than collapsing the last two, because a caller renders them
    differently: absent has nothing to say, stale has the answer and only needs
    to mark it as dated. Fail-safe like every other reader here — any error
    degrades to ABSENT, never a raise, since the statusline hook is downstream.
    """
    try:
        usable = _read_usable(path)
        if usable is None:
            return DriverStatus(DRIVER_ABSENT)
        data, age = usable
        itype = data.get("inclusion_type")
        return DriverStatus(
            DRIVER_STALE if age > max_age_seconds else DRIVER_FRESH,
            f"{data.get('top_label')} ×{data.get('occurrences')}",
            itype if isinstance(itype, str) and itype else None,
            int(age // 86400),
        )
    except Exception:  # a display helper must never raise
        return DriverStatus(DRIVER_ABSENT)


def format_driver(*, path: Path | None = None) -> tuple[str | None, str | None]:
    """``(display_label, inclusion_type)`` for a CURRENT cached top driver.

    The fresh-only view of :func:`resolve_driver`, for callers that want a
    driver they can present without qualification: ``display_label`` is the
    ``"<label> ×<count>"`` string (label rendering stays as shipped), and
    ``inclusion_type`` is the driver's classified kind (``file_read`` /
    ``search`` / ``prompt`` / ``tool_output``) or ``None`` for a pre-upgrade
    cache that didn't record it. Fail-safe: any error or a missing / stale /
    malformed cache degrades to ``(None, None)``. A caller that wants to
    DISTINGUISH stale from absent (the statusline does — see this module's
    docstring) calls :func:`resolve_driver` instead.
    """
    status = resolve_driver(path=path)
    if status.state != DRIVER_FRESH:
        return None, None
    return status.label, status.inclusion_type


def format_driver_label(*, path: Path | None = None) -> str | None:
    """The cached top driver formatted as ``"<label> ×<count>"``, or ``None``.

    Thin wrapper over :func:`format_driver` returning just the display label,
    kept for the resume-brief (``cli/cmd_resume_brief``) which needs only the
    label. The statusline calls :func:`format_driver` for the type too. Both
    still route through the ONE cache reader so the two surfaces can't drift on
    field names or formatting.
    """
    return format_driver(path=path)[0]


def _age_seconds(computed_at: str) -> float | None:
    try:
        from datetime import datetime

        from tokenjam.utils.time_parse import utcnow

        ts = datetime.fromisoformat(computed_at)
        now = utcnow()
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=now.tzinfo)
        return (now - ts).total_seconds()
    except Exception:  # 1
        return None


def refresh_attribution_cache(
    conn: Any, capture: Any, *, path: Path | None = None
) -> None:
    """Compute the window's top recurring-inclusion driver and cache it.

    Two callers, both processes that already hold a live DuckDB connection and
    the ``[capture]`` flags: ``ingest_claude_code`` after a backfill (the same
    path ``tj onboard`` uses), and the daemon's analyzer scan cycle as its last
    leg. The daemon one is what keeps the display current on a machine that
    never runs a backfill — see this module's docstring. Best-effort: any
    failure (empty window, capture off, query error) leaves any existing cache
    file untouched rather than raising or clobbering it with an empty result.
    """
    try:
        from tokenjam.core.context_diagnostic import compute_context_diagnostic
        from tokenjam.utils.time_parse import utcnow

        tool_inputs = bool(getattr(capture, "tool_inputs", False))
        prompts = bool(getattr(capture, "prompts", False))
        tool_outputs = bool(getattr(capture, "tool_outputs", False))
        if not (tool_inputs or prompts or tool_outputs):
            return

        until = utcnow()
        since = until - timedelta(days=ATTRIBUTION_WINDOW_DAYS)
        diag = compute_context_diagnostic(
            conn, since, until,
            tool_inputs_captured=tool_inputs,
            prompts_captured=prompts,
            tool_outputs_captured=tool_outputs,
        )
        if not diag.recurring:
            return
        top = diag.recurring[0]
        write_attribution_cache(
            _short_label(top), top.occurrences, top.sessions,
            top.inclusion_type, path=path,
        )
    except Exception as exc:  # must never break the ingest (or pass) it follows
        # Still swallowed — this is a cache refresh and neither caller may die
        # for it — but CLASSIFIED first. A DuckDB fatal invalidates the whole
        # database instance for the process, and a silent `pass` here would
        # leave the daemon serving a database it can no longer read with
        # nothing recorded (Critical Rule 45(b): note it where it is
        # recognised, so the scan cycle's `finally` recovers off the
        # process-wide record whoever swallowed the exception).
        try:
            from tokenjam.core.db import handle_if_fatal

            handle_if_fatal(exc, what="attribution cache refresh")
        except Exception:
            pass


def _short_label(inclusion: Any) -> str:
    """A terse display label for a recurring inclusion (basename for files)."""
    from tokenjam.core.context_diagnostic import INCLUSION_FILE_READ

    if inclusion.inclusion_type == INCLUSION_FILE_READ:
        return Path(inclusion.target).name or inclusion.target
    return inclusion.label[:40]
