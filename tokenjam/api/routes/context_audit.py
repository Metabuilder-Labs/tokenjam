"""/api/v1/context-audit — the "everything that enters your Claude Code
sessions" inventory (Lens's context-audit page).

Thin by convention: all classification lives in
``core/summarize/context_audit``. This module only resolves WHICH project
roots to scan (every root the local transcript corpus has recorded a session
in, via the same ``count_invocations`` + ``resolve_roots`` derivation the
optimize `summarize` analyzer uses) and caches the result, since a filesystem
+ transcript scan has no business running on every request from a page that
polls.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Body, Depends, HTTPException, Query, Request

from tokenjam.api.deps import require_api_key, require_relearn_write_auth

router = APIRouter()

#: How long a computed audit is served before the next request re-scans.
#: Filesystem + transcript-corpus reads are not free, and this page has no
#: write path that would need an immediate invalidation. An unbounded corpus
#: scan measured at ~38s on a real multi-year, 26-project corpus — an hour is
#: long enough that a user re-opening the page minutes later never re-pays
#: that cost, while a `refresh=true` (the page's own Rescan button) still
#: gets a same-request fresh answer whenever asked.
_CACHE_TTL_SECONDS = 3600

#: How far back the transcript corpus is walked to discover project roots.
#: Unbounded (`count_invocations` over the corpus's full history) is what
#: made the first, uncached request take ~38 seconds on a real multi-year
#: corpus — every `.jsonl` ever written gets parsed just to answer "which
#: directories has this user worked in". A rolling window trades "every
#: project since the beginning of time" for "every project touched
#: recently", which is the population this page's own reader actually cares
#: about (a repo abandoned three years ago contributing a CLAUDE.md nobody's
#: context has carried since is not part of today's context floor).
_PROJECT_DISCOVERY_WINDOW_DAYS = 180

_cache: dict[str, Any] = {"result": None, "at": 0.0}


def _config(request: Request):
    return request.app.state.config


def _resolved_project_roots(config) -> list[str]:
    """Every project root the local transcript corpus has recorded a session
    in over the last `_PROJECT_DISCOVERY_WINDOW_DAYS` days — the same
    derivation `core/optimize/analyzers/summarize.py` uses for its
    corpus-wide project scope (`_invocation_counts` + `resolve_roots`),
    reused rather than re-implemented, just windowed for this page's own
    performance budget (see `_PROJECT_DISCOVERY_WINDOW_DAYS`).
    """
    from datetime import datetime, timedelta, timezone

    from tokenjam.core.summarize.invocations import count_invocations
    from tokenjam.core.summarize.repo_roots import resolve_roots
    from tokenjam.core.transcript_cache import default_cache_dir

    try:
        until = datetime.now(timezone.utc)
        invocations = count_invocations(
            until - timedelta(days=_PROJECT_DISCOVERY_WINDOW_DAYS), until,
            cache_dir=default_cache_dir(config) if config is not None else None,
        )
    except Exception:
        return []
    if not invocations.observed:
        return []
    resolved = resolve_roots(invocations.session_cwds)
    return [str(r) for r in resolved.roots]


def _compute(config):
    from tokenjam.core.summarize.context_audit import run_context_audit

    roots = _resolved_project_roots(config)
    return run_context_audit(roots)


def _cached_result(config, *, refresh: bool = False):
    """The audit RESULT object (not its dict), cached.

    The object is what is cached rather than the payload because removal
    resolves a row_id against the rows this scan produced — see
    ``ContextAuditResult.find_removable``. Caching only the dict would leave
    the remove endpoint re-running a scan measured in tens of seconds just to
    look up one row.
    """
    now = time.monotonic()
    if not refresh and _cache["result"] is not None and (now - _cache["at"]) < _CACHE_TTL_SECONDS:
        return _cache["result"]
    result = _compute(config)
    _cache["result"] = result
    _cache["at"] = now
    return result


def _invalidate() -> None:
    """Drop the cache after a removal: the page's own numbers are now stale by
    exactly the row that just went, and a Remove followed by a stale re-read
    showing the row still there reads as a failed removal."""
    _cache["result"] = None
    _cache["at"] = 0.0


@router.get("/context-audit", dependencies=[Depends(require_api_key)])
def get_context_audit(
    request: Request,
    refresh: bool = Query(False, description="Force a fresh scan, bypassing the cache."),
) -> dict[str, Any]:
    """The full context-audit payload: global scope + one entry per scanned
    project root, cached for `_CACHE_TTL_SECONDS`. `refresh=true` forces a
    fresh scan (e.g. a user-triggered "Rescan" button)."""
    return _cached_result(_config(request), refresh=refresh).to_dict()


@router.get("/context-audit/removals", dependencies=[Depends(require_api_key)])
def get_removals() -> dict[str, Any]:
    """Everything removed from this page so far, newest first, each with
    whether it can still be restored."""
    from tokenjam.core.summarize.context_quarantine import list_removals, quarantine_root

    return {"removals": list_removals(), "quarantine_dir": str(quarantine_root())}


# Both mutating routes carry the always-on local write guard as well as the
# API key. These move files on the user's machine and edit their settings
# JSON, which is strictly more destructive than the relearn writes the guard
# was first built for; a page reachable from a browser must not be able to do
# that on a cross-origin request or without the process-local token.
@router.post("/context-audit/remove",
             dependencies=[Depends(require_api_key), Depends(require_relearn_write_auth)])
def post_remove(request: Request, body: dict[str, Any] = Body(...)) -> dict[str, Any]:
    """Reversibly remove one audited row.

    409 rather than 500 on every refusal (symlink, already gone, hook entry not
    present): these are all "the machine is not in the state you saw", which is
    a conflict the user resolves by rescanning, not a server fault.
    """
    from tokenjam.core.summarize.context_quarantine import remove_file, remove_hook
    from tokenjam.core.summarize.session import SummarizeRefused

    row_id = str(body.get("row_id") or "")
    result = _cached_result(_config(request))
    row = result.find_removable(row_id)
    if row is None:
        raise HTTPException(
            status_code=404,
            detail="that row is not in the current audit — rescan and try again.")
    try:
        if row.removal_kind == "hook":
            rec = remove_hook(Path(row.origin_path), row.hook_event, row.hook_matcher,
                              row.hook_command, label=row.source)
        else:
            rec = remove_file(Path(row.origin_path), label=Path(row.origin_path).name,
                              detail=row.trigger)
    except SummarizeRefused as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    _invalidate()
    return {"removed": rec.to_dict()}


@router.post("/context-audit/restore",
             dependencies=[Depends(require_api_key), Depends(require_relearn_write_auth)])
def post_restore(body: dict[str, Any] = Body(...)) -> dict[str, Any]:
    """Put one removal back where it came from."""
    from tokenjam.core.summarize.context_quarantine import restore
    from tokenjam.core.summarize.session import SummarizeRefused

    try:
        out = restore(str(body.get("record_id") or ""))
    except SummarizeRefused as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    _invalidate()
    return {"restored": out}
