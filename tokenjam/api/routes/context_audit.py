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
from typing import Any

from fastapi import APIRouter, Depends, Query, Request

from tokenjam.api.deps import require_api_key

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


def _compute(config) -> dict[str, Any]:
    from tokenjam.core.summarize.context_audit import run_context_audit

    roots = _resolved_project_roots(config)
    return run_context_audit(roots).to_dict()


@router.get("/context-audit", dependencies=[Depends(require_api_key)])
def get_context_audit(
    request: Request,
    refresh: bool = Query(False, description="Force a fresh scan, bypassing the cache."),
) -> dict[str, Any]:
    """The full context-audit payload: global scope + one entry per scanned
    project root, cached for `_CACHE_TTL_SECONDS`. `refresh=true` forces a
    fresh scan (e.g. a user-triggered "Rescan" button)."""
    now = time.monotonic()
    if not refresh and _cache["result"] is not None and (now - _cache["at"]) < _CACHE_TTL_SECONDS:
        return _cache["result"]
    result = _compute(_config(request))
    _cache["result"] = result
    _cache["at"] = now
    return result
