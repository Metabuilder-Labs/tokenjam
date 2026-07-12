"""GET/POST /api/v1/pothole/* — the self-improve loop's pothole review inbox.

Serves the (expensive, full-corpus) pothole-detector result from the on-disk
cache the serve-time background job keeps warm (``core.optimize.
pothole_store``) — this route NEVER computes the finding inline on a request,
which would block the UI for the tens of seconds a full local corpus scan
takes. ``POST /refresh`` kicks a background recompute on a fresh DuckDB
connection (the retention-job pattern from ``cli/cmd_serve.py``) so it never
contends with the live request connection's write lock.

Phase 1 (detect + surface) was read-only. Phase 2 (this module's ``/apply``,
``/{id}/enable``, ``/{id}/disable``, ``/{id}/revert``, ``/applied``) adds the
Approve stage: writes route through ``core.optimize.pothole_apply`` for every
rung-routing / backup / git-commit / fail-open guarantee — this route only
translates HTTP <-> that module's ``PotholeApplyRefused`` (-> 409) contract,
it never hand-rolls a parallel write path.

Phase 3 (Verify + Compound, ``core.optimize.pothole_verify``) rides the SAME
``/refresh`` cadence: every background/manual recompute also re-measures each
applied fix's recurrence and writes its verdict back into the ledger (see
``pothole_store.recompute_now``). ``GET /applied`` now also returns a
``ledger`` summary (``pothole_verify.compound_ledger``) — realized token
savings, estimated/correlational, across every verified fix.
"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from tokenjam.api.deps import require_api_key
from tokenjam.core.optimize import pothole_apply, pothole_store, pothole_verify

router = APIRouter()


def _config(request: Request):
    config = request.app.state.config
    if config is None:
        raise HTTPException(status_code=503, detail="Server not fully initialised (config missing).")
    return config


def _conn(request: Request) -> Any | None:
    db = getattr(request.app.state, "db", None)
    return getattr(db, "conn", None) if db is not None else None


@router.get("/pothole/proposals", dependencies=[Depends(require_api_key)])
def get_pothole_proposals(request: Request) -> dict[str, Any]:
    """Cached pothole-detector proposals for the Review inbox.

    Returns ``{"status": "ready"|"computing"|"never_run", "computed_at":
    iso|null, "finding": <PotholeFinding dict>|null}``. A fresh install (no
    background pass has completed yet, and none is running) reports
    ``"never_run"`` — the inbox renders its empty state for that, not an
    error. ``"computing"`` means a recompute is in flight right now; the
    ``finding``/``computed_at`` fields (when present) are still the last
    GOOD result, so the UI can keep showing it while a refresh runs.
    """
    cached = pothole_store.read_cache()
    computing = pothole_store.is_computing()
    if cached is None:
        return {
            "status": "computing" if computing else "never_run",
            "computed_at": None,
            "finding": None,
        }
    return {
        "status": "computing" if computing else "ready",
        "computed_at": cached.get("computed_at"),
        "finding": cached.get("finding"),
    }


@router.post("/pothole/refresh", dependencies=[Depends(require_api_key)])
def refresh_pothole_proposals(request: Request) -> dict[str, Any]:
    """Kick a background recompute. A recompute already in flight is a no-op
    (returns ``already_running``) — never queued twice."""
    config = request.app.state.config
    if config is None:
        raise HTTPException(status_code=503, detail="Server not fully initialised.")

    from tokenjam.core.db import DuckDBBackend

    started = pothole_store.trigger_background_recompute(
        lambda: DuckDBBackend(config.storage), config=config,
    )
    return {"status": "started" if started else "already_running"}


# --------------------------------------------------------------------------- #
# Apply stage (Phase 2) — every write routes through `core.optimize.
# pothole_apply`, which owns the rung-routing / backup / git-commit /
# fail-open / active-session-guard guarantees. Default is a DRY-RUN
# (go=False): the UI's card shows the diff before the user commits to it.
# --------------------------------------------------------------------------- #

class ApplyPotholeRequest(BaseModel):
    # The full cluster the client already has from GET /pothole/proposals —
    # re-posted rather than re-looked-up server-side so a stale cache can't
    # silently apply a DIFFERENT cluster than the one the human reviewed.
    signature:      str
    family_key:     str | None = None
    title:          str
    proposed_fix:   str = ""
    rung:           int
    sessions:       int = 0
    occurrences:    int = 0
    repos:          list[str] = []
    examples:       list[dict[str, Any]] = []
    # Scope override (§7 — "repo-identity is noisy"): the human confirms both
    # before Approve, never inferred silently.
    scope:          str
    target_path:    str
    go:             bool = False
    force:          bool = False   # bypass the active-session warning


@router.post("/pothole/apply", dependencies=[Depends(require_api_key)])
def post_pothole_apply(request: Request, body: ApplyPotholeRequest) -> dict[str, Any]:
    """Dry-run (default) or write (``go=true``) an approved fix at its rung.

    409s (via ``PotholeApplyRefused``) on: an unknown rung, a create-only
    target (skill/hook) that already holds a non-TokenJam file, or — unless
    ``force=true`` — a live session just seen in the target repo (§7: never
    apply mid-session). The UI's re-send-with-force is the explicit
    "apply anyway" the spec calls for.
    """
    cluster = body.model_dump(exclude={"scope", "target_path", "go", "force"})
    try:
        return pothole_apply.apply_pothole_fix(
            _config(request), cluster,
            target_path=body.target_path, scope=body.scope,
            go=body.go, conn=_conn(request), force=body.force,
        )
    except pothole_apply.PotholeApplyRefused as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.get("/pothole/applied", dependencies=[Depends(require_api_key)])
def get_pothole_applied(request: Request) -> dict[str, Any]:
    """Every applied fix (applied + reverted) — the inbox's 'Applied' section,
    plus the Phase 3 Compound ledger summary (``core.optimize.pothole_verify.
    compound_ledger``): total realized savings across every VERIFIED fix."""
    applied = pothole_apply.list_applied(_config(request))
    return {"applied": applied, "ledger": pothole_verify.compound_ledger(applied)}


class EnableEnforcementRequest(BaseModel):
    confirm: bool = False


@router.post("/pothole/{fix_id}/enable", dependencies=[Depends(require_api_key)])
def post_pothole_enable(request: Request, fix_id: str, body: EnableEnforcementRequest) -> dict[str, Any]:
    """Wire a generated rung 3-5 hook into settings.json. Requires an explicit
    ``confirm: true`` — the UI's "this intercepts your tools" warning."""
    try:
        return pothole_apply.enable_enforcement(_config(request), fix_id, confirm=body.confirm)
    except pothole_apply.PotholeApplyRefused as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/pothole/{fix_id}/disable", dependencies=[Depends(require_api_key)])
def post_pothole_disable(request: Request, fix_id: str) -> dict[str, Any]:
    """Unwire a hook from settings.json (the hook file itself stays on disk)."""
    try:
        return pothole_apply.disable_enforcement(_config(request), fix_id)
    except pothole_apply.PotholeApplyRefused as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/pothole/{fix_id}/revert", dependencies=[Depends(require_api_key)])
def post_pothole_revert(request: Request, fix_id: str) -> dict[str, Any]:
    """One-step revert: disables enforcement first if live, restores the
    pre-image (or deletes a freshly-created file), commits the revert when
    the target is git-tracked."""
    try:
        return pothole_apply.revert_applied_fix(_config(request), fix_id)
    except pothole_apply.PotholeApplyRefused as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
