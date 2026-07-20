"""GET/POST /api/v1/relearn/* — the self-improve loop's relearn review inbox.

Serves the (expensive, full-corpus) relearn-detector result from the on-disk
cache the serve-time background job keeps warm (``core.optimize.
relearn_store``) — this route NEVER computes the finding inline on a request,
which would block the UI for the tens of seconds a full local corpus scan
takes. ``POST /refresh`` kicks a background recompute on a fresh DuckDB
connection (the retention-job pattern from ``cli/cmd_serve.py``) so it never
contends with the live request connection's write lock.

Phase 1 (detect + surface) was read-only. Phase 2 (this module's ``/apply``,
``/{id}/enable``, ``/{id}/disable``, ``/{id}/revert``, ``/applied``) adds the
Approve stage: writes route through ``core.optimize.relearn_apply`` for every
rung-routing / backup / git-commit / fail-open guarantee — this route only
translates HTTP <-> that module's ``RelearnApplyRefused`` (-> 409) contract,
it never hand-rolls a parallel write path.

Phase 3 (Verify + Compound, ``core.optimize.relearn_verify``) rides the SAME
``/refresh`` cadence: every background/manual recompute also re-measures each
applied fix's recurrence and writes its verdict back into the ledger (see
``relearn_store.recompute_now``). ``GET /applied`` now also returns a
``ledger`` summary (``relearn_verify.compound_ledger``) — realized token
savings, estimated/correlational, across every verified fix.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from tokenjam.api.deps import require_api_key, require_relearn_write_auth
from tokenjam.core.optimize import (
    cost_apply,
    cost_proposals as cost_proposals_mod,
    cost_verify,
    relearn_apply,
    relearn_store,
    relearn_verify,
)

router = APIRouter()

# Write endpoints (apply/enable/disable/revert/refresh) always require BOTH
# the optional global api-key check (a no-op unless api.auth.enabled) AND the
# unconditional local write-token check (require_relearn_write_auth) — see
# api/deps.py's docstring for why the latter can't be skipped by config.
_WRITE_AUTH = [Depends(require_api_key), Depends(require_relearn_write_auth)]


def _reject_target_outside_home(target_path: str) -> None:
    """Defense-in-depth (must-fix #1): even with write-auth enforced, refuse
    to write anywhere outside the user's home directory. Every legitimate
    target (a project's CLAUDE.md/skill/hook, or a user-global ~/.claude/*
    file) lives under $HOME — this just makes a bug or a maliciously-crafted
    ``target_path`` (e.g. ``/etc/...``, ``/root/...``) fail closed rather than
    relying solely on the overwrite/symlink guards inside relearn_apply.
    """
    if not target_path:
        return   # relearn_apply itself refuses an empty target_path (409)
    try:
        resolved = Path(target_path).expanduser().resolve(strict=False)
        home = Path.home().resolve(strict=False)
    except (OSError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=f"unresolvable target_path: {exc}") from exc
    if resolved != home and home not in resolved.parents:
        raise HTTPException(
            status_code=403,
            detail=f"target_path {resolved} is outside the allowed root ({home}) — refusing.",
        )


def _config(request: Request):
    config = request.app.state.config
    if config is None:
        raise HTTPException(status_code=503, detail="Server not fully initialised (config missing).")
    return config


def _conn(request: Request) -> Any | None:
    db = getattr(request.app.state, "db", None)
    return getattr(db, "conn", None) if db is not None else None


@router.get("/relearn/proposals", dependencies=[Depends(require_api_key)])
def get_relearn_proposals(request: Request) -> dict[str, Any]:
    """Cached relearn-detector proposals for the Review inbox.

    Returns ``{"status": "ready"|"computing"|"never_run", "computed_at":
    iso|null, "finding": <RelearnFinding dict>|null}``. A fresh install (no
    background pass has completed yet, and none is running) reports
    ``"never_run"`` — the inbox renders its empty state for that, not an
    error. ``"computing"`` means a recompute is in flight right now; the
    ``finding``/``computed_at`` fields (when present) are still the last
    GOOD result, so the UI can keep showing it while a refresh runs.
    """
    cached = relearn_store.read_cache(config=_config(request))
    computing = relearn_store.is_computing()
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


@router.post("/relearn/refresh", dependencies=_WRITE_AUTH)
def refresh_relearn_proposals(request: Request) -> dict[str, Any]:
    """Kick a background recompute. A recompute already in flight is a no-op
    (returns ``already_running``) — never queued twice."""
    config = request.app.state.config
    if config is None:
        raise HTTPException(status_code=503, detail="Server not fully initialised.")

    from tokenjam.core.db import DuckDBBackend

    started = relearn_store.trigger_background_recompute(
        lambda: DuckDBBackend(config.storage), config=config,
    )
    return {"status": "started" if started else "already_running"}


# --------------------------------------------------------------------------- #
# Apply stage (Phase 2) — every write routes through `core.optimize.
# relearn_apply`, which owns the rung-routing / backup / git-commit /
# fail-open / active-session-guard guarantees. Default is a DRY-RUN
# (go=False): the UI's card shows the diff before the user commits to it.
# --------------------------------------------------------------------------- #

class ApplyRelearnRequest(BaseModel):
    # The full cluster the client already has from GET /relearn/proposals —
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
    # Model-routing apply kinds (``core.optimize.model_apply``): setting an
    # agent file's `model:` key, or swapping one exact model id in a repo the
    # user registered. Empty on every rung-ladder fix, which is every other
    # proposal. The model ids travel with the request for the same reason the
    # cluster does: the human approved THESE values on the card.
    apply_kind:     str = ""
    agent_name:     str = ""
    current_model:  str = ""
    proposed_model: str = ""
    source_path:    str = ""
    # Which cost analyzer's card this write came from, so the same apply also
    # opens a cost-verify exposure window and the realized delta lands in the
    # ledger the receipts surface reads dollars from. Empty for rung-ladder
    # fixes, whose receipts are recurrence-based rather than priced.
    analyzer:       str = ""
    agent_id:       str = ""


@router.post("/relearn/apply", dependencies=_WRITE_AUTH)
def post_relearn_apply(request: Request, body: ApplyRelearnRequest) -> dict[str, Any]:
    """Dry-run (default) or write (``go=true``) an approved fix at its rung.

    409s (via ``RelearnApplyRefused``) on: an unknown rung, a create-only
    target (skill/hook) that already holds a non-TokenJam file, or — unless
    ``force=true`` — a live session just seen in the target repo (§7: never
    apply mid-session). The UI's re-send-with-force is the explicit
    "apply anyway" the spec calls for. 403s when ``target_path`` resolves
    outside the user's home directory (defense-in-depth allowlist).
    """
    _reject_target_outside_home(body.target_path)
    cluster = body.model_dump(
        exclude={"scope", "target_path", "go", "force", "analyzer", "agent_id"},
    )
    try:
        result = relearn_apply.apply_relearn_fix(
            _config(request), cluster,
            target_path=body.target_path, scope=body.scope,
            go=body.go, conn=_conn(request), force=body.force,
        )
    except relearn_apply.RelearnApplyRefused as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if body.go and body.apply_kind and body.analyzer:
        result["cost_marker"] = _open_cost_verify_window(request, body)
    return result


def _open_cost_verify_window(request: Request, body: ApplyRelearnRequest) -> dict[str, Any] | None:
    """Start the priced exposure window for a model-routing write.

    A model swap is a cost fix that happens to have a file to edit, so its
    receipt has to be a measured dollar delta on spans, not a recurrence count.
    That measurement hangs off the cost-applied ledger, so the same approval
    that wrote the file also opens the window. Best-effort: a marker that cannot
    be created must never fail an apply that already succeeded on disk.
    """
    from tokenjam.core.optimize import cost_apply

    proposal = {
        "signature": body.signature,
        "analyzer": body.analyzer,
        "title": body.title,
        "agent_id": body.agent_id,
        "advise_text": body.proposed_fix,
        "target_key": {
            "models": [body.current_model] if body.current_model else [],
            "subagent": body.analyzer == "subagent",
            "agent_name": body.agent_name,
        },
        "baseline": {"proposed_model": body.proposed_model},
        "estimated_recoverable_usd": None,
        "estimated_recoverable_tokens": None,
        "estimate_basis": "",
    }
    try:
        return cost_apply.mark_applied(_conn(request), _config(request), proposal)
    except Exception:
        return None


@router.get("/relearn/applied", dependencies=[Depends(require_api_key)])
def get_relearn_applied(request: Request) -> dict[str, Any]:
    """Every applied fix (applied + reverted) — the inbox's 'Applied' section,
    plus the Phase 3 Compound ledger summary (``core.optimize.relearn_verify.
    compound_ledger``): total realized savings across every VERIFIED fix."""
    applied = relearn_apply.list_applied(_config(request))
    return {"applied": applied, "ledger": relearn_verify.compound_ledger(applied)}


class EnableEnforcementRequest(BaseModel):
    confirm: bool = False


@router.post("/relearn/{fix_id}/enable", dependencies=_WRITE_AUTH)
def post_relearn_enable(request: Request, fix_id: str, body: EnableEnforcementRequest) -> dict[str, Any]:
    """Wire a generated rung 3-5 hook into settings.json. Requires an explicit
    ``confirm: true`` — the UI's "this intercepts your tools" warning."""
    try:
        return relearn_apply.enable_enforcement(_config(request), fix_id, confirm=body.confirm)
    except relearn_apply.RelearnApplyRefused as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/relearn/{fix_id}/disable", dependencies=_WRITE_AUTH)
def post_relearn_disable(request: Request, fix_id: str) -> dict[str, Any]:
    """Unwire a hook from settings.json (the hook file itself stays on disk)."""
    try:
        return relearn_apply.disable_enforcement(_config(request), fix_id)
    except relearn_apply.RelearnApplyRefused as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/relearn/{fix_id}/revert", dependencies=_WRITE_AUTH)
def post_relearn_revert(request: Request, fix_id: str) -> dict[str, Any]:
    """One-step revert: disables enforcement first if live, restores the
    pre-image (or deletes a freshly-created file), commits the revert when
    the target is git-tracked."""
    try:
        return relearn_apply.revert_applied_fix(_config(request), fix_id)
    except relearn_apply.RelearnApplyRefused as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


# --------------------------------------------------------------------------- #
# Cost proposals — the same Review inbox, a distinct `kind`. These are the
# downsize/cache/trim analyzers' findings adapted into advise-only proposals
# (core.optimize.cost_proposals). They carry NO apply path (the fix lives in the
# user's own code); "apply" is a marker the delta-verify pass measures against.
# --------------------------------------------------------------------------- #

@router.get("/relearn/cost-proposals", dependencies=[Depends(require_api_key)])
def get_cost_proposals(request: Request) -> dict[str, Any]:
    """Cost proposals for the Review inbox, listed beside relearn proposals.

    Returns ``{"status": "ready"|"never_run", "computed_at": iso|null,
    "proposals": [dict, ...]}``. A fresh install (no cost recompute has run)
    reports ``never_run`` with an empty list — the inbox renders its empty
    state, not an error."""
    block = relearn_store.read_cost_proposals(config=_config(request))
    if block is None:
        return {"status": "never_run", "computed_at": None, "proposals": []}
    return {
        "status": "ready",
        "computed_at": block.get("cost_computed_at"),
        "proposals": block.get("cost_proposals") or [],
    }


@router.post("/relearn/cost-proposals/refresh", dependencies=_WRITE_AUTH)
def refresh_cost_proposals(request: Request) -> dict[str, Any]:
    """Recompute cost proposals over the default window AND re-measure the
    realized delta of every applied cost fix (same cadence). Degrades to
    ``{"status": "unavailable"}`` when the daemon has no direct DB connection
    (e.g. a proxy) rather than erroring."""
    config = _config(request)
    db = getattr(request.app.state, "db", None)
    if db is None or getattr(db, "conn", None) is None:
        return {"status": "unavailable", "reason": "no direct database connection"}
    proposals = cost_proposals_mod.recompute_cost_proposals(db, config)
    verify = cost_verify.rescan_all(db, config)
    return {"status": "ready", "proposals": len(proposals), "verified": verify}


class MarkCostAppliedRequest(BaseModel):
    # The full proposal the client already holds from GET /relearn/cost-proposals
    # — re-posted so a stale cache can't mark a DIFFERENT proposal than the one
    # the human reviewed (same guard as ApplyRelearnRequest).
    signature:   str
    analyzer:    str = ""
    title:       str = ""
    target_key:  dict[str, Any] = {}
    baseline:    dict[str, Any] = {}
    advise_text: str = ""
    agent_id:    str = ""
    estimated_recoverable_usd:    float | None = None
    estimated_recoverable_tokens: int | None = None
    estimate_basis: str = ""


@router.post("/relearn/cost-proposals/apply", dependencies=_WRITE_AUTH)
def post_cost_mark_applied(request: Request, body: MarkCostAppliedRequest) -> dict[str, Any]:
    """Mark a cost proposal applied: create the fix marker (an Expectation) and
    a ledger record the delta-verify pass measures against. There is NO code
    write — cost proposals are advise-only. 409 on a malformed proposal or when
    the marker can't be created (no writable DB)."""
    db = getattr(request.app.state, "db", None)
    if db is None:
        raise HTTPException(status_code=503, detail="Server not fully initialised (db missing).")
    try:
        return cost_apply.mark_applied(db, _config(request), body.model_dump())
    except cost_apply.CostApplyRefused as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


class ApplyWorkspaceCostRequest(BaseModel):
    # The subagent (CC-origin) proposal the client holds, plus the confirmed
    # write target. Re-posted so a stale cache can't apply a DIFFERENT proposal.
    signature:    str
    analyzer:     str = ""
    title:        str = ""
    target_key:   dict[str, Any] = {}
    baseline:     dict[str, Any] = {}
    advise_text:  str = ""
    agent_id:     str = ""
    estimated_recoverable_usd:    float | None = None
    estimated_recoverable_tokens: int | None = None
    estimate_basis: str = ""
    # Apply plumbing (adapter-supplied). rung is 1 for the sizing-rubric note.
    proposed_fix: str = ""
    rung:         int = 1
    scope:        str = "project"
    target_path:  str = ""
    go:           bool = False
    force:        bool = False


@router.post("/relearn/cost-proposals/apply-workspace", dependencies=_WRITE_AUTH)
def post_cost_apply_workspace(request: Request, body: ApplyWorkspaceCostRequest) -> dict[str, Any]:
    """Apply a CC-origin subagent proposal's sizing-rubric note to the workspace.

    Unlike the three advise-only analyzers, a subagent right-sizing finding has a
    writable surface (a rung-1 CLAUDE.md sizing rubric the orchestrating agent
    reads before spawning subagents). This routes the actual write through the
    EXISTING relearn apply path (``relearn_apply.apply_relearn_fix``) — same
    reversible, git-committed, human-gated (dry-run first) discipline — then
    records the cost marker so the delta-verify pass measures the fan-out
    model-mix cost delta after it. ``go=false`` returns the dry-run diff; a
    second call with ``go=true`` writes. 403 outside home; 409 on a refusal.
    """
    _reject_target_outside_home(body.target_path)
    config = _config(request)
    db = getattr(request.app.state, "db", None)
    # The cluster shape relearn_apply renders a rung-1 note from.
    cluster = {
        "signature": body.signature,
        "family_key": "subagent_rightsizing",
        "title": body.title or body.signature,
        "proposed_fix": body.proposed_fix,
        "rung": body.rung,
        "sessions": int(body.baseline.get("flagged_subagents", 0) or 0),
        "repos": [],
        "examples": [],
    }
    try:
        applied = relearn_apply.apply_relearn_fix(
            config, cluster, target_path=body.target_path, scope=body.scope,
            go=body.go, conn=_conn(request), force=body.force,
        )
    except relearn_apply.RelearnApplyRefused as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    # Dry-run: return the diff, don't touch the cost ledger.
    if not body.go or applied.get("dry_run"):
        return {"applied": applied, "cost_record": None}

    # Real write happened: drop the cost marker so the realized fan-out model-mix
    # delta is measured against this moment.
    cost_record = None
    if db is not None:
        try:
            cost_record = cost_apply.mark_applied(
                db, config, body.model_dump(exclude={
                    "proposed_fix", "rung", "scope", "target_path", "go", "force",
                }),
            )
        except cost_apply.CostApplyRefused:
            cost_record = None
    return {"applied": applied, "cost_record": cost_record}


@router.get("/relearn/cost-applied", dependencies=[Depends(require_api_key)])
def get_cost_applied(request: Request) -> dict[str, Any]:
    """Every applied (and reverted) cost fix, plus the realized-dollars ledger
    summary (``cost_verify.cost_compound_ledger``)."""
    applied = cost_apply.list_applied(_config(request))
    return {"applied": applied, "ledger": cost_verify.cost_compound_ledger(applied)}


@router.post("/relearn/cost-applied/{record_id}/revert", dependencies=_WRITE_AUTH)
def post_cost_revert(request: Request, record_id: str) -> dict[str, Any]:
    """Mark a cost fix reverted (the user undid their change). Advise-only, so
    there is no file to restore — this just stops the ledger counting its
    realized delta."""
    try:
        return cost_apply.revert_applied(_config(request), record_id)
    except cost_apply.CostApplyRefused as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
