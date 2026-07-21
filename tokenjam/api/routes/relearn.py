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
it never hand-rolls a parallel write path. ``/apply`` names a STORED proposal
(``core.optimize.relearn_proposals``) and never accepts cluster content from
the caller, so what gets written is always something the detector produced.

"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, ConfigDict

from tokenjam.api.deps import require_api_key, require_relearn_write_auth
from tokenjam.core.framing import (
    WindowSummary,
    compute_framing,
    plan_determination_mix,
)
from tokenjam.core.optimize import (
    cost_apply,
    cost_proposals as cost_proposals_mod,
    relearn_apply,
    relearn_proposals,
    relearn_store,
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


def _framing(request: Request) -> dict[str, Any]:
    """Plan-tier framing block for this module's dollar-bearing payloads.

    Same single compute path every other cost surface uses
    (``core.framing.compute_framing``) so the UI never re-derives the
    suppress-dollars-for-subscription rule in JS. The mix is
    window-INDEPENDENT (``plan_determination_mix``, as on /status): the
    cost-proposal figures are cumulative-to-date, not scoped to a window.
    Degrades to the config-declared plan when the daemon has no direct DB
    connection (e.g. a proxy), exactly as ``compute_framing`` already handles
    an empty mix.
    """
    db = getattr(request.app.state, "db", None)
    conn = getattr(db, "conn", None) if db is not None else None
    mix = plan_determination_mix(conn) if conn is not None else {}
    framing = compute_framing(
        _config(request),
        WindowSummary(plan_tier_mix=mix, sessions=sum(mix.values())),
    )
    return framing.to_dict()


def _conn(request: Request) -> Any | None:
    db = getattr(request.app.state, "db", None)
    return getattr(db, "conn", None) if db is not None else None


def _resolvable_session_ids(conn: Any | None, session_ids: list[str]) -> set[str]:
    """Subset of `session_ids` that exist as rows in the sessions table."""
    if conn is None or not session_ids:
        return set()
    placeholders = ", ".join(f"${i}" for i in range(1, len(session_ids) + 1))
    try:
        rows = conn.execute(
            f"SELECT session_id FROM sessions WHERE session_id IN ({placeholders})",
            session_ids,
        ).fetchall()
    except Exception:
        return set()
    return {str(r[0]) for r in rows}


def _with_example_resolvability(finding: Any, conn: Any | None) -> Any:
    """Copy of `finding` with each cluster example stamped `session_resolvable`.

    The detector sources example session ids from Claude Code transcript files
    on disk (`<projects_root>/**/<session_id>.jsonl`), so an example can name a
    session that was never ingested into the sessions table. Resolvability is
    computed at read time rather than baked into the cached finding, so it also
    covers findings stored before this field existed and stays correct as more
    sessions get ingested. The Review inbox links only resolvable examples and
    renders the rest as plain evidence text instead of a link to a dead page.
    """
    if not isinstance(finding, dict):
        return finding
    clusters = finding.get("clusters")
    if not isinstance(clusters, list):
        return finding
    ids = sorted({
        str(ex["session_id"])
        for c in clusters if isinstance(c, dict)
        for ex in (c.get("examples") or [])
        if isinstance(ex, dict) and ex.get("session_id")
    })
    resolvable = _resolvable_session_ids(conn, ids)
    return {
        **finding,
        "clusters": [
            c if not isinstance(c, dict) else {
                **c,
                "examples": [
                    ex if not isinstance(ex, dict) else {
                        **ex,
                        "session_resolvable": str(ex.get("session_id")) in resolvable,
                    }
                    for ex in (c.get("examples") or [])
                ],
            }
            for c in clusters
        ],
    }


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
    finding = cached.get("finding")
    if isinstance(finding, dict):
        # Re-stamp on read as well as on write, so a cache written before the
        # proposal id or the advise-only reason existed still resolves without
        # waiting for a recompute. Idempotent.
        finding = relearn_proposals.stamp_proposal_ids(finding)
    return {
        "status": "computing" if computing else "ready",
        "computed_at": cached.get("computed_at"),
        "finding": _with_example_resolvability(finding, _conn(request)),
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
    """A named STORED proposal plus the human's confirmed write target.

    The cluster content itself is never accepted from the client: it is looked
    up server-side from the detector's own stored proposals
    (``core.optimize.relearn_proposals``). ``extra="forbid"`` makes that
    explicit rather than silent, so a caller still posting a hand-built
    cluster gets a 422 telling it what changed instead of having its payload
    quietly ignored.
    """
    model_config = ConfigDict(extra="forbid")

    proposal_id:    str
    # Scope override (§7 — "repo-identity is noisy"): the human confirms both
    # before Approve, never inferred silently.
    scope:          str
    target_path:    str
    go:             bool = False
    force:          bool = False   # bypass the active-session warning

    # Nothing else. The model-routing values (apply_kind / agent_name /
    # current_model / proposed_model / source_path) and the cost-verify routing
    # (analyzer / agent_id) all come off the stored proposal, because the card
    # the human approved was rendered FROM that stored proposal. Reading them
    # back out of the request would be trusting the caller to echo faithfully
    # something the server already knows — and would let any holder of a valid
    # proposal_id aim source_path at an unregistered repo, which is the exact
    # precondition the model_swap safety case rests on.


@router.post("/relearn/apply", dependencies=_WRITE_AUTH)
def post_relearn_apply(request: Request, body: ApplyRelearnRequest) -> dict[str, Any]:
    """Dry-run (default) or write (``go=true``) an approved fix at its rung.

    Takes a ``proposal_id`` from ``GET /relearn/proposals``. 404s when no
    stored proposal carries that ID: a client-constructed cluster has no way
    into the write machinery, which is what makes "human-gated" a property of
    the server rather than of the UI flow.

    409s (via ``RelearnApplyRefused``) on: an unknown rung, a family with no
    matcher at an enforcement rung, a create-only target (skill/hook) that
    already holds a non-TokenJam file, or (unless ``force=true``) a live
    session just seen in the target repo (§7: never apply mid-session). The
    UI's re-send-with-force is the explicit "apply anyway" the spec calls for.
    403s when ``target_path`` resolves outside the user's home directory
    (defense-in-depth allowlist).
    """
    _reject_target_outside_home(body.target_path)
    stored = relearn_proposals.get_proposal(body.proposal_id, config=_config(request))
    if stored is None:
        raise HTTPException(
            status_code=404,
            detail=f"no stored proposal {body.proposal_id}. Refresh the proposals "
                   f"and apply one the detector actually produced.",
        )
    cluster = relearn_proposals.cluster_for_apply(stored)
    missing = relearn_proposals.missing_apply_fields(cluster)
    if missing:
        raise HTTPException(
            status_code=409,
            detail=f"stored proposal {body.proposal_id} is missing "
                   f"{', '.join(missing)}, which its "
                   f"{cluster.get('apply_kind')} apply cannot be built without. "
                   f"Recompute the proposals and retry.",
        )
    try:
        result = relearn_apply.apply_relearn_fix(
            _config(request), cluster,
            target_path=body.target_path, scope=body.scope,
            go=body.go, conn=_conn(request), force=body.force,
        )
    except relearn_apply.RelearnApplyRefused as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if body.go and cluster.get("apply_kind") and stored.get("analyzer"):
        result["cost_marker"] = _open_cost_verify_window(request, stored, cluster)
    return result


def _open_cost_verify_window(
    request: Request, stored: dict[str, Any], cluster: dict[str, Any],
) -> dict[str, Any] | None:
    """Start the priced exposure window for a model-routing write.

    A model swap is a cost fix that happens to have a file to edit, so its
    receipt has to be a measured dollar delta on spans, not a recurrence count.
    That measurement hangs off the cost-applied ledger, so the same approval
    that wrote the file also opens the window. Best-effort: a marker that cannot
    be created must never fail an apply that already succeeded on disk.

    Every value here comes from the STORED proposal, never the request body,
    for the same reason the write itself does: the ledger must record what the
    detector produced and the human reviewed, not what a caller asserts.
    """
    from tokenjam.core.optimize import cost_apply

    analyzer = str(stored.get("analyzer") or "")
    current_model = str(cluster.get("current_model") or "")
    proposal = {
        "signature": str(cluster.get("signature", "")),
        "analyzer": analyzer,
        "title": str(cluster.get("title", "")),
        "agent_id": str(stored.get("agent_id") or ""),
        "advise_text": str(cluster.get("proposed_fix", "")),
        "target_key": {
            "models": [current_model] if current_model else [],
            "subagent": analyzer == "subagent",
            "agent_name": str(cluster.get("agent_name") or ""),
        },
        "baseline": {"proposed_model": str(cluster.get("proposed_model") or "")},
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
    """Every applied fix (applied + reverted) — the inbox's 'Applied' section."""
    applied = relearn_apply.list_applied(_config(request))
    return {"applied": applied}


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
    "proposals": [dict, ...], "rollup": dict}``. A fresh install (no cost
    recompute has run) reports ``never_run`` with an empty list — the inbox
    renders its empty state, not an error.

    ``rollup`` is Component E's single "estimated recoverable" headline
    (``cost_proposals.estimated_recoverable_rollup``): the sum of
    ``estimated_recoverable_usd`` across the OPEN proposals only — every
    proposal whose signature isn't already in the (non-reverted) cost-applied
    ledger. Computed here, not client-side, so the headline reflects every
    viewer's state consistently (a browser's local "dismiss" never affects
    this figure — dismissing hides a card from one person's view, it doesn't
    change what's actually still outstanding)."""
    config = _config(request)
    block = relearn_store.read_cost_proposals(config=config)
    # Listed WITH their proposal_ids: a model-routing card's Approve names an
    # ID and nothing else, so the ID has to travel with the card it belongs to.
    proposals: list[dict[str, Any]] = (
        relearn_proposals.list_cost_proposals(config) if block is not None else []
    )
    applied_sigs = {
        rec.get("signature") for rec in cost_apply.list_applied(config)
        if rec.get("state") != "reverted"
    }
    open_proposals = [p for p in proposals if p.get("signature") not in applied_sigs]
    rollup = cost_proposals_mod.estimated_recoverable_rollup(open_proposals)
    # Same plan-tier framing the cost-applied payload carries, so a dollar
    # figure rendered here never disagrees with its sibling surfaces.
    framing = _framing(request)
    if block is None:
        return {
            "status": "never_run", "computed_at": None, "proposals": [],
            "rollup": rollup, "framing": framing,
        }
    return {
        "status": "ready",
        "computed_at": block.get("cost_computed_at"),
        "proposals": proposals,
        "rollup": rollup,
        "framing": framing,
    }


@router.post("/relearn/cost-proposals/refresh", dependencies=_WRITE_AUTH)
def refresh_cost_proposals(request: Request) -> dict[str, Any]:
    """Recompute cost proposals over the default window. Degrades to
    ``{"status": "unavailable"}`` when the daemon has no direct DB connection
    (e.g. a proxy) rather than erroring."""
    config = _config(request)
    db = getattr(request.app.state, "db", None)
    if db is None or getattr(db, "conn", None) is None:
        return {"status": "unavailable", "reason": "no direct database connection"}
    proposals = cost_proposals_mod.recompute_cost_proposals(db, config)
    return {"status": "ready", "proposals": len(proposals)}


def _stored_cost_proposal(request: Request, proposal_id: str) -> dict[str, Any]:
    """The stored cost proposal with this ID, or a 404.

    Resolved from the detector's own stored proposals
    (``relearn_proposals.list_cost_proposals``) rather than from the request
    body, so the ledger records what the detector produced and the human
    reviewed.
    """
    for proposal in relearn_proposals.list_cost_proposals(_config(request)):
        if proposal.get("proposal_id") == proposal_id:
            return proposal
    raise HTTPException(
        status_code=404,
        detail=f"no stored cost proposal {proposal_id}. Refresh the cost "
               f"proposals and apply one the detector actually produced.",
    )


class MarkCostAppliedRequest(BaseModel):
    """A named STORED cost proposal, and nothing else.

    The proposal's content (signature, analyzer, target_key, baseline, the
    estimate and its basis) is looked up server-side from the detector's own
    stored cost proposals, never accepted from the client — the same guard
    ``ApplyRelearnRequest`` uses. ``extra="forbid"`` makes that explicit rather
    than silent, so a caller still posting a hand-built proposal gets a 422
    telling it what changed instead of having its numbers quietly ignored.
    """
    model_config = ConfigDict(extra="forbid")

    proposal_id: str


@router.post("/relearn/cost-proposals/apply", dependencies=_WRITE_AUTH)
def post_cost_mark_applied(request: Request, body: MarkCostAppliedRequest) -> dict[str, Any]:
    """Mark a cost proposal applied: create the fix marker (an Expectation) and
    a ledger record of what was approved. There is NO code write — cost
    proposals are advise-only.

    Takes a ``proposal_id`` from ``GET /relearn/cost-proposals``. 404s when no
    stored cost proposal carries that ID. 409 on a malformed proposal or when
    the marker can't be created (no writable DB)."""
    db = getattr(request.app.state, "db", None)
    if db is None:
        raise HTTPException(status_code=503, detail="Server not fully initialised (db missing).")
    stored = _stored_cost_proposal(request, body.proposal_id)
    try:
        return cost_apply.mark_applied(db, _config(request), stored)
    except cost_apply.CostApplyRefused as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


class ApplyWorkspaceCostRequest(BaseModel):
    """A named STORED cost proposal plus the human's confirmed write target.

    Same split as ``ApplyRelearnRequest``: the proposal's content and its apply
    plumbing (``proposed_fix``, ``rung``) come off the store, because the card
    the human approved was rendered FROM that stored proposal; only the write
    target and the go/force confirmations are the caller's to choose.
    """
    model_config = ConfigDict(extra="forbid")

    proposal_id:  str
    scope:        str = "project"
    target_path:  str = ""
    go:           bool = False
    force:        bool = False


@router.post("/relearn/cost-proposals/apply-workspace", dependencies=_WRITE_AUTH)
def post_cost_apply_workspace(request: Request, body: ApplyWorkspaceCostRequest) -> dict[str, Any]:
    """Apply an ``apply_capable`` cost proposal's workspace note/skill.

    Covers every analyzer whose fix is a workspace surface an orchestrating
    agent (or the model itself) reads before acting, rather than a file this
    proposal can edit outright: ``subagent`` (rung-1 sizing rubric),
    ``script`` (rung-2 deterministic-workflow skill), ``reuse`` (rung-1
    planning-skeleton note), ``verbosity`` (rung-1 output-brevity note). This
    routes the actual write through the EXISTING relearn apply path
    (``relearn_apply.apply_relearn_fix``) — same reversible, git-committed,
    human-gated (dry-run first) discipline — then records the cost marker so
    the delta-verify pass measures the realized delta after it. ``go=false``
    returns the dry-run diff; a second call with ``go=true`` writes. 404 on an
    unknown ``proposal_id``; 403 outside home; 409 on a refusal.
    """
    _reject_target_outside_home(body.target_path)
    config = _config(request)
    db = getattr(request.app.state, "db", None)
    stored = _stored_cost_proposal(request, body.proposal_id)
    signature = str(stored.get("signature") or "")
    analyzer = str(stored.get("analyzer") or "")
    baseline = dict(stored.get("baseline") or {})
    # The cluster shape relearn_apply renders a rung-1/2 note/skill from,
    # projected from the STORE: a caller-supplied proposed_fix would be
    # arbitrary text written into the user's workspace under a reviewed
    # proposal's name. `apply_sessions` falls back to the subagent analyzer's
    # own baseline key (`flagged_subagents`) so this generalization doesn't
    # change that analyzer's existing behavior.
    cluster = {
        "signature": signature,
        "family_key": f"cost_{analyzer}" if analyzer else "cost_proposal",
        "title": str(stored.get("title") or "") or signature,
        "proposed_fix": str(stored.get("proposed_fix") or ""),
        "rung": int(stored.get("rung") or 1),
        "sessions": int(
            baseline.get("apply_sessions", baseline.get("flagged_subagents", 0)) or 0
        ),
        "repos": list(baseline.get("apply_repos") or []),
        "examples": list(baseline.get("apply_examples") or []),
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
            cost_record = cost_apply.mark_applied(db, config, stored)
        except cost_apply.CostApplyRefused:
            cost_record = None
    return {"applied": applied, "cost_record": cost_record}


@router.get("/relearn/cost-applied", dependencies=[Depends(require_api_key)])
def get_cost_applied(request: Request) -> dict[str, Any]:
    """Every applied (and reverted) cost fix, plus the plan-tier ``framing``
    block so any dollar figure a caller renders from it is suppressed /
    reframed for subscription users like every other cost surface."""
    applied = cost_apply.list_applied(_config(request))
    return {
        "applied": applied,
        "framing": _framing(request),
    }


@router.post("/relearn/cost-applied/{record_id}/revert", dependencies=_WRITE_AUTH)
def post_cost_revert(request: Request, record_id: str) -> dict[str, Any]:
    """Mark a cost fix reverted (the user undid their change). Advise-only, so
    there is no file to restore."""
    try:
        return cost_apply.revert_applied(_config(request), record_id)
    except cost_apply.CostApplyRefused as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
