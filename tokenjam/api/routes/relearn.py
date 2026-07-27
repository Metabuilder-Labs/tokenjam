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

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, ConfigDict

from tokenjam.api.deps import require_api_key, require_relearn_write_auth
from tokenjam.core.data_span import available_data_span
from tokenjam.core.framing import (
    WindowSummary,
    agent_persona_mix,
    compute_framing,
    config_declared_plan,
    dominant_persona,
    plan_determination_mix,
)
from tokenjam.core.optimize import (
    scope as scope_mod,
)
from tokenjam.core.optimize import (
    cost_apply,
    cost_proposals as cost_proposals_mod,
    inbox_contribution,
    relearn_apply,
    relearn_proposals,
    relearn_store,
)
from tokenjam.core.optimize.relearn_window import resolve_window_label, window_report

router = APIRouter()

# Write endpoints (apply/enable/disable/revert/refresh) always require BOTH
# the optional global api-key check (a no-op unless api.auth.enabled) AND the
# unconditional local write-token check (require_relearn_write_auth) — see
# api/deps.py's docstring for why the latter can't be skipped by config.
_WRITE_AUTH = [Depends(require_api_key), Depends(require_relearn_write_auth)]


def _allowed_write_root(config: Any) -> Path:
    """The one directory tree an approved write may land in, for THIS run.

    Resolved from the same ``core.optimize.scope`` contract the apply-target
    suggestion comes from (``resolve_write_scope``), never from the process's
    own ``Path.home()``. With ``--projects-root`` pointed outside ``$HOME``
    the suggestion followed the scoped home while this guard did not, so the
    UI suggested a target and the API then 403'd that exact write — the
    suggestion and the authorization disagreed. One helper, one root.

    Fail-closed: a scope that cannot be resolved raises rather than widening
    to "anywhere". With no scope override this is the real ``$HOME``, exactly
    as before.
    """
    try:
        root = scope_mod.resolve_write_scope(config).allowed_root
        return root.expanduser().resolve(strict=False)
    except (OSError, RuntimeError) as exc:
        raise HTTPException(
            status_code=403,
            detail=f"cannot resolve the allowed write root for this run ({exc}) — refusing.",
        ) from exc


def _reject_target_outside_home(target_path: str, config: Any) -> None:
    """Defense-in-depth (must-fix #1): even with write-auth enforced, refuse
    to write anywhere outside this run's allowed root (``_allowed_write_root``
    — ``$HOME`` unless the run is scoped). Every legitimate target (a
    project's CLAUDE.md/skill/hook, or a user-global ~/.claude/* file) lives
    under that root — this just makes a bug or a maliciously-crafted
    ``target_path`` (e.g. ``/etc/...``, ``/root/...``, or a ``..`` traversal
    out of the scope) fail closed rather than relying solely on the
    overwrite/symlink guards inside relearn_apply.

    ``resolve(strict=False)`` is deliberate on both sides: the target usually
    does not exist yet (that is the point of an apply), and resolving it
    collapses ``..`` segments AND follows any symlink on the way, so a symlink
    pointing out of the root is judged by where it LANDS, not by where it sits.
    """
    if not target_path:
        return   # relearn_apply itself refuses an empty target_path (409)
    root = _allowed_write_root(config)
    try:
        resolved = Path(target_path).expanduser().resolve(strict=False)
    except (OSError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=f"unresolvable target_path: {exc}") from exc
    if resolved != root and root not in resolved.parents:
        raise HTTPException(
            status_code=403,
            detail=f"target_path {resolved} is outside the allowed root ({root}) — refusing.",
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


def _persona(request: Request) -> str:
    """Dominant user persona, full-corpus (relearn is the unbounded-history
    detector — see its module docstring — so its own empty-state copy needs
    the same unbounded classification, not a windowed one that could
    disagree with what the daemon actually gated relearn's write levers on
    in ``relearn_store.recompute_now``). Mirrors that same computation;
    degrades to ``"unknown"`` on any error so a persona-classification
    failure never breaks the inbox itself.
    """
    try:
        conn = _conn(request)
        if conn is None:
            return "unknown"
        return dominant_persona(
            agent_persona_mix(conn), declared_plan=config_declared_plan(_config(request)),
        )
    except Exception:
        return "unknown"


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


#: The window the Review inbox headline is LABELLED with. Both inbox
#: endpoints (this one and ``/relearn/proposals``) resolve it through the
#: SAME function so a row can never publish a contribution a headline built
#: elsewhere never counted -- and so does the CLI's ``tj relearn
#: cost-proposals`` (``cli/cost_proposal_verbs.py``), which is why the
#: helper lives on the shared ``core/optimize/inbox_contribution.py`` module
#: rather than staying private to this route.
_headline_window_days = inbox_contribution.headline_window_days


_NO_WINDOWED_FIGURES = (
    "this cached result was produced before bounded window figures existed, so "
    "no window could be applied. The rows below are the full unbounded "
    "observation. Refresh the proposals to get windowed figures"
)


def _apply_window(
    finding: Any, since: str | None,
) -> tuple[Any, dict[str, Any], dict[str, Any] | None]:
    """Bound a cached finding to a trailing window.

    Returns ``(finding, window_block, windowed_total)``. The bounded figures were
    computed by the DETECTOR and cached (see ``core/optimize/relearn_window.py``
    for why they cannot be derived here: the cached cluster keeps a scalar
    occurrence count, and the only surviving dates are on a three-item
    newest-first example list). This function therefore SELECTS a precomputed
    window; it never computes one.

    Two populations exist in the result and both are named rather than blended.
    The rows are the clusters with at least one occurrence inside the window; the
    finding's own ``past_overspend_*`` totals still cover every cluster,
    including the omitted ones, and ``past_overspend_windowed`` is the figure
    that covers exactly the rows. ``window_block`` says how many were omitted, so
    a shorter list is never mistaken for a quieter corpus.

    A cache with no windowed figures at all cannot honor the window. Returning an
    empty list would claim the window observed nothing and returning the
    unbounded rows under the requested label would publish a corpus-wide total as
    a 24-hour one, so the rows pass through unfiltered and the block states that
    no window was applied and why.
    """
    available = list((finding or {}).get("past_overspend_windows") or {}) \
        if isinstance(finding, dict) else []
    if since is None:
        return finding, window_report(
            since=None, applied=None, available=available,
        ), None
    if not isinstance(finding, dict) or not available:
        return finding, window_report(
            since=since, applied=None, available=available,
            unavailable_reason=_NO_WINDOWED_FIGURES,
        ), None

    label = resolve_window_label(since, available)
    total = dict(finding["past_overspend_windows"][label])
    clusters = finding.get("clusters") or []
    kept: list[Any] = []
    omitted = 0
    for cluster in clusters:
        windows = cluster.get("past_overspend_windows") if isinstance(cluster, dict) else None
        bucket = (windows or {}).get(label) if isinstance(windows, dict) else None
        if bucket is None:
            # No windowed figure for this cluster: its occurrences carry no
            # parseable timestamps, so whether it recurred in the window is
            # UNKNOWN. Kept (dropping it would assert it did not) and stamped so
            # a reader is not shown an absent figure as a zero one.
            kept.append({**cluster, "window": None, "window_unknown": True})
            continue
        if not bucket.get("occurrences"):
            omitted += 1
            continue
        kept.append({**cluster, "window": bucket, "window_unknown": False})

    bounded = {**finding, "clusters": kept}
    return bounded, window_report(
        since=since, applied=total, available=available,
        clusters_in_window=len(kept), clusters_omitted=omitted,
    ), total


@router.get("/relearn/proposals", dependencies=[Depends(require_api_key)])
def get_relearn_proposals(
    request: Request,
    since: str | None = Query(
        None,
        description="Lookback window (e.g. 30d, 7d, 24h). Bounds the clusters "
                    "and the dollar figures to occurrences inside the window. "
                    "Omit for the full unbounded observation.",
    ),
) -> dict[str, Any]:
    """Cached relearn-detector proposals for the Review inbox.

    Returns ``{"status": "ready"|"computing"|"never_run", "computed_at":
    iso|null, "finding": <RelearnFinding dict>|null, "framing": dict}``. A
    fresh install (no background pass has completed yet, and none is
    running) reports ``"never_run"`` — the inbox renders its empty state for
    that, not an error. ``"computing"`` means a recompute is in flight right
    now; the ``finding``/``computed_at`` fields (when present) are still the
    last GOOD result, so the UI can keep showing it while a refresh runs.

    ``framing`` is the same plan-tier dollar-suppression block the cost-
    proposals endpoint carries (``core.framing.compute_framing``) — relearn
    clusters carry their own blended-rate ``past_overspend_usd``, the one
    canonical dollar field (no relearn-specific forward claim any more), so
    this surface needs the same suppression rule the cost-advisory dollars
    already respect rather than a second, un-gated dollar figure on the page.

    ``persona`` is the full-corpus dominant persona (see ``_persona``) — the
    inbox's empty state for this tab needs it to disclose that ``relearn``
    reads only on-disk Claude Code transcripts and therefore never surfaces
    anything for an SDK-dominant window, rather than reading as "you're doing
    great."

    ``since`` bounds the result to a trailing window, using the same helper and
    the same 400-on-malformed contract ``/cost``, ``/alerts``, ``/traces`` and
    ``/optimize`` use. This endpoint previously took no window at all, so the
    Dashboard's window selector governed three of its five Health tiles and
    silently did nothing to "Pending fixes": the identical cluster list came back
    for ``24h`` and ``90d``. The bounded figures come off the cache (the detector
    computes them; see ``_apply_window``), so a ``since`` with no precomputed
    bucket resolves to the NEAREST one and ``window.applied`` names which. Never
    the label that was asked for, when that is not the label the figures were
    computed under.

    ``window``, ``past_overspend_windowed`` and ``data_span`` are always present.
    ``past_overspend_windowed`` is ``None`` when no window was applied: an absent
    figure, not a zero one.

    Every cluster carries ``inbox_contribution_usd``/``_tokens``/``_window``/
    ``_basis``: what that row contributed to the Review inbox's ONE headline
    total, which ``/relearn/cost-proposals`` publishes and which now covers
    relearn's rows too (see ``core/optimize/inbox_contribution.py``). It is the
    bounded figure for the HEADLINE's window, net of the re-read share the
    ``resend`` proposal prices in full, and it is deliberately NOT re-based by a
    caller's own ``since``: the inbox's noise floor, its collapsed-tail combined
    figure and its headline have to read one quantity whatever the reader is
    filtering, and the selected window's own figure already travels on each row's
    ``window`` bucket. ``None`` means UNPRICED on that basis, never zero.
    """
    since_error: str | None = None
    if since is not None:
        from tokenjam.utils.time_parse import parse_since

        try:
            parse_since(since)
        except ValueError as exc:
            since_error = str(exc)
    if since_error is not None:
        raise HTTPException(status_code=400, detail=f"Invalid since: {since_error}")

    cached = relearn_store.read_cache(config=_config(request))
    computing = relearn_store.is_computing()
    conn = _conn(request)
    data_span = available_data_span(conn).to_dict()
    if cached is None:
        return {
            "status": "computing" if computing else "never_run",
            "computed_at": None,
            "finding": None,
            "framing": _framing(request),
            "persona": _persona(request),
            "window": window_report(
                since=since, applied=None, available=[],
                unavailable_reason=(
                    "no detector result has been cached yet, so there is nothing "
                    "to bound to a window"
                ) if since is not None else None,
            ),
            "past_overspend_windowed": None,
            "data_span": data_span,
        }
    finding = cached.get("finding")
    if isinstance(finding, dict):
        # Re-stamp on read as well as on write, so a cache written before the
        # proposal id or the advise-only reason existed still resolves without
        # waiting for a recompute. Idempotent.
        finding = relearn_proposals.stamp_proposal_ids(finding)
        # What each cluster contributed to the inbox's ONE headline total, on
        # the headline's own window and net of the re-read share the context
        # re-send proposal already prices. This is the field the noise floor is
        # tested against and the collapsed tail sums, so the rows, the hidden-set
        # note and the headline are one quantity over one population by
        # construction. See `core/optimize/inbox_contribution.py`; stamped
        # BEFORE `_apply_window` so a reader's own `since` cannot change which
        # window the headline's figure was taken from.
        finding = inbox_contribution.stamp_relearn_contributions(
            finding,
            label=inbox_contribution.contribution_window_label(
                finding, _headline_window_days(cached),
            ),
        )
    finding, window, windowed_total = _apply_window(finding, since)
    return {
        "status": "computing" if computing else "ready",
        "computed_at": cached.get("computed_at"),
        "finding": _with_example_resolvability(finding, conn),
        "framing": _framing(request),
        "persona": _persona(request),
        "window": window,
        "past_overspend_windowed": windowed_total,
        "data_span": data_span,
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
    403s when ``target_path`` resolves outside this run's allowed write
    root (defense-in-depth allowlist — ``$HOME`` unless the run is scoped;
    see ``_allowed_write_root``).
    """
    # Config first: the allowed write root is resolved FROM it, and a missing
    # config must 503 rather than let the guard fall back to a wider root.
    apply_config = _config(request)
    _reject_target_outside_home(body.target_path, apply_config)
    stored = relearn_proposals.get_proposal(body.proposal_id, config=apply_config)
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
        "past_overspend_usd": None,
        "past_overspend_tokens": None,
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


def _revert_linked_cost_record(config: Any, reverted_fix: dict[str, Any]) -> dict[str, Any] | None:
    """Revert the cost-applied ledger record (if any) linked to a just-reverted
    relearn fix, so the savings ledger stops counting a fix that was undone.

    The two ledgers (``relearn_apply``'s ``applied_fixes.json`` and
    ``cost_apply``'s ``cost_applied.json``) have no cross-reference field —
    they're linked solely by matching ``signature``. This is the only place
    that link is walked, so a revert here is the only way a cost-applied
    record ever gets un-counted for this path.

    Best-effort and idempotent: a missing or already-reverted cost record (or
    any lookup error) must never turn an already-successful file revert into
    an error — this degrades to ``None`` rather than raising.
    """
    signature = str(reverted_fix.get("signature") or "")
    if not signature:
        return None
    try:
        for rec in cost_apply.list_applied(config):
            if rec.get("signature") == signature and rec.get("state") != "reverted":
                return cost_apply.revert_applied(config, str(rec.get("id")))
    except Exception:
        return None
    return None


@router.post("/relearn/{fix_id}/revert", dependencies=_WRITE_AUTH)
def post_relearn_revert(request: Request, fix_id: str) -> dict[str, Any]:
    """One-step revert: disables enforcement first if live, restores the
    pre-image (or deletes a freshly-created file), commits the revert when
    the target is git-tracked.

    Also closes out any cost-applied ledger record linked to this fix by
    ``signature`` (see ``_revert_linked_cost_record``) — otherwise a reverted
    file change leaves the savings ledger counting a saving that was undone.
    """
    try:
        result = relearn_apply.revert_applied_fix(_config(request), fix_id)
    except relearn_apply.RelearnApplyRefused as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    result["cost_record_reverted"] = _revert_linked_cost_record(_config(request), result)
    return result


# --------------------------------------------------------------------------- #
# Cost proposals — the same Review inbox, a distinct `kind`. These are the
# downsize/cache/trim analyzers' findings adapted into advise-only proposals
# (core.optimize.cost_proposals). They carry NO apply path (the fix lives in the
# user's own code); "apply" is a marker the delta-verify pass measures against.
# --------------------------------------------------------------------------- #

@router.get("/relearn/cost-proposals", dependencies=[Depends(require_api_key)])
def get_cost_proposals(request: Request) -> dict[str, Any]:
    """Cost proposals for the Review inbox, listed beside relearn proposals.

    Returns ``{"status": "ready"|"computing"|"never_run"|"error",
    "computed_at": iso|null, "proposals": [dict, ...], "rollup": dict,
    "degraded": bool, "last_error": str|null, "last_error_at": iso|null}``.

    ``status`` is ``"computing"`` while a background recompute is in flight
    (scheduled job or a manual "Rescan now" — see ``cost_proposals.
    is_computing_cost_proposals``); the ``proposals``/``computed_at`` fields,
    when present, are still the last GOOD result, exactly like the relearn
    endpoint above. A fresh install with no recompute EVER completed and no
    recorded failure reports ``"never_run"`` with an empty list. A fresh
    install whose only recompute attempt(s) FAILED reports ``"error"`` — an
    empty Cost-advisories tab must never read as "nothing to report" when the
    real reason is "the scan never succeeded" (behavioral requirement #5).
    ``degraded``/``last_error``/``last_error_at`` surface a failed recompute
    even when a PRIOR good result still renders (``status`` stays
    ``"ready"``/``"computing"`` in that case) — the inbox shows a small
    inline warning rather than pretending the last refresh succeeded.

    ``past_overspend`` is THE aggregate — the ONLY one this endpoint returns
    (``cost_proposals.past_overspend_rollup``): the sum of
    ``past_overspend_usd``/``_tokens`` across the OPEN rows of the Review inbox —
    every row whose signature isn't already in a (non-reverted) applied ledger.
    It is the AVOIDABLE portion of what the flagged behaviours already cost over
    the analyzed window, observed rather than projected. Computed here, not
    client-side, so the Dashboard hero and the Review inbox headline read one
    server-computed figure and cannot disagree (a browser's local "dismiss" never
    affects it — dismissing hides a card from one person's view, it doesn't
    change what's actually still outstanding).

    **IT COVERS EVERY ROW OF THE INBOX, INCLUDING RELEARN'S.** The inbox is one
    list fed by this endpoint and by ``/relearn/proposals``, so a headline summed
    over one feed left the other's rows outside it: the collapsed tail's combined
    figure summed rows of BOTH kinds, and the below-floor note said the hidden
    items were "still counted in the total above" when most of that money had
    never entered the total. Each open relearn cluster therefore contributes an
    ordinary row on the canonical field (``core/optimize/inbox_contribution.py``),
    carrying the detector's own bounded figure for the window this rollup is
    labelled with, net of the re-read share the ``resend`` proposal already prices
    in full. One window over every row, and every dollar once. A cluster whose
    money cannot be put on that window is disclosed through ``excluded`` rather
    than counted as nothing.

    There is deliberately no second aggregate and no forward/paced one. A
    ``rollup`` key carrying ``estimated_recoverable_*`` plus a
    ``projected_usd_30d`` used to sit beside this block; the first had become
    the same number under another name and the second was it times a pace
    ratio, so a surface could render any of three near-identical dollar
    figures. Both are gone.

    An ``observed_cost_usd`` key used to sit here too — a second total, summed
    over whichever proposals carried a full-cost figure — under a disclosure
    saying the headline was a subset of it. It was not: on live data the headline
    summed 13 proposals and that total covered 2, so most of the headline sat
    outside the figure it was described as part of. Key and disclosure are both
    deleted. **Every figure this block publishes now covers the same set of
    proposals**, which is the invariant that failure exposed.

    The block still carries ``excluded`` (e.g. ``{"summarize": {...}}`` when that
    analyzer found something), waste this headline deliberately does NOT sum in
    because it has its own review surface, stated instead of silently dropped.

    Each proposal carries the same figures per-card
    (``past_overspend_usd``/``_tokens``/``_basis``, plus ``coverage_note`` where
    the avoidable figure was computed over a subset) so a card never re-derives
    its own headline.

    Every proposal ALSO carries ``inbox_contribution_usd``/``_tokens``/
    ``_window``/``_basis``: exactly what that row contributed to the headline
    above. ``/relearn/proposals`` stamps the same four fields on every relearn
    cluster, so ONE field spans both feeds and the inbox's noise floor, its
    collapsed-tail combined figure and this headline are the same quantity over
    the same population by construction. ``None`` there means UNPRICED, never
    zero: the floor may not hide such a row and no combined figure may include
    it. For a cost proposal the contribution IS ``past_overspend_usd`` unchanged;
    relearn's differs from its row's unbounded figure, which is why the field
    exists rather than each surface picking a number per row kind."""
    config = _config(request)
    block = relearn_store.read_cost_proposals(config=config)
    computing = cost_proposals_mod.is_computing_cost_proposals()
    # Listed WITH their proposal_ids: a model-routing card's Approve names an
    # ID and nothing else, so the ID has to travel with the card it belongs to.
    proposals: list[dict[str, Any]] = (
        relearn_proposals.list_cost_proposals(config)
        if block is not None and block.get("cost_computed_at") else []
    )
    applied_sigs = {
        str(rec.get("signature") or "") for rec in cost_apply.list_applied(config)
        if rec.get("state") != "reverted"
    }
    open_proposals = [
        p for p in proposals
        if not cost_apply.signature_is_applied(str(p.get("signature") or ""), applied_sigs)
    ]
    # The window this batch of proposals was actually computed over — stored
    # alongside them at recompute time, never re-derived here, so the window the
    # headline names is the window the figures were observed over. Deliberately
    # NOT accompanied by `active_days`/`n_sessions`: the rollup is a window
    # observation, so there is no pace to project it at and nothing to project
    # from (see `past_overspend_rollup`).
    window_days = _headline_window_days(block)
    # RELEARN'S ROWS ARE INBOX ROWS, SO THEIR MONEY IS IN THIS TOTAL. The inbox
    # is one list fed by two endpoints; a headline summed over one of them left
    # the other's rows outside it, which made the collapsed tail's combined
    # figure and the below-floor "still counted in the total above" note false
    # for the money they described. Each open cluster arrives here as an
    # ORDINARY row on the one canonical field — no parameter on the rollup, no
    # second aggregate, no second key — carrying the detector's own bounded
    # figure for THIS window, net of the re-read share the resend proposal
    # already prices in full. `core/optimize/inbox_contribution.py` owns that
    # design and why it is neither of the two mechanisms this repo retired.
    relearn_cache = relearn_store.read_cache(config=config)
    relearn_finding = (relearn_cache or {}).get("finding")
    relearn_label = inbox_contribution.contribution_window_label(
        relearn_finding, window_days,
    )
    relearn_applied_sigs = {
        str(rec.get("signature") or "")
        for rec in relearn_apply.list_applied(config)
        if rec.get("state") != "reverted"
    }
    relearn_rows = inbox_contribution.relearn_contribution_rows(
        relearn_finding, label=relearn_label,
        applied_signatures=relearn_applied_sigs,
    )
    # Clusters whose money could NOT be put on this window's basis (a cache
    # written before bounded figures, or occurrences with no parseable
    # timestamp). Absent is never zero: stated through the rollup's `excluded`
    # channel, summed into nothing.
    unrepresented = inbox_contribution.unrepresented_relearn(
        relearn_finding, label=relearn_label,
        applied_signatures=relearn_applied_sigs,
    )
    excluded = {
        **((block.get("cost_excluded") or {}) if block else {}),
        **inbox_contribution.relearn_excluded_entry(
            unrepresented, reason=inbox_contribution.NO_BOUNDED_WINDOW_REASON,
        ),
    }
    past_overspend = cost_proposals_mod.past_overspend_rollup(
        open_proposals + relearn_rows, window_days=window_days, excluded=excluded,
    )
    # Every row a reader sees carries what it contributed, cost and relearn
    # alike, so the noise floor and the tail's combined figure read the SAME
    # quantity the headline sums instead of each re-deriving one.
    proposals = [
        inbox_contribution.stamp_cost_contribution(p, window=f"{window_days}d")
        for p in proposals
    ]
    # Same plan-tier framing the cost-applied payload carries, so a dollar
    # figure rendered here never disagrees with its sibling surfaces.
    framing = _framing(request)
    last_error = block.get("cost_proposals_error") if block else None
    last_error_at = block.get("cost_proposals_error_at") if block else None
    has_good_result = bool(block and block.get("cost_computed_at"))
    if computing:
        status = "computing"
    elif has_good_result:
        status = "ready"
    elif last_error:
        status = "error"
    else:
        status = "never_run"
    return {
        "status": status,
        "computed_at": block.get("cost_computed_at") if block else None,
        "proposals": proposals,
        "past_overspend": past_overspend,
        "framing": framing,
        "degraded": bool(last_error),
        "last_error": last_error,
        "last_error_at": last_error_at,
    }


@router.post("/relearn/cost-proposals/refresh", dependencies=_WRITE_AUTH)
def refresh_cost_proposals(request: Request) -> dict[str, Any]:
    """Recompute cost proposals over the default window, on the request
    thread (unchanged, synchronous contract — the client awaits this before
    reading the refreshed list). Degrades to ``{"status": "unavailable"}``
    when the daemon has no direct DB connection (e.g. a proxy) rather than
    erroring. Locked against the scheduled background job (``tj serve``'s
    cost-proposals job, behavioral requirement #5) via ``cost_proposals.
    recompute_cost_proposals``'s own lock — this request either runs the
    recompute or, if the scheduled job already holds the lock, returns the
    unchanged last-good proposals rather than the two racing each other's
    cache write."""
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
    unknown ``proposal_id``; 403 outside the allowed write root; 409 on a
    refusal.
    """
    config = _config(request)
    _reject_target_outside_home(body.target_path, config)
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


class RegisterSourcePathRequest(BaseModel):
    """A named STORED cost proposal plus the ONE fact only the user can supply:
    where the flagged agent's source lives.

    Same split as its siblings — everything about WHAT is written comes off the
    store; the caller supplies only where the agent's checkout is, plus the
    go/force confirmations.
    """
    model_config = ConfigDict(extra="forbid")

    proposal_id:  str
    source_path:  str = ""
    scope:        str = "project"
    go:           bool = False
    force:        bool = False


@router.post("/relearn/cost-proposals/register-source-path", dependencies=_WRITE_AUTH)
def post_register_source_path(
    request: Request, body: RegisterSourcePathRequest,
) -> dict[str, Any]:
    """Register an agent's source path, then apply its model swap.

    The gap this closes: eleven live ``downsize`` model-swap proposals carried a
    real fix snippet and ``apply_capable: false``, for one reason — nobody had
    ever told tokenjam where those agents' source lives, and tokenjam refuses to
    scan a filesystem looking for it (``config.AgentConfig.source_path``: opt-in,
    never inferred). So the fix was deterministic, the target was one grep away
    from a path the user knows, and the row still offered nothing but a copy box.

    ``go=false`` previews: it resolves the path, re-runs every
    ``model_swap_precheck`` gate against it, and returns the diff it WOULD write,
    changing nothing on disk or in the config. ``go=true`` registers the path in
    the user's own config and then performs the swap through the existing
    ``relearn_apply`` machinery — backup, git commit, ledger, revert — with no
    second discipline invented here.

    **The write still reads its target out of the config, not out of this body.**
    That is the safety property that made ``source_path`` config-only in the first
    place (see ``relearn_proposals.APPLY_CLUSTER_FIELDS``), and registration does
    not weaken it: a caller cannot aim an edit at an arbitrary repo, it can only
    ask the user's config to be changed — deliberately, from their own machine,
    durably, and inspectably afterwards.

    Registration is per AGENT, so one answer unlocks every other proposal for the
    same agent at the next recompute. Several of the eleven share an ``agent_id``.

    403 outside the allowed write root; 409 on any refusal (a path that is not a
    directory, not in
    a git repo, a model id in several files, a dirty target, a live session in
    the repo unless ``force``); 404 on an unknown ``proposal_id``.
    """
    from tokenjam.core.optimize import model_apply

    config = _config(request)
    _reject_target_outside_home(body.source_path, config)
    stored = _stored_cost_proposal(request, body.proposal_id)
    if not stored.get("needs_source_path"):
        raise HTTPException(
            status_code=409,
            detail=f"stored proposal {body.proposal_id} is not waiting on a "
                   f"source path. Refresh the cost proposals and use the apply "
                   f"path its own card offers.",
        )
    agent_id = str(stored.get("agent_id") or "")
    current_model = str(stored.get("current_model") or "")
    proposed_model = str(stored.get("proposed_model") or "")

    # Resolve and gate BEFORE touching the config, so a preview never leaves a
    # registration behind and a refused apply never leaves a path pointing at a
    # repo the swap turned out not to be possible in.
    try:
        resolved = model_apply.resolve_source_path(body.source_path)
        check = model_apply.model_swap_precheck(resolved, current_model)
        if not check["ok"]:
            raise relearn_apply.RelearnApplyRefused(check["reason"])
        if not body.go:
            return {
                "applied": model_apply.preview_model_swap(
                    check["target_path"], current_model, proposed_model,
                ),
                "cost_record": None,
                "target_path": check["target_path"],
                "source_path": resolved,
            }
        # The cluster is projected from the STORE plus the resolved path, never
        # from the request body — the same rule `cluster_for_apply` states. It
        # is built from `resolved` directly (not a config re-read) so nothing
        # here depends on the config having been written yet: registration is
        # deferred until AFTER the apply succeeds, so a refused apply leaves the
        # on-disk config completely untouched.
        cluster = {
            "signature": str(stored.get("signature") or ""),
            "title": str(stored.get("title") or ""),
            "proposed_fix": str(stored.get("proposed_fix") or ""),
            "apply_kind": model_apply.APPLY_KIND_MODEL_SWAP,
            "current_model": current_model,
            "proposed_model": proposed_model,
            "source_path": resolved,
            "rung": int(stored.get("rung") or 1),
            "sessions": 0,
            "repos": [],
            "examples": [],
        }
        applied = relearn_apply.apply_relearn_fix(
            config, cluster, target_path=check["target_path"], scope=body.scope,
            go=True, conn=_conn(request), force=body.force,
        )
        # Only persist the registration once the swap itself has actually
        # succeeded — a refused apply (e.g. the active-session gate below)
        # must never leave a path pointing at a repo the swap turned out not
        # to be possible in.
        model_apply.register_agent_source_path(config, agent_id, resolved)
    except relearn_apply.RelearnApplyRefused as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    db = getattr(request.app.state, "db", None)
    cost_record = None
    if db is not None:
        try:
            cost_record = cost_apply.mark_applied(db, config, stored)
        except cost_apply.CostApplyRefused:
            cost_record = None
    return {
        "applied": applied,
        "cost_record": cost_record,
        "target_path": check["target_path"],
        "source_path": resolved,
    }


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
