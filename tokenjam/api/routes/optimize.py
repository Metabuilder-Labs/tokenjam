"""
GET /api/v1/optimize — the STORED analyzer report. Never a live analyzer run.

This route used to call `build_report(...)` on the request thread, which
dispatched every registered analyzer over the corpus — including `relearn`,
whose own store exists precisely because it is "far too slow to compute per
HTTP request". `fast=true` did not save it: it dropped only `trim`, applied no
timeout, and still ran the full-corpus scans. Measured on a real corpus the
Review inbox (which reads a stored block) painted instantly while the
Dashboard's recoverable-waste panel and Budget-at-risk card took tens of
seconds to minutes.

So no request path runs an analyzer any more. `core.optimize.report_store`
holds the report; the `tj serve` daemon computes it at boot, on the configured
interval, and when a user presses Rescan (`POST /optimize/rescan`, which starts
a BACKGROUND pass and returns immediately). This route reads that store and
returns the stored body plus the freshness envelope (`status`, `computed_at`,
the window it was observed over).

Ingestion is untouched: traces, spans, sessions, maps, approach and timeline
still update continuously. Only the analyzer layer moved off the request path.

**A cold store is not an empty result.** `status: "never_run"` comes back with
NO report body — never a zeroed one. A zero here would read as "no waste
found", which is a reassurance the data does not support.
"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from tokenjam.api.deps import require_api_key, require_relearn_write_auth
from tokenjam.cli.cmd_optimize import _rank_findings
from tokenjam.core.data_span import available_data_span
from tokenjam.core.framing import (
    PERSONAS,
    WindowSummary,
    agent_persona_mix,
    compute_framing,
    plan_tier_mix,
)
from tokenjam.core.optimize import (
    ANALYZER_REGISTRY,
    disabled_analyzers_for_persona,
    findings_for_persona,
    report_store,
)
from tokenjam.core.persona_scope import persona_scopes_population
from tokenjam.utils.time_parse import parse_since, utcnow

router = APIRouter()

# Rescan is a write-shaped action (it starts a full-corpus pass), so it takes
# the same local write-token gate the relearn write endpoints use — an
# unauthenticated caller must not be able to make the daemon scan on demand.
_WRITE_AUTH = [Depends(require_api_key), Depends(require_relearn_write_auth)]


@router.get("/optimize", dependencies=[Depends(require_api_key)])
def get_optimize(
    request: Request,
    since: str = Query(
        "30d",
        description="Accepted for backwards compatibility and echoed back as "
                    "requested_since. The response is the STORED report, which "
                    "was computed over [optimize] scan_window_days — see "
                    "window_days / scan_since / scan_until.",
    ),
    agent_id: str | None = Query(None, alias="agent_id"),
    finding: list[str] | None = Query(
        None, description="Accepted and echoed back; the stored report always "
                          "contains every analyzer the persona gate allowed.",
    ),
    budget_provider: str | None = Query(None),
    budget_usd: float | None = Query(None),
    persona: str | None = Query(
        None,
        description="Serve the report AS this persona (one of core.framing."
                    "PERSONAS). The analyzer set is gated for the REQUESTED "
                    "persona, not only for the corpus's dominant one, so the "
                    "dashboard's 'Viewing as' picker gets an answer that is "
                    "correct for what it is showing. Omitted = the stored "
                    "report's own dominant persona.",
    ),
    fast: bool = Query(
        False,
        description="Accepted for backwards compatibility and ignored: no "
                    "analyzer runs on this request at any speed.",
    ),
) -> dict[str, Any]:
    """Serve the stored analyzer report plus its freshness envelope.

    When a report exists, the stored body is returned at the TOP LEVEL (so
    `report_from_dict(payload)` keeps working for the CLI) with the envelope
    keys merged alongside it. When the store is cold or has only ever failed,
    the envelope comes back on its own with `report_available: false` — the
    caller renders "not yet computed", never a zero.

    **`persona` is a VIEW of one stored artifact, never a second computation.**
    No analyzer runs here at any speed, so a per-persona request cannot be
    answered by recomputing — and it must not be answered by serving the
    dominant persona's findings under the requested persona's name either. The
    daemon's pass therefore dispatches the UNION of what each gated persona has
    a lever for (`runner.build_report`'s `personas`), stores ONE report, and
    this route narrows it: findings the requested persona has no lever for are
    dropped, `persona_disabled_analyzers` names them, and
    `persona_unanswered_analyzers` names the ones the requested persona DOES
    have a lever for that this pass never dispatched. That last list is the
    honest "not yet known" channel (root anti-pattern 22) — a surface renders it
    as unresolved, never as "found nothing". It is empty on a report written by
    a build that computes the union, and non-empty only for one written before.
    """
    db = request.app.state.db
    config = request.app.state.config
    if db is None or config is None:
        raise HTTPException(
            status_code=503,
            detail="Server not fully initialised (db or config missing).",
        )

    # `since` is still validated so a malformed window is a 400 rather than
    # being silently ignored, even though it no longer selects the data.
    try:
        parse_since(since)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid --since: {exc}") from exc

    # An unrecognised persona is a 400, never a silent fallback to the dominant
    # one: a caller that mistypes it would otherwise be served a report gated
    # for a persona it did not ask for, with nothing on the wire saying so.
    if persona is not None and persona not in PERSONAS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown persona {persona!r}. Expected one of {sorted(PERSONAS)}.",
        )

    envelope = report_store.stored_report_block(config)
    envelope["requested_since"] = since
    envelope["requested_findings"] = list(finding) if finding else None
    envelope["requested_persona"] = persona
    envelope["scan_interval_hours"] = getattr(config.optimize, "scan_interval_hours", None)
    envelope["scan_enabled"] = getattr(config.optimize, "scan_enabled", True)
    envelope["ui_poll_seconds"] = getattr(config.optimize, "scan_ui_poll_seconds", 0)

    body = report_store.stored_report_dict(config)
    if body is None:
        # COLD (or error-only). No report body, no finding list, no rollups —
        # emphatically no zeros. Everything downstream must render this as
        # "not computed yet", which is a different claim from "found nothing".
        envelope["report_available"] = False
        return envelope

    # THE SCOPED BODY, when one was asked for. Slicing the analyzer SET for a
    # persona (further down) was only ever half the gate: the stored top-level
    # report is computed over the whole corpus, so every figure surviving that
    # slice was still the mixed corpus's money wearing the reader's persona
    # label. The scan cycle stores one fully-scoped report per persona
    # (`runner.build_persona_reports`) precisely because a population, unlike a
    # set of names, cannot be narrowed after the fact.
    view_persona = persona or str(body.get("persona") or "unknown")
    if persona_scopes_population(view_persona):
        scoped = (body.get("persona_reports") or {}).get(view_persona)
        if not isinstance(scoped, dict):
            # An artifact written before per-persona passes existed. It holds
            # corpus-wide figures and cannot answer for this persona. Serving
            # them anyway is the bug; serving them as this persona's ZERO would
            # be the same bug in the reassuring direction. So this reads as
            # COLD — not computed yet — which is the state a rescan resolves.
            envelope["report_available"] = False
            envelope["unavailable_reason"] = "persona_unscoped"
            envelope["view_persona"] = view_persona
            envelope["report_persona"] = body.get("persona")
            return envelope
        body = scoped

    # The STORED DICT verbatim. Deliberately NOT `report_to_dict(rehydrated)`:
    # the store already holds exactly what the serializer produced, so passing
    # it through a rehydration step could only ever lose something. The typed
    # report below is used for the two derivations that need real objects, and
    # never re-serialized onto the wire.
    payload: dict[str, Any] = dict(body)
    payload.update(envelope)
    payload["report_available"] = True
    payload["findings"] = _with_window_scoped_relearn(
        payload.get("findings"), envelope.get("window_days"), config=config,
    )

    # Rehydrated from THE SAME dict the payload was built from — the scoped one
    # when a persona narrowed it. Reading the top-level report here instead
    # would derive the ranking, the window and the framing from the whole
    # corpus while the findings beside them came from one persona: two
    # populations in one response, which is the class of defect this whole
    # change closes.
    report = report_store.report_from_stored_dict(body)
    if report is None:
        # The stored dict is present but un-rehydratable (a corrupt or
        # far-future payload). Serve the body — it is what the analyzers
        # wrote — and omit only the derivations that need typed objects.
        payload["finding_rank"] = []
        payload["persona_disabled_analyzers"] = []
        payload["persona_unanswered_analyzers"] = []
        payload["skipped_analyzers"] = []
        return payload

    # The persona this response is ANSWERING FOR, resolved above so the scoped
    # body could be selected with it. `report.persona` stays on the payload
    # untouched (it is what the corpus IS, and the CLI round-trips it);
    # everything gated below keys off `view_persona`, which is what the reader
    # asked to see.
    payload["report_persona"] = report.persona
    payload["view_persona"] = view_persona
    # WHAT POPULATION THESE FIGURES COVER. `None` means the whole corpus, which
    # is the right answer for a persona that narrows nothing (`mixed` /
    # `unknown`) and for an unnarrowed request. A client publishing a figure
    # under a persona label must check this matches.
    payload["persona_scope"] = getattr(report, "persona_scope", None)

    # `fast` no longer skips anything (nothing runs here), so nothing is
    # "skipped for speed". The key stays for wire compatibility.
    persona_disabled = disabled_analyzers_for_persona(view_persona)
    payload["skipped_analyzers"] = []
    # The names the persona gate dropped, so the UI can tell "ran, found
    # nothing" (render the empty state) from "not run for this persona"
    # (render nothing at all).
    payload["persona_disabled_analyzers"] = sorted(persona_disabled)

    # NOT-YET-KNOWN, kept strictly separate from both of the above. These are
    # analyzers the requested persona HAS a lever for and this pass never
    # dispatched, so this report holds no answer about them — which is not the
    # same claim as "they found nothing" and must not render as one (root
    # anti-pattern 22). `computed_analyzers` is empty on a report written before
    # the field existed; an empty list there means "unknown", so nothing is
    # declared unanswered rather than everything being.
    computed = set(getattr(report, "computed_analyzers", None) or [])
    payload["persona_unanswered_analyzers"] = sorted(
        (set(ANALYZER_REGISTRY) - persona_disabled) - computed,
    ) if computed else []

    # SLICED FOR THE REQUESTED PERSONA. Dropping them here rather than leaving
    # it to the client is what makes every consumer of this payload correct by
    # construction: a surface that forgets to read
    # `persona_disabled_analyzers` can no longer render a finding for a lever
    # this persona does not have.
    payload["findings"] = findings_for_persona(payload.get("findings") or {}, view_persona)
    if "downsize" in persona_disabled:
        payload["downgrade"] = None

    # Biggest-waste-first ranking — the same `_rank_findings` the CLI's text
    # view ranks by, so the web doesn't fall back to Object.keys() insertion
    # order. `share` of None means "no quantified estimate" (unranked), which
    # is NOT zero — the UI must not sort those away as de-minimis.
    payload["finding_rank"] = [
        {"name": name, "share": share}
        for name, share in _rank_findings(report, None)
        if name not in persona_disabled
    ]

    # Plan-tier / persona mix are cheap direct queries (no analyzer), so they
    # stay live: they describe the corpus as it is right now, and a stale mix
    # would frame a fresh figure under the wrong pricing mode.
    conn = getattr(db, "conn", None)
    since_dt = report.window.since
    until_dt = report.window.until or utcnow()
    payload["plan_tier_mix"] = _mix(plan_tier_mix, conn, since_dt, until_dt, agent_id)
    payload["agent_persona_mix"] = _mix(agent_persona_mix, conn, since_dt, until_dt, agent_id)

    w = report.window
    payload["framing"] = compute_framing(
        config,
        WindowSummary(
            total_cost_usd=float(getattr(w, "total_cost_usd", 0.0) or 0.0),
            total_tokens=int(getattr(w, "total_tokens", 0) or 0),
            sessions=int(getattr(w, "sessions", 0) or 0),
            plan_tier_mix=payload["plan_tier_mix"],
        ),
    ).to_dict()

    # `available_days` (core/data_span.py) so the Optimize window selector can
    # derive its options from what the store actually holds, the same way the
    # Dashboard's does — instead of a fixed 7d/30d/90d list.
    payload["data_span"] = available_data_span(conn).to_dict()

    return payload


@router.get("/optimize/analyzers", dependencies=[Depends(require_api_key)])
def get_optimize_analyzers() -> dict[str, Any]:
    """Which analyzers run for each persona — the resolved gate, not the map.

    The dashboard's analyzer guide has to say "these are the checks that run
    for your setup", which means enumerating the gate for a persona OTHER
    than the one the current window happens to resolve to. `GET /optimize`
    can only ever publish `persona_disabled_analyzers` for the stored
    report's own persona, so a guide built on it would have had to re-declare
    the gating map as a JS literal — the exact desync
    `PERSONA_DISABLED_ANALYZERS` is a single source of truth to prevent.

    Everything here is derived: `ANALYZER_REGISTRY` for what exists,
    `disabled_analyzers_for_persona` for what each persona has no lever for,
    and `runs` is the difference. `disabled` is the raw map entry, so it can
    legitimately name a sub-check that is not itself a registry entry
    (`placement`, which `downsize` attaches).

    Static config, no corpus read: it answers identically on a cold store,
    which is what lets the guide render before any scan has completed.
    """
    from tokenjam.core.framing import PERSONAS
    from tokenjam.core.optimize import ANALYZER_REGISTRY

    registered = sorted(ANALYZER_REGISTRY)
    personas: dict[str, Any] = {}
    for persona in PERSONAS:
        disabled = disabled_analyzers_for_persona(persona)
        personas[persona] = {
            "runs": [name for name in registered if name not in disabled],
            "disabled": sorted(disabled),
        }
    return {"registered": registered, "personas": personas}


#: The tile-level fields a window-scoped finding publishes. Present on relearn
#: whenever a bucket for the report's own window exists; the surface renders
#: THESE, never the unbounded `past_overspend_usd` beside them.
WINDOW_SCOPED_USD = "window_scoped_past_overspend_usd"
WINDOW_SCOPED_TOKENS = "window_scoped_past_overspend_tokens"
WINDOW_SCOPED_WINDOW = "window_scoped_window"
WINDOW_SCOPED_BASIS = "window_scoped_basis"

#: Why a finding that HAS a window vocabulary published no figure for this
#: report's window. A surface may not fall back to the unbounded figure on
#: seeing this — that fallback is the defect this whole field exists to close.
WINDOW_SCOPED_UNAVAILABLE_BASIS = (
    "unknown, not zero, and emphatically not the unbounded figure beside it. "
    "This analyzer measures over all retained history and publishes bounded "
    "buckets for a fixed vocabulary of windows; none of them equals the window "
    "this report was computed over, so it has no figure that can sit beside "
    "this report's other findings. Refresh the analyzer pass to fold this in"
)


def _with_window_scoped_relearn(
    findings: Any, window_days: Any, *, config: Any = None,
) -> Any:
    """``findings`` with relearn carrying a figure on THIS report's window.

    Relearn is the one analyzer whose ``past_overspend_usd`` is unbounded by
    design — its signal is recurrence across history, so ``run(ctx)``
    deliberately does not forward the report's ``since``. Every other finding on
    this payload is scoped to ``window_days``. The Dashboard's recoverable-waste
    row rendered all of them as peers, so relearn's all-history figure sat
    unmarked beside five window-scoped ones and a reader summing the row got a
    total on no basis at all.

    The bounded figure already existed: the detector precomputes a windowed
    bucket vocabulary while it still holds the per-occurrence dates
    (``core/optimize/relearn_window``). This selects the one matching this
    report's own window and nets it through the SAME helper the Review inbox
    row uses, so the two surfaces publish one relearn number per window from
    one code path rather than two that happen to agree.

    Derived on read, not stored: the stored dict stays exactly what the
    analyzers wrote — the unbounded fields are the full-corpus observation and
    must not be shrunk in place.

    Immutable: returns new dicts, never writes into the stored body.
    """
    from tokenjam.core.optimize.inbox_contribution import window_scoped_finding_figure

    if not isinstance(findings, dict):
        return findings
    relearn = findings.get("relearn")
    if not isinstance(relearn, dict):
        return findings
    # The clusters the user has ALREADY fixed, excluded here for the same
    # reason the Review inbox excludes them: a headline states what is still
    # outstanding. Without this the tile kept counting recovered money, so
    # applying a fix moved the inbox and left the Dashboard unchanged. Failing
    # to resolve them degrades toward the whole-population figure, which is
    # the old behaviour rather than a new wrong one.
    applied: set[str] = set()
    try:
        from tokenjam.core.optimize import relearn_apply

        applied = set(relearn_apply.applied_signatures(config))
    except Exception:
        applied = set()
    figure = window_scoped_finding_figure(
        relearn, days=window_days, applied_signatures=applied,
    )
    scoped = {
        WINDOW_SCOPED_USD: None if figure is None else figure["usd"],
        WINDOW_SCOPED_TOKENS: None if figure is None else figure["tokens"],
        WINDOW_SCOPED_WINDOW: None if figure is None else figure["window"],
        WINDOW_SCOPED_BASIS: (
            WINDOW_SCOPED_UNAVAILABLE_BASIS if figure is None else figure["basis"]
        ),
    }
    return {**findings, "relearn": {**relearn, **scoped}}


def _mix(fn: Any, conn: Any, since_dt: Any, until_dt: Any, agent_id: str | None) -> dict:
    """Best-effort mix query; `{}` when the storage layer exposes no connection
    (e.g. a proxy backend) rather than failing the whole read."""
    if conn is None:
        return {}
    try:
        return dict(fn(conn, since_dt, until_dt, agent_id))
    except Exception:
        return {}


@router.post("/optimize/rescan", dependencies=_WRITE_AUTH)
def rescan_optimize(request: Request) -> dict[str, Any]:
    """Start a background analyzer scan and return immediately.

    Three rails, all always-on rather than staged:

    * **Overlap guard** — `report_store.trigger_background_recompute` no-ops
      when a scan is already in flight, so pressing Rescan twice (or pressing
      it while the scheduled job runs) costs one pass, not two.
    * **Rate limit** — a request inside `[optimize] scan_min_rescan_seconds`
      is answered `throttled` with the stored result untouched, so a user
      cannot hammer full-corpus passes.
    * **Own connection** — the scan runs on a daemon thread against a FRESH
      `DuckDBBackend`, never this request's connection, so it never contends
      with the live writer and never blocks this response.
    """
    config = request.app.state.config
    db = getattr(request.app.state, "db", None)
    if config is None or db is None or getattr(db, "conn", None) is None:
        return {"status": "unavailable", "reason": "no direct database connection"}

    if report_store.is_computing():
        return {**report_store.stored_report_block(config), "started": False,
                "reason": "a scan is already running"}
    if report_store.rescan_throttled(config):
        return {**report_store.stored_report_block(config), "started": False,
                "throttled": True,
                "reason": "rescanned too recently; showing the stored result"}

    from tokenjam.core.db import DuckDBBackend
    from tokenjam.core.optimize.scan_cycle import trigger_scan_cycle

    # EVERY analyzer store, not just the report. This endpoint used to refresh
    # the report alone while the Review inbox's own Refresh refreshed the other
    # two, so "Rescan" meant something different depending on which screen you
    # pressed it from — and the Dashboard's tiles could end up hours fresher
    # than the inbox headline they are naturally compared against. One cycle,
    # one meaning: see `core/optimize/scan_cycle.py`.
    # force=True: an explicit rescan bypasses the ingestion-watermark gate
    # (`core/optimize/scan_cycle.py`) — a human asking must always attempt a
    # pass, never get deferred to the next scheduled tick just because the
    # corpus hasn't grown.
    started = trigger_scan_cycle(
        lambda: DuckDBBackend(config.storage), config, force=True,
    )
    any_started = any(started.values())
    return {
        **report_store.stored_report_block(config),
        # True when ANY pass started. A `False` per store is its own overlap
        # guard declining because a pass is already in flight — a no-op, not a
        # failure — so the per-store detail travels alongside rather than
        # collapsing into a single misleading `false`.
        "started": any_started,
        "started_by_store": started,
        # EVERY `started: false` carries a reason, on this path too. The two
        # early returns above have said why since they were written; this one
        # answered 200 with a bare `false` and nothing to render, so a refusal
        # arrived at the client indistinguishable from a successful start. It
        # fires when the cycle's own in-flight guard declines — a pass launched
        # by the daemon's startup kick, or one whose report leg has landed while
        # relearn and the cost proposals are still being built (exactly when an
        # impatient user presses the button).
        **({} if any_started else {"reason": "a scan is already running"}),
    }
