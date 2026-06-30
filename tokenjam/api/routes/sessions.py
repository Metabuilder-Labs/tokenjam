"""Session routes.

GET /api/v1/sessions — session enumeration (one row per session). Unlike
/status (which collapses to one record per agent via
`ORDER BY started_at DESC LIMIT 1`), this route returns every matching session,
so a single agent running multiple concurrent sessions is fully represented.
The MCP `list_active_sessions` tool proxies here in serve mode so its output
matches the direct-DB path (#35).

POST /api/v1/sessions/close — mark a terminal's sessions closed. Claude Code
emits no "session closed" telemetry, so tj can't passively know a terminal
exited. The `claude` shell wrapper installed by `tj onboard --claude-code`
reports the exit explicitly by POSTing here (via `tj session-end`) when
`claude` returns or is interrupted. This is a write endpoint gated by the same
ingest Bearer auth as POST /api/v1/spans (see IngestAuthMiddleware.PROTECTED_PATHS),
so it is not additionally gated by the read-side API key.

POST /api/v1/sessions/{id}/label — set (or clear) a user-supplied display name
for a session (the dashboard's right-click rename). API-key gated.

GET /api/v1/sessions/{session_id} — per-session detail rollup for the dashboard
Session Detail view. Read-only; guarded by `require_api_key` like other GET
endpoints. Includes a per-subagent cost/token breakdown (`subagents`) for
Claude Code sessions backfilled with sub_agent_id; the live OTLP path is a
flat 2-level tree carrying no subagent identity, so those sessions show an
empty breakdown (re-run `tj backfill claude-code --reingest` to populate
history ingested before the column existed).
"""
from __future__ import annotations

import bisect
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse

from tokenjam.api.deps import require_api_key
from tokenjam.api.routes.runs import _child_sessions, _run_sessions
from tokenjam.core.db import delete_session_label, set_session_label
from tokenjam.core.distill import distill_titles_cached, peek_cached_titles
from tokenjam.core.method_capture import capture_session_method, load_session_method
from tokenjam.core.framing import (
    WindowSummary,
    compute_framing,
    plan_determination_mix,
)
from tokenjam.core.method_spine import build_method_spine
from tokenjam.core.models import AlertFilters, SessionRecord
from tokenjam.core.runlink import scan_transcript_run_ids
from tokenjam.core.sessionmap import build_session_map
from tokenjam.core.transcript import (
    build_session_asks,
    build_session_story,
    resolve_projects_root,
    session_transcript_mtime,
)
from tokenjam.core.workmap import build_work_map
from tokenjam.otel.semconv import GenAIAttributes

router = APIRouter()

# Max tools to surface in the per-session breakdown (most-called first).
MAX_SESSION_TOOLS = 15
# Max alerts to surface for the session.
SESSION_ALERT_LIMIT = 50
# Max traces to list for the session.
SESSION_TRACE_LIMIT = 100
# Max points emitted in the context-growth series. Sessions with more
# llm.call spans are downsampled (first + last always kept) so the payload
# stays bounded for the dashboard's inline visualization.
MAX_CONTEXT_POINTS = 120


@router.get("/sessions", dependencies=[Depends(require_api_key)])
async def list_sessions(
    request: Request,
    status: str | None = None,
    agent_id: str | None = None,
) -> dict:
    """Enumerate sessions one row per session, newest first.

    `status` filters by session status (e.g. 'active'); `agent_id` scopes to a
    single agent. Both are optional.
    """
    db = request.app.state.db
    conn = getattr(db, "conn", None)
    if conn is None:
        return {"sessions": [], "count": 0}

    clauses: list[str] = []
    params: list = []
    if status:
        params.append(status)
        clauses.append(f"status = ${len(params)}")
    if agent_id:
        params.append(agent_id)
        clauses.append(f"agent_id = ${len(params)}")
    where = (" WHERE " + " AND ".join(clauses)) if clauses else ""

    rows = conn.execute(
        "SELECT session_id, agent_id, started_at, total_cost_usd, "
        "input_tokens, output_tokens, tool_call_count, error_count "
        f"FROM sessions{where} ORDER BY started_at DESC",
        params,
    ).fetchall()
    sessions = [
        {
            "session_id": r[0],
            "agent_id": r[1],
            "started_at": r[2].isoformat() if r[2] else None,
            "total_cost_usd": float(r[3]) if r[3] else 0.0,
            "input_tokens": r[4] or 0,
            "output_tokens": r[5] or 0,
            "tool_call_count": r[6] or 0,
            "error_count": r[7] or 0,
        }
        for r in rows
    ]
    return {"sessions": sessions, "count": len(sessions)}


@router.post("/sessions/close")
async def close_sessions(request: Request) -> JSONResponse:
    """Close all active sessions matching an instance_id and/or session_id.

    Body: ``{"instance_id": "<id>"}`` and/or ``{"session_id": "<id>"}`` —
    at least one is required. Idempotent: closing already-closed sessions is a
    no-op. Returns ``{"closed": <count>}``.
    """
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content={"error": "Invalid JSON body"})

    if not isinstance(body, dict):
        return JSONResponse(
            status_code=400, content={"error": "Expected a JSON object"}
        )

    instance_id = body.get("instance_id")
    session_id = body.get("session_id")
    if not instance_id and not session_id:
        return JSONResponse(
            status_code=400,
            content={"error": "Provide instance_id and/or session_id"},
        )

    db = request.app.state.db

    # Identify the sessions we're about to close BEFORE closing them, so we can
    # snapshot each one's reconstructed method (M1). The `claude` wrapper closes
    # by instance_id, so resolve that to its currently-active session ids; an
    # explicit session_id is included directly.
    target_ids: set[str] = set()
    if session_id:
        target_ids.add(session_id)
    if instance_id and hasattr(db, "conn"):
        rows = db.conn.execute(
            "SELECT session_id FROM sessions "
            "WHERE service_instance_id = $1 AND status = 'active'",
            [instance_id],
        ).fetchall()
        target_ids.update(r[0] for r in rows if r[0])

    closed = 0
    if instance_id:
        closed += db.close_sessions_by_instance(instance_id)
    if session_id:
        closed += db.close_session_by_id(session_id)

    # Persist a method snapshot for each closed session so an ephemeral agent's
    # Story survives Claude Code pruning the on-disk transcript. Best-effort:
    # capture_session_method never raises, so it can't break the close.
    if target_ids:
        override = getattr(request.app.state, "claude_projects_root", None)
        projects_root = resolve_projects_root(override)
        for sid in target_ids:
            capture_session_method(
                db, sid, projects_dir=projects_root, source="live-transcript"
            )

    return JSONResponse(status_code=200, content={"closed": closed})


# Max length of a user-supplied session label; longer input is truncated.
MAX_SESSION_LABEL_LEN = 120


@router.post(
    "/sessions/{session_id}/label",
    dependencies=[Depends(require_api_key)],
)
async def set_session_label_endpoint(
    request: Request, session_id: str
) -> JSONResponse:
    """Set (or clear) a user-supplied display name for a session.

    Body: ``{"label": "<str>"}``. The label is stripped and truncated to
    ``MAX_SESSION_LABEL_LEN``; an empty (or whitespace-only) label CLEARS any
    existing rename, reverting the card to its default (service.instance.id ->
    short session id). The /status route overlays this onto the tile/archive
    label, taking precedence over the OTel instance id but NOT over a config
    ``[session_labels]`` entry (see ``status._session_label``).

    Dashboard action gated by ``require_api_key`` (the UI's ``apiPost`` sends the
    Bearer api key) — deliberately NOT added to the ingest-secret PROTECTED_PATHS.
    Returns ``{"session_id": ..., "label": <str|None>}`` (None when cleared).
    """
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content={"error": "Invalid JSON body"})

    if not isinstance(body, dict):
        return JSONResponse(
            status_code=400, content={"error": "Expected a JSON object"}
        )

    label = (body.get("label") or "").strip()[:MAX_SESSION_LABEL_LEN]
    db = request.app.state.db
    if not label:
        delete_session_label(db, session_id)
        return JSONResponse(
            status_code=200, content={"session_id": session_id, "label": None}
        )
    set_session_label(db, session_id, label)
    return JSONResponse(
        status_code=200, content={"session_id": session_id, "label": label}
    )


def _session_tools(db: Any, session_id: str) -> list[dict]:
    """Per-session tool breakdown (tool / count / failures), top by count.

    Tool spans carry ``tool_name``; the count is per distinct tool and the
    error count is the number of those spans with a failing status. The
    ``StorageBackend`` protocol doesn't cover this query, so we read
    ``db.conn`` directly (consistent with CostEngine / cmd_status).
    """
    if not hasattr(db, "conn"):
        return []
    rows = db.conn.execute(
        "SELECT tool_name, COUNT(*) AS call_count, "
        "SUM(CASE WHEN status_code = 'error' THEN 1 ELSE 0 END) AS error_count "
        "FROM spans WHERE session_id = $1 AND tool_name IS NOT NULL "
        "GROUP BY tool_name ORDER BY call_count DESC LIMIT $2",
        [session_id, MAX_SESSION_TOOLS],
    ).fetchall()
    return [
        {"tool_name": r[0], "count": int(r[1]), "error_count": int(r[2] or 0)}
        for r in rows
    ]


def _session_conversation_count(db: Any, session_id: str) -> int:
    """COUNT(DISTINCT conversation_id) across the session's spans."""
    if not hasattr(db, "conn"):
        return 0
    row = db.conn.execute(
        "SELECT COUNT(DISTINCT conversation_id) FROM spans "
        "WHERE session_id = $1 AND conversation_id IS NOT NULL",
        [session_id],
    ).fetchone()
    return int(row[0]) if row and row[0] is not None else 0


def _session_traces(db: Any, session_id: str) -> list[dict]:
    """Traces belonging to the session, newest first (reuses trace shape).

    Rolls spans up by ``trace_id`` for the session — the same shape the
    ``/traces`` listing returns, so the frontend can drill into the existing
    waterfall via ``#/traces/<trace_id>``. Reads ``db.conn`` directly because
    ``TraceFilters`` has no session_id dimension.
    """
    if not hasattr(db, "conn"):
        return []
    rows = db.conn.execute(
        "SELECT trace_id, "
        "FIRST(name ORDER BY start_time) AS name, "
        "MIN(start_time) AS start_time, "
        "SUM(duration_ms) AS duration_ms, "
        "SUM(cost_usd) AS cost_usd, "
        "CASE WHEN SUM(CASE WHEN status_code='error' THEN 1 ELSE 0 END) > 0 THEN 'error' "
        "     ELSE 'ok' END AS status_code, "
        "COUNT(*) AS span_count "
        "FROM spans WHERE session_id = $1 "
        "GROUP BY trace_id ORDER BY start_time DESC LIMIT $2",
        [session_id, SESSION_TRACE_LIMIT],
    ).fetchall()
    return [
        {
            "trace_id": r[0],
            "name": r[1],
            "start_time": r[2].isoformat() if r[2] else None,
            "duration_ms": float(r[3]) if r[3] is not None else 0.0,
            "cost_usd": float(r[4]) if r[4] is not None else 0.0,
            "status_code": r[5],
            "span_count": int(r[6]),
        }
        for r in rows
    ]


def _session_drift(db: Any, agent_id: str) -> dict | None:
    """Latest drift baseline summary for the session's agent, else None."""
    baseline = db.get_baseline(agent_id)
    if baseline is None:
        return None
    return {
        "sessions_sampled": baseline.sessions_sampled,
        "computed_at": (
            baseline.computed_at.isoformat() if baseline.computed_at else None
        ),
        "avg_input_tokens": baseline.avg_input_tokens,
        "stddev_input_tokens": baseline.stddev_input_tokens,
        "avg_output_tokens": baseline.avg_output_tokens,
        "stddev_output_tokens": baseline.stddev_output_tokens,
        "avg_session_duration_s": baseline.avg_session_duration_s,
        "stddev_session_duration": baseline.stddev_session_duration,
        "avg_tool_call_count": baseline.avg_tool_call_count,
        "stddev_tool_call_count": baseline.stddev_tool_call_count,
    }


def _session_turn_count(db: Any, session_id: str) -> int:
    """Number of ``gen_ai.llm.call`` spans for the session (one per turn/LLM
    call). Honest descriptive count — not a routing-quality measure."""
    if not hasattr(db, "conn"):
        return 0
    row = db.conn.execute(
        "SELECT COUNT(*) FROM spans WHERE session_id = $1 AND name = $2",
        [session_id, GenAIAttributes.SPAN_LLM_CALL],
    ).fetchone()
    return int(row[0]) if row and row[0] is not None else 0


def _session_model_mix(db: Any, session_id: str) -> list[dict]:
    """Per-model rollup over the session's LLM calls, ordered by call count.

    Descriptive only — shows *how this session split across models*. Makes no
    claim that any model could be substituted for another. Reads ``db.conn``
    directly (the StorageBackend protocol doesn't cover this query).
    """
    if not hasattr(db, "conn"):
        return []
    rows = db.conn.execute(
        "SELECT model, COUNT(*) AS calls, "
        "SUM(COALESCE(input_tokens, 0)) AS input_tokens, "
        "SUM(COALESCE(output_tokens, 0)) AS output_tokens, "
        "SUM(COALESCE(cache_tokens, 0)) AS cache_tokens, "
        "SUM(COALESCE(cost_usd, 0)) AS cost_usd "
        "FROM spans WHERE session_id = $1 AND name = $2 AND model IS NOT NULL "
        "GROUP BY model ORDER BY calls DESC, model ASC",
        [session_id, GenAIAttributes.SPAN_LLM_CALL],
    ).fetchall()
    return [
        {
            "model": r[0],
            "calls": int(r[1]),
            "input_tokens": int(r[2] or 0),
            "output_tokens": int(r[3] or 0),
            "cache_tokens": int(r[4] or 0),
            "cost_usd": float(r[5]) if r[5] is not None else 0.0,
        }
        for r in rows
    ]


def _session_subagents(db: Any, session_id: str) -> dict:
    """Per-subagent (Task-tool) cost/token breakdown for the session.

    Groups the session's spans by ``sub_agent_id`` and tags each subagent with
    the same structural right-sizing flags the ``subagent`` optimize analyzer
    uses (over_powered / over_provisioned) — imported from there so the
    heuristic has a single source of truth. Returns an empty breakdown for
    sessions with no subagent spans (SDK/live sessions, or history ingested
    before the column existed — re-run backfill with ``--reingest``).
    """
    if not hasattr(db, "conn"):
        return {"rows": [], "total": 0, "cost_usd": 0.0, "tokens": 0, "flagged": 0}
    from tokenjam.core.optimize.analyzers.subagent_rightsizing import _flags_for

    rows = db.conn.execute(
        "SELECT sub_agent_id, "
        "arg_max(model, COALESCE(input_tokens, 0) + COALESCE(output_tokens, 0)) AS model, "
        "COUNT(*) FILTER (WHERE name = $2) AS llm_calls, "
        "COUNT(*) FILTER (WHERE tool_name IS NOT NULL) AS tool_calls, "
        "SUM(COALESCE(input_tokens, 0)) AS input_tokens, "
        "SUM(COALESCE(output_tokens, 0)) AS output_tokens, "
        "SUM(COALESCE(cache_tokens, 0)) AS cache_tokens, "
        "SUM(COALESCE(cache_write_tokens, 0)) AS cache_write_tokens, "
        "SUM(COALESCE(cost_usd, 0)) AS cost_usd "
        "FROM spans WHERE session_id = $1 AND sub_agent_id IS NOT NULL "
        "GROUP BY sub_agent_id ORDER BY cost_usd DESC",
        [session_id, GenAIAttributes.SPAN_LLM_CALL],
    ).fetchall()

    out: list[dict] = []
    for r in rows:
        in_t, out_t = int(r[4] or 0), int(r[5] or 0)
        cache_t, cw_t = int(r[6] or 0), int(r[7] or 0)
        cost = float(r[8] or 0.0)
        model = str(r[1] or "unknown")
        tool_calls = int(r[3] or 0)
        out.append({
            "sub_agent_id": str(r[0]),
            "model": model,
            "llm_calls": int(r[2] or 0),
            "tool_calls": tool_calls,
            "input_tokens": in_t,
            "output_tokens": out_t,
            "cache_tokens": cache_t,
            "cache_write_tokens": cw_t,
            "cost_usd": cost,
            "flags": _flags_for(
                model=model, output_tokens=out_t, tool_calls=tool_calls,
                input_tokens=in_t, cache_tokens=cache_t, cost_usd=cost,
            ),
        })
    return {
        "rows": out,
        "total": len(out),
        "cost_usd": sum(x["cost_usd"] for x in out),
        "tokens": sum(
            x["input_tokens"] + x["output_tokens"]
            + x["cache_tokens"] + x["cache_write_tokens"]
            for x in out
        ),
        "flagged": sum(1 for x in out if x["flags"]),
    }


def _session_context_series(db: Any, session_id: str) -> list[dict]:
    """Time-ordered per-turn context (input) tokens for the session's LLM calls.

    Each point keeps that turn's real measured ``input_tokens`` — the size of
    the context the model saw on that call ("context utilized" curve). When the
    session has more than ``MAX_CONTEXT_POINTS`` LLM calls, the series is
    downsampled by even index buckets (every Nth row, N = ceil(count / max))
    so the payload stays bounded; the first and last turns are always kept.
    """
    if not hasattr(db, "conn"):
        return []
    rows = db.conn.execute(
        "SELECT start_time, "
        "COALESCE(input_tokens, 0) AS input_tokens, "
        "COALESCE(cache_tokens, 0) AS cache_tokens, "
        "COALESCE(output_tokens, 0) AS output_tokens "
        "FROM spans WHERE session_id = $1 AND name = $2 "
        "ORDER BY start_time ASC",
        [session_id, GenAIAttributes.SPAN_LLM_CALL],
    ).fetchall()

    total = len(rows)
    if total > MAX_CONTEXT_POINTS:
        # Downsample by even index buckets, preserving first + last points.
        step = -(-total // MAX_CONTEXT_POINTS)  # ceil(total / MAX_CONTEXT_POINTS)
        kept = [rows[i] for i in range(0, total, step)]
        if kept[-1] is not rows[-1]:
            kept.append(rows[-1])
        rows = kept

    return [
        {
            "t": r[0].isoformat() if r[0] else None,
            "input_tokens": int(r[1] or 0),
            "cache_tokens": int(r[2] or 0),
            "output_tokens": int(r[3] or 0),
        }
        for r in rows
    ]


def _session_framing(db: Any, config: Any, session: SessionRecord) -> dict:
    """Plan-tier framing block for the session detail view (#191).

    SessionDetailView routes its Overview / subagents / traces cost cells
    through ``fmtFramedDollar(..., framing)``; without this block the UI silently
    falls back to raw dollars — the exact honesty issue #191 closed on every
    other dollar-bearing read route. Reuses ``compute_framing`` (single source of
    truth, ``core/framing.py``) with a window-INDEPENDENT plan mix scoped to this
    session's agent (``plan_determination_mix``, as ``/status`` and ``/traces``
    do). Window totals are this session's own tokens/cost so the subscription
    token-share ("% of cycle") math has a denominator.
    """
    conn = getattr(db, "conn", None)
    mix = plan_determination_mix(conn, session.agent_id) if conn is not None else {}
    total_tokens = (
        session.input_tokens + session.output_tokens
        + session.cache_tokens + session.cache_write_tokens
    )
    total_cost = (
        float(session.total_cost_usd)
        if session.total_cost_usd is not None else 0.0
    )
    return compute_framing(
        config,
        WindowSummary(
            total_cost_usd=total_cost,
            total_tokens=total_tokens,
            sessions=sum(mix.values()),
            plan_tier_mix=mix,
        ),
    ).to_dict()


@router.get(
    "/sessions/{session_id}",
    response_model=None,
    dependencies=[Depends(require_api_key)],
)
async def get_session_detail(request: Request, session_id: str):
    """Per-session rollup + tools + alerts + drift + traces.

    Returns 404 (as a JSONResponse) when the session id is unknown — hence
    ``response_model=None`` (FastAPI can't model a ``dict | JSONResponse``
    union).
    """
    db = request.app.state.db
    session = db.get_session(session_id)
    if session is None:
        return JSONResponse(
            status_code=404,
            content={"error": f"Session {session_id} not found"},
        )

    # Active (unacknowledged, unsuppressed) alerts attributed to this session.
    all_alerts = db.get_alerts(
        AlertFilters(agent_id=session.agent_id, limit=SESSION_ALERT_LIMIT)
    )
    session_alerts = [a for a in all_alerts if a.session_id == session_id]
    active_alerts = [
        a for a in session_alerts if not a.acknowledged and not a.suppressed
    ]

    tools = _session_tools(db, session_id)
    conversation_count = _session_conversation_count(db, session_id)
    drift = _session_drift(db, session.agent_id)
    traces = _session_traces(db, session_id)
    turn_count = _session_turn_count(db, session_id)
    model_mix = _session_model_mix(db, session_id)
    context_series = _session_context_series(db, session_id)
    subagents = _session_subagents(db, session_id)
    config = getattr(request.app.state, "config", None)
    framing = _session_framing(db, config, session)

    # CC session liveness: a fresh transcript mtime rescues a live session whose
    # backfilled spans have gone stale (see _transcript_aware_status).
    override = getattr(request.app.state, "claude_projects_root", None)
    projects_root = resolve_projects_root(override)
    status = _transcript_aware_status(session, projects_root)

    return {
        "session": {
            "session_id": session.session_id,
            "agent_id": session.agent_id,
            "label": session.service_instance_id,
            "namespace": session.service_namespace,
            "run_id": session.run_id,
            "parent_session_id": session.parent_session_id,
            "status": status,
            "plan_tier": session.plan_tier,
            "pricing_mode": session.pricing_mode,
            "started_at": (
                session.started_at.isoformat() if session.started_at else None
            ),
            "last_span_time": (
                session.ended_at.isoformat() if session.ended_at else None
            ),
            "duration_seconds": session.duration_seconds,
            "total_cost_usd": (
                float(session.total_cost_usd)
                if session.total_cost_usd is not None else 0.0
            ),
            "input_tokens": session.input_tokens,
            "output_tokens": session.output_tokens,
            "cache_tokens": session.cache_tokens,
            "tool_call_count": session.tool_call_count,
            "error_count": session.error_count,
            "conversation_count": conversation_count,
            "active_alerts": len(active_alerts),
        },
        "turn_count": turn_count,
        "model_mix": model_mix,
        "subagents": subagents,
        "context_series": context_series,
        "tools": tools,
        "alerts": [
            {
                "fired_at": a.fired_at.isoformat() if a.fired_at else None,
                "severity": a.severity.value,
                "type": a.type.value,
                "title": a.title,
            }
            for a in session_alerts
        ],
        "drift": drift,
        "traces": traces,
        "framing": framing,
    }


# Stable user-facing reason when a session has no on-disk CC transcript.
_NO_TRANSCRIPT_REASON = (
    "No on-disk transcript for this session "
    "(SDK session, or transcript pruned)."
)


@router.get(
    "/sessions/{session_id}/story",
    response_model=None,
    dependencies=[Depends(require_api_key)],
)
async def get_session_story(
    request: Request, session_id: str, subagents: bool = True
):
    """Deterministic step-by-step story from the session's CC JSONL transcript.

    Surfaces the agent's own narration + literal tool calls + ok/error outcomes
    (no LLM, no generation). Found -> ``{"available": true, ...}``. No transcript
    on disk -> ``{"available": false, "reason": ...}`` with HTTP 200 (a normal
    "no data" state for SDK sessions, not an error).

    When the session spawned subagents (Claude Code ``Task``/``Agent`` steps),
    each such step carries a recursive ``subagent`` object (same step schema)
    so the Story is the COMPLETE nested log of the session and everything it
    spawned. Pass ``?subagents=false`` for the cheaper flat single-session story.

    The projects root resolves from ``app.state.claude_projects_root`` (tests),
    then the ``TJ_CLAUDE_PROJECTS_ROOT`` env var, then ``~/.claude/projects``.
    """
    override = getattr(request.app.state, "claude_projects_root", None)
    projects_root = resolve_projects_root(override)

    story = build_session_story(
        session_id, projects_root=projects_root, include_subagents=subagents
    )
    if story is not None:
        # Live transcript present -> unchanged behavior (byte-identical to before).
        return {"available": True, **story}

    # Transcript gone (pruned). Fall back to a method snapshot persisted at
    # session close (M1), so a killed agent's method survives the prune.
    snapshot = load_session_method(request.app.state.db, session_id)
    if snapshot and snapshot.get("story"):
        return {"available": True, "from_snapshot": True, **snapshot["story"]}

    return {"available": False, "reason": _NO_TRANSCRIPT_REASON}


def _join_delegation_costs(
    spine: list[dict[str, Any]], by_id: dict[str, dict[str, Any]]
) -> None:
    """Join span-derived cost/tokens/flags/status onto each delegation in-place.

    Walks the spine recursively; for every delegation, matches its ``agent_id``
    against the per-subagent rollup (keyed by ``sub_agent_id``) and writes
    ``cost_usd`` / ``tokens`` / ``flags``, plus a ``status`` of ``"capped"`` (a
    recursion guard tripped) or ``"ended"``. Leaves the cost fields ``None`` when
    no matching span row exists (live/SDK sessions carry no subagent identity).
    """
    for move in spine:
        for deleg in move.get("delegations") or []:
            row = by_id.get(deleg.get("agent_id") or "")
            if row is not None:
                deleg["cost_usd"] = row["cost_usd"]
                deleg["tokens"] = (
                    row["input_tokens"] + row["output_tokens"]
                    + row["cache_tokens"] + row["cache_write_tokens"]
                )
                deleg["flags"] = row["flags"]
            else:
                deleg["cost_usd"] = None
                deleg["tokens"] = None
                deleg["flags"] = []
            deleg["status"] = "capped" if deleg.get("capped") else "ended"
            _join_delegation_costs(deleg.get("spine") or [], by_id)


def _flatten_delegation_agents(
    spine: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Preorder-flatten every delegation in the spine into rail-agent summaries.

    Each delegation (after ``_join_delegation_costs``) becomes one entry; deeper
    delegations follow their parent (depth-first) so the rail reads top-down as
    the tree was spawned.
    """
    agents: list[dict[str, Any]] = []
    for move in spine:
        for deleg in move.get("delegations") or []:
            agents.append({
                "name": deleg.get("name"),
                "agent_id": deleg.get("agent_id"),
                "depth": deleg.get("depth"),
                "provenance": "in_session_subagent",
                "status": deleg.get("status"),
                "cost_usd": deleg.get("cost_usd"),
                "tokens": deleg.get("tokens"),
                "capture_completeness": (
                    "session-level" if deleg.get("capped") else "full"
                ),
            })
            agents.extend(_flatten_delegation_agents(deleg.get("spine") or []))
    return agents


def _spine_counts(spine: list[dict[str, Any]]) -> dict[str, int]:
    """Totals over the whole (nested) spine: moves / delegations / dead-ends /
    verifies — the header stats the Approach card surfaces."""
    counts = {"moves": 0, "delegations": 0, "dead_ends": 0, "verifies": 0}

    def walk(moves: list[dict[str, Any]]) -> None:
        for move in moves:
            counts["moves"] += 1
            kind = move.get("kind")
            if kind == "dead_end":
                counts["dead_ends"] += 1
            elif kind == "verify":
                counts["verifies"] += 1
            for deleg in move.get("delegations") or []:
                counts["delegations"] += 1
                walk(deleg.get("spine") or [])

    walk(spine)
    return counts


def _main_agent_summary(
    story: dict[str, Any], session: SessionRecord | None
) -> dict[str, Any]:
    """The depth-0 rail entry for the main agent (the session itself)."""
    if session is not None:
        status = "active" if session.effective_status == "active" else "completed"
        tokens: int | None = (
            session.input_tokens + session.output_tokens
            + session.cache_tokens + session.cache_write_tokens
        )
        cost: float | None = (
            float(session.total_cost_usd)
            if session.total_cost_usd is not None else None
        )
        name = story.get("name") or session.agent_id
        agent_id = session.agent_id
    else:
        status, tokens, cost = None, None, None
        name = story.get("name")
        agent_id = story.get("agent_id")
    return {
        "name": name,
        "agent_id": agent_id,
        "depth": 0,
        "provenance": "main_session",
        "status": status,
        "cost_usd": cost,
        "tokens": tokens,
        "capture_completeness": "full",
    }


def _session_meta(session: SessionRecord | None) -> dict[str, Any]:
    """Session-level ``{cost_usd, tokens}`` for the Approach header stats."""
    if session is None:
        return {"cost_usd": None, "tokens": None}
    return {
        "cost_usd": (
            float(session.total_cost_usd)
            if session.total_cost_usd is not None else None
        ),
        "tokens": (
            session.input_tokens + session.output_tokens
            + session.cache_tokens + session.cache_write_tokens
        ),
    }


def _child_method_spine(
    db: Any, child: SessionRecord, projects_root: Any
) -> list[dict[str, Any]]:
    """The cross-terminal child's OWN method spine, or ``[]`` when none.

    A cross-terminal child is a separate session, so its method comes from its
    own live transcript (``build_session_story``) or — once the transcript is
    pruned — the M1 snapshot persisted at its close (``load_session_method``),
    exactly the read-through ``/approach`` uses for the parent. Returns the folded
    spine when a story is recoverable, else ``[]`` (the honest "session-level
    only" signal — no method to splice).
    """
    story = build_session_story(
        child.session_id, projects_root=projects_root, include_subagents=True
    )
    if story is None:
        snapshot = load_session_method(db, child.session_id)
        if snapshot and snapshot.get("story"):
            story = snapshot["story"]
    if not story:
        return []
    return build_method_spine(story)


def _cross_terminal_children(
    db: Any, session_id: str, projects_root: Any
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Cross-terminal child sessions spliced into the Approach view (M2b).

    In-session subagents (Task sidechains) already live in the spine + rail; a
    cross-terminal child is a *separate* ``SessionRecord`` a harness spawned in
    another terminal, linked only by a declared ``parent_session_id``. For each
    such child this returns:

    * a rail-agent summary — ``{name, agent_id (the child session_id), depth 1,
      provenance "cross_terminal_child", status, cost_usd, tokens,
      capture_completeness}``. ``capture_completeness`` is ``"full"`` when the
      child's own method is recoverable (live transcript or M1 snapshot) else
      ``"session_level"`` (we have the cost/identity but not the *how*).
    * — only when its method IS available — a ``cross_terminal`` spine entry
      ``{name, agent_id, provenance, spine}`` so its method nests like an
      in-session delegation.

    Returns ``([], [])`` when the session launched no cross-terminal children
    (the common case), so the payload's ``cross_terminal`` is ``[]`` and no extra
    rail agents are added.
    """
    children = _child_sessions(db, session_id)
    agents: list[dict[str, Any]] = []
    cross_terminal: list[dict[str, Any]] = []
    for child in children:
        spine = _child_method_spine(db, child, projects_root)
        completeness = "full" if spine else "session_level"
        status = "active" if child.effective_status == "active" else "ended"
        name = (
            child.service_instance_id or child.agent_id or child.session_id[:8]
        )
        tokens = (
            child.input_tokens + child.output_tokens
            + child.cache_tokens + child.cache_write_tokens
        )
        cost = (
            float(child.total_cost_usd)
            if child.total_cost_usd is not None else None
        )
        agents.append({
            "name": name,
            "agent_id": child.session_id,
            "depth": 1,
            "provenance": "cross_terminal_child",
            "status": status,
            "cost_usd": cost,
            "tokens": tokens,
            "capture_completeness": completeness,
        })
        if spine:
            cross_terminal.append({
                "name": name,
                "agent_id": child.session_id,
                "provenance": "cross_terminal_child",
                "spine": spine,
            })
    return agents, cross_terminal


def _add_cross_terminal_counts(
    counts: dict[str, int], cross_terminal: list[dict[str, Any]]
) -> None:
    """Fold each spliced cross-terminal child's spine into the header ``counts``.

    Each child counts as one delegation (the splice edge) plus the moves /
    dead-ends / verifies / nested delegations of its own spine, so the Approach
    header totals stay honest once a child's method is nested in. No-op when
    nothing was spliced (the common case)."""
    for child in cross_terminal:
        child_counts = _spine_counts(child.get("spine") or [])
        counts["moves"] += child_counts["moves"]
        counts["dead_ends"] += child_counts["dead_ends"]
        counts["verifies"] += child_counts["verifies"]
        counts["delegations"] += child_counts["delegations"] + 1


def _approach_payload(
    story: dict[str, Any],
    db: Any,
    session_id: str,
    *,
    from_snapshot: bool = False,
    projects_root: Any = None,
) -> dict[str, Any]:
    """Render-ready Approach body: the recursive method spine joined to the
    span-derived per-subagent cost/status, plus the delegation-tree rail summary
    (``agents``), the header ``counts``, session ``meta``, and the cross-terminal
    children (``cross_terminal``, M2b) spliced in.

    The UI reads everything it draws straight from this payload (repo rule: the
    UI never aggregates) — the rail, the header stats, and the per-delegation
    cost chips are all assembled here, not in the browser.
    """
    spine = build_method_spine(story)

    subs = _session_subagents(db, session_id)
    by_id = {r["sub_agent_id"]: r for r in subs.get("rows", [])}
    _join_delegation_costs(spine, by_id)

    session = db.get_session(session_id)
    main = _main_agent_summary(story, session)
    agents = [main, *_flatten_delegation_agents(spine)]
    counts = _spine_counts(spine)

    # M2b: splice cross-terminal child sessions (separate SessionRecords linked by
    # parent_session_id) into the rail + spine, honestly marked by completeness.
    ct_agents, cross_terminal = _cross_terminal_children(
        db, session_id, projects_root
    )
    agents.extend(ct_agents)
    _add_cross_terminal_counts(counts, cross_terminal)

    payload: dict[str, Any] = {
        "available": True,
        "name": main["name"],
        "task": story.get("task") or "",
        "outcome": story.get("outcome") or "",
        "spine": spine,
        "agents": agents,
        "cross_terminal": cross_terminal,
        "counts": counts,
        "meta": _session_meta(session),
    }
    if from_snapshot:
        payload["from_snapshot"] = True
    return payload


@router.get(
    "/sessions/{session_id}/approach",
    response_model=None,
    dependencies=[Depends(require_api_key)],
)
async def get_session_approach(request: Request, session_id: str):
    """Deterministic **method spine** for a session — the *how* of the work.

    Folds the session's reconstructed Story into an ordered list of intent-tagged
    moves (``core/method_spine.build_method_spine``): ``delegate`` /
    ``dead_end`` / ``verify`` / ``act``, recursively for subagents. Honesty-bounded
    (Critical Rule 14): only structurally-determinable intent is emitted — richer
    labels are the opt-in distill layer, not this route.

    Found -> ``{"available": true, "name", "task", "outcome", "spine",
    "agents", "counts", "meta"}``. ``spine`` carries the recursive moves (each
    ``delegate`` move's ``delegations`` joined to the span-derived
    cost/tokens/flags/status); ``agents`` is the preorder delegation-tree rail
    (main agent first); ``counts`` rolls up moves/delegations/dead-ends/verifies
    over the whole tree; ``meta`` is the session ``{cost_usd, tokens}``. When the
    live transcript is gone, falls back to the method snapshot persisted at
    session close (M1), marked ``"from_snapshot": true``. Neither ->
    ``{"available": false, "reason": ...}`` with HTTP 200 (same contract as
    ``/story``).
    """
    override = getattr(request.app.state, "claude_projects_root", None)
    projects_root = resolve_projects_root(override)
    db = request.app.state.db

    story = build_session_story(
        session_id, projects_root=projects_root, include_subagents=True
    )
    if story is not None:
        return _approach_payload(
            story, db, session_id, projects_root=projects_root
        )

    # Transcript pruned -> read-through the persisted snapshot's story slice (M1).
    snapshot = load_session_method(db, session_id)
    if snapshot and snapshot.get("story"):
        return _approach_payload(
            snapshot["story"], db, session_id,
            from_snapshot=True, projects_root=projects_root,
        )

    return {"available": False, "reason": _NO_TRANSCRIPT_REASON}


def _parse_iso(ts: Any) -> datetime | None:
    """Parse a transcript ISO-8601 timestamp to a tz-aware datetime, or None."""
    if not isinstance(ts, str) or not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        return None


def _coerce_utc(dt: datetime | None) -> datetime | None:
    """Coerce a datetime to tz-aware UTC (naive -> assume UTC), or pass None.

    The Map board mixes transcript timestamps (parsed tz-aware via
    ``_parse_iso``) with DB span timestamps (DuckDB hands back naive datetimes).
    Coercing both to UTC-aware lets them be compared/subtracted on one axis
    without a naive-vs-aware ``TypeError``.
    """
    if dt is None:
        return None
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)


def _bucket_tokens_by_ask(
    db: Any, session_id: str, asks: list[dict]
) -> tuple[dict[int, int], dict[int, float]]:
    """Sum each ask's tokens/cost from LLM-call spans, bucketed by start_time.

    Each ask carries a start ``ts`` (the exchange boundary). A span belongs to
    the latest ask whose ts precedes it (spans before the first ask fall to it).
    Returns ``({ask_n: tokens}, {ask_n: cost_usd})``; empty when timestamps or
    spans are unavailable.
    """
    tokens: dict[int, int] = {}
    costs: dict[int, float] = {}
    if not hasattr(db, "conn") or not asks:
        return tokens, costs

    bounds = sorted(
        ((a["n"], dt) for a in asks if (dt := _parse_iso(a.get("ts"))) is not None),
        key=lambda b: b[1],
    )
    if not bounds:
        return tokens, costs
    bound_times = [b[1] for b in bounds]
    bound_ns = [b[0] for b in bounds]

    rows = db.conn.execute(
        "SELECT start_time, "
        "COALESCE(input_tokens, 0) + COALESCE(output_tokens, 0) "
        "+ COALESCE(cache_tokens, 0) + COALESCE(cache_write_tokens, 0) AS toks, "
        "COALESCE(cost_usd, 0) AS cost "
        "FROM spans WHERE session_id = $1 AND name = $2 ORDER BY start_time",
        [session_id, GenAIAttributes.SPAN_LLM_CALL],
    ).fetchall()

    for start_time, toks, cost in rows:
        if start_time is None:
            continue
        idx = bisect.bisect_right(bound_times, start_time) - 1
        if idx < 0:
            idx = 0
        n = bound_ns[idx]
        tokens[n] = tokens.get(n, 0) + int(toks or 0)
        costs[n] = costs.get(n, 0.0) + float(cost or 0.0)
    return tokens, costs


def _run_card(
    run_id: str, source: str, members: list[SessionRecord], self_id: str
) -> dict[str, Any]:
    """Render-ready run card from a run's member sessions.

    ``source`` is ``"tagged"`` (the viewed session carries ``tokenjam.run_id``)
    or ``"inferred"`` (the run id was scraped from the launcher's transcript and
    confirmed against real run data). Totals come straight off the session
    records — no extra DB round-trips per member.
    """
    sessions = [
        {
            "session_id": s.session_id,
            "label": s.service_instance_id,
            "status": s.effective_status,
            "is_self": s.session_id == self_id,
        }
        for s in members
    ]
    starts = [s.started_at for s in members if s.started_at]
    ends = [s.ended_at or s.started_at for s in members if (s.ended_at or s.started_at)]
    return {
        "run_id": run_id,
        "source": source,
        "session_count": len(members),
        "total_cost_usd": sum(
            float(s.total_cost_usd) for s in members if s.total_cost_usd is not None
        ),
        "tool_call_count": sum(s.tool_call_count for s in members),
        "started_at": min(starts).isoformat() if starts else None,
        "last_activity": max(ends).isoformat() if ends else None,
        "sessions": sessions,
    }


def _launched_run(
    db: Any, session: SessionRecord, projects_root: Any
) -> dict[str, Any] | None:
    """Best-effort run this session launched or belongs to, for the Map card.

    Tries, in order: the session's own ``run_id`` (tagged), then any run ids its
    transcript announced (inferred). A candidate becomes a card only when its
    run has at least one OTHER member session — a run of just this session isn't
    a fan-out worth surfacing. Returns ``None`` when nothing qualifies.
    """
    candidates: list[tuple[str, str]] = []
    if session.run_id:
        candidates.append((session.run_id, "tagged"))
    for rid in scan_transcript_run_ids(session.session_id, projects_root):
        if rid != session.run_id:
            candidates.append((rid, "inferred"))

    for run_id, source in candidates:
        members = _run_sessions(db, run_id)
        others = [s for s in members if s.session_id != session.session_id]
        if not others:
            continue
        return _run_card(run_id, source, members, session.session_id)
    return None


def _transcript_aware_status(session: SessionRecord, projects_root: Any) -> str:
    """Session status, rescued from a stale span signal by transcript activity.

    Claude Code spans are backfilled periodically, so a live CC session can read
    ``idle``/``stale`` once its last backfilled span ages past the threshold —
    even while its transcript is still being written. When the deterministic
    status is idle/stale but the transcript was touched within the active
    window, report ``active``. Only stats the transcript when needed (the base
    status is idle/stale), so active and non-CC sessions cost nothing extra.
    """
    base = session.effective_status
    if base not in ("idle", "stale"):
        return base
    mtime = session_transcript_mtime(session.session_id, projects_root)
    return session.status_with_transcript_mtime(mtime)


@router.get(
    "/sessions/{session_id}/workmap",
    response_model=None,
    dependencies=[Depends(require_api_key)],
)
async def get_session_workmap(request: Request, session_id: str):
    """Graphical "work map" of a session as a list of *asks* (exchanges).

    A session isn't one task — it's a sequence of human asks fired into the same
    terminal until the context window fills. This returns each ask (newest first)
    with its activity rollup, its bucketed token/cost total, and the subagent
    subtree it spawned (joined to the span-derived per-subagent cost/flags). No
    LLM, no interpretation — it reports what happened so a human can judge it. No
    transcript on disk -> ``{"available": false, ...}`` with HTTP 200 (the same
    contract as ``/story``).
    """
    override = getattr(request.app.state, "claude_projects_root", None)
    projects_root = resolve_projects_root(override)

    db = request.app.state.db

    asks_payload = build_session_asks(
        session_id, projects_root=projects_root, include_subagents=True
    )
    from_snapshot = False
    if asks_payload is None:
        # Transcript gone (pruned). Fall back to the method snapshot persisted at
        # session close (M1); the span-derived rollups below still come from the DB.
        snapshot = load_session_method(db, session_id)
        if snapshot and snapshot.get("asks"):
            asks_payload = snapshot["asks"]
            from_snapshot = True
        else:
            return {"available": False, "reason": _NO_TRANSCRIPT_REASON}

    subagents = _session_subagents(db, session_id)
    ask_tokens, ask_costs = _bucket_tokens_by_ask(
        db, session_id, asks_payload["asks"]
    )

    session = db.get_session(session_id)
    session_tokens = (
        session.input_tokens + session.output_tokens
        + session.cache_tokens + session.cache_write_tokens
        if session is not None else None
    )
    session_cost = (
        float(session.total_cost_usd)
        if session and session.total_cost_usd is not None else None
    )

    workmap = build_work_map(
        asks_payload, subagents,
        ask_tokens=ask_tokens, ask_costs=ask_costs,
        session_tokens=session_tokens, session_cost_usd=session_cost,
    )
    result: dict[str, Any] = {"available": True, **workmap}
    if from_snapshot:
        result["from_snapshot"] = True
    if session is not None:
        launched = _launched_run(db, session, projects_root)
        if launched is not None:
            result["launched_run"] = launched
    return result


def _session_map_spans(db: Any, session_id: str) -> list[dict[str, Any]]:
    """The session's LLM-call spans, time-ordered, with the columns the Map's
    context/cost time-series need.

    Mirrors ``_session_context_series`` / ``_bucket_tokens_by_ask``: the context
    and cost curves live on ``gen_ai.llm.call`` spans (tool/event spans carry no
    tokens or cost), so the series is built from those, ordered by start_time.
    """
    if not hasattr(db, "conn"):
        return []
    rows = db.conn.execute(
        "SELECT start_time, "
        "COALESCE(input_tokens, 0), COALESCE(cache_tokens, 0), "
        "COALESCE(cache_write_tokens, 0), COALESCE(output_tokens, 0), "
        "COALESCE(cost_usd, 0.0), model "
        "FROM spans WHERE session_id = $1 AND name = $2 "
        "ORDER BY start_time ASC",
        [session_id, GenAIAttributes.SPAN_LLM_CALL],
    ).fetchall()
    return [
        {
            "start_time": r[0],
            "input_tokens": int(r[1] or 0),
            "cache_tokens": int(r[2] or 0),
            "cache_write_tokens": int(r[3] or 0),
            "output_tokens": int(r[4] or 0),
            "cost_usd": float(r[5] or 0.0),
            "model": r[6] if isinstance(r[6], str) else None,
        }
        for r in rows
    ]


def _session_map_series(
    spans: list[dict[str, Any]],
    t0: datetime | None,
    session: SessionRecord | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Build the context-growth + cost-burn series and the meta block.

    ``context_series`` carries the *running* context size (cumulative
    input+cache+cache_write tokens); ``cost_series`` carries each span's own
    ``cost_usd`` (per-span burn, not cumulative). ``t_s`` is seconds from ``t0``
    (the session start). Spans with no start_time are skipped.
    """
    context_series: list[dict[str, Any]] = []
    cost_series: list[dict[str, Any]] = []
    cumulative_tokens = 0
    total_tokens = 0
    total_cost = 0.0
    model_freq: dict[str, int] = {}

    for span in spans:
        start = _coerce_utc(span["start_time"])
        if start is None:
            continue
        t_s = (start - t0).total_seconds() if t0 is not None else 0.0
        context_tokens = (
            span["input_tokens"] + span["cache_tokens"] + span["cache_write_tokens"]
        )
        cumulative_tokens += context_tokens
        context_series.append({"t_s": t_s, "tokens": cumulative_tokens})
        cost_series.append({"t_s": t_s, "usd": span["cost_usd"]})

        total_tokens += context_tokens + span["output_tokens"]
        total_cost += span["cost_usd"]
        model = span["model"]
        if model:
            model_freq[model] = model_freq.get(model, 0) + 1

    dominant_model = (
        max(model_freq, key=lambda m: model_freq[m]) if model_freq else None
    )
    if context_series:
        duration_s: float | None = context_series[-1]["t_s"]
    elif session is not None:
        duration_s = session.duration_seconds
    else:
        duration_s = None

    started_at = t0.isoformat() if t0 is not None else None
    meta = {
        "started_at": started_at,
        "duration_s": duration_s,
        "total_tokens": total_tokens,
        "total_cost_usd": total_cost,
        "model": dominant_model,
    }
    return context_series, cost_series, meta


def _session_map_axis(
    events: list[dict[str, Any]],
    spans: list[dict[str, Any]],
    session: SessionRecord | None,
) -> tuple[datetime | None, float]:
    """Unified ``(t0, duration_s)`` for the Map board over the UNION of event +
    span timestamps, so every lane shares one clock.

    The board overlays two clocks: the tool/phase *event* lanes carry transcript
    timestamps (``event.ts``), while the context/cost series + subagent windows
    carry *span* timestamps. On a backfilled/resumed session whose span
    ``start_time`` postdates (or otherwise diverges from) its transcript, a
    span-only ``t0`` drives every event's wall-clock offset negative — the UI
    clamps those to 0 and the whole event lane collapses to the left edge while
    the series spread out. Anchoring instead on::

        t0   = min(earliest event ts, earliest span start_time)
        tEnd = max(latest   event ts, latest   span start_time)

    gives both lanes a common origin and span, so they line up. ``duration_s`` is
    ``(tEnd - t0)`` seconds, floored to ``1.0`` (empty / zero-width -> ``1.0``) so
    downstream division is always safe. All datetimes are coerced to tz-aware UTC
    first (``_coerce_utc``) so naive DB spans and tz-aware transcript ts compare
    cleanly. When nothing carries a usable time, falls back to the session start.
    """
    times: list[datetime] = []
    for event in events:
        dt = _parse_iso(event.get("ts"))
        if dt is not None:
            times.append(dt)
    for span in spans:
        dt = _coerce_utc(span.get("start_time"))
        if dt is not None:
            times.append(dt)
    if not times:
        fallback = _coerce_utc(session.started_at) if session else None
        return fallback, 1.0
    t0 = min(times)
    duration = (max(times) - t0).total_seconds()
    return t0, duration if duration > 0 else 1.0


def _subagent_name_index(asks_payload: dict[str, Any] | None) -> dict[str, str]:
    """Map ``sub_agent_id`` -> human name from the story's nested subagent nodes.

    Walks every ask's steps recursively, collecting the ``name`` each subagent
    node carries (keyed by its ``agent_id``). The first name seen for an id
    wins. Empty when there's no story (the caller then falls back to a synthetic
    ``agent-<id[:8]>`` label).
    """
    names: dict[str, str] = {}
    if not asks_payload:
        return names

    def walk(steps: Any) -> None:
        for step in steps or []:
            if not isinstance(step, dict) or "omitted" in step:
                continue
            subs: list[dict[str, Any]] = []
            if isinstance(step.get("subagent"), dict):
                subs.append(step["subagent"])
            for sub in step.get("subagents") or []:
                if isinstance(sub, dict):
                    subs.append(sub)
            for sub in subs:
                aid = str(sub.get("agent_id") or "")
                nm = sub.get("name")
                if aid and nm and aid not in names:
                    names[aid] = nm
                walk(sub.get("steps") or [])

    for ask in asks_payload.get("asks") or []:
        if isinstance(ask, dict):
            walk(ask.get("steps") or [])
    return names


def _session_map_subagent_windows(
    db: Any, session_id: str
) -> dict[str, dict[str, Any]]:
    """Per-subagent time window — ``min(start_time)`` / ``max(end_time)`` grouped
    by ``sub_agent_id`` over the session's spans.

    The window spans the subagent's whole footprint regardless of span kind (LLM
    calls + tool events both carry ``sub_agent_id``). ``end_time`` is nullable on
    older rows, so it coalesces to ``start_time``. Empty for sessions with no
    subagent spans.
    """
    if not hasattr(db, "conn"):
        return {}
    rows = db.conn.execute(
        "SELECT sub_agent_id, MIN(start_time) AS start_ts, "
        "MAX(COALESCE(end_time, start_time)) AS end_ts "
        "FROM spans WHERE session_id = $1 AND sub_agent_id IS NOT NULL "
        "GROUP BY sub_agent_id",
        [session_id],
    ).fetchall()
    return {
        str(r[0]): {"start_ts": r[1], "end_ts": r[2]}
        for r in rows
        if r[0] and r[1] is not None
    }


def _map_window_to_ordinals(
    events: list[dict[str, Any]],
    start_dt: datetime,
    end_dt: datetime,
) -> tuple[int | None, int | None]:
    """Map a subagent's [start_dt, end_dt] window onto main-thread event ordinals.

    When the window overlaps main events, returns the inner edges: the first
    event at/after ``start_dt`` and the last event at/before ``end_dt``. A
    subagent often runs in a *gap* where the main thread emitted no events (it
    was waiting on the subagent), so that inner mapping inverts; in that case it
    *brackets* the gap instead — the event just before the window to the event
    just after — clamping to the first/last ordinal at the session edges. Returns
    ``(None, None)`` only when there are no time-anchored events at all (the UI
    then falls back to ts-only positioning in time mode).
    """
    parsed: list[tuple[datetime, int]] = []
    for event in events:
        dt = _parse_iso(event.get("ts"))
        if dt is not None:
            parsed.append((dt, int(event["ordinal"])))
    if not parsed:
        return None, None
    try:
        inner_start = next((o for dt, o in parsed if dt >= start_dt), None)
        inner_end: int | None = None
        for dt, o in parsed:
            if dt <= end_dt:
                inner_end = o
        # Clean overlap: the window contains at least one main event in order.
        if (
            inner_start is not None and inner_end is not None
            and inner_start <= inner_end
        ):
            return inner_start, inner_end
        # Gap / edge: bracket with the surrounding events, clamped to the axis.
        before = [o for dt, o in parsed if dt <= start_dt]
        after = [o for dt, o in parsed if dt >= end_dt]
        s = before[-1] if before else parsed[0][1]
        e = after[0] if after else parsed[-1][1]
    except TypeError:
        # tz-aware vs naive mismatch — give up on ordinals, keep ts positioning.
        return None, None
    return (s, e) if s <= e else (e, s)


def _session_map_subagents(
    db: Any,
    session_id: str,
    events: list[dict[str, Any]],
    asks_payload: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """In-session subagents as positioned time windows for the Map's sub lane.

    One entry per ``sub_agent_id`` with a usable span window: its name (from the
    story, else ``agent-<id[:8]>``), absolute ts range, the window mapped onto
    main-thread event ordinals (``None`` when it can't map cleanly), and its
    summed tokens/cost (reusing ``_session_subagents``). Sorted by start ts.
    Empty when the session spawned no subagents.
    """
    windows = _session_map_subagent_windows(db, session_id)
    if not windows:
        return []
    by_id = {r["sub_agent_id"]: r for r in _session_subagents(db, session_id)["rows"]}
    names = _subagent_name_index(asks_payload)

    out: list[dict[str, Any]] = []
    for aid, win in windows.items():
        start_dt = win["start_ts"]
        end_dt = win["end_ts"] or start_dt
        row = by_id.get(aid)
        tokens = (
            row["input_tokens"] + row["output_tokens"]
            + row["cache_tokens"] + row["cache_write_tokens"]
        ) if row else 0
        cost = float(row["cost_usd"]) if row else 0.0
        start_ord, end_ord = _map_window_to_ordinals(events, start_dt, end_dt)
        out.append({
            "name": names.get(aid) or f"agent-{aid[:8]}",
            "agent_id": aid,
            "start_ts": start_dt.isoformat() if start_dt else None,
            "end_ts": end_dt.isoformat() if end_dt else None,
            "start_ordinal": start_ord,
            "end_ordinal": end_ord,
            "tokens": tokens,
            "cost_usd": cost,
        })
    out.sort(key=lambda s: s["start_ts"] or "")
    return out


@router.get(
    "/sessions/{session_id}/sessionmap",
    response_model=None,
    dependencies=[Depends(require_api_key)],
)
async def get_session_map(request: Request, session_id: str):
    """Board data for the Map's synchronized swimlanes (lens ①).

    Folds the session's reconstructed Story into a flat tool-event list +
    contiguous phase spans (``core.sessionmap.build_session_map``) and pairs them
    with span-derived context-growth and cost-burn series over a shared
    seconds-from-start axis. The Story comes from ``build_session_asks`` with the
    same persisted-snapshot read-through as ``/workmap`` (marked
    ``from_snapshot``); the series come from ``db.conn``.

    Returns ``{"available": true, "events", "phases", "subagents",
    "context_series", "cost_series", "meta", "from_snapshot"}``. ``subagents`` is
    one entry per in-session subagent with a usable span window (name, ts range,
    the window mapped onto event ordinals, summed tokens/cost); ``[]`` when none.
    When the session has neither a Story nor any spans ->
    ``{"available": false, "reason": ...}`` with HTTP 200 (the same contract as
    ``/story`` and ``/workmap``).
    """
    override = getattr(request.app.state, "claude_projects_root", None)
    projects_root = resolve_projects_root(override)

    db = request.app.state.db

    asks_payload = build_session_asks(
        session_id, projects_root=projects_root, include_subagents=True
    )
    from_snapshot = False
    if asks_payload is None:
        # Transcript gone (pruned). Fall back to the method snapshot persisted at
        # session close (M1); the span-derived series below still come from the DB.
        snapshot = load_session_method(db, session_id)
        if snapshot and snapshot.get("asks"):
            asks_payload = snapshot["asks"]
            from_snapshot = True

    if asks_payload is not None:
        session_map = build_session_map(asks_payload)
        events = session_map["events"]
        phases = session_map["phases"]
    else:
        events, phases = [], []

    session = db.get_session(session_id)
    spans = _session_map_spans(db, session_id)

    # Neither a story nor any spans -> nothing to draw.
    if asks_payload is None and not spans:
        return {"available": False, "reason": _NO_TRANSCRIPT_REASON}

    # One axis basis over the UNION of event (transcript) + span timestamps, so
    # the event/phase lanes and the context/cost series share a single clock. A
    # span-only t0 collapses the event lane to x=0 on backfilled/resumed sessions
    # whose spans postdate the transcript (see _session_map_axis).
    t0, duration_s = _session_map_axis(events, spans, session)

    context_series, cost_series, meta = _session_map_series(spans, t0, session)
    meta["duration_s"] = duration_s
    subagents = _session_map_subagents(db, session_id, events, asks_payload)

    return {
        "available": True,
        "events": events,
        "phases": phases,
        "subagents": subagents,
        "context_series": context_series,
        "cost_series": cost_series,
        "meta": meta,
        "from_snapshot": from_snapshot,
    }


# Min outcome length (chars) before an ask is worth distilling a title for. Short
# outcomes already read as a title; sending them just burns a CLI round-trip.
DISTILL_MIN_OUTCOME_CHARS = 40


@router.get(
    "/sessions/{session_id}/distill",
    response_model=None,
    dependencies=[Depends(require_api_key)],
)
def get_session_distill(
    request: Request, session_id: str, cached_only: bool = False
):
    """On-demand LLM-distilled crisp titles for a session's ask outcomes.

    The Map's deterministic headlines (the first sentence of each ask outcome)
    are faithful but often long. This shells to the user's local ``claude`` CLI
    (via ``core.distill``) to crunch each long outcome into a <=6-word title,
    cached per session. It holds no API key — it reuses the user's CLI.

    ``cached_only=true`` returns titles **only if already cached** and never calls
    ``claude`` (used to auto-apply a distilled session on load, at zero cost).

    Defined as a **sync** ``def`` on purpose: the ``claude`` subprocess can take
    15-40s, so FastAPI runs this in its threadpool and the event loop stays
    free (an ``async def`` would block it for the whole call).

    No transcript on disk -> ``{"available": false, "reason": ...}``. Otherwise
    ``{"available": true, "model": "haiku", "titles": {...}, "candidate_count":
    N, "cached": bool}``. ``candidate_count`` lets the UI tell "nothing long
    enough to distill" (N==0, a success) apart from "claude was unreachable"
    (N>0 but ``titles`` empty), instead of conflating the two.
    """
    override = getattr(request.app.state, "claude_projects_root", None)
    projects_root = resolve_projects_root(override)

    asks_payload = build_session_asks(session_id, projects_root=projects_root)
    if asks_payload is None:
        return {"available": False, "reason": _NO_TRANSCRIPT_REASON}

    # Only distill asks whose outcome is long enough to be worth crunching.
    candidates = [
        {"n": a["n"], "outcome": a["outcome"]}
        for a in (asks_payload.get("asks") or [])
        if (a.get("outcome") or "").strip()
        and len((a["outcome"]).strip()) >= DISTILL_MIN_OUTCOME_CHARS
    ]
    if cached_only:
        titles = peek_cached_titles(session_id, candidates)
    else:
        titles = distill_titles_cached(session_id, candidates)
    return {
        "available": True,
        "model": "haiku",
        "titles": {str(n): t for n, t in titles.items()},
        "candidate_count": len(candidates),
        "cached": cached_only,
    }
