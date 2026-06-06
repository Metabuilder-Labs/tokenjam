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

from typing import Any

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse

from tokenjam.api.deps import require_api_key
from tokenjam.core.db import delete_session_label, set_session_label
from tokenjam.core.models import AlertFilters
from tokenjam.core.transcript import build_session_story, resolve_projects_root
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
    closed = 0
    if instance_id:
        closed += db.close_sessions_by_instance(instance_id)
    if session_id:
        closed += db.close_session_by_id(session_id)

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

    return {
        "session": {
            "session_id": session.session_id,
            "agent_id": session.agent_id,
            "label": session.service_instance_id,
            "namespace": session.service_namespace,
            "run_id": session.run_id,
            "parent_session_id": session.parent_session_id,
            "status": session.effective_status,
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
    if story is None:
        return {"available": False, "reason": _NO_TRANSCRIPT_REASON}

    return {"available": True, **story}
