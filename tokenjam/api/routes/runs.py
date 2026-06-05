"""Run endpoints — cross-session run grouping.

A fan-out harness (e.g. the governor) stamps ``tokenjam.run_id`` on every worker
session it spawns. tj groups those sessions into a *run*. Linkage is declared by
the spawner via OTel resource attributes, never reverse-engineered — Claude Code
OTLP carries no native parent<->child edge.

GET /api/v1/runs/{run_id}
    Run rollup: aggregate totals over the run's member sessions, the member
    sessions with a lighter per-session rollup, and a tree derived from each
    session's ``parent_session_id`` (a flat list under the run when no parent
    edges exist). 404 (JSONResponse) for an unknown run_id — hence
    ``response_model=None`` (FastAPI can't model a ``dict | JSONResponse`` union).

GET /api/v1/runs
    Lightweight index of all runs, newest first, with totals.

Both are read-only and guarded by ``require_api_key`` like other GET endpoints.

Cost framing honesty: a run reports a ``pricing_mode`` aggregated over its member
sessions (``mixed`` when they disagree). The dashboard uses it to render the same
plan-tier-honest framing as the session detail view — implied API value for
subscription sessions, token-only for local, dollars suppressed for unknown.
"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse

from tokenjam.api.deps import require_api_key
from tokenjam.core.db import _row_to_session
from tokenjam.core.models import SessionRecord

router = APIRouter()

# Max runs to surface in the index listing (newest first).
MAX_RUNS = 200


def _run_sessions(db: Any, run_id: str) -> list[SessionRecord]:
    """All sessions belonging to a run, newest activity first.

    Reads ``db.conn`` directly because the ``StorageBackend`` protocol has no
    run dimension (consistent with the session-detail helpers and CostEngine).
    """
    if not hasattr(db, "conn"):
        return []
    cur = db.conn.execute(
        "SELECT * FROM sessions WHERE run_id = $1 "
        "ORDER BY COALESCE(ended_at, started_at) DESC",
        [run_id],
    )
    rows = cur.fetchall()
    cols = [d[0] for d in cur.description]
    return [_row_to_session(r, cols) for r in rows]


def _session_span_tool_counts(db: Any, session_id: str) -> tuple[int, int]:
    """(span_count, tool_call_count) over a session's spans."""
    if not hasattr(db, "conn"):
        return (0, 0)
    row = db.conn.execute(
        "SELECT COUNT(*), "
        "SUM(CASE WHEN tool_name IS NOT NULL THEN 1 ELSE 0 END) "
        "FROM spans WHERE session_id = $1",
        [session_id],
    ).fetchone()
    if not row:
        return (0, 0)
    return (int(row[0] or 0), int(row[1] or 0))


def _session_summary(db: Any, s: SessionRecord) -> dict[str, Any]:
    """Lighter per-session rollup for the run view (one row per session)."""
    span_count, tool_count = _session_span_tool_counts(db, s.session_id)
    return {
        "session_id": s.session_id,
        "agent_id": s.agent_id,
        "label": s.service_instance_id,
        "namespace": s.service_namespace,
        "parent_session_id": s.parent_session_id,
        "status": s.effective_status,
        "plan_tier": s.plan_tier,
        "pricing_mode": s.pricing_mode,
        "started_at": s.started_at.isoformat() if s.started_at else None,
        "last_span_time": s.ended_at.isoformat() if s.ended_at else None,
        "duration_seconds": s.duration_seconds,
        "total_cost_usd": (
            float(s.total_cost_usd) if s.total_cost_usd is not None else 0.0
        ),
        "input_tokens": s.input_tokens,
        "output_tokens": s.output_tokens,
        "cache_tokens": s.cache_tokens,
        "cache_creation_tokens": s.cache_creation_tokens,
        "tool_call_count": s.tool_call_count,
        "error_count": s.error_count,
        "span_count": span_count,
    }


def _aggregate_pricing_mode(sessions: list[SessionRecord]) -> str:
    """One pricing_mode for the whole run, or 'mixed' when members disagree.

    Empty run -> 'unknown'. The dashboard keys honest cost framing off this.
    """
    modes = {s.pricing_mode for s in sessions}
    if not modes:
        return "unknown"
    if len(modes) == 1:
        return next(iter(modes))
    return "mixed"


def _build_tree(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Parent-edge tree over the run's sessions.

    Each node is ``{session_id, children: [...]}``. A session whose
    ``parent_session_id`` points at another member of this run becomes that
    session's child; everything else (no parent, or a parent outside the run)
    is a root. Self-references and cycles are guarded — a node already placed
    under a parent is never also added as a root.
    """
    by_id = {s["session_id"]: s for s in summaries}
    nodes: dict[str, dict[str, Any]] = {
        s["session_id"]: {"session_id": s["session_id"], "children": []}
        for s in summaries
    }
    roots: list[dict[str, Any]] = []
    for s in summaries:
        sid = s["session_id"]
        parent = s.get("parent_session_id")
        if parent and parent != sid and parent in by_id:
            nodes[parent]["children"].append(nodes[sid])
        else:
            roots.append(nodes[sid])
    return roots


@router.get(
    "/runs",
    response_model=None,
    dependencies=[Depends(require_api_key)],
)
async def list_runs(request: Request):
    """Index of runs (newest first) with per-run totals and session counts."""
    db = request.app.state.db
    if not hasattr(db, "conn"):
        return {"runs": []}
    rows = db.conn.execute(
        "SELECT run_id, COUNT(*) AS session_count, "
        "COALESCE(SUM(total_cost_usd), 0.0) AS total_cost_usd, "
        "COALESCE(SUM(input_tokens), 0) AS input_tokens, "
        "COALESCE(SUM(output_tokens), 0) AS output_tokens, "
        "MIN(started_at) AS started_at, "
        "MAX(COALESCE(ended_at, started_at)) AS last_activity "
        "FROM sessions WHERE run_id IS NOT NULL "
        "GROUP BY run_id ORDER BY last_activity DESC LIMIT $1",
        [MAX_RUNS],
    ).fetchall()
    runs = []
    for r in rows:
        sessions = _run_sessions(db, r[0])
        runs.append({
            "run_id": r[0],
            "session_count": int(r[1]),
            "total_cost_usd": float(r[2] or 0.0),
            "input_tokens": int(r[3] or 0),
            "output_tokens": int(r[4] or 0),
            "started_at": r[5].isoformat() if r[5] else None,
            "last_activity": r[6].isoformat() if r[6] else None,
            "pricing_mode": _aggregate_pricing_mode(sessions),
        })
    return {"runs": runs}


@router.get(
    "/runs/{run_id}",
    response_model=None,
    dependencies=[Depends(require_api_key)],
)
async def get_run_detail(request: Request, run_id: str):
    """Run rollup: totals + member sessions + parent-edge tree.

    404 (JSONResponse) when no session carries this run_id.
    """
    db = request.app.state.db
    sessions = _run_sessions(db, run_id)
    if not sessions:
        return JSONResponse(
            status_code=404,
            content={"error": f"Run {run_id} not found"},
        )

    summaries = [_session_summary(db, s) for s in sessions]

    total_cost = sum(s["total_cost_usd"] for s in summaries)
    total_input = sum(s["input_tokens"] for s in summaries)
    total_output = sum(s["output_tokens"] for s in summaries)
    total_cache = sum(s["cache_tokens"] for s in summaries)
    total_cache_creation = sum(s["cache_creation_tokens"] for s in summaries)
    total_tools = sum(s["tool_call_count"] for s in summaries)
    total_errors = sum(s["error_count"] for s in summaries)
    total_spans = sum(s["span_count"] for s in summaries)

    starts = [s.started_at for s in sessions if s.started_at]
    ends = [s.ended_at or s.started_at for s in sessions if (s.ended_at or s.started_at)]
    started_at = min(starts) if starts else None
    last_activity = max(ends) if ends else None
    time_span_seconds = (
        (last_activity - started_at).total_seconds()
        if started_at and last_activity else None
    )

    return {
        "run": {
            "run_id": run_id,
            "session_count": len(sessions),
            "pricing_mode": _aggregate_pricing_mode(sessions),
            "total_cost_usd": total_cost,
            "input_tokens": total_input,
            "output_tokens": total_output,
            "cache_tokens": total_cache,
            "cache_creation_tokens": total_cache_creation,
            "tool_call_count": total_tools,
            "error_count": total_errors,
            "span_count": total_spans,
            "started_at": started_at.isoformat() if started_at else None,
            "last_activity": last_activity.isoformat() if last_activity else None,
            "time_span_seconds": time_span_seconds,
        },
        "sessions": summaries,
        "tree": _build_tree(summaries),
    }
