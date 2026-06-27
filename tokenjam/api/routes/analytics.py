"""
GET /api/v1/analytics — the generalized group-by behind the Analytics explorer
(#210).

One pivot endpoint: (metric × group_by × optional stack_by × filters × window)
-> a time-bucketed grouped series + KPI totals + the plan-tier framing block.
It generalizes the Wave-1 `/cost` group-by (which carried provider + the token
split): the explorer's line / bar / hbar views are all client-side pivots of
the single `rows` shape this returns, so there is one server compute path.

Honesty / framing: the `spend` metric is dollar-bearing, so the response always
carries the framing block (single source, #110) AND a `tokens` value per row,
letting the UI render token-share for subscription/local users without
re-deriving the suppression rules (Critical Rule 14). Token / session / event
metrics are plan-independent counts.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request

from tokenjam.api.deps import require_api_key
from tokenjam.api.routes.cost import _framing_block
from tokenjam.utils.time_parse import parse_since, utcnow

router = APIRouter(dependencies=[Depends(require_api_key)])

# tool_name -> coarse category. Centralized here so "tool_category" is a real
# explorer dimension without a stored column. Unknown tools fall to 'other'.
_TOOL_CATEGORY_CASE = (
    "CASE "
    "WHEN tool_name IN ('Read','Write','Edit','MultiEdit','NotebookEdit') THEN 'file' "
    "WHEN tool_name IN ('Bash','BashOutput','KillShell') THEN 'shell' "
    "WHEN tool_name IN ('Grep','Glob','LS') THEN 'search' "
    "WHEN tool_name IN ('WebFetch','WebSearch') THEN 'web' "
    "WHEN tool_name IN ('Task','Agent') THEN 'agent' "
    "WHEN tool_name IN ('TodoWrite') THEN 'planning' "
    "WHEN tool_name IS NULL THEN NULL "
    "ELSE 'other' END"
)

# Dimension name -> a SAFE SQL expression (never user input; whitelisted so no
# injection is possible — Rule 7). "day" is the time bucket, handled specially.
_DIMENSION_EXPR: dict[str, str] = {
    "agent": "agent_id",
    "provider": "provider",
    "model": "model",
    "tool": "tool_name",
    "tool_category": _TOOL_CATEGORY_CASE,
    "kind": "kind",
    "request_type": "request_type",
    "day": "__bucket__",  # sentinel: resolved to the time-bucket expression
}

# Metric name -> (SQL aggregate, value unit). spend is the only dollar-bearing
# (framing-sensitive) metric.
_METRIC_EXPR: dict[str, tuple[str, str]] = {
    "spend": ("COALESCE(SUM(cost_usd), 0.0)", "usd"),
    "tokens": ("COALESCE(SUM(COALESCE(input_tokens,0)+COALESCE(output_tokens,0)), 0)", "tokens"),
    "events": ("COUNT(*)", "count"),
    "sessions": ("COUNT(DISTINCT session_id)", "count"),
}

_METRIC_LABEL = {
    "spend": "Spend", "tokens": "Tokens", "events": "Events", "sessions": "Sessions",
}

# Equality filters the explorer exposes (param name -> span column).
_FILTER_COLUMN = {
    "agent_id": "agent_id",
    "provider": "provider",
    "model": "model",
    "tool": "tool_name",
}

_TOKENS_EXPR = "COALESCE(SUM(COALESCE(input_tokens,0)+COALESCE(output_tokens,0)), 0)"


def _bucket_unit(since_dt, until_dt) -> str:
    start = since_dt
    end = until_dt or utcnow()
    span_days = ((end - start).total_seconds() / 86400.0) if start is not None else None
    return "hour" if (span_days is not None and span_days <= 2) else "day"


def _bucket_expr(unit: str) -> str:
    # `unit` is a provably-internal literal ('hour'/'day' from _bucket_unit),
    # never user input, so inlining it keeps a single shared param list across
    # the grouped + KPI queries (still no f-string SQL on any user value, Rule 7).
    assert unit in ("hour", "day")
    return f"CAST(epoch(date_trunc('{unit}', start_time AT TIME ZONE 'UTC')) AS BIGINT)"


@router.get("/analytics")
async def get_analytics(
    request: Request,
    metric: str = "spend",
    group_by: str = "model",
    stack_by: str | None = None,
    since: str | None = None,
    until: str | None = None,
    agent_id: str | None = None,
    provider: str | None = None,
    model: str | None = None,
    tool: str | None = None,
) -> dict:
    """Return a grouped series for (metric, group_by, optional stack_by, filters)."""
    if metric not in _METRIC_EXPR:
        raise HTTPException(400, f"Unknown metric: {metric}")
    if group_by not in _DIMENSION_EXPR:
        raise HTTPException(400, f"Unknown group_by: {group_by}")
    if stack_by in ("", "none"):
        stack_by = None
    if stack_by is not None and stack_by not in _DIMENSION_EXPR:
        raise HTTPException(400, f"Unknown stack_by: {stack_by}")

    db = request.app.state.db
    config = request.app.state.config
    since_dt = parse_since(since) if since else None
    until_dt = parse_since(until) if until else None
    conn = getattr(db, "conn", None)

    bucket = _bucket_unit(since_dt, until_dt)
    metric_expr, value_unit = _METRIC_EXPR[metric]

    meta = {
        "metric": metric,
        "metric_label": _METRIC_LABEL[metric],
        "value_unit": value_unit,
        "group_by": group_by,
        "stack_by": stack_by,
        "series_bucket": bucket,
        "window_start": int(since_dt.timestamp()) if since_dt is not None else None,
        "window_end": int((until_dt or utcnow()).timestamp()),
        "available_metrics": list(_METRIC_EXPR.keys()),
        "available_dimensions": list(_DIMENSION_EXPR.keys()),
    }
    empty = {
        **meta, "rows": [], "groups": [], "stacks": [], "totals_by_group": {},
        "kpis": {"spend": 0.0, "tokens": 0, "events": 0, "sessions": 0},
        "kpi_series": [], "kpi_prev": None, "kpi_deltas": {},
        "framing": _framing_block(db, config, agent_id, 0.0, 0),
    }
    if conn is None:
        return empty

    def resolve(dim: str) -> str:
        expr = _DIMENSION_EXPR[dim]
        return _bucket_expr(bucket) if expr == "__bucket__" else expr

    # WHERE — every user value bound via $N; filters/since/until only (the bucket
    # unit is inlined as a validated literal). One shared param list serves both
    # the grouped query and the KPI query.
    #
    # Span-subtype gate, dimension-aware: a model/provider breakdown should only
    # see LLM spans (tool spans have NULL model and would add a noisy '(none)'
    # group), while a tool/tool_category breakdown should only see tool spans.
    # Other dimensions (agent / day / kind) span all rows. A model+tool pivot is
    # incoherent and correctly yields nothing.
    dims = {group_by} | ({stack_by} if stack_by else set())
    # Subtype gate (no params) cleans the BREAKDOWN; it is NOT applied to the KPI
    # totals, which reflect the true window scoped only by the user's filters +
    # time — so "events by tool" doesn't zero out the Spend KPI.
    subtype: list[str] = []
    if dims & {"model", "provider"}:
        subtype.append("model IS NOT NULL")
    if dims & {"tool", "tool_category"}:
        subtype.append("tool_name IS NOT NULL")
    # The non-time equality filters (agent/provider/model/tool), reused to build
    # the current window, the per-bucket KPI series, and the prior-period window.
    filter_pairs = [(_FILTER_COLUMN[p], v) for p, v in
                    (("agent_id", agent_id), ("provider", provider),
                     ("model", model), ("tool", tool)) if v]

    def where_and_params(*, with_subtype: bool, lo, hi) -> tuple[str, list]:
        """Build a (where, params) over the filters + a [lo, hi) time window.
        `with_subtype` adds the breakdown's span-subtype gate (no params)."""
        cl: list[str] = list(subtype) if with_subtype else []
        ps: list = []
        for col, val in filter_pairs:
            ps.append(val)
            cl.append(col + " = $" + str(len(ps)))
        if lo is not None:
            ps.append(lo)
            cl.append("start_time >= $" + str(len(ps)))
        if hi is not None:
            ps.append(hi)
            cl.append("start_time < $" + str(len(ps)))
        return (" AND ".join(cl) if cl else "1=1", ps)

    end_dt = until_dt or utcnow()
    where, params = where_and_params(with_subtype=True, lo=since_dt, hi=end_dt)
    kpi_where, kpi_params = where_and_params(with_subtype=False, lo=since_dt, hi=end_dt)

    g1 = resolve(group_by)
    g2 = resolve(stack_by) if stack_by else None
    select_cols = [_bucket_expr(bucket) + " AS b", g1 + " AS g1"]
    group_cols = ["b", "g1"]
    if g2 is not None:
        select_cols.append(g2 + " AS g2")
        group_cols.append("g2")
    select_cols.append(metric_expr + " AS val")
    select_cols.append(_TOKENS_EXPR + " AS toks")
    select_cols.append("COALESCE(SUM(cost_usd), 0.0) AS cost")
    select_cols.append("COALESCE(SUM(input_tokens), 0) AS input_toks")
    select_cols.append("COALESCE(SUM(output_tokens), 0) AS output_toks")
    select_cols.append("COALESCE(SUM(cache_tokens), 0) AS cache_read_toks")
    select_cols.append("COALESCE(SUM(cache_write_tokens), 0) AS cache_write_toks")
    select_cols.append("COUNT(*) AS events")
    select_cols.append("COUNT(DISTINCT session_id) AS sessions")

    sql = (
        "SELECT " + ", ".join(select_cols) + " FROM spans WHERE " + where
        + " GROUP BY " + ", ".join(group_cols) + " ORDER BY b"
    )
    raw = conn.execute(sql, params).fetchall()

    rows: list[dict] = []
    totals: dict[str, float] = {}
    stacks: dict[str, float] = {}
    for r in raw:
        b = int(r[0])
        g1v = r[1] if r[1] is not None else "(none)"
        if g2 is not None:
            g2v = r[2] if r[2] is not None else "(none)"
            val = float(r[3] or 0.0)
            toks = int(r[4] or 0)
            cost = float(r[5] or 0.0)
            input_toks = int(r[6] or 0)
            output_toks = int(r[7] or 0)
            cache_read = int(r[8] or 0)
            cache_write = int(r[9] or 0)
            events = int(r[10] or 0)
            sessions = int(r[11] or 0)
        else:
            g2v = None
            val = float(r[2] or 0.0)
            toks = int(r[3] or 0)
            cost = float(r[4] or 0.0)
            input_toks = int(r[5] or 0)
            output_toks = int(r[6] or 0)
            cache_read = int(r[7] or 0)
            cache_write = int(r[8] or 0)
            events = int(r[9] or 0)
            sessions = int(r[10] or 0)
        rows.append({
            "bucket": b, "group": str(g1v),
            "stack": (str(g2v) if g2v is not None else None),
            "value": val, "tokens": toks,
            "cost": cost, "input_tokens": input_toks,
            "output_tokens": output_toks, "cache_read_tokens": cache_read,
            "cache_write_tokens": cache_write, "events": events,
            "sessions": sessions,
        })
        totals[str(g1v)] = totals.get(str(g1v), 0.0) + val
        if g2v is not None:
            stacks[str(g2v)] = stacks.get(str(g2v), 0.0) + val

    groups = sorted(totals, key=lambda k: totals[k], reverse=True)
    stack_keys = sorted(stacks, key=lambda k: stacks[k], reverse=True)

    # KPI totals over the window (independent of group_by). DISTINCT for sessions
    # so a session spanning models/buckets is counted once. The four-metric
    # SELECT is reused for the prior window too.
    _kpi_cols = ("COALESCE(SUM(cost_usd),0.0), " + _TOKENS_EXPR
                 + ", COUNT(*), COUNT(DISTINCT session_id)")

    def _kpi_totals(where_sql: str, ps: list) -> dict:
        row = conn.execute("SELECT " + _kpi_cols + " FROM spans WHERE " + where_sql, ps).fetchone()
        return {
            "spend": float(row[0] or 0.0) if row else 0.0,
            "tokens": int(row[1] or 0) if row else 0,
            "events": int(row[2] or 0) if row else 0,
            "sessions": int(row[3] or 0) if row else 0,
        }

    kpis = _kpi_totals(kpi_where, kpi_params)

    # Per-bucket series for ALL FOUR KPI metrics (the sparklines, #217). One
    # query, server-side — the UI zero-fills across the window and never
    # aggregates per-bucket in JS (single compute path).
    kpi_series_sql = (
        "SELECT " + _bucket_expr(bucket) + " AS b, " + _kpi_cols
        + " FROM spans WHERE " + kpi_where + " GROUP BY b ORDER BY b"
    )
    kpi_series = [
        {"bucket": int(r[0]), "spend": float(r[1] or 0.0), "tokens": int(r[2] or 0),
         "events": int(r[3] or 0), "sessions": int(r[4] or 0)}
        for r in conn.execute(kpi_series_sql, kpi_params).fetchall()
    ]

    # Period-over-period delta vs the prior equal-length window (#217), same
    # convention as the Cost/Overview --compare headline. Computed server-side.
    kpi_prev: dict | None = None
    kpi_deltas: dict = {}
    if since_dt is not None:
        prev_since = since_dt - (end_dt - since_dt)
        pwhere, pparams = where_and_params(with_subtype=False, lo=prev_since, hi=since_dt)
        kpi_prev = _kpi_totals(pwhere, pparams)
        for m in ("spend", "tokens", "events", "sessions"):
            prev_v = kpi_prev[m]
            cur_v = kpis[m]
            kpi_deltas[m] = (round((cur_v - prev_v) / prev_v * 100.0, 1)
                             if prev_v else None)

    return {
        **meta,
        "rows": rows,
        "groups": groups,
        "stacks": stack_keys,
        "totals_by_group": {k: round(v, 8) for k, v in totals.items()},
        "kpis": kpis,
        "kpi_series": kpi_series,
        "kpi_prev": kpi_prev,
        "kpi_deltas": kpi_deltas,
        "framing": _framing_block(db, config, agent_id, kpis["spend"], kpis["tokens"]),
    }
