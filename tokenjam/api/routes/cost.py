"""GET /api/v1/cost — aggregated cost data."""
from __future__ import annotations

import math

from fastapi import APIRouter, Depends, Request

from tokenjam.api.deps import require_api_key
from tokenjam.core.cycle import cycle_bounds, effective_cycle_start_day
from tokenjam.core.framing import (
    WindowSummary,
    compute_framing,
    plan_determination_mix,
)
from tokenjam.core.models import CostFilters
from tokenjam.utils.time_parse import parse_since, utcnow

router = APIRouter(dependencies=[Depends(require_api_key)])


def _cycle_block(config) -> dict:
    """Current billing-cycle bounds for the run-rate caption (#138).

    Honors `[budget.<provider>] cycle_start_day` when configured, falling back
    to the calendar month (start_day=1). The UI projects the run-rate to
    `cycle.end` instead of assuming a calendar-month boundary, so a user on a
    non-calendar cycle gets an honest "by <cycle end>" caption.
    """
    now = utcnow()
    start_day = effective_cycle_start_day(config)
    cs, ce = cycle_bounds(now, start_day)
    days_remaining = max(1, math.ceil((ce - now).total_seconds() / 86400.0))
    return {
        "start": int(cs.timestamp()),
        "end": int(ce.timestamp()),
        "days_remaining": days_remaining,
        "start_day": start_day,
    }


def _window_series(conn, agent_id, since_dt, until_dt) -> dict:
    """Window-bucketed cost/tokens per (bucket, agent, model) for charting.

    Buckets hourly for short windows (≤ 2 days) and daily otherwise, returning
    epoch-second bucket keys plus the window bounds so the UI can render the
    FULL selected window with zero-fill and window-matched tick density (#133) —
    the grouped `rows` collapse the agent/model dimension and only cover days
    with data. Buckets are UTC (AT TIME ZONE 'UTC', Rule 1).
    """
    start = since_dt
    end = until_dt or utcnow()
    span_days = ((end - start).total_seconds() / 86400.0) if start is not None else None
    bucket = "hour" if (span_days is not None and span_days <= 2) else "day"
    out: dict = {
        "series": [],
        "series_bucket": bucket,
        "window_start": int(start.timestamp()) if start is not None else None,
        "window_end": int(end.timestamp()),
    }
    if conn is None:
        return out
    # $1 is the date_trunc unit (a controlled 'hour'/'day' literal, bound as a
    # parameter — no f-string SQL, Rule 7).
    clauses = ["model IS NOT NULL"]
    params: list = [bucket]
    if agent_id:
        params.append(agent_id)
        clauses.append("agent_id = $" + str(len(params)))
    if since_dt is not None:
        params.append(since_dt)
        clauses.append("start_time >= $" + str(len(params)))
    if until_dt is not None:
        params.append(until_dt)
        clauses.append("start_time <= $" + str(len(params)))
    where = " AND ".join(clauses)
    # Grouped by (bucket, agent, model, provider) with the full token-component
    # split per row. This is the reusable group-by series shape the cost charts
    # (#213 stacked by model/agent) and the future analytics explorer (#210)
    # consume — callers pick which dimension(s) to stack client-side.
    sql = (
        "SELECT CAST(epoch(date_trunc($1, start_time AT TIME ZONE 'UTC')) AS BIGINT) AS b, "
        "agent_id, model, provider, COALESCE(SUM(cost_usd), 0.0), "
        "COALESCE(SUM(input_tokens), 0), COALESCE(SUM(output_tokens), 0), "
        "COALESCE(SUM(cache_tokens), 0), COALESCE(SUM(cache_write_tokens), 0) "
        "FROM spans WHERE " + where + " GROUP BY b, agent_id, model, provider ORDER BY b"
    )
    rows = conn.execute(sql, params).fetchall()
    out["series"] = [
        {
            "bucket": int(r[0]), "agent_id": r[1], "model": r[2], "provider": r[3],
            "cost_usd": float(r[4] or 0.0),
            "input_tokens": int(r[5] or 0), "output_tokens": int(r[6] or 0),
            "cache_tokens": int(r[7] or 0), "cache_write_tokens": int(r[8] or 0),
        }
        for r in rows
    ]
    return out


def _framing_block(db, config, agent_id, total_cost, total_tokens) -> dict:
    """Build the plan-tier framing block (single source, #110), window-independent
    plan mix (#177). Shared by /cost and /cost/cache so neither re-derives the
    suppression rules client-side."""
    conn = getattr(db, "conn", None)
    mix = plan_determination_mix(conn, agent_id) if conn is not None else {}
    return compute_framing(
        config,
        WindowSummary(
            total_cost_usd=total_cost,
            total_tokens=total_tokens,
            sessions=sum(mix.values()),
            plan_tier_mix=mix,
        ),
    ).to_dict()


def _cache_series(conn, agent_id, since_dt, until_dt) -> dict:
    """Per-bucket cache hit-rate + cumulative captured cache savings (#212).

    Hit-rate is `cache_read / (cache_read + input)` per bucket — a token ratio,
    always meaningful regardless of plan tier. "Captured" savings is the dollars
    already saved by the cache reads that DID happen, priced per (provider,
    model) as `cache_read_tokens × (input_rate − cache_read_rate)` — a measured
    figure (so it's "captured", not "estimated"). The recoverable *additional*
    estimate is attached by the route from the cache analyzer. Buckets are UTC
    (Rule 1) and mirror `_window_series` so the chart x-axes line up.
    """
    from tokenjam.core.pricing import get_rates

    start = since_dt
    end = until_dt or utcnow()
    span_days = ((end - start).total_seconds() / 86400.0) if start is not None else None
    bucket = "hour" if (span_days is not None and span_days <= 2) else "day"
    out: dict = {
        "series": [],
        "series_bucket": bucket,
        "window_start": int(start.timestamp()) if start is not None else None,
        "window_end": int(end.timestamp()),
    }
    if conn is None:
        return out
    clauses = ["model IS NOT NULL"]
    params: list = [bucket]
    if agent_id:
        params.append(agent_id)
        clauses.append("agent_id = $" + str(len(params)))
    if since_dt is not None:
        params.append(since_dt)
        clauses.append("start_time >= $" + str(len(params)))
    if until_dt is not None:
        params.append(until_dt)
        clauses.append("start_time <= $" + str(len(params)))
    where = " AND ".join(clauses)
    sql = (
        "SELECT CAST(epoch(date_trunc($1, start_time AT TIME ZONE 'UTC')) AS BIGINT) AS b, "
        "provider, model, COALESCE(SUM(input_tokens), 0), "
        "COALESCE(SUM(cache_tokens), 0), COALESCE(SUM(cache_write_tokens), 0) "
        "FROM spans WHERE " + where + " GROUP BY b, provider, model ORDER BY b"
    )
    rows = conn.execute(sql, params).fetchall()

    # Fold per-(provider, model) rows into per-bucket aggregates + priced savings.
    by_bucket: dict[int, dict] = {}
    for b, provider, model, in_tok, cr_tok, cw_tok in rows:
        b = int(b)
        agg = by_bucket.setdefault(
            b, {"input": 0, "cache_read": 0, "cache_write": 0, "captured_usd": 0.0}
        )
        agg["input"] += int(in_tok or 0)
        agg["cache_read"] += int(cr_tok or 0)
        agg["cache_write"] += int(cw_tok or 0)
        rates = get_rates(provider, model) if model else None
        # Only claim captured savings when the model has a real discounted
        # cache-read rate; otherwise stay silent (honest — no invented savings).
        if rates is not None and rates.cache_read_per_mtok > 0:
            delta = max(0.0, rates.input_per_mtok - rates.cache_read_per_mtok)
            agg["captured_usd"] += (int(cr_tok or 0) * delta) / 1_000_000.0

    series = []
    for b in sorted(by_bucket):
        a = by_bucket[b]
        denom = a["input"] + a["cache_read"]
        hit_rate = (a["cache_read"] / denom) if denom > 0 else 0.0
        series.append({
            "bucket": b,
            "input_tokens": a["input"],
            "cache_read_tokens": a["cache_read"],
            "cache_write_tokens": a["cache_write"],
            "hit_rate": round(hit_rate, 4),
            "captured_usd": round(a["captured_usd"], 8),
            "captured_tokens": a["cache_read"],
        })
    out["series"] = series
    return out


# Component a given analyzer's recoverable savings primarily acts on. "call"
# means whole-call (a model swap or call elimination), NOT a single token
# component — used for analyzers whose savings can't be honestly pinned to one
# component (downsize / script / reuse). New analyzers default to "call", so the
# overlay stays registry-driven (#211).
_ANALYZER_COMPONENT = {
    "cache": "input",     # raising cache efficacy shrinks the input component
    "trim": "input",      # trims low-significance input/prompt tokens
    "downsize": "call",   # whole-call model swap
    "script": "call",     # eliminates whole deterministic calls
    "reuse": "call",      # eliminates whole repeated-planning calls
}
_ANALYZER_TITLE = {
    "downsize": "Downsize", "cache": "Cache", "script": "Script",
    "trim": "Trim", "reuse": "Reuse",
}

_COMPONENT_LABELS = [
    ("input", "Input"),
    ("output", "Output"),
    ("cache_read", "Cache read"),
    ("cache_write", "Cache write"),
]


def _component_costs(conn, agent_id, since_dt, until_dt) -> dict:
    """Split the window's spend into the four token components (#211).

    Computes each component's cost per (provider, model) via the pricing table
    so the split is exact rather than apportioned from the aggregate cost_usd.
    Returns the four components with both cost and token volume (the UI shows
    tokens for subscription/local framing where dollars are suppressed).
    """
    from tokenjam.core.pricing import get_rates

    comp = {k: {"cost_usd": 0.0, "tokens": 0} for k, _ in _COMPONENT_LABELS}
    if conn is None:
        return comp
    clauses = ["model IS NOT NULL"]
    params: list = []
    if agent_id:
        params.append(agent_id)
        clauses.append("agent_id = $" + str(len(params)))
    if since_dt is not None:
        params.append(since_dt)
        clauses.append("start_time >= $" + str(len(params)))
    if until_dt is not None:
        params.append(until_dt)
        clauses.append("start_time <= $" + str(len(params)))
    where = " AND ".join(clauses)
    sql = (
        "SELECT provider, model, COALESCE(SUM(input_tokens), 0), "
        "COALESCE(SUM(output_tokens), 0), COALESCE(SUM(cache_tokens), 0), "
        "COALESCE(SUM(cache_write_tokens), 0) "
        "FROM spans WHERE " + where + " GROUP BY provider, model"
    )
    for provider, model, in_t, out_t, cr_t, cw_t in conn.execute(sql, params).fetchall():
        in_t, out_t, cr_t, cw_t = int(in_t or 0), int(out_t or 0), int(cr_t or 0), int(cw_t or 0)
        comp["input"]["tokens"] += in_t
        comp["output"]["tokens"] += out_t
        comp["cache_read"]["tokens"] += cr_t
        comp["cache_write"]["tokens"] += cw_t
        rates = get_rates(provider, model) if model else None
        if rates is None:
            continue
        comp["input"]["cost_usd"] += in_t * rates.input_per_mtok / 1_000_000.0
        comp["output"]["cost_usd"] += out_t * rates.output_per_mtok / 1_000_000.0
        comp["cache_read"]["cost_usd"] += cr_t * rates.cache_read_per_mtok / 1_000_000.0
        comp["cache_write"]["cost_usd"] += cw_t * rates.cache_write_per_mtok / 1_000_000.0
    return comp


def _collect_recoverable(report) -> list[dict]:
    """Registry-driven per-analyzer recoverable list (#211 overlay).

    Iterates the typed downgrade slot + every wave-2 finding carrying the #111
    recoverable contract field, so a new analyzer appears with no code change
    here. Each entry keeps the analyzer's own caveat + estimate_basis verbatim
    (Rule 14) and the component its savings act on. Only positive estimates are
    surfaced (a None/0 estimate is "nothing to recover", not an overlay bar).

    Returned biggest-first (by USD, then tokens) so a caller can render "largest
    opportunity + N more" without re-deriving the order — and so index 0 is
    always the entry `largest_recoverable_*` below is drawn from."""
    out: list[dict] = []

    def add(name: str, finding) -> None:
        if finding is None:
            return
        usd = getattr(finding, "estimated_recoverable_usd", None)
        tok = getattr(finding, "estimated_recoverable_tokens", None)
        if not ((usd is not None and usd > 0) or (tok is not None and tok > 0)):
            return
        out.append({
            "analyzer": name,
            "title": _ANALYZER_TITLE.get(name, name.replace("-", " ").title()),
            "component": _ANALYZER_COMPONENT.get(name, "call"),
            "estimated_recoverable_usd": usd,
            "estimated_recoverable_tokens": tok,
            "estimate_basis": getattr(finding, "estimate_basis", "") or "",
            "caveat": getattr(finding, "caveat", "") or "",
        })

    add("downsize", getattr(report, "downgrade", None))
    for name, finding in (getattr(report, "findings", None) or {}).items():
        if name == "downsize":
            continue
        if hasattr(finding, "estimated_recoverable_usd"):
            add(name, finding)
    out.sort(
        key=lambda r: (r["estimated_recoverable_usd"] or 0.0, r["estimated_recoverable_tokens"] or 0),
        reverse=True,
    )
    return out


# A1 (analyzer-audit #482, self-improve-loop): every analyzer above estimates
# waste from its own angle over the SAME underlying spans — a `cache` fix and
# a `downsize` swap can both price the same call's waste, a `reuse` hit can
# double-count a `trim` hit's planning call, and so on. `total_recoverable_usd`
# below is therefore a GROSS ceiling across N overlapping analyzers, not a
# simultaneously-achievable total — summing the list's own figures would be
# summing waste that was measured twice. This is a presentation fix only: no
# individual analyzer's `estimated_recoverable_usd` is touched or reduced here
# (house rule: never quietly deflate a figure the user can act on). The single
# largest entry, in contrast, IS honest on its own — acting on it alone
# recovers at least that much, because it isn't a sum of anything.
def _recoverable_overlap_note(recoverable: list[dict]) -> str:
    if len(recoverable) < 2:
        return ""
    return (
        f"These {len(recoverable)} estimates are computed from overlapping angles on "
        "the same sessions (for example, a cache fix and a model downsize can both "
        "price the same call's waste), so they do not add up to an amount you could "
        "actually recover. The figure above is a ceiling across every analyzer; the "
        "largest single line is the safest number to act on first."
    )


@router.get("/cost/components")
async def get_cost_components(
    request: Request,
    agent_id: str | None = None,
    since: str | None = None,
    until: str | None = None,
) -> dict:
    """Cost-by-component split + per-analyzer recoverable-waste overlay (#211).

    The component bars are MEASURED spend split into input / output / cache-read
    / cache-write. The overlay is each analyzer's *estimated recoverable* waste —
    registry-driven, carrying the analyzer's own caveat verbatim. "Estimated
    recoverable" is never conflated with the measured cost and never called
    "saved" (Critical Rule 14). Every figure routes through the framing block so
    subscription/local users see token-share, not raw dollars."""
    db = request.app.state.db
    config = request.app.state.config
    since_dt = parse_since(since) if since else None
    until_dt = parse_since(until) if until else None
    conn = getattr(db, "conn", None)

    comp = _component_costs(conn, agent_id, since_dt, until_dt)
    components = [
        {"key": key, "label": label,
         "cost_usd": round(comp[key]["cost_usd"], 8), "tokens": comp[key]["tokens"]}
        for key, label in _COMPONENT_LABELS
    ]
    total_cost = sum(c["cost_usd"] for c in components)
    total_tokens = sum(c["tokens"] for c in components)

    recoverable: list[dict] = []
    if conn is not None:
        try:
            from tokenjam.core.optimize import ANALYZER_REGISTRY, build_report

            # `relearn` is a full-corpus scan the analyzer's OWN docstring
            # says is too heavy for per-request HTTP use ("callers that serve
            # this over HTTP MUST cache the result, not compute it per-
            # request" — core/optimize/analyzers/relearn.py). Worse, its
            # `RelearnFinding` never carries `estimated_recoverable_usd`, so
            # `_collect_recoverable` below silently discards its result no
            # matter what — running it here was guaranteed dead work on every
            # request. Excluding it by name changes nothing about what this
            # endpoint returns (verified: no output field ever came from it)
            # while removing that tax; the Review inbox
            # (api/routes/relearn.py) already serves relearn's finding from
            # its own background-refreshed cache. `deadweight` DOES
            # contribute (it has `estimated_recoverable_usd`) so it still
            # runs here, but now via the persistent transcript parse cache
            # (core.transcript_cache, wired into its `run(ctx)` entry point)
            # so a warm cache makes repeat requests cheap instead of
            # re-scanning every transcript from scratch each time.
            findings = [name for name in ANALYZER_REGISTRY if name != "relearn"]
            report = build_report(
                db=db, config=config,
                since=since_dt or utcnow(), until=until_dt or utcnow(),
                agent_id=agent_id, findings=findings,
            )
            recoverable = _collect_recoverable(report)
        except Exception:
            recoverable = []

    total_rec_usd = sum(r["estimated_recoverable_usd"] or 0.0 for r in recoverable)
    total_rec_tokens = sum(r["estimated_recoverable_tokens"] or 0 for r in recoverable)
    largest = recoverable[0] if recoverable else None

    return {
        "components": components,
        "total_cost_usd": round(total_cost, 8),
        "total_tokens": total_tokens,
        "recoverable": recoverable,
        # Gross ceiling, magnitude unchanged (see _recoverable_overlap_note) —
        # NOT a claim that this much is simultaneously recoverable.
        "total_recoverable_usd": round(total_rec_usd, 8),
        "total_recoverable_tokens": total_rec_tokens,
        "recoverable_additive": False,
        "recoverable_overlap_note": _recoverable_overlap_note(recoverable),
        # The one entry in `recoverable` that is honest as a standalone claim:
        # it isn't a sum of anything, so it's the floor a reader can act on.
        "largest_recoverable_usd": largest["estimated_recoverable_usd"] if largest else None,
        "largest_recoverable_tokens": largest["estimated_recoverable_tokens"] if largest else None,
        "largest_recoverable_analyzer": largest["analyzer"] if largest else None,
        "framing": _framing_block(db, config, agent_id, total_cost, total_tokens),
    }


@router.get("/cost/cache")
async def get_cost_cache(
    request: Request,
    agent_id: str | None = None,
    since: str | None = None,
    until: str | None = None,
) -> dict:
    """Cache-savings time-series (#212): per-bucket hit-rate + cumulative captured
    savings, plus the window-level *estimated recoverable* from the cache analyzer.

    "Captured" is measured (priced from real cache reads); "estimated
    recoverable" is the analyzer's heuristic for additional savings if caching
    were expanded — never conflated, never called "saved" (Critical Rule 14)."""
    db = request.app.state.db
    config = request.app.state.config
    since_dt = parse_since(since) if since else None
    until_dt = parse_since(until) if until else None
    conn = getattr(db, "conn", None)

    block = _cache_series(conn, agent_id, since_dt, until_dt)
    total_captured = sum(p["captured_usd"] for p in block["series"])
    total_captured_tokens = sum(p["captured_tokens"] for p in block["series"])

    # Window-level estimated recoverable from the cache-efficacy analyzer (#111
    # recoverable contract). Best-effort: skip silently if the report can't run.
    recoverable_usd: float | None = None
    recoverable_tokens: int | None = None
    estimate_basis = ""
    if conn is not None:
        try:
            from tokenjam.core.optimize import build_report
            report = build_report(
                db=db, config=config,
                since=since_dt or utcnow(), until=until_dt or utcnow(),
                agent_id=agent_id, findings=["cache"],
            )
            cache_finding = (report.findings or {}).get("cache")
            if cache_finding is not None:
                recoverable_usd = getattr(cache_finding, "estimated_recoverable_usd", None)
                recoverable_tokens = getattr(cache_finding, "estimated_recoverable_tokens", None)
                estimate_basis = getattr(cache_finding, "estimate_basis", "") or ""
        except Exception:
            pass

    return {
        **block,
        "total_captured_usd": round(total_captured, 8),
        "total_captured_tokens": total_captured_tokens,
        "estimated_recoverable_usd": recoverable_usd,
        "estimated_recoverable_tokens": recoverable_tokens,
        "estimate_basis": estimate_basis,
        "framing": _framing_block(db, config, agent_id, total_captured, total_captured_tokens),
    }


@router.get("/cost")
async def get_cost(
    request: Request,
    agent_id: str | None = None,
    since: str | None = None,
    until: str | None = None,
    group_by: str = "day",
) -> dict:
    db = request.app.state.db
    config = request.app.state.config
    since_dt = parse_since(since) if since else None
    until_dt = parse_since(until) if until else None
    filters = CostFilters(
        agent_id=agent_id,
        since=since_dt,
        until=until_dt,
        group_by=group_by,
    )
    rows = db.get_cost_summary(filters)
    total = sum(r.cost_usd for r in rows)
    total_tokens = sum(r.input_tokens + r.output_tokens for r in rows)

    # Plan-tier framing block — single source shared with the CLI (#110). Lets
    # the local web UI render the same suppressed/qualified dollar figures.
    # The mix is window-INDEPENDENT (#177): the pricing mode + qualifier banner
    # are a property of the user's plan, so the chart's tokens-vs-dollars unit
    # stays consistent as the user switches windows. Only the window totals
    # (above) and the `series` (below) are window-scoped.
    conn = getattr(db, "conn", None)
    framing = _framing_block(db, config, agent_id, total, total_tokens)
    return {
        "rows": [
            {
                "group": r.group,
                "agent_id": r.agent_id,
                "model": r.model,
                "input_tokens": r.input_tokens,
                "output_tokens": r.output_tokens,
                "cache_tokens": r.cache_tokens,
                "cache_write_tokens": r.cache_write_tokens,
                "cost_usd": r.cost_usd,
            }
            for r in rows
        ],
        "total_cost_usd": total,
        "total_tokens": total_tokens,
        "total_cache_tokens": sum(r.cache_tokens for r in rows),
        "total_cache_write_tokens": sum(r.cache_write_tokens for r in rows),
        **_window_series(conn, agent_id, since_dt, until_dt),
        "cycle": _cycle_block(config),
        "framing": framing,
    }
