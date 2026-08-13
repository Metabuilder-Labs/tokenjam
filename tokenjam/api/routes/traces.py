"""GET /api/v1/traces — trace listing and detail."""
from __future__ import annotations


from fastapi import APIRouter, Depends, HTTPException, Request

from tokenjam.api.deps import require_api_key
from tokenjam.core.data_span import available_data_span
from tokenjam.core.framing import (
    PERSONAS,
    WindowSummary,
    compute_framing,
    plan_determination_mix,
)
from tokenjam.core.models import TraceCostStats, TraceFilters
from tokenjam.utils.time_parse import parse_since

router = APIRouter(dependencies=[Depends(require_api_key)])


def _traces_framing(request: Request, agent_id: str | None) -> dict:
    """Plan-tier framing block for trace cost figures (#187).

    Traces and trace-detail are not window-scoped views, so the plan is derived
    from a window-INDEPENDENT session mix (`plan_determination_mix`) — the same
    helper `/cost` uses. The web UI consumes this block to suppress / reframe raw
    dollar costs for subscription / local users (honesty discipline, Rule 14)
    instead of re-deriving the suppression rules in JS (single compute path).
    """
    db = request.app.state.db
    config = request.app.state.config
    conn = getattr(db, "conn", None)
    mix = plan_determination_mix(conn, agent_id) if conn is not None else {}
    framing = compute_framing(
        config,
        WindowSummary(plan_tier_mix=mix, sessions=sum(mix.values())),
    )
    return framing.to_dict()


# Valid values for the `sort` query param. Anything else falls back to
# "recent" rather than erroring — same forgiving-default philosophy as the
# web UI's readParam() (bad/old bookmarked URLs degrade gracefully).
_VALID_TRACE_SORTS = ("recent", "cost")


@router.get("/traces")
async def list_traces(
    request: Request,
    agent_id: str | None = None,
    since: str | None = None,
    until: str | None = None,
    limit: int = 200,
    offset: int = 0,
    status: str | None = None,
    span_name: str | None = None,
    sort: str | None = None,
    min_cost_usd: float | None = None,
    persona: str | None = None,
) -> dict:
    """Trace list, scoped to one side of the "Viewing as" picker.

    ``persona`` narrows to interactive coding agents (`claude-code`) or to
    everything else (`sdk`); `mixed` / `unknown` / omitted narrow nothing,
    matching `GET /sessions`. It reaches the SQL through ``TraceFilters`` so the
    rows, the total count and the outlier rule below all describe the same
    population — a filtered list beside an unfiltered total is the recurring
    defect in this product, not a rounding error.
    """
    db = request.app.state.db
    if persona is not None and persona not in PERSONAS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown persona {persona!r}. Expected one of {sorted(PERSONAS)}.",
        )
    try:
        since_dt = parse_since(since) if since else None
        until_dt = parse_since(until) if until else None
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid --since: {exc}") from exc
    filters = TraceFilters(
        agent_id=agent_id,
        since=since_dt,
        until=until_dt,
        limit=limit,
        offset=offset,
        status=status,
        span_name=span_name,
        sort=sort if sort in _VALID_TRACE_SORTS else "recent",
        min_cost_usd=min_cost_usd,
        persona=persona,
    )
    traces = db.get_traces(filters)
    total_count = db.count_traces(filters) if hasattr(db, "count_traces") else len(traces)
    stats = db.get_trace_cost_stats(filters) if hasattr(db, "get_trace_cost_stats") else None
    conn = getattr(db, "conn", None)
    return {
        "traces": [
            {
                "trace_id": t.trace_id,
                "agent_id": t.agent_id,
                "name": t.name,
                "start_time": t.start_time.isoformat() if t.start_time else None,
                "duration_ms": t.duration_ms,
                "cost_usd": t.cost_usd,
                "status_code": t.status_code,
                "span_count": t.span_count,
                # Per-trace token totals — the UI renders per-row cost as TOKENS
                # for subscription/local users (#249), never "% of cycle".
                "input_tokens": t.input_tokens,
                "output_tokens": t.output_tokens,
                # Statistical cost-outlier flag for this window; see
                # `outlier_rule` below for the plain-language explanation and
                # the numbers behind it.
                "is_outlier": t.is_outlier,
            }
            for t in traces
        ],
        "count": len(traces),
        "total_count": total_count,
        "framing": _traces_framing(request, agent_id),
        "outlier_rule": _outlier_rule_dict(stats),
        # `available_days` (core/data_span.py) so the Traces window selector
        # can derive its options from what the store actually holds, the same
        # way the Dashboard's does — instead of a fixed 24h/7d/30d/90d list.
        "data_span": available_data_span(conn).to_dict(),
    }


def _outlier_rule_dict(stats: TraceCostStats | None) -> dict | None:
    """Plain-language + numeric explanation of the `is_outlier` flag.

    The UI renders this verbatim so a user never has to read source to know
    what "outlier" means. `threshold_usd` is None when there weren't enough
    priced traces in the window to compute a reliable range — in that case no
    trace in the response is flagged, and the UI should say so rather than
    imply a rule ran and found nothing.
    """
    if stats is None:
        return None
    return {
        "method": stats.method,
        "sample_size": stats.sample_size,
        "min_sample": stats.min_sample,
        "q1_usd": stats.q1_usd,
        "q3_usd": stats.q3_usd,
        "threshold_usd": stats.threshold_usd,
    }


# How many of a trace's own spans to surface as "the costliest spans in this
# trace" — a runaway retry loop or a single blown-up tool call should be
# findable without scanning the whole waterfall by eye.
TOP_COST_SPAN_LIMIT = 5

# Hard cap on how many spans the waterfall payload carries (#653). A big
# fan-out/agent trace can hold tens of thousands of spans; shipping them all
# (previously each with its FULL `attributes` dict of captured prompt/tool
# content) produced ~1 GB JSON responses the browser could neither fetch nor
# render, so the detail pane hung on its skeleton forever. We ship at most this
# many spans and surface `truncated` + the true `span_count` so the UI can say
# "showing first N of M spans" — an honest disclosure, never a silent drop
# (Rule 14). A 45k-span trace still renders (its costliest spans first) instead
# of hanging.
TRACE_SPAN_CAP = 2000


@router.get("/traces/{trace_id}")
async def get_trace(request: Request, trace_id: str, attributes: bool = True) -> dict:
    """Trace detail.

    **DELIBERATELY PERSONA-BLIND, and this is the reason.** A trace id addresses
    ONE trace, which belongs to exactly one agent and therefore to exactly one
    side of the picker already. A `persona` parameter here could only do one of
    two things, and both are wrong: silently return nothing for a trace the
    reader is looking straight at (a deep link, a row they just clicked), or
    filter within a single trace's spans, which would produce a partial
    waterfall whose totals no longer add up to the trace. Scoping belongs on
    the LIST (`GET /traces`), which is what decides whether this trace is
    offered at all. Reachability by direct URL is intentional — the picker is a
    view over a list, not an access control.

    Defaults to FULL spans (each with its `attributes` dict) so
    `ApiBackend.get_trace_spans` and every existing complete-span consumer
    (exports, backfill re-reads) keep the same contract they had before #653.

    The Lens waterfall passes `?attributes=false` for the lightweight payload
    (#653): a big fan-out/agent trace can hold tens of thousands of spans, and
    shipping every span's FULL captured prompt/tool content produced ~1 GB JSON
    responses the browser could neither fetch nor render. The attribute-free
    payload carries only the tree/timing/tokens/cost the waterfall needs; the
    span-detail panel then fetches ONE span's attributes lazily on expand via
    GET /traces/{trace_id}/spans/{span_id}. Making the light payload opt-in (not
    the default) is deliberate — the previous default silently truncated the
    daemon-backed shim's attributes for all non-UI consumers.
    """
    db = request.app.state.db
    spans = db.get_trace_spans(trace_id)
    total = len(spans)
    # Scope the plan determination to this trace's agent when known (falls back
    # to the whole install) so subscription / local cost suppression matches the
    # Traces list and Cost screen.
    agent_id = next((s.agent_id for s in spans if getattr(s, "agent_id", None)), None)
    # Rank spans WITHIN this trace by cost. Computed from the (full) span list
    # already fetched (bounded by one trace's span count — never a separate DB
    # scan), since the waterfall needs the tree regardless of rank; only the
    # ranking is new.
    priced = [s for s in spans if (getattr(s, "cost_usd", None) or 0) > 0]
    top_cost_spans = sorted(priced, key=lambda s: s.cost_usd or 0, reverse=True)[:TOP_COST_SPAN_LIMIT]
    top_cost_span_ids = [s.span_id for s in top_cost_spans]

    truncated = total > TRACE_SPAN_CAP
    if truncated:
        # Keep the tree navigable: always include the costliest spans (so the
        # "jump to costliest" badges resolve), plus the head of the trace up to
        # the cap. Ordering is otherwise the DB's (chronological), which is what
        # the waterfall expects.
        keep_ids = set(top_cost_span_ids)
        capped: list = []
        for s in spans:
            if len(capped) >= TRACE_SPAN_CAP and s.span_id not in keep_ids:
                continue
            capped.append(s)
        spans = capped

    return {
        "trace_id": trace_id,
        # FULL attributes by default (complete-span consumers / the shim);
        # attribute-free ONLY when the caller opts in with ?attributes=false
        # (the Lens waterfall). See the docstring above (#653).
        "spans": [_span_to_dict(s, include_attributes=attributes) for s in spans],
        # Whether this response carries per-span `attributes` at all — the light
        # (waterfall) payload does not; the default (full) payload does.
        "attributes_included": attributes,
        # True total, always — even when the returned list is capped.
        "span_count": total,
        "returned_count": len(spans),
        "truncated": truncated,
        "span_cap": TRACE_SPAN_CAP,
        "top_cost_span_ids": top_cost_span_ids,
        "framing": _traces_framing(request, agent_id),
    }


@router.get("/traces/{trace_id}/spans/{span_id}")
async def get_trace_span(request: Request, trace_id: str, span_id: str) -> dict:
    """Single span WITH its full `attributes` (#653).

    The waterfall payload is deliberately attribute-free so a 45k-span trace
    doesn't ship ~1 GB of captured prompt/tool content. The span-detail panel
    still shows captured content — it just fetches the one selected span's
    attributes lazily through here, so the cost is paid per-expand for one span
    rather than upfront for all of them.
    """
    db = request.app.state.db
    # Targeted single-span fetch (#653): a WHERE span_id=? lookup, NOT the whole
    # trace. Loading + deserializing every captured attribute of a 45k-span trace
    # to pick one is O(trace) per expand and re-runs on every re-selection, which
    # kept the "lazy" detail path slow and memory-heavy — the exact thing the
    # attribute-free waterfall was supposed to avoid.
    span = db.get_span(trace_id, span_id) if hasattr(db, "get_span") else next(
        (s for s in db.get_trace_spans(trace_id) if s.span_id == span_id), None
    )
    if span is None:
        raise HTTPException(status_code=404, detail="span not found in trace")
    return _span_to_dict(span, include_attributes=True)


def _span_to_dict(span: object, include_attributes: bool = False) -> dict:
    """Serialise a NormalizedSpan to a JSON-safe dict.

    `attributes` (captured prompt/completion/tool content, potentially many KB
    per span) is included ONLY when `include_attributes` is set — the waterfall
    payload omits it (#653) and the detail panel fetches it lazily per span.
    """
    from tokenjam.core.models import NormalizedSpan
    assert isinstance(span, NormalizedSpan)
    out: dict = {
        "span_id": span.span_id,
        "trace_id": span.trace_id,
        "parent_span_id": span.parent_span_id,
        "name": span.name,
        "kind": span.kind.value,
        "status_code": span.status_code.value,
        "status_message": span.status_message,
        "start_time": span.start_time.isoformat() if span.start_time else None,
        "end_time": span.end_time.isoformat() if span.end_time else None,
        "duration_ms": span.duration_ms,
        "agent_id": span.agent_id,
        "session_id": span.session_id,
        "provider": span.provider,
        "model": span.model,
        "tool_name": span.tool_name,
        "input_tokens": span.input_tokens,
        "output_tokens": span.output_tokens,
        "cache_tokens": span.cache_tokens,            # cache-READ tokens
        "cache_write_tokens": span.cache_write_tokens,  # cache-CREATE tokens
        "cost_usd": span.cost_usd,
        "request_type": span.request_type,
        "conversation_id": span.conversation_id,
        # Cheap boolean so the UI knows whether a lazy attributes-fetch is worth
        # making for this span (no attributes → don't show the fetch affordance).
        "has_attributes": bool(span.attributes),
    }
    if include_attributes:
        out["attributes"] = span.attributes
    return out
