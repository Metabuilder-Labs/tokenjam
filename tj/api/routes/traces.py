"""GET /api/v1/traces — trace listing and detail."""
from __future__ import annotations


from fastapi import APIRouter, Depends, Request

from tj.api.deps import require_api_key
from tj.core.models import TraceFilters
from tj.utils.time_parse import parse_since

router = APIRouter(dependencies=[Depends(require_api_key)])


@router.get("/traces")
async def list_traces(
    request: Request,
    agent_id: str | None = None,
    since: str | None = None,
    until: str | None = None,
    limit: int = 50,
    offset: int = 0,
    status: str | None = None,
    span_name: str | None = None,
) -> dict:
    db = request.app.state.db
    filters = TraceFilters(
        agent_id=agent_id,
        since=parse_since(since) if since else None,
        until=parse_since(until) if until else None,
        limit=limit,
        offset=offset,
        status=status,
        span_name=span_name,
    )
    traces = db.get_traces(filters)
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
            }
            for t in traces
        ],
        "count": len(traces),
    }


@router.get("/traces/{trace_id}")
async def get_trace(request: Request, trace_id: str) -> dict:
    db = request.app.state.db
    spans = db.get_trace_spans(trace_id)
    return {
        "trace_id": trace_id,
        "spans": [_span_to_dict(s) for s in spans],
        "span_count": len(spans),
    }


def _span_to_dict(span: object) -> dict:
    """Serialise a NormalizedSpan to a JSON-safe dict."""
    from tj.core.models import NormalizedSpan
    assert isinstance(span, NormalizedSpan)
    return {
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
        "cache_tokens": span.cache_tokens,
        "cost_usd": span.cost_usd,
        "request_type": span.request_type,
        "conversation_id": span.conversation_id,
        "attributes": span.attributes,
    }
