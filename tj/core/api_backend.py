"""
API-based backend for CLI commands when DuckDB is locked by tj serve.
Implements the subset of StorageBackend used by CLI commands by routing
queries through the REST API.
"""
from __future__ import annotations

from datetime import date, datetime

import httpx

from tj.core.models import (
    Alert,
    AlertFilters,
    AlertType,
    CostFilters,
    CostRow,
    NormalizedSpan,
    Severity,
    SessionRecord,
    SpanKind,
    SpanStatus,
    TraceFilters,
    TraceRecord,
)
from tj.utils.time_parse import utcnow


class ApiBackend:
    """Read-only backend that queries tj serve instead of DuckDB."""

    def __init__(self, base_url: str, api_key: str | None = None) -> None:
        self.base_url = base_url.rstrip("/")
        headers: dict[str, str] = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self.client = httpx.Client(base_url=self.base_url, headers=headers, timeout=10)

    def _get(self, path: str, params: dict | None = None) -> dict:
        resp = self.client.get(path, params=params)
        resp.raise_for_status()
        return resp.json()

    def get_traces(self, filters: TraceFilters) -> list[TraceRecord]:
        params: dict[str, str | int] = {"limit": filters.limit, "offset": filters.offset}
        if filters.agent_id:
            params["agent_id"] = filters.agent_id
        if filters.since:
            params["since"] = filters.since.isoformat()
        if filters.until:
            params["until"] = filters.until.isoformat()
        if filters.status:
            params["status"] = filters.status
        if filters.span_name:
            params["span_name"] = filters.span_name
        data = self._get("/api/v1/traces", params)
        return [
            TraceRecord(
                trace_id=t["trace_id"],
                agent_id=t["agent_id"],
                name=t["name"],
                start_time=datetime.fromisoformat(t["start_time"]) if t.get("start_time") else None,
                duration_ms=t.get("duration_ms"),
                cost_usd=t.get("cost_usd"),
                status_code=t.get("status_code", "ok"),
                span_count=t.get("span_count", 0),
            )
            for t in data.get("traces", [])
        ]

    def get_trace_spans(self, trace_id: str) -> list[NormalizedSpan]:
        data = self._get(f"/api/v1/traces/{trace_id}")
        return [_dict_to_span(s) for s in data.get("spans", [])]

    def get_cost_summary(self, filters: CostFilters) -> list[CostRow]:
        params: dict[str, str] = {}
        if filters.agent_id:
            params["agent_id"] = filters.agent_id
        if filters.since:
            params["since"] = filters.since.isoformat()
        if filters.until:
            params["until"] = filters.until.isoformat()
        if filters.group_by:
            params["group_by"] = filters.group_by
        data = self._get("/api/v1/cost", params)
        return [
            CostRow(
                group=r["group"],
                agent_id=r["agent_id"],
                model=r["model"],
                input_tokens=r.get("input_tokens", 0),
                output_tokens=r.get("output_tokens", 0),
                cost_usd=r.get("cost_usd", 0.0),
            )
            for r in data.get("rows", [])
        ]

    def get_alerts(self, filters: AlertFilters) -> list[Alert]:
        params: dict[str, str | int | bool] = {}
        if filters.agent_id:
            params["agent_id"] = filters.agent_id
        if filters.since:
            params["since"] = filters.since.isoformat()
        if filters.severity:
            params["severity"] = filters.severity.value
        if filters.type:
            params["type"] = filters.type.value
        if filters.unread:
            params["unread"] = True
        data = self._get("/api/v1/alerts", params)
        return [
            Alert(
                alert_id=a["alert_id"],
                fired_at=datetime.fromisoformat(a["fired_at"]),
                type=AlertType(a["type"]),
                severity=Severity(a["severity"]),
                title=a["title"],
                detail=a.get("detail", {}),
                agent_id=a.get("agent_id"),
                session_id=a.get("session_id"),
                span_id=a.get("span_id"),
                acknowledged=a.get("acknowledged", False),
                suppressed=a.get("suppressed", False),
            )
            for a in data.get("alerts", [])
        ]

    def get_tool_calls(
        self, agent_id: str | None, since: datetime | None, tool_name: str | None,
    ) -> list[dict]:
        params: dict[str, str] = {}
        if agent_id:
            params["agent_id"] = agent_id
        if since:
            params["since"] = since.isoformat()
        if tool_name:
            params["tool_name"] = tool_name
        data = self._get("/api/v1/tools", params)
        return data.get("tools", [])

    def get_daily_cost(self, agent_id: str, day: date) -> float:
        params: dict[str, str] = {
            "agent_id": agent_id,
            "since": datetime(day.year, day.month, day.day).isoformat(),
            "group_by": "day",
        }
        data = self._get("/api/v1/cost", params)
        return data.get("total_cost_usd", 0.0)

    def get_completed_sessions(self, agent_id: str, limit: int) -> list[SessionRecord]:
        # Use /api/v1/status which already returns the latest session per agent
        # with token counts and tool_call_count populated. Without this,
        # `tj status` over a running server shows every agent as "idle" with
        # zeros even when sessions exist (U3).
        try:
            data = self._get("/api/v1/status", {"agent_id": agent_id})
        except (httpx.HTTPError, ValueError):
            return []
        agents = data.get("agents", [])
        if not agents:
            return []
        a = agents[0]
        if not a.get("session_id"):
            return []
        started_at = a.get("started_at")
        return [SessionRecord(
            session_id=a["session_id"],
            agent_id=a.get("agent_id") or agent_id,
            started_at=datetime.fromisoformat(started_at) if started_at else utcnow(),
            ended_at=None,
            conversation_id=None,
            status=a.get("status", "completed"),
            total_cost_usd=a.get("total_cost_usd"),
            input_tokens=a.get("input_tokens", 0) or 0,
            output_tokens=a.get("output_tokens", 0) or 0,
            cache_tokens=0,
            tool_call_count=a.get("tool_call_count", 0) or 0,
            error_count=a.get("error_count", 0) or 0,
        )][:limit]

    def get_completed_session_count(self, agent_id: str) -> int:
        return 0

    def get_session_cost(self, session_id: str) -> float:
        return 0.0

    def get_recent_spans(self, session_id: str, limit: int) -> list[NormalizedSpan]:
        return []

    def close(self) -> None:
        self.client.close()


def _dict_to_span(d: dict) -> NormalizedSpan:
    return NormalizedSpan(
        span_id=d["span_id"],
        trace_id=d["trace_id"],
        name=d["name"],
        kind=SpanKind(d.get("kind", "internal")),
        status_code=SpanStatus(d.get("status_code", "ok")),
        start_time=datetime.fromisoformat(d["start_time"]) if d.get("start_time") else None,
        parent_span_id=d.get("parent_span_id"),
        session_id=d.get("session_id"),
        agent_id=d.get("agent_id"),
        end_time=datetime.fromisoformat(d["end_time"]) if d.get("end_time") else None,
        duration_ms=d.get("duration_ms"),
        status_message=d.get("status_message"),
        attributes=d.get("attributes", {}),
        events=d.get("events", []),
        provider=d.get("provider"),
        model=d.get("model"),
        tool_name=d.get("tool_name"),
        input_tokens=d.get("input_tokens"),
        output_tokens=d.get("output_tokens"),
        cache_tokens=d.get("cache_tokens"),
        cost_usd=d.get("cost_usd"),
        request_type=d.get("request_type"),
        conversation_id=d.get("conversation_id"),
    )


def probe_api(host: str, port: int, api_key: str | None = None) -> ApiBackend | None:
    """Check if tj serve is running and return an ApiBackend if so."""
    base_url = f"http://{host}:{port}"
    try:
        headers: dict[str, str] = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        resp = httpx.get(f"{base_url}/api/v1/traces", params={"limit": 1},
                         headers=headers, timeout=2)
        if resp.status_code in (200, 401):
            return ApiBackend(base_url, api_key)
    except (httpx.ConnectError, httpx.TimeoutException):
        pass
    return None
