"""
API-based backend for CLI commands when DuckDB is locked by tj serve.
Implements the subset of StorageBackend used by CLI commands by routing
queries through the REST API.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from typing import Any

import httpx

from tokenjam.core.models import (
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
from tokenjam.utils.time_parse import utcnow


class ApiBackend:
    """Read-only backend that queries tj serve instead of DuckDB."""

    #: Blanket read timeout for the cheap shim reads (traces/cost/status/…).
    #: These are near-instant DB lookups the daemon serves in well under a
    #: second, so a tight ceiling keeps a wedged daemon from hanging the CLI.
    _DEFAULT_TIMEOUT = 10

    #: Read timeout for the heavy *computed* endpoints (quota-audit, context).
    #: These re-aggregate a user's whole history server-side, so on a large DB
    #: they can legitimately take longer than the cheap-read ceiling — e.g. a
    #: 3,424-session / 150k-turn history computes the Opus quota audit in ~13s.
    #: Under the blanket 10s ceiling that raised ReadTimeout and the very next
    #: step onboarding advertises (`tj quota-audit`) failed deterministically
    #: for exactly the large-history users tj most wants. A generous override
    #: absorbs that compute while still bounding a genuinely stuck daemon.
    _HEAVY_ENDPOINT_TIMEOUT = 60

    def __init__(self, base_url: str, api_key: str | None = None) -> None:
        self.base_url = base_url.rstrip("/")
        headers: dict[str, str] = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self.client = httpx.Client(
            base_url=self.base_url, headers=headers, timeout=self._DEFAULT_TIMEOUT
        )

    def _get(
        self, path: str, params: dict | None = None, *, timeout: float | None = None
    ) -> dict:
        """GET `path` and return the parsed JSON body.

        `timeout` overrides the client's blanket read timeout for a single
        request — used by the heavy computed endpoints whose server-side
        aggregation can outlast the cheap-read ceiling on large histories.
        Cheap shim reads pass nothing and keep the tight default.
        """
        if timeout is None:
            resp = self.client.get(path, params=params)
        else:
            resp = self.client.get(path, params=params, timeout=timeout)
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
                # Cache fields added in #149; the API serializes them but this
                # shim was missed, so `tj cost` silently showed 0s in the cache
                # columns whenever the daemon was up. Mirror the wider contract.
                cache_tokens=r.get("cache_tokens", 0),
                cache_write_tokens=r.get("cache_write_tokens", 0),
                cost_usd=r.get("cost_usd", 0.0),
            )
            for r in data.get("rows", [])
        ]

    def fetch_cost_framing(
        self, *, since: str = "7d", agent_id: str | None = None,
    ) -> dict | None:
        """Return the plan-tier `framing` block from /api/v1/cost.

        `tj cost` renders the COST column plan-tier-aware (#175). When the
        daemon holds the DB lock the CLI can't compute framing locally, so it
        reuses the block the API already emits (`Framing.to_dict()`); the CLI
        reconstructs a `Framing` from it. Returns None if the response carries
        no framing.
        """
        params: dict[str, str] = {"since": since}
        if agent_id:
            params["agent_id"] = agent_id
        data = self._get("/api/v1/cost", params)
        return data.get("framing")

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
        start = datetime(day.year, day.month, day.day, tzinfo=timezone.utc)
        end = start + timedelta(days=1)
        params: dict[str, str] = {
            "agent_id": agent_id,
            "since": start.isoformat(),
            "until": end.isoformat(),
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
            cache_write_tokens=0,
            tool_call_count=a.get("tool_call_count", 0) or 0,
            error_count=a.get("error_count", 0) or 0,
        )][:limit]

    def get_completed_session_count(self, agent_id: str) -> int:
        return 0

    def get_session_cost(self, session_id: str) -> float:
        return 0.0

    def get_recent_spans(self, session_id: str, limit: int) -> list[NormalizedSpan]:
        return []

    def get_baseline(self, agent_id: str):
        """
        Fetch a drift baseline for a single agent via /api/v1/drift?agent_id=X.

        Returns DriftBaseline or None. Mirrors the StorageBackend method so
        cmd_drift can call it transparently in API-shim mode (#68 §3).
        """
        from tokenjam.core.models import DriftBaseline
        try:
            data = self._get("/api/v1/drift", {"agent_id": agent_id})
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                return None
            raise
        b = data.get("baseline")
        if not b:
            return None
        return DriftBaseline(
            agent_id=agent_id,
            sessions_sampled=int(b.get("sessions_sampled", 0)),
            computed_at=(
                datetime.fromisoformat(b["computed_at"])
                if b.get("computed_at") else utcnow()
            ),
            avg_input_tokens=b.get("avg_input_tokens"),
            stddev_input_tokens=b.get("stddev_input_tokens"),
            avg_output_tokens=b.get("avg_output_tokens"),
            stddev_output_tokens=b.get("stddev_output_tokens"),
            avg_session_duration_s=b.get("avg_session_duration_s"),
            stddev_session_duration=b.get("stddev_session_duration"),
            avg_tool_call_count=b.get("avg_tool_call_count"),
            stddev_tool_call_count=b.get("stddev_tool_call_count"),
        )

    def list_baseline_agents(self) -> list[str]:
        """
        Enumerate agent IDs that have a drift baseline. Hits /api/v1/drift
        (no agent_id) which returns {"agents": [...]} for every baseline.

        Used by cmd_drift to discover agents under API-shim mode (#68 §3).
        """
        try:
            data = self._get("/api/v1/drift")
        except Exception:
            return []
        return [
            a["agent_id"] for a in data.get("agents", [])
            if a.get("agent_id")
        ]

    def fetch_cost_compare(
        self,
        *,
        since: str = "7d",
        compare: str = "previous",
        agent_id: str | None = None,
        top_n: int = 5,
    ) -> dict:
        """
        Fetch a window-vs-window cost diff from tj serve. Mirrors
        compute_cost_diff's output schema; used by cmd_cost and cmd_optimize
        when the daemon holds the DB lock (#68 §12 follow-up).
        """
        params: dict[str, Any] = {
            "since": since,
            "compare": compare,
            "top_n": top_n,
        }
        if agent_id:
            params["agent_id"] = agent_id
        return self._get(
            "/api/v1/cost/compare", params, timeout=self._HEAVY_ENDPOINT_TIMEOUT
        )

    def fetch_optimize_report(
        self,
        *,
        since: str = "30d",
        agent_id: str | None = None,
        findings: list[str] | None = None,
        budget_provider: str | None = None,
        budget_usd: float | None = None,
    ) -> dict:
        """
        Fetch a serialized optimize report from `tj serve`.

        Used by cmd_optimize when the local DuckDB connection is unavailable
        (daemon holds the write lock). Returns the dict that `report_to_dict`
        produced server-side; the CLI passes it through `report_from_dict`
        before rendering.

        See issue #68 §12 for the rationale.
        """
        params: dict[str, Any] = {"since": since}
        if agent_id:
            params["agent_id"] = agent_id
        if findings:
            # FastAPI accepts repeated query params for list values.
            params["finding"] = findings
        if budget_provider:
            params["budget_provider"] = budget_provider
        if budget_usd is not None:
            params["budget_usd"] = budget_usd
        return self._get(
            "/api/v1/optimize", params, timeout=self._HEAVY_ENDPOINT_TIMEOUT
        )

    def fetch_reuse_clusters(
        self,
        *,
        since: str = "30d",
        agent_id: str | None = None,
    ) -> dict:
        """
        Fetch the Reuse finding + skeleton-rendering data from `tj serve`.

        Used by `tj report --reuse` when the local DuckDB connection is
        unavailable (daemon holds the write lock). The response is
        `report_to_dict(report)` plus `planning_texts` ({session_id: completion
        text or null}) and `pricing_mode`; the CLI reconstructs the finding via
        `report_from_dict` and renders without direct DB access. See issue #154.
        """
        params: dict[str, Any] = {"since": since}
        if agent_id:
            params["agent_id"] = agent_id
        return self._get(
            "/api/v1/reuse/clusters", params, timeout=self._HEAVY_ENDPOINT_TIMEOUT
        )

    #: `/sessions` page size for `find_last_substantial_session`. The floor is
    #: usually 1 tool call, so the newest row almost always qualifies — this
    #: cap just needs to survive a handful of non-substantial rows at the head
    #: of the list without pulling a user's entire (possibly thousands-row)
    #: sessions table over HTTP just to inspect the first few (Greptile, PR #448).
    _LAST_SESSION_PAGE_SIZE = 25

    def find_last_substantial_session(
        self, min_tool_calls: int = 1
    ) -> str | None:
        """Most-recent session id with at least `min_tool_calls` tool calls.

        Mirrors the direct-DB `--last` resolution in `tj session-story` when the
        daemon holds the write lock. `/sessions` already returns rows newest
        first, so the first qualifying row within the fetched page is the
        answer. Fetches only `_LAST_SESSION_PAGE_SIZE` rows via `/sessions`'
        `limit` param, not the whole table. Returns None when no session in
        that page clears the floor (or the list is empty) — a session further
        back than the page never gets picked as "last", which is the honest
        trade-off for not transferring a user's entire session history.
        """
        data = self._get(
            "/api/v1/sessions", {"limit": self._LAST_SESSION_PAGE_SIZE}
        )
        for s in data.get("sessions", []):
            if (s.get("tool_call_count") or 0) >= min_tool_calls:
                sid = s.get("session_id")
                if sid:
                    return sid
        return None

    def fetch_session_story(
        self, session_id: str, subagents: bool = True
    ) -> dict:
        """Fetch a session's reconstructed story from `tj serve`.

        Used by `tj session-story` when the local DuckDB connection is
        unavailable (daemon holds the write lock). The daemon already did the
        transcript reconstruction + snapshot fallback, so this returns the raw
        `{"available": bool, ...}` payload (`available`/`reason`/`from_snapshot`
        + the story fields) verbatim for the CLI to render. See issue #63 for
        the same daemon-availability pattern behind `tj context`.
        """
        return self._get(
            f"/api/v1/sessions/{session_id}/story",
            {"subagents": subagents},
        )

    def fetch_context_diagnostic(
        self,
        *,
        since: str = "30d",
        agent_id: str | None = None,
    ) -> dict:
        """
        Fetch the server-computed context-cost diagnostic from `tj serve`.

        Used by `tj context` when the local DuckDB connection is unavailable
        (the daemon holds the write lock — the launch-hero availability gap,
        #63). The diagnostic reads the raw `attributes` column, which the shim
        can't expose, so the daemon (which owns the direct connection) computes
        it and returns `diagnostic_to_dict(diag)` plus the `framing` block. The
        CLI reconstructs via `diagnostic_from_dict` + `Framing(**framing)`.
        """
        params: dict[str, Any] = {"since": since}
        if agent_id:
            params["agent_id"] = agent_id
        return self._get(
            "/api/v1/context", params, timeout=self._HEAVY_ENDPOINT_TIMEOUT
        )

    def fetch_opus_quota_audit(
        self,
        *,
        since: str = "30d",
        agent_id: str | None = None,
    ) -> dict:
        """
        Fetch the server-computed Opus quota audit from `tj serve`.

        Used by `tj quota-audit` when the local DuckDB connection is unavailable
        (the daemon holds the write lock). The audit aggregates per-session
        token/model metadata the shim can't expose at this grain, so the daemon
        (which owns the direct connection) computes it and returns
        `audit_to_dict(audit)` plus the `framing` block. The CLI reconstructs via
        `audit_from_dict` + `Framing(**framing)`. Mirrors
        `fetch_context_diagnostic` / `/api/v1/context`.
        """
        params: dict[str, Any] = {"since": since}
        if agent_id:
            params["agent_id"] = agent_id
        return self._get(
            "/api/v1/quota-audit", params, timeout=self._HEAVY_ENDPOINT_TIMEOUT
        )

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
        cache_write_tokens=d.get("cache_write_tokens"),
        cost_usd=d.get("cost_usd"),
        request_type=d.get("request_type"),
        conversation_id=d.get("conversation_id"),
        billing_account=d.get("billing_account"),
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
