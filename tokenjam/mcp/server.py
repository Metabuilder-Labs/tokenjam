"""TokenJam MCP server — exposes observability data to Claude Code via stdio."""
from __future__ import annotations

try:
    from fastmcp import FastMCP
except ImportError as _import_err:  # pragma: no cover - defensive
    # fastmcp moved into the base dependencies in v0.3.5 (issue #101). This
    # fallback only triggers if a user manually uninstalled fastmcp or
    # installed via an unusual route that excluded base deps. Surface a
    # clean error pointing at the fix instead of the raw ImportError that
    # Claude Code surfaces as "MCP server failed to start".
    raise ImportError(
        "tokenjam.mcp requires fastmcp, but it is not installed. "
        "Reinstall TokenJam with `pipx install --force tokenjam` "
        "(or `pip install --upgrade tokenjam` in a venv) — fastmcp is "
        "shipped in the base install as of v0.3.5."
    ) from _import_err

mcp = FastMCP("tj")

# Module-level state initialised by init() or cmd_mcp.py
_ro_conn = None       # duckdb read-only connection
_config = None        # TjConfig
_ro_db = None         # _ReadOnlyDB or _HttpDB
_serve_url: str | None = None  # base URL for tj serve HTTP API when DuckDB is locked


def init(ro_conn, config, serve_url: str | None = None) -> None:
    """Inject DB connection and config. Called by cmd_mcp.py and tests."""
    global _ro_conn, _config, _ro_db, _serve_url
    _ro_conn, _config = ro_conn, config
    _serve_url = serve_url
    if ro_conn is not None:
        _ro_db = _ReadOnlyDB(ro_conn)
    elif serve_url is not None:
        _ro_db = _HttpDB()
    else:
        _ro_db = None


def _no_config() -> dict:
    return {
        "error": (
            "No TokenJam config found. "
            "Run 'tj onboard --claude-code' (Claude Code) "
            "or 'tj onboard --codex' (Codex CLI) to set up."
        )
    }


def _http_get(path: str, params: dict | None = None) -> dict:
    """Issue a GET request to the tj serve HTTP API. Uses module-level _serve_url."""
    import json
    import urllib.parse
    import urllib.request

    url = f"{_serve_url}{path}"
    if params:
        filtered = {k: str(v) for k, v in params.items() if v is not None}
        if filtered:
            url += "?" + urllib.parse.urlencode(filtered)
    req = urllib.request.Request(url)
    if _config and _config.api.auth.enabled:
        req.add_header("Authorization", f"Bearer {_config.api.auth.api_key}")
    with urllib.request.urlopen(req, timeout=5) as resp:
        return json.loads(resp.read())


class _HttpDB:
    """DB-like object that proxies reads to a running tj serve HTTP API."""

    @property
    def conn(self) -> None:
        return None

    def get_cost_summary(self, filters):
        try:
            from types import SimpleNamespace
            params = {}
            if filters.agent_id:
                params["agent_id"] = filters.agent_id
            if filters.since:
                params["since"] = filters.since.isoformat()
            if filters.group_by:
                params["group_by"] = filters.group_by
            data = _http_get("/api/v1/cost", params)
            return [
                SimpleNamespace(
                    group=r.get("group"),
                    agent_id=r.get("agent_id"),
                    model=r.get("model"),
                    input_tokens=r.get("input_tokens", 0),
                    output_tokens=r.get("output_tokens", 0),
                    cost_usd=r.get("cost_usd", 0.0),
                )
                for r in data.get("rows", [])
            ]
        except Exception:
            return []

    def get_alerts(self, filters):
        try:
            from datetime import datetime
            from types import SimpleNamespace
            params = {}
            if filters.agent_id:
                params["agent_id"] = filters.agent_id
            if filters.severity:
                params["severity"] = filters.severity.value
            if filters.unread:
                params["unread"] = "true"
            data = _http_get("/api/v1/alerts", params)
            results = []
            for a in data.get("alerts", []):
                results.append(SimpleNamespace(
                    alert_id=a["alert_id"],
                    fired_at=datetime.fromisoformat(a["fired_at"]),
                    type=SimpleNamespace(value=a["type"]),
                    severity=SimpleNamespace(value=a["severity"]),
                    title=a["title"],
                    agent_id=a["agent_id"],
                    acknowledged=a["acknowledged"],
                    suppressed=a["suppressed"],
                ))
            return results
        except Exception:
            return []

    def get_traces(self, filters):
        try:
            from datetime import datetime
            from types import SimpleNamespace
            params = {}
            if filters.agent_id:
                params["agent_id"] = filters.agent_id
            if filters.since:
                params["since"] = filters.since.isoformat()
            if filters.limit:
                params["limit"] = filters.limit
            data = _http_get("/api/v1/traces", params)
            results = []
            for t in data.get("traces", []):
                results.append(SimpleNamespace(
                    trace_id=t["trace_id"],
                    agent_id=t.get("agent_id"),
                    name=t.get("name"),
                    start_time=datetime.fromisoformat(t["start_time"]) if t.get("start_time") else None,
                    duration_ms=t.get("duration_ms"),
                    cost_usd=t.get("cost_usd"),
                    status_code=t.get("status_code"),
                    span_count=t.get("span_count", 0),
                ))
            return results
        except Exception:
            return []

    def get_trace_spans(self, trace_id: str):
        try:
            from datetime import datetime
            from types import SimpleNamespace
            data = _http_get(f"/api/v1/traces/{trace_id}")
            results = []
            for s in data.get("spans", []):
                results.append(SimpleNamespace(
                    span_id=s["span_id"],
                    parent_span_id=s.get("parent_span_id"),
                    name=s.get("name"),
                    kind=SimpleNamespace(value=s.get("kind", "")),
                    status_code=SimpleNamespace(value=s.get("status_code", "")),
                    start_time=datetime.fromisoformat(s["start_time"]) if s.get("start_time") else None,
                    end_time=datetime.fromisoformat(s["end_time"]) if s.get("end_time") else None,
                    duration_ms=s.get("duration_ms"),
                    provider=s.get("provider"),
                    model=s.get("model"),
                    tool_name=s.get("tool_name"),
                    input_tokens=s.get("input_tokens"),
                    output_tokens=s.get("output_tokens"),
                    cost_usd=s.get("cost_usd"),
                ))
            return results
        except Exception:
            return []

    def get_tool_calls(self, agent_id, since, tool_name):
        try:
            params = {}
            if agent_id:
                params["agent_id"] = agent_id
            if since:
                params["since"] = since.isoformat()
            if tool_name:
                params["tool_name"] = tool_name
            data = _http_get("/api/v1/tools", params)
            return data.get("tools", [])
        except Exception:
            return []

    def get_baseline(self, agent_id: str):
        from datetime import datetime
        from types import SimpleNamespace
        try:
            data = _http_get("/api/v1/drift", {"agent_id": agent_id})
        except Exception:
            return None
        if "error" in data or data.get("baseline") is None:
            return None
        b = data["baseline"]
        computed_at = datetime.fromisoformat(b["computed_at"]) if b.get("computed_at") else None
        return SimpleNamespace(
            sessions_sampled=b.get("sessions_sampled"),
            computed_at=computed_at,
            avg_input_tokens=b.get("avg_input_tokens"),
            stddev_input_tokens=b.get("stddev_input_tokens"),
            avg_output_tokens=b.get("avg_output_tokens"),
            stddev_output_tokens=b.get("stddev_output_tokens"),
            avg_session_duration_s=b.get("avg_session_duration_s"),
            avg_tool_call_count=b.get("avg_tool_call_count"),
        )

    def get_completed_sessions(self, agent_id: str, limit: int):
        from types import SimpleNamespace
        try:
            data = _http_get("/api/v1/status", {"agent_id": agent_id})
        except Exception:
            return []
        agents = data.get("agents", [])
        results = []
        for a in agents[:limit]:
            results.append(SimpleNamespace(
                session_id=a.get("session_id"),
                input_tokens=a.get("input_tokens", 0),
                output_tokens=a.get("output_tokens", 0),
                tool_call_count=a.get("tool_call_count", 0),
                duration_seconds=a.get("duration_seconds"),
            ))
        return results


class _ReadOnlyDB:
    """Wraps a read-only duckdb connection to satisfy StorageBackend protocol methods."""
    def __init__(self, conn):
        self.conn = conn

    def get_cost_summary(self, filters):
        from tokenjam.core.db import DuckDBBackend
        return DuckDBBackend.get_cost_summary(self, filters)

    def get_alerts(self, filters):
        from tokenjam.core.db import DuckDBBackend
        return DuckDBBackend.get_alerts(self, filters)

    def get_traces(self, filters):
        from tokenjam.core.db import DuckDBBackend
        return DuckDBBackend.get_traces(self, filters)

    def get_trace_spans(self, trace_id):
        from tokenjam.core.db import DuckDBBackend
        return DuckDBBackend.get_trace_spans(self, trace_id)

    def get_tool_calls(self, agent_id, since, tool_name):
        from tokenjam.core.db import DuckDBBackend
        return DuckDBBackend.get_tool_calls(self, agent_id, since, tool_name)

    def get_baseline(self, agent_id):
        from tokenjam.core.db import DuckDBBackend
        return DuckDBBackend.get_baseline(self, agent_id)

    def get_completed_sessions(self, agent_id, limit):
        from tokenjam.core.db import DuckDBBackend
        return DuckDBBackend.get_completed_sessions(self, agent_id, limit)

    def _decision_where(self, filters):
        from tokenjam.core.db import DuckDBBackend
        return DuckDBBackend._decision_where(self, filters)

    def get_policy_decisions(self, filters):
        from tokenjam.core.db import DuckDBBackend
        return DuckDBBackend.get_policy_decisions(self, filters)

    def get_savings_entries(self, filters):
        from tokenjam.core.db import DuckDBBackend
        return DuckDBBackend.get_savings_entries(self, filters)


# ---------------------------------------------------------------------------
# Handler functions — called by @mcp.tool() wrappers and directly in tests
# ---------------------------------------------------------------------------

def _tool_get_status(conn, config, agent_id: str | None = None) -> dict:
    if config is None:
        return _no_config()

    # HTTP mode: proxy to tj serve
    if conn is None:
        if _serve_url is None:
            return {"error": "No database connection and no serve URL configured."}
        params: dict = {}
        if agent_id:
            params["agent_id"] = agent_id
        data = _http_get("/api/v1/status", params)
        # If filtering by a single agent_id, extract that agent's record from the list
        if agent_id:
            agents = data.get("agents", [])
            matching = [a for a in agents if a.get("agent_id") == agent_id]
            if matching:
                raw = dict(matching[0])
                raw["cost_today_usd"] = raw.pop("cost_today", raw.get("cost_today_usd", 0.0))
                return raw
            return {
                "agent_id": agent_id,
                "session_id": None,
                "status": "idle",
                "input_tokens": 0,
                "output_tokens": 0,
                "tool_call_count": 0,
                "error_count": 0,
                "cost_today_usd": 0.0,
                "active_alerts": 0,
            }
        agents = []
        for a in data.get("agents", []):
            a = dict(a)
            a["cost_today_usd"] = a.pop("cost_today", a.get("cost_today_usd", 0.0))
            agents.append(a)
        return {"agents": agents}

    from tokenjam.utils.time_parse import utcnow

    if agent_id:
        aids = [agent_id]
    else:
        rows = conn.execute(
            "SELECT DISTINCT agent_id FROM sessions ORDER BY agent_id"
        ).fetchall()
        aids = [r[0] for r in rows]

    results = []
    for aid in aids:
        row = conn.execute(
            "SELECT session_id, status, started_at, ended_at, input_tokens, "
            "output_tokens, tool_call_count, error_count, total_cost_usd "
            "FROM sessions WHERE agent_id = $1 AND status = 'active' "
            "ORDER BY started_at DESC LIMIT 1",
            [aid],
        ).fetchone()
        if row is None:
            row = conn.execute(
                "SELECT session_id, status, started_at, ended_at, input_tokens, "
                "output_tokens, tool_call_count, error_count, total_cost_usd "
                "FROM sessions WHERE agent_id = $1 "
                "ORDER BY started_at DESC LIMIT 1",
                [aid],
            ).fetchone()

        active_alerts = conn.execute(
            "SELECT COUNT(*) FROM alerts WHERE agent_id = $1 "
            "AND acknowledged = false AND suppressed = false",
            [aid],
        ).fetchone()[0]

        today_cost = conn.execute(
            "SELECT COALESCE(SUM(cost_usd), 0.0) FROM spans "
            "WHERE agent_id = $1 AND CAST(start_time AT TIME ZONE 'UTC' AS DATE) = $2",
            [aid, utcnow().date()],
        ).fetchone()[0]

        if row:
            results.append({
                "agent_id": aid,
                "session_id": row[0],
                "status": row[1],
                "input_tokens": row[4] or 0,
                "output_tokens": row[5] or 0,
                "tool_call_count": row[6] or 0,
                "error_count": row[7] or 0,
                "cost_today_usd": float(today_cost),
                "active_alerts": active_alerts,
            })
        else:
            results.append({
                "agent_id": aid,
                "session_id": None,
                "status": "idle",
                "input_tokens": 0,
                "output_tokens": 0,
                "tool_call_count": 0,
                "error_count": 0,
                "cost_today_usd": float(today_cost),
                "active_alerts": active_alerts,
            })

    if agent_id:
        return results[0] if results else {
            "agent_id": agent_id,
            "session_id": None,
            "status": "idle",
            "input_tokens": 0,
            "output_tokens": 0,
            "tool_call_count": 0,
            "error_count": 0,
            "cost_today_usd": 0.0,
            "active_alerts": 0,
        }
    return {"agents": results}


def _tool_get_budget_headroom(conn, config, agent_id: str) -> dict:
    if config is None:
        return _no_config()
    from tokenjam.core.config import resolve_effective_budget

    # HTTP mode: fetch both limits and spend from the live serve API so budget edits
    # made via the UI are reflected without restarting the MCP server.
    if conn is None:
        if _serve_url is None:
            return {"error": "No database connection and no serve URL configured."}
        budget_data = _http_get("/api/v1/budget")
        agent_budget = budget_data.get("agents", {}).get(agent_id, {})
        effective = agent_budget.get("effective", {})
        daily_limit = effective.get("daily_usd")
        session_limit = effective.get("session_usd")

        status_data = _http_get("/api/v1/status", {"agent_id": agent_id})
        agents = status_data.get("agents", [])
        today_cost = 0.0
        session_cost = 0.0
        if agents:
            a = agents[0]
            today_cost = float(a.get("cost_today", 0.0))
            session_cost = float(a.get("total_cost_usd", 0.0))
        return {
            "agent_id": agent_id,
            "daily_limit_usd": daily_limit,
            "daily_spent_usd": today_cost,
            "daily_remaining_usd": (daily_limit - today_cost) if daily_limit else None,
            "session_limit_usd": session_limit,
            "session_spent_usd": session_cost,
            "session_remaining_usd": (session_limit - session_cost) if session_limit else None,
        }

    budget = resolve_effective_budget(agent_id, config)

    from tokenjam.utils.time_parse import utcnow

    today_cost = float(conn.execute(
        "SELECT COALESCE(SUM(cost_usd), 0.0) FROM spans "
        "WHERE agent_id = $1 AND CAST(start_time AT TIME ZONE 'UTC' AS DATE) = $2",
        [agent_id, utcnow().date()],
    ).fetchone()[0])

    active_session = conn.execute(
        "SELECT COALESCE(total_cost_usd, 0.0) FROM sessions "
        "WHERE agent_id = $1 AND status = 'active' ORDER BY started_at DESC LIMIT 1",
        [agent_id],
    ).fetchone()
    session_cost = float(active_session[0]) if active_session else 0.0

    return {
        "agent_id": agent_id,
        "daily_limit_usd": budget.daily_usd,
        "daily_spent_usd": today_cost,
        "daily_remaining_usd": (budget.daily_usd - today_cost) if budget.daily_usd else None,
        "session_limit_usd": budget.session_usd,
        "session_spent_usd": session_cost,
        "session_remaining_usd": (budget.session_usd - session_cost) if budget.session_usd else None,
    }


def _tool_list_agents(conn) -> dict:
    # HTTP mode: proxy to serve status endpoint
    if conn is None:
        if _serve_url is None:
            return _no_config()
        data = _http_get("/api/v1/agents")
        return {
            "agents": [
                {
                    "agent_id": a["agent_id"],
                    "first_seen": a.get("first_seen"),
                    "last_seen": a.get("last_seen"),
                    "lifetime_cost_usd": float(a.get("lifetime_cost_usd", 0.0)),
                }
                for a in data.get("agents", [])
            ]
        }

    rows = conn.execute(
        "SELECT a.agent_id, a.first_seen, a.last_seen, "
        "COALESCE(SUM(s.cost_usd), 0.0) AS lifetime_cost "
        "FROM agents a LEFT JOIN spans s ON a.agent_id = s.agent_id "
        "GROUP BY a.agent_id, a.first_seen, a.last_seen "
        "ORDER BY a.last_seen DESC"
    ).fetchall()
    return {
        "agents": [
            {
                "agent_id": r[0],
                "first_seen": r[1].isoformat() if r[1] else None,
                "last_seen": r[2].isoformat() if r[2] else None,
                "lifetime_cost_usd": float(r[3]),
            }
            for r in rows
        ]
    }


def _tool_list_active_sessions(conn) -> dict:
    # HTTP mode: proxy to the per-session endpoint so concurrent sessions for
    # the same agent are all surfaced. /api/v1/status collapses to one record
    # per agent and undercounted parallel sessions (#35); /api/v1/sessions
    # returns one row per session, matching the direct-DB path below.
    if conn is None:
        if _serve_url is None:
            return _no_config()
        data = _http_get("/api/v1/sessions", {"status": "active"})
        sessions = [
            {
                "session_id": s.get("session_id"),
                "agent_id": s.get("agent_id"),
                "started_at": s.get("started_at"),
                "total_cost_usd": float(s.get("total_cost_usd", 0.0)),
                "input_tokens": s.get("input_tokens", 0),
                "output_tokens": s.get("output_tokens", 0),
                "tool_call_count": s.get("tool_call_count", 0),
                "error_count": s.get("error_count", 0),
            }
            for s in data.get("sessions", [])
        ]
        return {"sessions": sessions, "count": len(sessions)}

    rows = conn.execute(
        "SELECT session_id, agent_id, started_at, total_cost_usd, "
        "input_tokens, output_tokens, tool_call_count, error_count "
        "FROM sessions WHERE status = 'active' ORDER BY started_at DESC"
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


def _tool_get_cost_summary(
    db, agent_id: str | None, since: str | None, group_by: str
) -> dict:
    from tokenjam.core.models import CostFilters
    from tokenjam.utils.time_parse import parse_since
    filters = CostFilters(
        agent_id=agent_id,
        since=parse_since(since) if since else None,
        group_by=group_by,
    )
    rows = db.get_cost_summary(filters)
    total = sum(r.cost_usd for r in rows)
    return {
        "rows": [
            {
                "group": r.group,
                "agent_id": r.agent_id,
                "model": r.model,
                "input_tokens": r.input_tokens,
                "output_tokens": r.output_tokens,
                "cost_usd": r.cost_usd,
            }
            for r in rows
        ],
        "total_cost_usd": total,
    }


def _tool_list_alerts(
    db, agent_id: str | None, severity: str | None, unread: bool
) -> dict:
    from tokenjam.core.models import AlertFilters, Severity
    filters = AlertFilters(
        agent_id=agent_id,
        severity=Severity(severity) if severity else None,
        unread=unread,
    )
    alerts = db.get_alerts(filters)
    return {
        "alerts": [
            {
                "alert_id": a.alert_id,
                "fired_at": a.fired_at.isoformat(),
                "type": a.type.value,
                "severity": a.severity.value,
                "title": a.title,
                "agent_id": a.agent_id,
                "acknowledged": a.acknowledged,
                "suppressed": a.suppressed,
            }
            for a in alerts
        ],
        "count": len(alerts),
    }


def _tool_list_traces(db, agent_id: str | None, since: str | None, limit: int) -> dict:
    from tokenjam.core.models import TraceFilters
    from tokenjam.utils.time_parse import parse_since
    filters = TraceFilters(
        agent_id=agent_id,
        since=parse_since(since) if since else None,
        limit=limit,
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


def _tool_get_trace(db, trace_id: str) -> dict:
    spans = db.get_trace_spans(trace_id)
    return {
        "trace_id": trace_id,
        "spans": [
            {
                "span_id": s.span_id,
                "parent_span_id": s.parent_span_id,
                "name": s.name,
                "kind": s.kind.value,
                "status_code": s.status_code.value,
                "start_time": s.start_time.isoformat() if s.start_time else None,
                "end_time": s.end_time.isoformat() if s.end_time else None,
                "duration_ms": s.duration_ms,
                "provider": s.provider,
                "model": s.model,
                "tool_name": s.tool_name,
                "input_tokens": s.input_tokens,
                "output_tokens": s.output_tokens,
                "cost_usd": s.cost_usd,
            }
            for s in spans
        ],
        "span_count": len(spans),
    }


def _tool_get_tool_stats(db, agent_id: str | None, since: str | None) -> dict:
    from tokenjam.utils.time_parse import parse_since
    since_dt = parse_since(since) if since else None
    rows = db.get_tool_calls(agent_id, since_dt, None)
    return {"tools": rows, "count": len(rows)}


def _tool_get_drift_report(db, agent_id: str | None) -> dict:
    if agent_id:
        baseline = db.get_baseline(agent_id)
        latest_sessions = db.get_completed_sessions(agent_id, limit=1)
        latest = None
        if latest_sessions:
            s = latest_sessions[0]
            latest = {
                "session_id": s.session_id,
                "input_tokens": s.input_tokens,
                "output_tokens": s.output_tokens,
                "tool_call_count": s.tool_call_count,
                "duration_seconds": s.duration_seconds,
            }
        return {
            "agent_id": agent_id,
            "baseline": {
                "sessions_sampled": baseline.sessions_sampled,
                "computed_at": baseline.computed_at.isoformat() if baseline.computed_at else None,
                "avg_input_tokens": baseline.avg_input_tokens,
                "stddev_input_tokens": baseline.stddev_input_tokens,
                "avg_output_tokens": baseline.avg_output_tokens,
                "stddev_output_tokens": baseline.stddev_output_tokens,
                "avg_session_duration_s": baseline.avg_session_duration_s,
                "avg_tool_call_count": baseline.avg_tool_call_count,
            } if baseline else None,
            "latest_session": latest,
        }
    # All agents with baselines
    if hasattr(db, "conn") and db.conn is not None:
        agents_with_baselines = db.conn.execute(
            "SELECT DISTINCT agent_id FROM drift_baselines ORDER BY agent_id"
        ).fetchall()
        return {
            "agents": [_tool_get_drift_report(db, row[0]) for row in agents_with_baselines]
        }
    elif _serve_url is not None:
        return _http_get("/api/v1/drift")
    return {"agents": []}


def _tool_acknowledge_alert(conn, alert_id: str) -> dict:
    """Update acknowledged flag. conn may be read-write (tests) or short-lived write (prod)."""
    if conn is None:
        return {
            "error": "Use the dashboard to acknowledge alerts while tj serve is running."
        }
    result = conn.execute(
        "SELECT alert_id FROM alerts WHERE alert_id = $1", [alert_id]
    ).fetchone()
    if result is None:
        return {"error": f"Alert {alert_id} not found"}
    conn.execute(
        "UPDATE alerts SET acknowledged = true WHERE alert_id = $1", [alert_id]
    )
    return {"acknowledged": True, "alert_id": alert_id}


def _tool_setup_project(
    config, config_path: str | None, agent_id: str | None, project_path: str | None
) -> dict:
    if config is None or config_path is None:
        return _no_config()
    import json
    import subprocess
    from pathlib import Path
    from tokenjam.core.config import write_config, AgentConfig

    # Derive agent_id if not provided
    if not agent_id:
        try:
            git_proc = subprocess.run(
                ["git", "remote", "get-url", "origin"],
                capture_output=True, text=True, timeout=3,
                cwd=project_path or ".",
            )
            if git_proc.returncode == 0:
                url = git_proc.stdout.strip()
                name = url.rstrip("/").split("/")[-1].split(":")[-1]
                name = name.removesuffix(".git").lower()
                agent_id = f"claude-code-{name}" if name else None
        except Exception:
            pass
        if not agent_id:
            cwd = Path(project_path) if project_path else Path.cwd()
            agent_id = f"claude-code-{cwd.name.lower()}"

    # Write project-level .claude/settings.json
    proj_dir = Path(project_path) if project_path else Path.cwd()
    claude_dir = proj_dir / ".claude"
    claude_dir.mkdir(parents=True, exist_ok=True)
    settings_path = claude_dir / "settings.json"

    existing: dict = {}
    if settings_path.exists():
        try:
            existing = json.loads(settings_path.read_text())
        except (json.JSONDecodeError, OSError):
            existing = {}

    env = existing.get("env", {})
    env["OTEL_RESOURCE_ATTRIBUTES"] = f"service.name={agent_id}"
    existing["env"] = env
    settings_path.write_text(json.dumps(existing, indent=2) + "\n")

    # Add agent entry to TokenJam config
    if agent_id not in config.agents:
        config.agents[agent_id] = AgentConfig()
        write_config(config, Path(config_path))

    # Warn if global OTLP endpoint not configured
    global_settings = Path.home() / ".claude" / "settings.json"
    warning = None
    if global_settings.exists():
        try:
            gs = json.loads(global_settings.read_text())
            if "OTEL_EXPORTER_OTLP_ENDPOINT" not in gs.get("env", {}):
                warning = "Global OTLP endpoint not configured. Run 'tj onboard --claude-code' to finish setup, then run 'claude mcp add tj --scope user -- tj mcp' to register the MCP server."
        except Exception:
            warning = "Could not read ~/.claude/settings.json."
    else:
        warning = "~/.claude/settings.json not found. Run 'tj onboard --claude-code' to configure the global OTLP endpoint and register the MCP server."

    result = {
        "agent_id": agent_id,
        "settings_path": str(settings_path),
    }
    if warning:
        result["warning"] = warning
    return result


def _tool_open_dashboard(config) -> dict:
    """Start tj serve in the background if not running, return the dashboard URL."""
    if config is None:
        return _no_config()
    import socket
    import subprocess
    import sys
    import time

    host = config.api.host
    port = config.api.port
    url = f"http://{host}:{port}/ui"

    # Check if something is already listening on the port
    already_running = False
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        try:
            s.connect((host, port))
            already_running = True
        except (ConnectionRefusedError, OSError):
            pass

    if already_running:
        return {"url": url, "started": False, "message": "tj serve is already running."}

    # Spawn tj serve detached from this process.
    # start_new_session is Unix-only; use DETACHED_PROCESS on Windows instead.
    import shutil as _shutil
    tj_bin = _shutil.which("tj") or sys.argv[0]
    import sys as _sys
    popen_kwargs: dict = {
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
    }
    if _sys.platform == "win32":
        popen_kwargs["creationflags"] = subprocess.DETACHED_PROCESS
    else:
        popen_kwargs["start_new_session"] = True
    try:
        subprocess.Popen([tj_bin, "serve"], **popen_kwargs)
    except (FileNotFoundError, OSError):
        return {"error": f"Could not find '{tj_bin}' on PATH. Run 'tj serve' manually."}

    # Wait up to 5 seconds for the port to open
    for _ in range(10):
        time.sleep(0.5)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.5)
            try:
                s.connect((host, port))
                return {"url": url, "started": True, "message": "Dashboard started."}
            except (ConnectionRefusedError, OSError):
                pass

    return {
        "url": url,
        "started": True,
        "message": "Server launched but not yet ready — try opening the URL in a moment.",
    }


# ---------------------------------------------------------------------------
# Enforcement-plane policy tools (#223). Read-only, suggest-mode, unvalidated.
# The MCP layer NEVER bypasses the api-only / pricing-mode invariants — it only
# reports what the gated, suggest-mode engine already recorded. In DB mode it
# reads via the read-only connection; when tj serve holds the lock it reads the
# /api/v1/policy/* routes. `db.conn is None` distinguishes HTTP mode.
# ---------------------------------------------------------------------------

def _tool_get_policy_status(db, config, limit: int = 20) -> dict:
    if config is None:
        return _no_config()
    if getattr(db, "conn", None) is None:  # HTTP mode
        return _http_get("/api/v1/policy/status", {"limit": limit})
    from tokenjam.proxy.recommend import policy_status
    return policy_status(db, config, limit=limit)


def _tool_get_savings_summary(db, since: str | None = None) -> dict:
    if getattr(db, "conn", None) is None:  # HTTP mode
        return _http_get("/api/v1/policy/savings", {"since": since})
    from tokenjam.proxy.audit import reconcile_savings
    from tokenjam.utils.time_parse import parse_since
    return reconcile_savings(db, since=parse_since(since) if since else None).to_dict()


def _tool_suggest_policies(db, config) -> dict:
    if config is None:
        return _no_config()
    if getattr(db, "conn", None) is None:  # HTTP mode
        return _http_get("/api/v1/policy/suggestions")
    from tokenjam.proxy.recommend import suggest_policies
    return suggest_policies(db, config)


# ---------------------------------------------------------------------------
# FastMCP tool registrations
# ---------------------------------------------------------------------------

@mcp.tool()
def get_status(agent_id: str | None = None) -> dict:
    """
    Return the current observability status for one or all agents — equivalent to running
    `tj status` but returns structured data directly. Use this whenever the user asks about
    agent status, what's running, token usage, cost today, or active alerts. Omit agent_id
    to get a summary of every known agent.
    """
    try:
        return _tool_get_status(_ro_conn, _config, agent_id)
    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
def get_budget_headroom(agent_id: str) -> dict:
    """
    Return daily and per-session budget limits vs current spend for a specific agent.
    Use this when the user asks how much budget is left, whether they're close to a limit,
    or wants to check spend against their configured daily_usd or session_usd cap.
    """
    try:
        return _tool_get_budget_headroom(_ro_conn, _config, agent_id)
    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
def list_agents() -> dict:
    """
    List all agents TokenJam has ever seen, with first/last seen timestamps and lifetime cost.
    Use this when the user asks which agents are being tracked, wants an overview of all
    projects, or asks about total spend across agents.
    """
    if _ro_conn is None and _serve_url is None:
        return _no_config()
    try:
        return _tool_list_agents(_ro_conn)
    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
def list_active_sessions() -> dict:
    """
    List every session currently in 'active' status — one row per session. Use this as the
    primary dashboard view: it shows all running agents at a glance, including parallel
    sessions for the same agent. Prefer this over get_status when the user wants to see
    what's running right now across all agents.
    """
    if _ro_conn is None and _serve_url is None:
        return _no_config()
    try:
        return _tool_list_active_sessions(_ro_conn)
    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
def get_cost_summary(
    agent_id: str | None = None,
    since: str | None = None,
    group_by: str = "day",
) -> dict:
    """
    Return a cost breakdown grouped by day, agent, or model — equivalent to `tj cost`.
    Use this when the user asks about spending, cost trends, which model is most expensive,
    or wants a breakdown over a time period. since accepts relative values like '24h', '7d'
    or an absolute date like '2026-04-01'. group_by accepts 'day', 'agent', or 'model'.
    """
    if _ro_db is None:
        return _no_config()
    try:
        return _tool_get_cost_summary(_ro_db, agent_id, since, group_by)
    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
def list_alerts(
    agent_id: str | None = None,
    severity: str | None = None,
    unread: bool = False,
) -> dict:
    """
    Return alert history — equivalent to `tj alerts`. Use this when the user asks about
    alerts, what fired while they were away, budget breaches, sensitive actions, or drift
    events. severity filters to 'critical', 'warning', or 'info'. Set unread=True to show
    only active (unacknowledged, unsuppressed) alerts.
    """
    if _ro_db is None:
        return _no_config()
    try:
        return _tool_list_alerts(_ro_db, agent_id, severity, unread)
    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
def list_traces(
    agent_id: str | None = None,
    since: str | None = None,
    limit: int = 20,
) -> dict:
    """
    List recent traces with cost, duration, and span count — equivalent to `tj traces`.
    Use this when the user wants to see recent LLM calls, browse trace history, or find a
    specific trace to drill into. since accepts '24h', '7d', or '2026-04-01'.
    """
    if _ro_db is None:
        return _no_config()
    try:
        return _tool_list_traces(_ro_db, agent_id, since, limit)
    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
def get_trace(trace_id: str) -> dict:
    """
    Return the full span waterfall for a single trace — equivalent to `tj trace <id>`.
    Use this when the user wants to inspect a specific trace in detail, understand the
    sequence of LLM and tool calls within it, or debug a particular agent run.
    """
    if _ro_db is None:
        return _no_config()
    try:
        return _tool_get_trace(_ro_db, trace_id)
    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
def get_tool_stats(
    agent_id: str | None = None,
    since: str | None = None,
) -> dict:
    """
    Return tool call counts and average duration per tool — equivalent to `tj tools`.
    Use this when the user asks which tools their agent uses most, average execution time,
    or wants to identify slow or frequently-called tools.
    """
    if _ro_db is None:
        return _no_config()
    try:
        return _tool_get_tool_stats(_ro_db, agent_id, since)
    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
def get_drift_report(agent_id: str | None = None) -> dict:
    """
    Return behavioral drift data: statistical baseline vs the latest session's actual
    behavior. Use this when the user asks whether their agent is behaving differently than
    usual, wants Z-score analysis on token usage or tool patterns, or wants to understand
    drift alerts. Omit agent_id to get reports for all agents with baselines.
    """
    if _ro_db is None:
        return _no_config()
    try:
        return _tool_get_drift_report(_ro_db, agent_id)
    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
def acknowledge_alert(alert_id: str) -> dict:
    """
    Mark a specific alert as acknowledged. Use this when the user says they've seen an
    alert and want to clear it, or when resolving active alerts shown by list_alerts.
    Does not suppress future alerts of the same type — only marks this one as read.
    """
    if _config is None:
        return _no_config()
    try:
        if _serve_url is not None:
            import json as _json
            import urllib.request as _urlreq
            url = f"{_serve_url}/api/v1/alerts/{alert_id}/acknowledge"
            req = _urlreq.Request(url, method="PATCH", data=b"")
            if _config.api.auth.enabled:
                req.add_header("Authorization", f"Bearer {_config.api.auth.api_key}")
            with _urlreq.urlopen(req, timeout=5) as resp:
                return _json.loads(resp.read())
        if _ro_conn is not None:
            # Read-only DuckDB fallback: a module-global read-only connection is
            # already open against this file. DuckDB forbids opening the same
            # file read-write while a read-only connection exists in the same
            # process, so attempting the UPDATE here raises a raw
            # ConnectionException (#34). Return actionable guidance instead.
            return {
                "error": (
                    "Acknowledging alerts requires a writable database, but it's "
                    "open read-only because tj serve holds the lock (or wasn't "
                    "reachable). Acknowledge from the dashboard, or stop tj serve "
                    "and retry."
                )
            }
        from pathlib import Path
        import duckdb as _duckdb
        db_path = str(Path(_config.storage.path).expanduser())
        with _duckdb.connect(db_path) as write_conn:
            return _tool_acknowledge_alert(write_conn, alert_id)
    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
def setup_project(agent_id: str | None = None, project_path: str | None = None) -> dict:
    """
    Configure the current project to send telemetry to TokenJam. Writes OTEL_RESOURCE_ATTRIBUTES
    into .claude/settings.json so Claude Code tags spans with the right agent ID. For Codex
    CLI users the agent ID is set in ~/.codex/config.toml via 'tj onboard --codex'. Use
    this when the user wants to start monitoring a new project, or asks how to set up TokenJam
    for this repo. Infers agent_id from the git remote if not provided.
    """
    try:
        from tokenjam.core.config import find_config_file
        cp = find_config_file()
        return _tool_setup_project(
            config=_config,
            config_path=str(cp) if cp else None,
            agent_id=agent_id,
            project_path=project_path,
        )
    except Exception as e:
        return {"error": str(e)}


def _tool_get_optimize_report(
    db, config, agent_id: str | None, since: str | None, findings: list[str] | None,
    budget_provider: str | None, budget_usd: float | None,
) -> dict:
    from tokenjam.core.optimize import build_report, report_to_dict
    from tokenjam.utils.time_parse import parse_since, utcnow
    if config is None:
        return _no_config()
    if db is None or getattr(db, "conn", None) is None:
        return {
            "error": (
                "Optimize requires a direct database connection. "
                "Stop tj serve or run via the CLI."
            )
        }
    since_dt = parse_since(since) if since else parse_since("30d")
    report = build_report(
        db=db,
        config=config,
        since=since_dt,
        until=utcnow(),
        agent_id=agent_id,
        findings=findings,
        budget_provider_filter=budget_provider,
        budget_usd_override=budget_usd,
    )
    return report_to_dict(report)


@mcp.tool()
def get_optimize_report(
    agent_id: str | None = None,
    since: str | None = "30d",
    findings: list[str] | None = None,
    budget_provider: str | None = None,
    budget_usd: float | None = None,
) -> dict:
    """
    Return cost-saving candidates and budget projections — equivalent to
    `tj optimize`. Use this when the user asks where they could save money,
    whether they're going to exceed a budget, what their monthly run rate
    looks like, which sessions look like they could have used a cheaper
    model, whether they're using their Claude plan efficiently, or whether
    they're getting their money's worth from a subscription plan. Includes a
    mandatory caveat field reminding callers that the downsize
    finding is a structural heuristic, not a quality judgment.

    `findings` is a list of analyzer names to run (e.g. ['downsize']
    or ['budget-projection']). Omit to run all registered analyzers.

    Output includes a `pricing_mode` field derived from the session's
    plan_tier: 'api' (per-token billing), 'subscription' (flat-rate plan;
    dollar figures shown as implied API value, not spend), 'local' (no
    marginal cost), or 'unknown' (plan not configured — dollar figures
    suppressed). Subscription-tier output reframes savings as token
    headroom freed against the plan cap rather than dollars saved.
    """
    if _config is None:
        return _no_config()
    try:
        # Pass the writable backend (not _ro_db) so the analyzers can read
        # via the same DuckDB connection the CLI uses.
        from tokenjam.core.db import open_db
        if _ro_conn is not None:
            class _Shim:
                conn = _ro_conn
            db = _Shim()
        else:
            db = open_db(_config.storage)
        return _tool_get_optimize_report(
            db, _config, agent_id, since, findings, budget_provider, budget_usd,
        )
    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
def open_dashboard() -> dict:
    """
    Open the TokenJam web dashboard in the browser. Starts `tj serve` in the background if it
    is not already running — do NOT start tj serve manually via Bash. Call this tool
    whenever the user asks to open the dashboard, view the UI, or browse observability data
    visually. Returns the URL to open. Safe to call repeatedly — detects if already running.
    """
    try:
        return _tool_open_dashboard(_config)
    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
def get_policy_status(limit: int = 20) -> dict:
    """
    Return the TokenJam enforcement-plane policy status: the defined policies
    (`[[policies]]`) and recent policy decisions recorded by the proxy. Use this
    when the user asks what policies are active, what the proxy has been deciding,
    or whether a policy would have blocked/modified traffic.

    SUGGEST MODE ONLY: every decision is what a policy WOULD do — nothing is
    enforced. Policies are `unvalidated` (there is no certification engine in the
    open tree). Never describe a decision as enforced or validated-safe.
    """
    if _ro_db is None:
        return _no_config()
    try:
        return _tool_get_policy_status(_ro_db, _config, limit)
    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
def get_savings_summary(since: str | None = None) -> dict:
    """
    Return the policy savings meter: the ESTIMATED-RECOVERABLE amount these
    policies WOULD have recovered if enforced, reconciled against actual spend
    (the same source `tj cost` reads). Use this when the user asks how much the
    policies could save or what the proxy's cost impact would be.

    CRITICAL: this is NOT realized savings. Suggest mode enforces nothing, so
    `realized` is always false and every figure is "estimated recoverable" /
    "would-have-saved" — NEVER present it as money saved. Dollar figures are
    api-only; the result is `unvalidated`. `since` accepts '7d', '30d', etc.
    """
    if _ro_db is None:
        return _no_config()
    try:
        return _tool_get_savings_summary(_ro_db, since)
    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
def suggest_policies() -> dict:
    """
    Recommend enforcement-plane policies from the user's actual usage — e.g. a
    `budget_cap` ceiling for an api-billed provider that has cycle spend but no
    ceiling configured. Use this when the user asks what policies they should add
    or how to start using the proxy.

    These are SUGGESTIONS in suggest mode — a suggested ceiling is a starting
    point to review, not validated-safe, and adding a policy enforces nothing.
    Dollar suggestions are api-only.
    """
    if _ro_db is None:
        return _no_config()
    try:
        return _tool_suggest_policies(_ro_db, _config)
    except Exception as e:
        return {"error": str(e)}
