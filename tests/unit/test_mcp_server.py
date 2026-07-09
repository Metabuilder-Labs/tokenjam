"""Unit tests for TokenJam MCP server tool handlers."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from tokenjam.core.db import InMemoryBackend
from tokenjam.core.config import TjConfig, AgentConfig, BudgetConfig, DefaultsConfig
from tokenjam.core.models import AlertType, Severity, Alert, DriftBaseline, AgentRecord
from tokenjam.utils.time_parse import utcnow
from tokenjam.utils.ids import new_uuid
from tests.factories import make_session, make_llm_span, make_tool_span

from tokenjam.mcp.server import (
    _tool_get_status,
    _tool_get_budget_headroom,
    _tool_list_agents,
    _tool_list_active_sessions,
    _tool_get_cost_summary,
    _tool_list_alerts,
    _tool_list_traces,
    _tool_get_trace,
    _tool_get_tool_stats,
    _tool_get_drift_report,
    _tool_acknowledge_alert,
    _tool_setup_project,
    _tool_list_summarize_candidates,
    _tool_summarize_prep,
    _tool_summarize_check,
    _tool_summarize_apply,
    _tool_summarize_undo,
)


def _make_config(agent_id: str = "test-agent", daily_usd: float | None = 5.0) -> TjConfig:
    return TjConfig(
        version="1",
        defaults=DefaultsConfig(budget=BudgetConfig(daily_usd=daily_usd)),
        agents={agent_id: AgentConfig(budget=BudgetConfig(daily_usd=daily_usd))},
    )


# --- get_status ---

def test_get_status_active_session():
    db = InMemoryBackend()
    session = make_session(agent_id="alpha", status="active", input_tokens=100, output_tokens=50)
    db.upsert_session(session)
    config = _make_config("alpha")

    result = _tool_get_status(db.conn, config, agent_id="alpha")

    assert result["agent_id"] == "alpha"
    assert result["status"] == "active"
    assert result["input_tokens"] == 100
    assert result["output_tokens"] == 50
    assert result["active_alerts"] == 0


def test_get_status_no_session():
    db = InMemoryBackend()
    config = _make_config("ghost")

    result = _tool_get_status(db.conn, config, agent_id="ghost")

    assert result["agent_id"] == "ghost"
    assert result["status"] == "idle"
    assert result["session_id"] is None


def test_get_status_no_config():
    db = InMemoryBackend()
    result = _tool_get_status(db.conn, None, agent_id="x")
    assert "error" in result


# --- get_budget_headroom ---

def test_get_budget_headroom_within_budget():
    db = InMemoryBackend()
    config = _make_config("alpha", daily_usd=10.0)
    span = make_llm_span(agent_id="alpha", cost_usd=2.50)
    db.insert_span(span)

    result = _tool_get_budget_headroom(db.conn, config, agent_id="alpha")

    assert result["agent_id"] == "alpha"
    assert result["daily_limit_usd"] == 10.0
    assert abs(result["daily_spent_usd"] - 2.50) < 0.01
    assert abs(result["daily_remaining_usd"] - 7.50) < 0.01


def test_get_budget_headroom_no_limit():
    db = InMemoryBackend()
    config = _make_config("alpha", daily_usd=None)

    result = _tool_get_budget_headroom(db.conn, config, agent_id="alpha")

    assert result["daily_limit_usd"] is None
    assert result["daily_remaining_usd"] is None


# --- list_agents ---

def test_list_agents_returns_all_known():
    db = InMemoryBackend()
    db.upsert_agent(AgentRecord(agent_id="a1", first_seen=utcnow(), last_seen=utcnow()))
    db.upsert_agent(AgentRecord(agent_id="a2", first_seen=utcnow(), last_seen=utcnow()))
    span = make_llm_span(agent_id="a1", cost_usd=1.50)
    db.insert_span(span)

    result = _tool_list_agents(db.conn)

    ids = [a["agent_id"] for a in result["agents"]]
    assert "a1" in ids and "a2" in ids
    a1 = next(a for a in result["agents"] if a["agent_id"] == "a1")
    assert abs(a1["lifetime_cost_usd"] - 1.50) < 0.01


def test_list_agents_empty():
    db = InMemoryBackend()
    result = _tool_list_agents(db.conn)
    assert result["agents"] == []


# --- list_active_sessions ---

def test_list_active_sessions_one_per_session():
    db = InMemoryBackend()
    s1 = make_session(agent_id="proj-a", status="active")
    s2 = make_session(agent_id="proj-a", status="active")
    s3 = make_session(agent_id="proj-b", status="active")
    s4 = make_session(agent_id="proj-b", status="completed")
    for s in [s1, s2, s3, s4]:
        db.upsert_session(s)

    result = _tool_list_active_sessions(db.conn)

    assert result["count"] == 3  # s1, s2, s3 — s4 excluded
    session_ids = {r["session_id"] for r in result["sessions"]}
    assert s1.session_id in session_ids
    assert s2.session_id in session_ids
    assert s3.session_id in session_ids
    assert s4.session_id not in session_ids


def test_list_active_sessions_empty():
    db = InMemoryBackend()
    result = _tool_list_active_sessions(db.conn)
    assert result["sessions"] == []
    assert result["count"] == 0


# --- get_cost_summary ---

def test_get_cost_summary_total():
    db = InMemoryBackend()
    s1 = make_llm_span(agent_id="a", cost_usd=1.00)
    s2 = make_llm_span(agent_id="a", cost_usd=2.50)
    db.insert_span(s1)
    db.insert_span(s2)

    result = _tool_get_cost_summary(db, agent_id="a", since=None, group_by="day")

    assert abs(result["total_cost_usd"] - 3.50) < 0.01
    assert len(result["rows"]) >= 1


def test_get_cost_summary_empty():
    db = InMemoryBackend()
    result = _tool_get_cost_summary(db, agent_id="nobody", since=None, group_by="day")
    assert result["total_cost_usd"] == 0.0
    assert result["rows"] == []


# --- list_alerts ---

def test_list_alerts_returns_alerts():
    db = InMemoryBackend()
    alert = Alert(
        alert_id=new_uuid(),
        fired_at=utcnow(),
        type=AlertType.COST_BUDGET_DAILY,
        severity=Severity.WARNING,
        title="Budget exceeded",
        detail={"cost": 6.0},
        agent_id="a",
        session_id=None,
        span_id=None,
        acknowledged=False,
        suppressed=False,
    )
    db.insert_alert(alert)

    result = _tool_list_alerts(db, agent_id="a", severity=None, unread=False)

    assert result["count"] == 1
    assert result["alerts"][0]["alert_id"] == alert.alert_id
    assert result["alerts"][0]["type"] == "cost_budget_daily"


def test_list_alerts_empty():
    db = InMemoryBackend()
    result = _tool_list_alerts(db, agent_id="x", severity=None, unread=False)
    assert result["count"] == 0
    assert result["alerts"] == []


# --- list_traces ---

def test_list_traces_returns_recent():
    db = InMemoryBackend()
    span = make_llm_span(agent_id="a", cost_usd=0.50)
    db.insert_span(span)

    result = _tool_list_traces(db, agent_id="a", since=None, limit=20)

    assert result["count"] >= 1
    assert result["traces"][0]["trace_id"] == span.trace_id


def test_list_traces_empty():
    db = InMemoryBackend()
    result = _tool_list_traces(db, agent_id="nobody", since=None, limit=20)
    assert result["count"] == 0
    assert result["traces"] == []


# --- get_trace ---

def test_get_trace_returns_spans():
    db = InMemoryBackend()
    span = make_llm_span(agent_id="a")
    db.insert_span(span)

    result = _tool_get_trace(db, trace_id=span.trace_id)

    assert result["trace_id"] == span.trace_id
    assert result["span_count"] == 1
    assert result["spans"][0]["span_id"] == span.span_id


def test_get_trace_unknown():
    db = InMemoryBackend()
    result = _tool_get_trace(db, trace_id="nonexistent-trace")
    assert result["span_count"] == 0
    assert result["spans"] == []


# --- get_tool_stats ---

def test_get_tool_stats_aggregates():
    db = InMemoryBackend()
    db.insert_span(make_tool_span(agent_id="a", tool_name="Read", duration_ms=100.0))
    db.insert_span(make_tool_span(agent_id="a", tool_name="Read", duration_ms=200.0))
    db.insert_span(make_tool_span(agent_id="a", tool_name="Edit", duration_ms=50.0))

    result = _tool_get_tool_stats(db, agent_id="a", since=None)

    tools = {t["tool_name"]: t for t in result["tools"]}
    assert tools["Read"]["call_count"] == 2
    assert tools["Edit"]["call_count"] == 1
    assert result["count"] == 2


def test_get_tool_stats_empty():
    db = InMemoryBackend()
    result = _tool_get_tool_stats(db, agent_id="nobody", since=None)
    assert result["tools"] == []
    assert result["count"] == 0


# --- get_drift_report ---

def test_get_drift_report_with_baseline():
    db = InMemoryBackend()
    baseline = DriftBaseline(
        agent_id="a",
        sessions_sampled=10,
        computed_at=utcnow(),
        avg_input_tokens=1000.0,
        stddev_input_tokens=100.0,
        avg_output_tokens=200.0,
        stddev_output_tokens=20.0,
        avg_session_duration_s=120.0,
        stddev_session_duration=15.0,
        avg_tool_call_count=5.0,
        stddev_tool_call_count=1.0,
    )
    db.upsert_baseline(baseline)

    result = _tool_get_drift_report(db, agent_id="a")

    assert result["agent_id"] == "a"
    assert result["baseline"]["sessions_sampled"] == 10
    assert result["baseline"]["avg_input_tokens"] == 1000.0


def test_get_drift_report_no_baseline():
    db = InMemoryBackend()
    result = _tool_get_drift_report(db, agent_id="ghost")
    assert result["agent_id"] == "ghost"
    assert result["baseline"] is None


# --- acknowledge_alert ---

def test_acknowledge_alert_sets_flag():
    db = InMemoryBackend()
    alert = Alert(
        alert_id=new_uuid(),
        fired_at=utcnow(),
        type=AlertType.RETRY_LOOP,
        severity=Severity.WARNING,
        title="Retry loop",
        detail={},
        agent_id="a",
        session_id=None,
        span_id=None,
        acknowledged=False,
        suppressed=False,
    )
    db.insert_alert(alert)

    result = _tool_acknowledge_alert(db.conn, alert.alert_id)

    assert result == {"acknowledged": True, "alert_id": alert.alert_id}
    row = db.conn.execute(
        "SELECT acknowledged FROM alerts WHERE alert_id = $1", [alert.alert_id]
    ).fetchone()
    assert row[0] is True


def test_acknowledge_alert_unknown_id():
    db = InMemoryBackend()
    result = _tool_acknowledge_alert(db.conn, "nonexistent-id")
    assert "error" in result


# --- setup_project ---

def test_setup_project_writes_settings(tmp_path):
    config = _make_config()
    config_path = tmp_path / "tokenjam.toml"
    config_path.write_text("")  # dummy

    result = _tool_setup_project(
        config=config,
        config_path=str(config_path),
        agent_id="my-project",
        project_path=str(tmp_path),
    )

    assert result["agent_id"] == "my-project"
    settings_file = tmp_path / ".claude" / "settings.json"
    assert settings_file.exists()
    data = json.loads(settings_file.read_text())
    assert data["env"]["OTEL_RESOURCE_ATTRIBUTES"] == "service.name=my-project"


def test_setup_project_warns_no_global_otlp(tmp_path, monkeypatch):
    # Simulate no global ~/.claude/settings.json by pointing home() to a fresh tmp dir
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "fake_home")
    config = _make_config()
    config_path = tmp_path / "tokenjam.toml"
    config_path.write_text("")

    result = _tool_setup_project(
        config=config,
        config_path=str(config_path),
        agent_id="proj",
        project_path=str(tmp_path),
    )

    assert "warning" in result


def test_setup_project_no_config():
    result = _tool_setup_project(
        config=None,
        config_path=None,
        agent_id=None,
        project_path=None,
    )
    assert "error" in result


# --- setup_harness (run-linkage instrumentation) ---

from tokenjam.mcp.server import _tool_setup_harness


def test_setup_harness_instrument_writes_helper(tmp_path):
    (tmp_path / "run-loop.sh").write_text(
        "#!/bin/bash\nclaude -p \"$ticket\" &\n", encoding="utf-8"
    )
    result = _tool_setup_harness(
        mode="instrument", project_path=str(tmp_path), runs_lookup=lambda: [],
    )
    assert result["mode"] == "instrument"
    assert result["attribute"] == "tokenjam.run_id"
    helper = tmp_path / ".tj" / "run-env.sh"
    assert helper.exists()
    assert "tokenjam.run_id=" in helper.read_text()
    # It located the spawn point and gives concrete wiring + the honest boundary.
    assert any(h["file"] == "run-loop.sh" for h in result["spawn_points"])
    assert result["shell_wiring"] == "source .tj/run-env.sh"
    assert result["next_steps"] and "boundary" in result


def test_setup_harness_map_makes_no_changes(tmp_path):
    runs = [{"run_id": "gov-1", "session_count": 3}]
    result = _tool_setup_harness(
        mode="map", project_path=str(tmp_path), runs_lookup=lambda: runs,
    )
    assert result["mode"] == "map"
    assert result["runs_visible"] == runs
    # map never writes the helper.
    assert not (tmp_path / ".tj" / "run-env.sh").exists()
    assert "recommendation" in result


def test_setup_harness_rejects_bad_mode(tmp_path):
    result = _tool_setup_harness(
        mode="nope", project_path=str(tmp_path), runs_lookup=lambda: [],
    )
    assert "error" in result


# --- open_dashboard ---

from unittest.mock import patch, MagicMock
from tokenjam.mcp.server import _tool_open_dashboard


def test_open_dashboard_already_running():
    config = _make_config()

    mock_sock = MagicMock()
    mock_sock.__enter__ = lambda s: s
    mock_sock.__exit__ = MagicMock(return_value=False)
    mock_sock.connect = MagicMock()  # connect succeeds = port is bound

    with patch("socket.socket", return_value=mock_sock):
        result = _tool_open_dashboard(config)

    assert result["started"] is False
    assert "7391" in result["url"]
    assert "/ui" in result["url"]


def test_open_dashboard_starts_server():
    config = _make_config()

    call_count = 0

    def fake_socket_factory(*args, **kwargs):
        nonlocal call_count
        s = MagicMock()
        s.__enter__ = lambda self: self
        s.__exit__ = MagicMock(return_value=False)
        call_count += 1
        if call_count == 1:
            # First call: port not bound
            s.connect = MagicMock(side_effect=ConnectionRefusedError)
        else:
            # Subsequent calls (polling): port is bound
            s.connect = MagicMock()
        return s

    with patch("socket.socket", side_effect=fake_socket_factory), \
         patch("subprocess.Popen") as mock_popen, \
         patch("time.sleep"):
        result = _tool_open_dashboard(config)

    mock_popen.assert_called_once()
    assert result["started"] is True
    assert "/ui" in result["url"]


def test_open_dashboard_no_config():
    result = _tool_open_dashboard(None)
    assert "error" in result


# ---------------------------------------------------------------------------
# HTTP mode tests (_HttpDB and HTTP-path handlers)
# ---------------------------------------------------------------------------

import tokenjam.mcp.server as _srv
from tokenjam.mcp.server import (
    _tool_list_traces,
    _tool_list_alerts,
    _tool_list_active_sessions,
    _tool_acknowledge_alert,
    _tool_get_status,
)


def _set_serve_url(url: str | None) -> None:
    _srv._serve_url = url


# --- test_get_status_http_mode ---

def test_get_status_http_mode():
    fake_response = {
        "agents": [
            {
                "agent_id": "alpha",
                "status": "active",
                "session_id": "s1",
                "input_tokens": 100,
                "output_tokens": 50,
                "tool_call_count": 5,
                "error_count": 0,
                "cost_today": 1.23,
                "active_alerts": 0,
            }
        ],
        "has_active_alerts": False,
    }
    config = _make_config("alpha")
    _set_serve_url("http://127.0.0.1:7391")
    try:
        with patch("tokenjam.mcp.server._http_get", return_value=fake_response):
            result = _tool_get_status(None, config, "alpha")
        assert result["status"] == "active"
        # cost_today from API must be renamed to cost_today_usd in the single-agent path
        assert "cost_today_usd" in result
        assert abs(result["cost_today_usd"] - 1.23) < 0.01
        assert "cost_today" not in result
    finally:
        _set_serve_url(None)


# --- test_list_traces_http_mode ---

def test_list_traces_http_mode():
    fake_response = {
        "traces": [
            {
                "trace_id": "abc",
                "agent_id": "a",
                "name": "gen_ai.llm.call",
                "start_time": "2026-04-11T10:00:00+00:00",
                "duration_ms": 1234,
                "cost_usd": 0.05,
                "status_code": "ok",
                "span_count": 2,
            }
        ],
        "count": 1,
    }
    _set_serve_url("http://127.0.0.1:7391")
    try:
        with patch("tokenjam.mcp.server._http_get", return_value=fake_response):
            db = _srv._HttpDB()
            result = _tool_list_traces(db, "a", None, 20)
        assert result["count"] == 1
        assert result["traces"][0]["trace_id"] == "abc"
    finally:
        _set_serve_url(None)


# --- test_list_alerts_http_mode ---

def test_list_alerts_http_mode():
    fake_response = {
        "alerts": [
            {
                "alert_id": "alert-1",
                "fired_at": "2026-04-11T09:00:00+00:00",
                "type": "cost_budget_daily",
                "severity": "warning",
                "title": "Budget exceeded",
                "agent_id": "alpha",
                "acknowledged": False,
                "suppressed": False,
            }
        ],
        "count": 1,
    }
    _set_serve_url("http://127.0.0.1:7391")
    try:
        with patch("tokenjam.mcp.server._http_get", return_value=fake_response):
            db = _srv._HttpDB()
            result = _tool_list_alerts(db, "alpha", None, False)
        assert result["count"] == 1
        assert result["alerts"][0]["type"] == "cost_budget_daily"
        assert result["alerts"][0]["severity"] == "warning"
    finally:
        _set_serve_url(None)


# --- test_list_active_sessions_http_mode ---

def test_list_active_sessions_http_mode():
    # Serve mode now proxies to /api/v1/sessions?status=active, which returns
    # one row per active session (not one per agent).
    fake_response = {
        "sessions": [
            {
                "session_id": "s1",
                "agent_id": "alpha",
                "started_at": "2026-04-11T10:00:00+00:00",
                "total_cost_usd": 1.23,
                "input_tokens": 100,
                "output_tokens": 50,
                "tool_call_count": 5,
                "error_count": 0,
            }
        ],
        "count": 1,
    }
    _set_serve_url("http://127.0.0.1:7391")
    try:
        with patch("tokenjam.mcp.server._http_get", return_value=fake_response):
            result = _tool_list_active_sessions(None)
        assert result["count"] == 1
        assert result["sessions"][0]["session_id"] == "s1"
    finally:
        _set_serve_url(None)


def test_list_active_sessions_http_mode_enumerates_per_session():
    """Serve mode must surface EVERY active session, including multiple
    concurrent sessions for the same agent — matching the direct-DB path (#35).
    The old code proxied to /api/v1/status (one record per agent) and
    undercounted; it must now hit a per-session endpoint."""
    sessions_response = {
        "sessions": [
            {
                "session_id": "s1", "agent_id": "alpha",
                "started_at": "2026-04-11T10:00:00+00:00", "total_cost_usd": 1.0,
                "input_tokens": 10, "output_tokens": 5,
                "tool_call_count": 1, "error_count": 0,
            },
            {
                "session_id": "s2", "agent_id": "alpha",
                "started_at": "2026-04-11T10:05:00+00:00", "total_cost_usd": 2.0,
                "input_tokens": 20, "output_tokens": 8,
                "tool_call_count": 2, "error_count": 0,
            },
        ],
        "count": 2,
    }
    # /status collapses alpha's two concurrent sessions to one record — the
    # shape the broken serve path used to read.
    status_response = {
        "agents": [
            {"agent_id": "alpha", "status": "active", "session_id": "s2"}
        ],
        "has_active_alerts": False,
    }

    def fake_http_get(path, params=None):
        if "/sessions" in path:
            assert params and params.get("status") == "active"
            return sessions_response
        return status_response

    _set_serve_url("http://127.0.0.1:7391")
    try:
        with patch("tokenjam.mcp.server._http_get", side_effect=fake_http_get):
            result = _tool_list_active_sessions(None)
        assert result["count"] == 2
        ids = {s["session_id"] for s in result["sessions"]}
        assert ids == {"s1", "s2"}
    finally:
        _set_serve_url(None)


# --- test_acknowledge_alert_http_mode ---

def test_acknowledge_alert_inner_conn_none_returns_error():
    """_tool_acknowledge_alert with conn=None returns a descriptive error (not a crash)."""
    result = _tool_acknowledge_alert(None, "some-id")
    assert "error" in result


def test_acknowledge_alert_http_mode_proxies_patch():
    """acknowledge_alert MCP wrapper proxies to PATCH endpoint when _serve_url is set."""
    from unittest.mock import MagicMock
    config = _make_config("alpha")
    _set_serve_url("http://127.0.0.1:7391")
    _srv._config = config
    try:
        mock_resp = MagicMock()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.read.return_value = b'{"acknowledged": true, "alert_id": "alert-1"}'

        with patch("urllib.request.urlopen", return_value=mock_resp) as mock_urlopen:
            result = _srv.acknowledge_alert("alert-1")

        call_args = mock_urlopen.call_args
        req = call_args[0][0]
        assert "alerts/alert-1/acknowledge" in req.full_url
        assert req.get_method() == "PATCH"
        assert result["acknowledged"] is True
    finally:
        _set_serve_url(None)
        _srv._config = None


def test_acknowledge_alert_read_only_fallback_returns_actionable_error(tmp_path):
    """In read-only DuckDB fallback (tj serve unreachable, _ro_conn open, no
    _serve_url) acknowledge_alert must NOT open a conflicting read-write
    connection to the same file — that raises a raw DuckDB ConnectionException.
    It should return a clear, actionable message instead (#34)."""
    import duckdb
    from tokenjam.core.config import StorageConfig
    from tokenjam.core.db import DuckDBBackend

    db_file = tmp_path / "telemetry.duckdb"
    # Seed the DB (schema + one alert) via a normal read-write backend, then
    # close it so the read-only connection below is the only one in process.
    backend = DuckDBBackend(StorageConfig(path=str(db_file)))
    alert = Alert(
        alert_id=new_uuid(),
        fired_at=utcnow(),
        type=AlertType.RETRY_LOOP,
        severity=Severity.WARNING,
        title="Retry loop",
        detail={},
        agent_id="a",
        session_id=None,
        span_id=None,
        acknowledged=False,
        suppressed=False,
    )
    backend.insert_alert(alert)
    backend._conn.close()

    ro_conn = duckdb.connect(str(db_file), read_only=True)
    config = _make_config("a")
    config.storage = StorageConfig(path=str(db_file))
    _srv._ro_conn = ro_conn
    _srv._config = config
    _set_serve_url(None)
    try:
        result = _srv.acknowledge_alert(alert.alert_id)
        assert "error" in result
        msg = result["error"].lower()
        # Actionable guidance — not a raw DuckDB connection exception.
        assert "read-only" in msg or "dashboard" in msg
        assert "configuration" not in msg  # raw DuckDB conflict phrasing
    finally:
        ro_conn.close()
        _srv._ro_conn = None
        _srv._config = None


# --- get_optimize_report under a running tj serve (#61) ---

def test_get_optimize_report_serve_holds_lock_routes_over_http(tmp_path):
    """When a tj serve holds the DuckDB write lock (_serve_url set, _ro_conn
    None), get_optimize_report must route the optimize computation through the
    serve HTTP API and NOT open its own connection to the locked file.

    The old code called open_db() (read-write) in this state, so DuckDB threw
    "Could not set lock on file" and the tool returned {"error": ...} on every
    call while a serve was up (#61). Holding a real write lock here proves the
    fix: if the buggy path still ran, open_db() would raise before we ever
    reached the (mocked) HTTP fetch."""
    from unittest.mock import patch
    from tokenjam.core.config import StorageConfig
    from tokenjam.core.db import DuckDBBackend

    db_file = tmp_path / "telemetry.duckdb"
    # A live "tj serve" holds the exclusive write lock on the file.
    serve_backend = DuckDBBackend(StorageConfig(path=str(db_file)))

    fake_report = {
        "window": {"total_cost_usd": 1.0, "total_tokens": 100, "sessions": 1},
        "findings": {"downsize": {"candidates": []}},
        "pricing_mode": "api",
    }

    config = _make_config("alpha")
    config.storage = StorageConfig(path=str(db_file))
    _srv._config = config
    _srv._ro_conn = None
    _set_serve_url("http://127.0.0.1:7391")
    try:
        with patch(
            "tokenjam.core.api_backend.ApiBackend.fetch_optimize_report",
            return_value=fake_report,
        ) as mock_fetch:
            result = _srv.get_optimize_report(since="7d", findings=["downsize"])

        # Routed over HTTP and returned the serve-built report — no raw lock error.
        assert result == fake_report
        assert "could not set lock" not in str(result.get("error", "")).lower()
        mock_fetch.assert_called_once()
        _, kwargs = mock_fetch.call_args
        assert kwargs["since"] == "7d"
        assert kwargs["findings"] == ["downsize"]
    finally:
        serve_backend._conn.close()
        _srv._config = None
        _srv._ro_conn = None
        _set_serve_url(None)


def test_get_optimize_report_no_connection_returns_graceful_message():
    """With neither a read-only connection nor a serve URL, get_optimize_report
    must return the graceful "requires a direct database connection" guidance
    rather than opening a competing read-write connection (#61)."""
    config = _make_config("alpha")
    _srv._config = config
    _srv._ro_conn = None
    _set_serve_url(None)
    try:
        result = _srv.get_optimize_report()
        assert "error" in result
        assert "direct database connection" in result["error"].lower()
        assert "could not set lock" not in result["error"].lower()
    finally:
        _srv._config = None


# --- test_get_budget_headroom_http_mode ---

def test_get_budget_headroom_http_mode_reads_live_limits():
    """Budget limits should come from the live /api/v1/budget endpoint, not stale _config."""
    budget_response = {
        "defaults": {"daily_usd": 5.0, "session_usd": None},
        "agents": {
            "alpha": {
                "configured": {"daily_usd": 20.0, "session_usd": None},
                "effective": {"daily_usd": 20.0, "session_usd": None},
            }
        },
    }
    status_response = {
        "agents": [{"agent_id": "alpha", "cost_today": 3.0, "status": "active"}],
        "has_active_alerts": False,
    }

    def fake_http_get(path, params=None):
        if "/budget" in path:
            return budget_response
        return status_response

    config = _make_config("alpha", daily_usd=5.0)  # stale config still has old $5
    _set_serve_url("http://127.0.0.1:7391")
    try:
        with patch("tokenjam.mcp.server._http_get", side_effect=fake_http_get):
            result = _tool_get_budget_headroom(None, config, "alpha")
        # Must use the live 20.0 limit, not the stale 5.0 from config
        assert result["daily_limit_usd"] == 20.0
        assert abs(result["daily_spent_usd"] - 3.0) < 0.01
        assert abs(result["daily_remaining_usd"] - 17.0) < 0.01
    finally:
        _set_serve_url(None)


# --- list_summarize_candidates (advisory scan; no DB) ---

@pytest.fixture
def _iso_summarize_catalog(monkeypatch):
    """Controlled catalog (no real ~/.claude globals) so the MCP scan is deterministic."""
    from tokenjam.core.summarize import candidates as _cand
    from tokenjam.core.summarize.catalog import Catalog
    fake = Catalog(project_files=frozenset({"CLAUDE.md", "AGENTS.md"}),
                   project_globs=(), global_paths=(), forbidden_roots=())
    monkeypatch.setattr(_cand, "load_catalog", lambda: fake)


def test_summarize_candidates_lists_prompt(tmp_path, _iso_summarize_catalog):
    (tmp_path / "CLAUDE.md").write_text("instructions " * 200)
    result = _tool_list_summarize_candidates(_make_config(), path=str(tmp_path))
    assert result["count"] == len(result["candidates"]) >= 1
    claude = [c for c in result["candidates"] if Path(c["path"]).name == "CLAUDE.md"]
    assert claude and claude[0]["kind"] == "prompt"
    assert claude[0]["est_tokens_saved"] > 0
    assert "review before adopting" in result["note"]   # honesty caveat survives the MCP layer


def test_summarize_candidates_recursive_passthrough(tmp_path, _iso_summarize_catalog):
    nested = tmp_path / "sub" / "deep"
    nested.mkdir(parents=True)
    (nested / "CLAUDE.md").write_text("instructions " * 200)
    flat = _tool_list_summarize_candidates(_make_config(), path=str(tmp_path))
    deep = _tool_list_summarize_candidates(_make_config(), path=str(tmp_path), recursive=True)
    assert not any("deep" in c["path"] for c in flat["candidates"])   # no walk without -r
    assert any("deep" in c["path"] for c in deep["candidates"])       # recursive flag reaches core
    assert deep["recursive"] is True


def test_summarize_candidates_no_config():
    result = _tool_list_summarize_candidates(None)
    assert "error" in result and "config" in result["error"].lower()  # _no_config sentinel


# --- summarize_prep / summarize_check (no scratch; re-derive + hash + staging) ---

def _summarize_config(tmp_path):
    """Config whose summarize anchor is a tmp dir — never the real ~/.tj."""
    from tokenjam.core.config import StorageConfig
    return TjConfig(version="1", storage=StorageConfig(path=str(tmp_path / "t.duckdb")))


def test_summarize_prep_then_check_roundtrip(tmp_path):
    config = _summarize_config(tmp_path)
    f = tmp_path / "CLAUDE.md"
    f.write_text("Always act carefully and never skip a required step. " * 30 + "\n```\nx = 1\n```\n")
    prep = _tool_summarize_prep(config, str(f), 0.5)
    assert prep["source_sha256"] and "<tj-keep" in prep["wrapped_prompt"]
    import re
    markers = re.findall(r'<tj-keep id="\d+"[^>]*?(?:/>|>.*?</tj-keep>)', prep["wrapped_prompt"], re.DOTALL)
    summary = "Be careful; never skip a step. " + " ".join(markers)
    verdict = _tool_summarize_check(config, str(f), summary, prep["source_sha256"])
    assert verdict["structure_ok"] is True and verdict["staged"] is True
    assert "x = 1" in verdict["restored"]             # structure restored verbatim through the MCP layer
    assert verdict["est_tokens_saved"] > 0


def test_summarize_check_refuses_changed_file(tmp_path):
    from tokenjam.core.summarize.session import SummarizeRefused
    config = _summarize_config(tmp_path)
    f = tmp_path / "CLAUDE.md"
    f.write_text("Always act carefully and never skip a required step. " * 30)
    prep = _tool_summarize_prep(config, str(f), 0.5)
    f.write_text("edited after prep")                 # file changed since prep
    with pytest.raises(SummarizeRefused, match="changed since"):
        _tool_summarize_check(config, str(f), "anything", prep["source_sha256"])


def test_summarize_check_records_in_session_provenance(tmp_path):
    """MCP front door = Claude rewrote the prompt in-session → produced_by == 'in-session' (DEC-027/028)."""
    config = _summarize_config(tmp_path)
    f = tmp_path / "CLAUDE.md"
    f.write_text("Always act carefully and never skip a required step. " * 30 + "\n```\nx = 1\n```\n")
    prep = _tool_summarize_prep(config, str(f), 0.5)
    import re
    markers = re.findall(r'<tj-keep id="\d+"[^>]*?(?:/>|>.*?</tj-keep>)', prep["wrapped_prompt"], re.DOTALL)
    verdict = _tool_summarize_check(config, str(f), "Careful; never skip. " + " ".join(markers),
                                    prep["source_sha256"])
    assert verdict["produced_by"] == "in-session"     # the MCP layer stamps the seam, not 'manual'


def test_summarize_prep_no_config():
    assert "error" in _tool_summarize_prep(None, "x.md", 0.5)


def test_summarize_check_no_config():
    assert "error" in _tool_summarize_check(None, "x.md", "summary", "deadbeef")


def test_summarize_apply_then_undo_roundtrip(tmp_path):
    config = _summarize_config(tmp_path)
    f = tmp_path / "CLAUDE.md"
    original = "Always act carefully and never skip a required step. " * 30 + "\n```\nx = 1\n```\n"
    f.write_text(original)
    prep = _tool_summarize_prep(config, str(f), 0.5)
    import re
    markers = re.findall(r'<tj-keep id="\d+"[^>]*?(?:/>|>.*?</tj-keep>)', prep["wrapped_prompt"], re.DOTALL)
    _tool_summarize_check(config, str(f), "Careful; never skip. " + " ".join(markers), prep["source_sha256"])

    dry = _tool_summarize_apply(config, str(f), False)           # default dry-run writes nothing
    assert dry["dry_run"] is True and f.read_text() == original
    done = _tool_summarize_apply(config, str(f), True)           # go writes
    assert done["applied"] and "x = 1" in f.read_text() and f.read_text() != original
    _tool_summarize_undo(config, str(f), True)                   # undo restores
    assert f.read_text() == original


def test_summarize_apply_no_config():
    assert "error" in _tool_summarize_apply(None, None, False)


def test_mcp_server_tool_count():
    """Verify that the MCP server registers the expected number of tools.
    This catches documentation drift when new tools are added or removed.
    If this test fails, update docs/claude-code-integration.md and
    docs/architecture.md with the new tool count and tool list.
    """
    import re
    from pathlib import Path
    from tokenjam.mcp import server as mcp_module

    # Read server.py and count @mcp.tool() decorators
    server_path = Path(mcp_module.__file__)
    server_src = server_path.read_text()

    # Find all @mcp.tool() decorators followed by function definitions
    tool_pattern = r'@mcp\.tool\(\)\s*\ndef\s+(\w+)\s*\('
    tools_found = re.findall(tool_pattern, server_src)

    # The expected tool count; update this when new tools are added
    expected_count = 23
    tool_count = len(tools_found)
    assert tool_count == expected_count, (
        f"MCP tool count mismatch: found {tool_count}, expected {expected_count}. "
        f"Update docs/claude-code-integration.md (tool table) and "
        f"docs/architecture.md (tool count reference) to match. "
        f"Tools: {sorted(tools_found)}"
    )
