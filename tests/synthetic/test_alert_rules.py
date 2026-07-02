"""Synthetic tests for alert rules — uses span factories + mock StorageBackend."""
from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock

from tokenjam.core.alerts import (
    AlertDispatcher,
    AlertEngine,
    FileChannel,
    StdoutChannel,
    WebhookChannel,
    _alert_to_dict,
    _strip_sensitive,
    SENSITIVE_DETAIL_KEYS,
)
from tokenjam.core.config import (
    AgentConfig,
    AlertChannelConfig,
    AlertsConfig,
    BudgetConfig,
    DefaultsConfig,
    TjConfig,
    SensitiveAction,
)
from tokenjam.core.models import Alert, AlertType, Severity, SpanStatus
from tokenjam.otel.semconv import TjAttributes
from tokenjam.utils.ids import new_uuid
from tokenjam.utils.time_parse import utcnow
from tests.factories import make_llm_span, make_session, make_tool_span


def _make_config(
    agents: dict | None = None,
    cooldown_seconds: int = 0,
    include_captured_content: bool = False,
) -> TjConfig:
    """Build a minimal TjConfig for alert tests."""
    return TjConfig(
        version="1",
        agents=agents or {},
        alerts=AlertsConfig(
            cooldown_seconds=cooldown_seconds,
            include_captured_content=include_captured_content,
            channels=[AlertChannelConfig(type="stdout")],
        ),
    )


def _make_engine(
    config: TjConfig | None = None,
    db: MagicMock | None = None,
) -> tuple[AlertEngine, MagicMock]:
    """Create an AlertEngine with a mock DB."""
    if db is None:
        db = MagicMock()
    if config is None:
        config = _make_config()
    engine = AlertEngine(db, config)
    # Silence stdout channel in tests
    engine.dispatcher.channels = []
    return engine, db


# ── Sensitive action ────���──────────────────────────────────────────────────

def test_sensitive_action_fires_on_matching_tool():
    config = _make_config(agents={
        "test-agent": AgentConfig(
            sensitive_actions=[SensitiveAction(name="send_email", severity="critical")],
        ),
    })
    engine, db = _make_engine(config)
    span = make_tool_span(agent_id="test-agent", tool_name="send_email")
    engine.evaluate(span)
    db.insert_alert.assert_called_once()
    alert: Alert = db.insert_alert.call_args[0][0]
    assert alert.type == AlertType.SENSITIVE_ACTION
    assert alert.severity == Severity.CRITICAL


def test_sensitive_action_does_not_fire_on_non_matching_tool():
    config = _make_config(agents={
        "test-agent": AgentConfig(
            sensitive_actions=[SensitiveAction(name="send_email")],
        ),
    })
    engine, db = _make_engine(config)
    span = make_tool_span(agent_id="test-agent", tool_name="read_file")
    engine.evaluate(span)
    db.insert_alert.assert_not_called()


def test_sensitive_action_uses_configured_severity():
    config = _make_config(agents={
        "test-agent": AgentConfig(
            sensitive_actions=[SensitiveAction(name="delete_db", severity="info")],
        ),
    })
    engine, db = _make_engine(config)
    span = make_tool_span(agent_id="test-agent", tool_name="delete_db")
    engine.evaluate(span)
    alert: Alert = db.insert_alert.call_args[0][0]
    assert alert.severity == Severity.INFO


# ── Retry loop ─────────────────────────────────────────────────────────────

def test_retry_loop_fires_at_4_calls_not_3():
    # A genuine loop = the SAME tool with IDENTICAL arguments repeated.
    config = _make_config()
    engine, db = _make_engine(config)
    session_id = new_uuid()
    trace_id = "a" * 32
    same = {"url": "http://x"}

    # 3 identical calls — no alert
    three_spans = [
        make_tool_span(tool_name="fetch_url", trace_id=trace_id, tool_input=same)
        for _ in range(3)
    ]
    db.get_recent_spans.return_value = three_spans
    span3 = make_tool_span(tool_name="fetch_url", trace_id=trace_id, tool_input=same)
    span3.session_id = session_id
    engine.evaluate(span3)
    db.insert_alert.assert_not_called()

    # 4th identical call — should fire
    four_spans = three_spans + [
        make_tool_span(tool_name="fetch_url", trace_id=trace_id, tool_input=same)]
    db.get_recent_spans.return_value = four_spans
    span4 = make_tool_span(tool_name="fetch_url", trace_id=trace_id, tool_input=same)
    span4.session_id = session_id
    engine.evaluate(span4)
    db.insert_alert.assert_called_once()
    alert: Alert = db.insert_alert.call_args[0][0]
    assert alert.type == AlertType.RETRY_LOOP


def test_retry_loop_ignores_different_arguments():
    # 4 calls to the SAME tool with DIFFERENT args is normal work, not a loop.
    config = _make_config()
    engine, db = _make_engine(config)
    session_id = new_uuid()
    spans = [
        make_tool_span(tool_name="Bash", tool_input={"cmd": f"echo {i}"})
        for i in range(4)
    ]
    db.get_recent_spans.return_value = spans
    span = make_tool_span(tool_name="Bash", tool_input={"cmd": "echo 5"})
    span.session_id = session_id
    engine.evaluate(span)
    db.insert_alert.assert_not_called()


def test_retry_loop_skipped_without_tool_arguments():
    # Telemetry with no tool args (Claude Code over OTLP) can't prove a repeat.
    config = _make_config()
    engine, db = _make_engine(config)
    session_id = new_uuid()
    spans = [make_tool_span(tool_name="Read") for _ in range(5)]  # no tool_input
    db.get_recent_spans.return_value = spans
    span = make_tool_span(tool_name="Read")
    span.session_id = session_id
    engine.evaluate(span)
    db.insert_alert.assert_not_called()


def test_retry_loop_ignores_decision_event_spans():
    # Non-execution spans (claude_code.tool_decision) must not count, even with
    # identical args — only real gen_ai.tool.call executions do.
    config = _make_config()
    engine, db = _make_engine(config)
    session_id = new_uuid()
    same = {"cmd": "ls"}
    spans = [
        make_tool_span(tool_name="Bash", tool_input=same,
                       name="claude_code.tool_decision")
        for _ in range(5)
    ]
    db.get_recent_spans.return_value = spans
    span = make_tool_span(tool_name="Bash", tool_input=same,
                          name="claude_code.tool_decision")
    span.session_id = session_id
    engine.evaluate(span)
    db.insert_alert.assert_not_called()


def test_retry_loop_ignores_spans_without_session():
    engine, db = _make_engine()
    span = make_tool_span(tool_name="fetch_url")
    span.session_id = None
    engine.evaluate(span)
    db.get_recent_spans.assert_not_called()


# ── Cost budgets ─────���─────────────────────────────────────────────────────

def test_cost_budget_session_fires_when_exceeded():
    config = _make_config(agents={
        "test-agent": AgentConfig(budget=BudgetConfig(session_usd=1.00)),
    })
    engine, db = _make_engine(config)
    session = make_session(agent_id="test-agent", total_cost_usd=1.50)
    engine.evaluate_session_end(session)
    db.insert_alert.assert_called()
    alert: Alert = db.insert_alert.call_args[0][0]
    assert alert.type == AlertType.COST_BUDGET_SESSION


def test_cost_budget_session_does_not_fire_under_budget():
    config = _make_config(agents={
        "test-agent": AgentConfig(budget=BudgetConfig(session_usd=5.00)),
    })
    engine, db = _make_engine(config)
    session = make_session(agent_id="test-agent", total_cost_usd=1.00)
    engine.evaluate_session_end(session)
    db.insert_alert.assert_not_called()


def test_cost_budget_daily_fires_when_exceeded():
    config = _make_config(agents={
        "test-agent": AgentConfig(budget=BudgetConfig(daily_usd=10.00)),
    })
    engine, db = _make_engine(config)
    db.get_daily_cost.return_value = 12.50
    session = make_session(agent_id="test-agent")
    engine.evaluate_session_end(session)
    db.insert_alert.assert_called()
    alert: Alert = db.insert_alert.call_args[0][0]
    assert alert.type == AlertType.COST_BUDGET_DAILY


def test_cost_budget_daily_inherits_from_defaults_when_agent_has_only_session():
    """Regression: agent with session_usd but no daily_usd should still enforce defaults.daily_usd."""
    config = _make_config(agents={
        "test-agent": AgentConfig(budget=BudgetConfig(session_usd=5.0)),
    })
    config.defaults = DefaultsConfig(budget=BudgetConfig(daily_usd=10.0))
    engine, db = _make_engine(config)
    db.get_daily_cost.return_value = 12.50
    session = make_session(agent_id="test-agent")
    engine.evaluate_session_end(session)
    alerts = [call[0][0] for call in db.insert_alert.call_args_list]
    daily_alerts = [a for a in alerts if a.type == AlertType.COST_BUDGET_DAILY]
    assert len(daily_alerts) == 1


# ── Session duration ────���─────────────────────────────────────���────────────

def test_session_duration_fires_when_exceeded():
    engine, db = _make_engine()
    session = make_session(agent_id="test-agent", duration_seconds=4000.0)
    engine.evaluate_session_end(session)
    db.insert_alert.assert_called()
    alert: Alert = db.insert_alert.call_args[0][0]
    assert alert.type == AlertType.SESSION_DURATION


def test_session_duration_does_not_fire_under_threshold():
    engine, db = _make_engine()
    session = make_session(agent_id="test-agent", duration_seconds=600.0)
    engine.evaluate_session_end(session)
    # No SESSION_DURATION alert should fire
    for call in db.insert_alert.call_args_list:
        alert: Alert = call[0][0]
        assert alert.type != AlertType.SESSION_DURATION


# ── Content stripping ──��───────────────────────────────────────────────────

def test_content_stripped_from_external_payload():
    detail = {
        "message": "test",
        "prompt_content": "secret prompt",
        "completion_content": "secret completion",
        "tool_input": "secret input",
        "tool_output": "secret output",
    }
    stripped = _strip_sensitive(detail)
    assert "message" in stripped
    for key in SENSITIVE_DETAIL_KEYS:
        assert key not in stripped


def test_content_not_stripped_when_include_is_true():
    detail = {
        "message": "test",
        "prompt_content": "visible prompt",
    }
    # _strip_sensitive is only called when strip=True
    full = _alert_to_dict(
        Alert(
            alert_id="x",
            fired_at=utcnow(),
            type=AlertType.SENSITIVE_ACTION,
            severity=Severity.WARNING,
            title="test",
            detail=detail,
        ),
        strip=False,
    )
    assert full["detail"]["prompt_content"] == "visible prompt"


def test_stdout_channel_always_gets_full_payload():
    """StdoutChannel never strips content — verified by its send() not calling _strip_sensitive."""
    ch = StdoutChannel()
    # StdoutChannel doesn't have _include_captured_content — it always shows full detail
    assert not hasattr(ch, "_include_captured_content")


def test_file_channel_always_gets_full_payload():
    """FileChannel always stores full detail regardless of config."""
    ch = FileChannel("/tmp/test.jsonl", include_captured_content=False)
    assert ch._include_captured_content is True


# ── Suppression ───────────────���────────────────────────────────────────────

def test_suppressed_alert_persisted_to_db():
    config = _make_config(cooldown_seconds=300)
    engine, db = _make_engine(config)
    config_with_agent = _make_config(
        agents={"test-agent": AgentConfig(
            sensitive_actions=[SensitiveAction(name="send_email")],
        )},
        cooldown_seconds=300,
    )
    engine, db = _make_engine(config_with_agent)

    span = make_tool_span(agent_id="test-agent", tool_name="send_email")

    # First alert — not suppressed
    engine.evaluate(span)
    assert db.insert_alert.call_count == 1
    first_alert: Alert = db.insert_alert.call_args_list[0][0][0]
    assert not first_alert.suppressed

    # Second alert — suppressed but still persisted
    engine.evaluate(span)
    assert db.insert_alert.call_count == 2
    second_alert: Alert = db.insert_alert.call_args_list[1][0][0]
    assert second_alert.suppressed


def test_suppressed_alert_not_dispatched_to_channels():
    config = _make_config(
        agents={"test-agent": AgentConfig(
            sensitive_actions=[SensitiveAction(name="send_email")],
        )},
        cooldown_seconds=300,
    )
    engine, db = _make_engine(config)
    mock_channel = MagicMock()
    engine.dispatcher.channels = [mock_channel]

    span = make_tool_span(agent_id="test-agent", tool_name="send_email")

    # First — dispatched
    engine.evaluate(span)
    assert mock_channel.send.call_count == 1

    # Second — suppressed, not dispatched
    engine.evaluate(span)
    assert mock_channel.send.call_count == 1  # still 1


# ── Sandbox events ──────────��──────────────────────────────────────────────

def test_sandbox_network_blocked_fires_correct_alert_type():
    engine, db = _make_engine()
    span = make_tool_span(
        agent_id="test-agent",
        tool_name="http_request",
    )
    span.attributes[TjAttributes.SANDBOX_EVENT] = "network_blocked"
    span.attributes[TjAttributes.EGRESS_HOST] = "evil.com"
    span.attributes[TjAttributes.EGRESS_PORT] = 443
    engine.evaluate(span)
    db.insert_alert.assert_called_once()
    alert: Alert = db.insert_alert.call_args[0][0]
    assert alert.type == AlertType.NETWORK_EGRESS_BLOCKED
    assert alert.severity == Severity.CRITICAL
    assert alert.detail["host"] == "evil.com"


def test_sandbox_fs_denied_fires():
    engine, db = _make_engine()
    span = make_tool_span(agent_id="test-agent", tool_name="write_file")
    span.attributes[TjAttributes.SANDBOX_EVENT] = "fs_denied"
    span.attributes[TjAttributes.FILESYSTEM_PATH] = "/etc/passwd"
    engine.evaluate(span)
    alert: Alert = db.insert_alert.call_args[0][0]
    assert alert.type == AlertType.FILESYSTEM_ACCESS_DENIED


def test_sandbox_syscall_denied_fires():
    engine, db = _make_engine()
    span = make_tool_span(agent_id="test-agent", tool_name="exec")
    span.attributes[TjAttributes.SANDBOX_EVENT] = "syscall_denied"
    span.attributes[TjAttributes.SYSCALL_NAME] = "ptrace"
    engine.evaluate(span)
    alert: Alert = db.insert_alert.call_args[0][0]
    assert alert.type == AlertType.SYSCALL_DENIED


def test_sandbox_inference_rerouted_fires():
    engine, db = _make_engine()
    span = make_llm_span(agent_id="test-agent")
    span.attributes[TjAttributes.SANDBOX_EVENT] = "inference_rerouted"
    engine.evaluate(span)
    alert: Alert = db.insert_alert.call_args[0][0]
    assert alert.type == AlertType.INFERENCE_REROUTED


def test_no_sandbox_event_is_noop():
    engine, db = _make_engine()
    span = make_tool_span(agent_id="test-agent", tool_name="normal_tool")
    engine.evaluate(span)
    # No sandbox event, no sensitive action, no session_id → no alerts
    db.insert_alert.assert_not_called()


# ── Failure rate ───────────────────────────────────────────────────────────

def test_failure_rate_fires_when_threshold_exceeded():
    engine, db = _make_engine()
    session_id = new_uuid()

    # 20 spans, 5 errors → 25% > 20% threshold
    # error_count=5 is divisible by 5, so it will fire
    spans = []
    for i in range(20):
        status = "error" if i < 5 else "ok"
        s = make_tool_span(agent_id="test-agent", tool_name=f"tool_{i}", status=status)
        spans.append(s)

    db.get_recent_spans.return_value = spans

    error_span = make_tool_span(
        agent_id="test-agent", tool_name="failing_tool", status="error"
    )
    error_span.session_id = session_id
    error_span.status_code = SpanStatus.ERROR
    engine.evaluate(error_span)
    db.insert_alert.assert_called()
    alert: Alert = db.insert_alert.call_args[0][0]
    assert alert.type == AlertType.FAILURE_RATE


def test_failure_rate_fires_once_per_session():
    # A struggling session is one incident — further errors must not re-fire.
    engine, db = _make_engine()
    session_id = new_uuid()

    def errored_window(n_errors):
        spans = []
        for i in range(20):
            status = "error" if i < n_errors else "ok"
            spans.append(make_tool_span(agent_id="test-agent",
                                        tool_name=f"t{i}", status=status))
        return spans

    def fire(n_errors):
        db.get_recent_spans.return_value = errored_window(n_errors)
        s = make_tool_span(agent_id="test-agent", tool_name="boom", status="error")
        s.session_id = session_id
        s.status_code = SpanStatus.ERROR
        engine.evaluate(s)

    fire(5)   # crosses 20% -> one alert
    fire(8)   # worse, same session -> must NOT re-fire
    fire(12)  # worse still -> must NOT re-fire
    failure_alerts = [
        c.args[0] for c in db.insert_alert.call_args_list
        if c.args[0].type == AlertType.FAILURE_RATE
    ]
    assert len(failure_alerts) == 1


# ── External fire() entry point ───────────���────────────────────────────────

def test_fire_external_creates_alert_from_span():
    engine, db = _make_engine()
    span = make_llm_span(agent_id="test-agent")
    engine.fire(
        AlertType.SCHEMA_VIOLATION,
        span,
        detail={"message": "schema mismatch"},
        severity=Severity.WARNING,
    )
    db.insert_alert.assert_called_once()
    alert: Alert = db.insert_alert.call_args[0][0]
    assert alert.type == AlertType.SCHEMA_VIOLATION
    assert alert.span_id == span.span_id


def test_fire_external_creates_alert_from_session():
    engine, db = _make_engine()
    session = make_session(agent_id="test-agent")
    engine.fire(
        AlertType.DRIFT_DETECTED,
        session,
        detail={"message": "drift detected"},
    )
    db.insert_alert.assert_called_once()
    alert: Alert = db.insert_alert.call_args[0][0]
    assert alert.type == AlertType.DRIFT_DETECTED
    assert alert.span_id is None
    assert alert.session_id == session.session_id
