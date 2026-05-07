"""Integration tests for CLI commands using CliRunner."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from tj.cli.main import cli
from tj.core.config import AgentConfig, BudgetConfig, TjConfig
from tj.core.db import InMemoryBackend
from tj.core.models import (
    AgentRecord,
    Alert,
    AlertType,
    Severity,
)
from tj.utils.ids import new_uuid
from tj.utils.time_parse import utcnow
from tests.factories import make_llm_span, make_session


@pytest.fixture
def db():
    backend = InMemoryBackend()
    yield backend
    backend.close()


@pytest.fixture
def config():
    return TjConfig(
        version="1",
        agents={"test-agent": AgentConfig(budget=BudgetConfig(daily_usd=5.0))},
    )


@pytest.fixture
def runner():
    return CliRunner()


def _invoke(runner, db, config, args):
    """Invoke CLI with patched db and config."""
    with patch("tj.cli.main.load_config", return_value=config), \
         patch("tj.cli.main.open_db", return_value=db):
        return runner.invoke(cli, args)


def _seed_agent_and_session(db, agent_id="test-agent"):
    """Insert an agent and session into the DB for tests that need them."""
    now = utcnow()
    agent = AgentRecord(
        agent_id=agent_id, first_seen=now, last_seen=now,
    )
    db.upsert_agent(agent)
    session = make_session(agent_id=agent_id, status="completed")
    db.upsert_session(session)
    return session


def _seed_alert(db, agent_id="test-agent", acknowledged=False, suppressed=False):
    """Insert an alert into the DB."""
    alert = Alert(
        alert_id=new_uuid(),
        fired_at=utcnow(),
        type=AlertType.COST_BUDGET_DAILY,
        severity=Severity.WARNING,
        title="Daily budget exceeded",
        detail={"message": "Agent exceeded $5.00 daily budget"},
        agent_id=agent_id,
        acknowledged=acknowledged,
        suppressed=suppressed,
    )
    db.insert_alert(alert)
    return alert


# -- status tests --

def test_status_exits_0_when_no_alerts(runner, db, config):
    _seed_agent_and_session(db)
    result = _invoke(runner, db, config, ["status", "--json"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["has_active_alerts"] is False


def test_status_exits_1_when_active_alerts(runner, db, config):
    _seed_agent_and_session(db)
    _seed_alert(db, acknowledged=False)
    result = _invoke(runner, db, config, ["status", "--json"])
    assert result.exit_code == 1
    data = json.loads(result.output)
    assert data["has_active_alerts"] is True


# -- traces tests --

def test_traces_json_output_is_valid_json(runner, db, config):
    _seed_agent_and_session(db)
    span = make_llm_span(agent_id="test-agent")
    db.upsert_agent(AgentRecord(
        agent_id="test-agent", first_seen=utcnow(), last_seen=utcnow(),
    ))
    db.insert_span(span)

    result = _invoke(runner, db, config, ["traces", "--json"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert isinstance(data, list)


def test_trace_id_shows_span_waterfall(runner, db, config):
    span = make_llm_span(agent_id="test-agent")
    db.upsert_agent(AgentRecord(
        agent_id="test-agent", first_seen=utcnow(), last_seen=utcnow(),
    ))
    session = make_session(agent_id="test-agent")
    db.upsert_session(session)
    db.insert_span(span)

    result = _invoke(runner, db, config, ["trace", span.trace_id, "--json"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert len(data) >= 1
    assert data[0]["span_id"] == span.span_id


# -- export tests --

def test_export_openevals_format_is_message_list(runner, db, config):
    span = make_llm_span(
        agent_id="test-agent",
        extra_attributes={
            "gen_ai.prompt.content": "Hello",
            "gen_ai.completion.content": "Hi there",
        },
    )
    db.upsert_agent(AgentRecord(
        agent_id="test-agent", first_seen=utcnow(), last_seen=utcnow(),
    ))
    session = make_session(agent_id="test-agent")
    db.upsert_session(session)
    db.insert_span(span)

    result = _invoke(runner, db, config, ["export", "--format", "openevals", "--since", "1h"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert isinstance(data, list)
    if data:
        assert "messages" in data[0]
        assert "trace_id" in data[0]


# -- doctor tests --

def test_doctor_exits_0_when_config_is_clean(runner, db, config, tmp_path):
    config_file = tmp_path / "tj.toml"
    config_file.write_text('version = "1"\n')
    # Set DB path to a writable temp location
    config.storage.path = str(tmp_path / "test.duckdb")
    # Set ingest secret so no warning fires
    config.security.ingest_secret = "test-secret"
    # Disable drift so no "insufficient sessions" warning fires
    config.agents["test-agent"].drift.enabled = False
    with patch("tj.cli.cmd_doctor.find_config_file", return_value=config_file):
        result = _invoke(runner, db, config, ["doctor", "--json"])
    assert result.exit_code == 0


def test_doctor_exits_1_when_warnings_present(runner, db, config, tmp_path):
    # No ingest secret => warning
    config.security.ingest_secret = ""
    config.storage.path = str(tmp_path / "test.duckdb")
    config.agents["test-agent"].drift.enabled = False
    config_file = tmp_path / "tj.toml"
    config_file.write_text('version = "1"\n')
    with patch("tj.cli.cmd_doctor.find_config_file", return_value=config_file):
        result = _invoke(runner, db, config, ["doctor", "--json"])
    assert result.exit_code == 1
    checks = json.loads(result.output)
    warnings = [c for c in checks if c["level"] == "warning"]
    assert len(warnings) > 0


def test_doctor_exits_2_when_errors_present(runner, db, config):
    # No config file found => error
    with patch("tj.cli.cmd_doctor.find_config_file", return_value=None):
        result = _invoke(runner, db, config, ["doctor", "--json"])
    assert result.exit_code == 2


def test_doctor_warns_on_schema_without_capture(runner, db, config, tmp_path):
    config.agents["test-agent"] = AgentConfig(output_schema="schema.json")
    config.capture.tool_outputs = False
    config_file = tmp_path / "tj.toml"
    config_file.write_text('version = "1"\n')
    with patch("tj.cli.cmd_doctor.find_config_file", return_value=config_file):
        result = _invoke(runner, db, config, ["doctor", "--json"])
    checks = json.loads(result.output)
    schema_checks = [c for c in checks if c["name"] == "Schema vs capture"]
    assert any(c["level"] == "warning" for c in schema_checks)


# -- since flag parsing --

def test_since_flag_parses_all_formats(runner, db, config):
    _seed_agent_and_session(db)
    for since_val in ["30m", "1h", "7d", "2026-03-01"]:
        result = _invoke(runner, db, config, ["traces", "--since", since_val, "--json"])
        assert result.exit_code == 0, f"Failed for --since {since_val}: {result.output}"


# -- drift tests --

def _seed_baseline(db, agent_id="test-agent"):
    """Insert a DriftBaseline into the DB."""
    from tj.core.models import DriftBaseline
    baseline = DriftBaseline(
        agent_id=agent_id,
        sessions_sampled=15,
        computed_at=utcnow(),
        avg_input_tokens=12400.0,
        stddev_input_tokens=3200.0,
        avg_output_tokens=1800.0,
        stddev_output_tokens=400.0,
        avg_session_duration_s=145.0,
        stddev_session_duration=32.0,
        avg_tool_call_count=24.0,
        stddev_tool_call_count=8.0,
        common_tool_sequences=[["Read", "Write", "Bash"]],
    )
    db.upsert_baseline(baseline)
    return baseline


def test_drift_no_baselines_exits_0(runner, db, config):
    result = _invoke(runner, db, config, ["drift", "--json"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["drifted"] is False
    assert data["agents"] == []


def test_drift_with_baseline_no_violations(runner, db, config):
    """Normal session (tokens within threshold) -> exit 0."""
    # Use a baseline without tool sequences so Jaccard doesn't fire
    from tj.core.models import DriftBaseline
    baseline = DriftBaseline(
        agent_id="test-agent",
        sessions_sampled=15,
        computed_at=utcnow(),
        avg_input_tokens=12400.0,
        stddev_input_tokens=3200.0,
        avg_output_tokens=1800.0,
        stddev_output_tokens=400.0,
        avg_session_duration_s=145.0,
        stddev_session_duration=32.0,
        avg_tool_call_count=24.0,
        stddev_tool_call_count=8.0,
        common_tool_sequences=None,  # skip Jaccard check
    )
    db.upsert_baseline(baseline)
    # Session with tokens close to baseline mean -> no drift
    session = make_session(
        agent_id="test-agent",
        input_tokens=12000,  # close to 12400 mean
        output_tokens=1750,
        tool_call_count=22,
        status="completed",
        duration_seconds=150.0,
    )
    db.upsert_session(session)

    result = _invoke(runner, db, config, ["drift"])
    assert result.exit_code == 0


def test_drift_with_violations_exits_1(runner, db, config):
    """Outlier session (far from baseline) -> exit 1."""
    _seed_baseline(db)
    # Session with very high input tokens -> drift
    session = make_session(
        agent_id="test-agent",
        input_tokens=50000,  # >> 12400 mean, z >> 2.0
        output_tokens=1800,
        tool_call_count=24,
        status="completed",
        duration_seconds=145.0,
    )
    db.upsert_session(session)

    result = _invoke(runner, db, config, ["drift"])
    assert result.exit_code == 1


def test_drift_json_output(runner, db, config):
    _seed_baseline(db)
    session = make_session(
        agent_id="test-agent",
        input_tokens=50000,
        output_tokens=1800,
        status="completed",
        duration_seconds=145.0,
    )
    db.upsert_session(session)

    result = _invoke(runner, db, config, ["drift", "--json"])
    assert result.exit_code == 1
    data = json.loads(result.output)
    assert "agents" in data
    assert "drifted" in data
    assert data["drifted"] is True
    assert len(data["agents"]) == 1
    agent_data = data["agents"][0]
    assert agent_data["agent_id"] == "test-agent"
    assert "violations" in agent_data
    assert "metrics" in agent_data


def test_drift_agent_filter(runner, db, config):
    """--agent filters output to the specified agent."""
    _seed_baseline(db, agent_id="test-agent")
    _seed_baseline(db, agent_id="other-agent")

    session = make_session(agent_id="test-agent", status="completed")
    db.upsert_session(session)

    result = _invoke(runner, db, config, ["drift", "--agent", "other-agent", "--json"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    # Only other-agent queried, no sessions -> no results
    assert all(a["agent_id"] == "other-agent" for a in data["agents"])


# -- onboard --claude-code tests --

def test_onboard_claude_code_writes_settings(runner, tmp_path):
    """--claude-code writes env vars to ~/.claude/settings.json."""
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    settings_path = fake_home / ".claude" / "settings.json"

    with patch("tj.cli.cmd_onboard.find_config_file", return_value=None), \
         patch("tj.cli.cmd_onboard.Path.home", return_value=fake_home), \
         patch("tj.cli.cmd_onboard.click.confirm", return_value=False):
        result = runner.invoke(cli, ["onboard", "--claude-code", "--no-daemon", "--budget", "5.0"])

    assert result.exit_code == 0
    assert settings_path.exists()
    data = json.loads(settings_path.read_text())
    assert "env" in data
    env = data["env"]
    assert env["CLAUDE_CODE_ENABLE_TELEMETRY"] == "1"
    assert env["OTEL_LOGS_EXPORTER"] == "otlp"
    assert "OTEL_EXPORTER_OTLP_ENDPOINT" in env


def test_onboard_claude_code_preserves_existing(runner, tmp_path):
    """Existing settings.json keys are not clobbered."""
    fake_home = tmp_path / "home"
    (fake_home / ".claude").mkdir(parents=True)
    settings_path = fake_home / ".claude" / "settings.json"
    settings_path.write_text(
        json.dumps({"theme": "dark", "env": {"MY_VAR": "preserved"}}) + "\n"
    )

    with patch("tj.cli.cmd_onboard.find_config_file", return_value=None), \
         patch("tj.cli.cmd_onboard.Path.home", return_value=fake_home), \
         patch("tj.cli.cmd_onboard.click.confirm", return_value=False):
        runner.invoke(cli, ["onboard", "--claude-code", "--no-daemon", "--budget", "5.0"])

    data = json.loads(settings_path.read_text())
    # Original top-level key preserved
    assert data.get("theme") == "dark"
    # Original env var preserved
    assert data["env"].get("MY_VAR") == "preserved"
    # New env vars added
    assert data["env"].get("CLAUDE_CODE_ENABLE_TELEMETRY") == "1"


def test_onboard_claude_code_creates_tj_config(runner, tmp_path):
    """ocw config is created when none exists."""
    fake_home = tmp_path / "home"
    fake_home.mkdir()

    with patch("tj.cli.cmd_onboard.find_config_file", return_value=None), \
         patch("tj.cli.cmd_onboard.Path.home", return_value=fake_home), \
         patch("tj.cli.cmd_onboard.click.confirm", return_value=False), \
         patch("tj.core.config.write_config") as mock_write:
        runner.invoke(cli, ["onboard", "--claude-code", "--no-daemon", "--budget", "5.0"])

    # write_config should have been called with an TjConfig containing a claude-code-* agent
    assert mock_write.called
    saved_config = mock_write.call_args[0][0]
    assert any(k.startswith("claude-code-") for k in saved_config.agents)


def test_onboard_claude_code_prompts_for_budget(runner, tmp_path):
    """Budget prompt is shown when --budget is not passed."""
    fake_home = tmp_path / "home"
    fake_home.mkdir()

    with patch("tj.cli.cmd_onboard.find_config_file", return_value=None), \
         patch("tj.cli.cmd_onboard.Path.home", return_value=fake_home), \
         patch("tj.cli.cmd_onboard.click.confirm", return_value=False), \
         patch("tj.core.config.write_config") as mock_write:
        result = runner.invoke(cli, ["onboard", "--claude-code", "--no-daemon"], input="7.0\n")

    assert result.exit_code == 0
    assert mock_write.called
    saved_config = mock_write.call_args[0][0]
    agent_id = next(k for k in saved_config.agents if k.startswith("claude-code-"))
    assert saved_config.agents[agent_id].budget.daily_usd == 7.0


def test_onboard_claude_code_resyncs_secret_on_rerun(runner, tmp_path):
    """Re-running --claude-code always writes the current secret, even if OTLP already configured.

    Regression test: previously the guard `if OTEL_EXPORTER_OTLP_ENDPOINT not in global_env`
    silently skipped updating the secret when settings.json already existed, causing 401s
    whenever the OCW config was regenerated without re-running onboard --claude-code.
    """
    fake_home = tmp_path / "home"
    (fake_home / ".claude").mkdir(parents=True)
    settings_path = fake_home / ".claude" / "settings.json"

    old_secret = "aabbccdd" * 8
    new_secret = "11223344" * 8

    # Simulate settings.json already configured with a now-stale secret
    settings_path.write_text(json.dumps({
        "env": {
            "CLAUDE_CODE_ENABLE_TELEMETRY": "1",
            "OTEL_LOGS_EXPORTER": "otlp",
            "OTEL_EXPORTER_OTLP_PROTOCOL": "http/json",
            "OTEL_EXPORTER_OTLP_ENDPOINT": "http://127.0.0.1:7391",
            "OTEL_EXPORTER_OTLP_HEADERS": f"Authorization=Bearer {old_secret}",
        }
    }) + "\n")

    # Global config path — --claude-code always uses ~/.config/tj/config.toml
    fake_config_path = fake_home / ".config" / "tj" / "config.toml"
    fake_config_path.parent.mkdir(parents=True)
    fake_config_path.touch()  # must exist so the "existing and not force" branch runs

    from tj.core.config import AgentConfig, TjConfig, SecurityConfig
    fake_config = TjConfig(
        version="1",
        agents={"claude-code-tokenjam": AgentConfig()},
        security=SecurityConfig(ingest_secret=new_secret),
    )

    with patch("tj.core.config.load_config", return_value=fake_config), \
         patch("tj.core.config.write_config"), \
         patch("tj.cli.cmd_onboard.Path.home", return_value=fake_home), \
         patch("tj.cli.cmd_onboard.click.confirm", return_value=False):
        result = runner.invoke(cli, ["onboard", "--claude-code", "--no-daemon", "--budget", "5.0"])

    assert result.exit_code == 0
    data = json.loads(settings_path.read_text())
    # Secret must be updated to match the current OCW config, not left as stale old value
    assert data["env"]["OTEL_EXPORTER_OTLP_HEADERS"] == f"Authorization=Bearer {new_secret}"


def test_onboard_claude_code_preserves_custom_otlp_headers(runner, tmp_path):
    """Re-running --claude-code does NOT overwrite manually customised OTEL_EXPORTER_OTLP_HEADERS.

    Only headers that were previously written by ocw (i.e. contain 'Authorization=Bearer')
    are eligible for syncing. A header set by the user to something else is left untouched.
    """
    fake_home = tmp_path / "home"
    (fake_home / ".claude").mkdir(parents=True)
    settings_path = fake_home / ".claude" / "settings.json"

    custom_header = "X-Custom-Token: my-own-value"

    settings_path.write_text(json.dumps({
        "env": {
            "CLAUDE_CODE_ENABLE_TELEMETRY": "1",
            "OTEL_LOGS_EXPORTER": "otlp",
            "OTEL_EXPORTER_OTLP_PROTOCOL": "http/json",
            "OTEL_EXPORTER_OTLP_ENDPOINT": "http://127.0.0.1:7391",
            "OTEL_EXPORTER_OTLP_HEADERS": custom_header,
        }
    }) + "\n")

    # Global config path — --claude-code always uses ~/.config/tj/config.toml
    fake_config_path = fake_home / ".config" / "tj" / "config.toml"
    fake_config_path.parent.mkdir(parents=True)
    fake_config_path.touch()  # must exist so the "existing and not force" branch runs

    from tj.core.config import AgentConfig, TjConfig, SecurityConfig
    fake_config = TjConfig(
        version="1",
        agents={"claude-code-tokenjam": AgentConfig()},
        security=SecurityConfig(ingest_secret="newsecret" * 4),
    )

    with patch("tj.core.config.load_config", return_value=fake_config), \
         patch("tj.core.config.write_config"), \
         patch("tj.cli.cmd_onboard.Path.home", return_value=fake_home), \
         patch("tj.cli.cmd_onboard.click.confirm", return_value=False):
        result = runner.invoke(cli, ["onboard", "--claude-code", "--no-daemon", "--budget", "5.0"])

    assert result.exit_code == 0
    data = json.loads(settings_path.read_text())
    # Manually customised header must survive the re-run
    assert data["env"]["OTEL_EXPORTER_OTLP_HEADERS"] == custom_header


def test_onboard_does_not_prompt_for_daemon(runner, tmp_path):
    """Regression: ocw onboard should auto-install the daemon without
    prompting. The prompt was removed in v0.1.6 but reappeared.
    See v0.1.7 fix."""
    with patch("tj.cli.cmd_onboard.find_config_file", return_value=None), \
         patch("tj.cli.cmd_onboard._install_daemon", return_value="Daemon installed") as mock_daemon:
        result = runner.invoke(cli, ["onboard", "--budget", "5.0"], input="")
    assert result.exit_code == 0
    # Daemon should be auto-installed (not prompted)
    mock_daemon.assert_called_once()
    # Should NOT contain the interactive prompt text
    assert "Install background daemon" not in result.output


def test_onboard_no_daemon_skips_install(runner, tmp_path):
    """--no-daemon flag should skip daemon installation entirely."""
    with patch("tj.cli.cmd_onboard.find_config_file", return_value=None), \
         patch("tj.cli.cmd_onboard._install_daemon") as mock_daemon:
        result = runner.invoke(cli, ["onboard", "--budget", "5.0", "--no-daemon"])
    assert result.exit_code == 0
    mock_daemon.assert_not_called()


def test_budget_show_displays_defaults(runner, db, config):
    """ocw budget with no flags shows current budgets."""
    result = _invoke(runner, db, config, ["budget"])
    assert result.exit_code == 0
    assert "5.00" in result.output  # fixture has daily_usd=5.0


def test_budget_set_global_writes_config(runner, db, config, tmp_path):
    """ocw budget --daily updates global defaults and writes config."""
    config_file = tmp_path / "config.toml"
    config_file.write_text("")

    with patch("tj.cli.main.load_config", return_value=config), \
         patch("tj.cli.main.open_db", return_value=db), \
         patch("tj.cli.cmd_budget.find_config_file", return_value=str(config_file)), \
         patch("tj.cli.cmd_budget.write_config") as mock_write:
        result = runner.invoke(cli, ["budget", "--daily", "8.0"])

    assert result.exit_code == 0
    assert mock_write.called
    saved_config = mock_write.call_args[0][0]
    assert saved_config.defaults.budget.daily_usd == 8.0


def test_budget_set_agent_writes_config(runner, db, config, tmp_path):
    """ocw budget --agent --daily --session updates per-agent budget."""
    config_file = tmp_path / "config.toml"
    config_file.write_text("")

    with patch("tj.cli.main.load_config", return_value=config), \
         patch("tj.cli.main.open_db", return_value=db), \
         patch("tj.cli.cmd_budget.find_config_file", return_value=str(config_file)), \
         patch("tj.cli.cmd_budget.write_config") as mock_write:
        result = runner.invoke(
            cli, ["budget", "--agent", "test-agent", "--daily", "3.0", "--session", "0.25"]
        )

    assert result.exit_code == 0
    assert mock_write.called
    saved_config = mock_write.call_args[0][0]
    assert saved_config.agents["test-agent"].budget.daily_usd == 3.0
    assert saved_config.agents["test-agent"].budget.session_usd == 0.25


def test_budget_set_negative_daily_rejected(runner, db, config, tmp_path):
    """ocw budget --daily -5 should error, not silently clear the limit."""
    config_file = tmp_path / "config.toml"
    config_file.write_text("")

    with patch("tj.cli.main.load_config", return_value=config), \
         patch("tj.cli.main.open_db", return_value=db), \
         patch("tj.cli.cmd_budget.find_config_file", return_value=str(config_file)), \
         patch("tj.cli.cmd_budget.write_config") as mock_write:
        result = runner.invoke(cli, ["budget", "--daily", "-5"])

    assert result.exit_code != 0
    assert "non-negative" in result.output.lower()
    mock_write.assert_not_called()
