"""Integration tests for CLI commands using CliRunner."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from tokenjam.cli.main import cli
from tokenjam.core.config import AgentConfig, BudgetConfig, TjConfig
from tokenjam.core.db import InMemoryBackend
from tokenjam.core.models import (
    AgentRecord,
    Alert,
    AlertType,
    Severity,
)
from tokenjam.utils.ids import new_uuid
from tokenjam.utils.time_parse import utcnow
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


@pytest.fixture(autouse=True)
def _isolate_real_world_side_effects(tmp_path_factory, monkeypatch):
    """Keep every CLI test off real user/daemon state so the file runs to
    completion deterministically, in any order (#23).

    The ``tj onboard``/``tj onboard --claude-code`` path otherwise reaches out
    to live, machine-global resources as a side effect of the command — none of
    which these tests are actually asserting on:

    * runs the real ``ingest_claude_code`` against ``~/.claude/projects`` (6 GB+
      on a dev box) and writes the real ``~/.tj/telemetry.duckdb``;
    * registers the real ``claude`` MCP server via ``subprocess``;
    * pokes the real ``tj serve`` daemon to release the DuckDB write lock.

    Because the DuckDB write lock is process- and order-dependent, whenever the
    daemon (or a sibling test) is holding it, ``ingest_claude_code`` blocks on
    ``upsert_session`` forever — so the file passes test-by-test but hangs when
    run as a group. Pointing every machine-global resource at a throwaway temp
    location (and stubbing the daemon/MCP shell-outs) makes the suite hermetic.
    """
    iso = tmp_path_factory.mktemp("cli-iso")

    # No Claude Code logs to ingest -> onboard skips the open_db + ingest block
    # entirely, so it never touches the real telemetry DB or its write lock.
    monkeypatch.setattr(
        "tokenjam.core.backfill.CLAUDE_CODE_PROJECTS_ROOT",
        iso / "no-such-claude-projects",
        raising=False,
    )
    # Never shell out to the real `claude` CLI to register the MCP server.
    monkeypatch.setattr("tokenjam.cli.cmd_onboard.shutil.which", lambda _name: None)
    monkeypatch.setattr(
        "tokenjam.cli.cmd_onboard.subprocess.run",
        lambda *a, **k: subprocess.CompletedProcess(a[0] if a else [], 0, b"", b""),
    )
    # Never touch the real `tj serve` daemon lifecycle.
    monkeypatch.setattr(
        "tokenjam.cli.cmd_onboard._stop_serve_for_db_write", lambda: False
    )
    # Storage paths resolve via os.path.expanduser($HOME); redirect them at the
    # default ``~/.tj/telemetry.duckdb`` so any stray open_db lands in tmp.
    monkeypatch.setenv("HOME", str(iso))


def _invoke(runner, db, config, args):
    """Invoke CLI with patched db and config."""
    with patch("tokenjam.cli.main.load_config", return_value=config), \
         patch("tokenjam.cli.main.open_db", return_value=db):
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


def _seed_alert(db, agent_id="test-agent", acknowledged=False, suppressed=False,
                 alert_type=AlertType.COST_BUDGET_DAILY, title="Daily budget exceeded"):
    """Insert an alert into the DB."""
    alert = Alert(
        alert_id=new_uuid(),
        fired_at=utcnow(),
        type=alert_type,
        severity=Severity.WARNING,
        title=title,
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


def test_status_json_includes_active_and_elapsed(runner, db, config):
    """Status JSON carries active (compute) time + wall-clock elapsed (#147)."""
    session = _seed_agent_and_session(db)
    for ms in (1500.0, 2500.0):
        sp = make_llm_span(agent_id="test-agent", duration_ms=ms)
        sp.session_id = session.session_id
        db.insert_span(sp)

    result = _invoke(runner, db, config, ["status", "--json"])
    assert result.exit_code == 0
    a = json.loads(result.output)["agents"][0]
    assert "active_seconds" in a and "duration_seconds" in a
    assert a["active_seconds"] == 4.0   # 4000 ms of spans


def test_status_dedupes_duplicate_alerts_of_same_type(runner, db, config):
    """The alert engine's in-memory dedup state (CooldownTracker,
    `_failure_rate_fired`) resets on process restart, so the DB can
    legitimately hold two rows for the same (type, agent) pair (#96). The
    human-readable `tj status` render must collapse repeats into one line
    with a count rather than printing the same alert line twice."""
    _seed_agent_and_session(db)
    _seed_alert(db, alert_type=AlertType.FAILURE_RATE, title="failure_rate — test-agent")
    _seed_alert(db, alert_type=AlertType.FAILURE_RATE, title="failure_rate — test-agent")
    result = _invoke(runner, db, config, ["status"])
    assert result.exit_code == 1
    assert result.output.count("failure_rate — test-agent") == 1
    assert "×2" in result.output


def test_status_distinct_alert_types_are_not_collapsed(runner, db, config):
    """Two DIFFERENT alert types for the same agent must both still render —
    dedup is keyed on (type, agent), not agent alone."""
    _seed_agent_and_session(db)
    _seed_alert(db, alert_type=AlertType.FAILURE_RATE, title="failure_rate — test-agent")
    _seed_alert(db, alert_type=AlertType.RETRY_LOOP, title="retry_loop — test-agent")
    result = _invoke(runner, db, config, ["status"])
    assert "failure_rate — test-agent" in result.output
    assert "retry_loop — test-agent" in result.output
    assert "×2" not in result.output


def test_status_subscription_plan_shows_no_raw_dollar_line(runner, db):
    """Subscription-tier users must not see a raw '$0.00' Cost today line
    (#96) — the workspace rule is subscription users see plan-share framing,
    never raw spend they don't pay. Mirrors `tj cost`'s honesty note."""
    from tokenjam.core.config import ProviderBudget

    session = make_session(agent_id="test-agent", plan_tier="max_5x")
    db.upsert_session(session)
    sub_config = TjConfig(version="1", budgets={"anthropic": ProviderBudget(plan="max_5x")})

    result = _invoke(runner, db, sub_config, ["status"])
    assert result.exit_code == 0
    assert "$0.00" not in result.output
    assert "$0.000000" not in result.output
    assert "Cost today:     0.0% of cycle" in result.output
    assert "Subscription plan" in result.output


def test_status_api_plan_keeps_raw_dollar_line(runner, db, config):
    """API-billed users keep the historical raw-dollar Cost today line."""
    _seed_agent_and_session(db)
    result = _invoke(runner, db, config, ["status"])
    assert result.exit_code == 0
    assert "Cost today:     $0.00" in result.output
    assert "Subscription plan" not in result.output


def test_status_subscription_plan_with_daily_limit_shows_literal_dollar_cap(runner, db):
    """A subscription-framed agent with a per-agent `daily_usd` limit must show
    that limit as its literal dollar amount with a `/day` qualifier, not as a
    percentage of the monthly subscription cycle — a user-configured DAILY
    dollar cap has no relationship to the MONTHLY cycle `render_dollar` uses
    for framed spend. Only the spend-so-far figure stays plan-tier-framed."""
    from tokenjam.core.config import ProviderBudget

    session = make_session(agent_id="test-agent", plan_tier="max_5x")
    db.upsert_session(session)
    sub_config = TjConfig(
        version="1",
        budgets={"anthropic": ProviderBudget(plan="max_5x")},
        agents={"test-agent": AgentConfig(budget=BudgetConfig(daily_usd=5.0))},
    )

    result = _invoke(runner, db, sub_config, ["status"])
    assert result.exit_code == 0
    assert "Cost today:     0.0% of cycle / $5.00/day limit" in result.output
    assert "% of cycle limit" not in result.output


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
    config_file = tmp_path / "tokenjam.toml"
    config_file.write_text('version = "1"\n')
    # Set DB path to a writable temp location
    config.storage.path = str(tmp_path / "test.duckdb")
    # Set ingest secret so no warning fires
    config.security.ingest_secret = "test-secret"
    # Disable drift so no "insufficient sessions" warning fires
    config.agents["test-agent"].drift.enabled = False
    with patch("tokenjam.cli.cmd_doctor.find_config_file", return_value=config_file):
        result = _invoke(runner, db, config, ["doctor", "--json"])
    assert result.exit_code == 0


def test_doctor_exits_1_when_warnings_present(runner, db, config, tmp_path):
    # No ingest secret => warning
    config.security.ingest_secret = ""
    config.storage.path = str(tmp_path / "test.duckdb")
    config.agents["test-agent"].drift.enabled = False
    config_file = tmp_path / "tokenjam.toml"
    config_file.write_text('version = "1"\n')
    with patch("tokenjam.cli.cmd_doctor.find_config_file", return_value=config_file):
        result = _invoke(runner, db, config, ["doctor", "--json"])
    assert result.exit_code == 1
    checks = json.loads(result.output)
    warnings = [c for c in checks if c["level"] == "warning"]
    assert len(warnings) > 0


def test_doctor_exits_2_when_errors_present(runner, db, config):
    # No config file found => error
    with patch("tokenjam.cli.cmd_doctor.find_config_file", return_value=None):
        result = _invoke(runner, db, config, ["doctor", "--json"])
    assert result.exit_code == 2


def _clean_doctor_config(config, tmp_path):
    """Configure `config` so only the check under test can produce a warning."""
    config.storage.path = str(tmp_path / "test.duckdb")
    config.security.ingest_secret = "test-secret"
    config.agents["test-agent"].drift.enabled = False


def test_doctor_no_staleness_warning_with_fresh_spans(runner, db, config, tmp_path):
    """A just-recorded span keeps the freshness check green (no false alarm)."""
    _clean_doctor_config(config, tmp_path)
    span = make_llm_span(agent_id="test-agent", start_time=utcnow())
    db.insert_span(span)
    config_file = tmp_path / "tokenjam.toml"
    config_file.write_text('version = "1"\n')
    with patch("tokenjam.cli.cmd_doctor.find_config_file", return_value=config_file):
        result = _invoke(runner, db, config, ["doctor", "--json"])
    assert result.exit_code == 0
    checks = json.loads(result.output)
    freshness = [c for c in checks if c["name"] == "Live-span freshness"]
    assert freshness and freshness[0]["level"] == "ok"


def test_doctor_warns_on_stale_spans(runner, db, config, tmp_path):
    """A newest span older than the 6h threshold warns (exit 1) with a restart hint."""
    from datetime import timedelta

    _clean_doctor_config(config, tmp_path)
    span = make_llm_span(agent_id="test-agent", start_time=utcnow() - timedelta(hours=10))
    db.insert_span(span)
    config_file = tmp_path / "tokenjam.toml"
    config_file.write_text('version = "1"\n')
    with patch("tokenjam.cli.cmd_doctor.find_config_file", return_value=config_file):
        result = _invoke(runner, db, config, ["doctor", "--json"])
    assert result.exit_code == 1
    checks = json.loads(result.output)
    freshness = [c for c in checks if c["name"] == "Live-span freshness"]
    assert freshness and freshness[0]["level"] == "warning"
    assert "restart" in freshness[0]["message"].lower()


def test_doctor_no_staleness_warning_when_no_spans(runner, db, config, tmp_path):
    """An empty DB (pre-onboard) must not be mistaken for a stalled connection."""
    _clean_doctor_config(config, tmp_path)
    config_file = tmp_path / "tokenjam.toml"
    config_file.write_text('version = "1"\n')
    with patch("tokenjam.cli.cmd_doctor.find_config_file", return_value=config_file):
        result = _invoke(runner, db, config, ["doctor", "--json"])
    assert result.exit_code == 0
    checks = json.loads(result.output)
    freshness = [c for c in checks if c["name"] == "Live-span freshness"]
    assert freshness and freshness[0]["level"] == "info"
    assert not any(c["level"] == "warning" for c in checks)


def test_doctor_onboarding_signal_info_when_silent(runner, db, config, tmp_path):
    """Onboarded-but-zero-spans (#80) surfaces an actionable info line — never a
    warning, so a fresh setup still exits 0."""
    _clean_doctor_config(config, tmp_path)
    config_file = tmp_path / "tokenjam.toml"
    config_file.write_text('version = "1"\n')
    with patch("tokenjam.cli.cmd_doctor.find_config_file", return_value=config_file):
        result = _invoke(runner, db, config, ["doctor", "--json"])
    assert result.exit_code == 0
    checks = json.loads(result.output)
    signal = [c for c in checks if c["name"] == "Onboarding signal"]
    assert signal and signal[0]["level"] == "info"
    assert "no spans" in signal[0]["message"].lower()


def test_doctor_onboarding_signal_ok_with_spans(runner, db, config, tmp_path):
    """Once any span exists, the onboarding-signal check goes green."""
    _clean_doctor_config(config, tmp_path)
    db.insert_span(make_llm_span(agent_id="test-agent", start_time=utcnow()))
    config_file = tmp_path / "tokenjam.toml"
    config_file.write_text('version = "1"\n')
    with patch("tokenjam.cli.cmd_doctor.find_config_file", return_value=config_file):
        result = _invoke(runner, db, config, ["doctor", "--json"])
    assert result.exit_code == 0
    checks = json.loads(result.output)
    signal = [c for c in checks if c["name"] == "Onboarding signal"]
    assert signal and signal[0]["level"] == "ok"


def test_doctor_warns_on_schema_without_capture(runner, db, config, tmp_path):
    config.agents["test-agent"] = AgentConfig(output_schema="schema.json")
    config.capture.tool_outputs = False
    config_file = tmp_path / "tokenjam.toml"
    config_file.write_text('version = "1"\n')
    with patch("tokenjam.cli.cmd_doctor.find_config_file", return_value=config_file):
        result = _invoke(runner, db, config, ["doctor", "--json"])
    checks = json.loads(result.output)
    schema_checks = [c for c in checks if c["name"] == "Schema vs capture"]
    assert any(c["level"] == "warning" for c in schema_checks)


def _duckdb_with_dropped_migration_7(tmp_path):
    """A real DuckDBBackend reproduced in the #55 state: migration 7 recorded
    applied, but its request_params/request_tools columns dropped after open
    (DuckDBBackend self-heals on open, so we corrupt the live conn afterward)."""
    from tokenjam.core.config import StorageConfig
    from tokenjam.core.db import DuckDBBackend

    backend = DuckDBBackend(StorageConfig(path=str(tmp_path / "telemetry.duckdb")))
    for idx in (
        "idx_spans_trace_id", "idx_spans_agent_id", "idx_spans_start_time",
        "idx_spans_tool_name", "idx_spans_conv_id",
    ):
        backend.conn.execute(f"DROP INDEX IF EXISTS {idx}")
    backend.conn.execute("ALTER TABLE spans DROP COLUMN request_params")
    backend.conn.execute("ALTER TABLE spans DROP COLUMN request_tools")
    return backend


def test_doctor_warns_on_missing_schema_column(runner, config, tmp_path):
    """`tj doctor` flags a recorded-but-unlanded migration (missing column, #55)."""
    _clean_doctor_config(config, tmp_path)
    backend = _duckdb_with_dropped_migration_7(tmp_path)
    config_file = tmp_path / "tokenjam.toml"
    config_file.write_text('version = "1"\n')
    try:
        with patch("tokenjam.cli.cmd_doctor.find_config_file", return_value=config_file):
            result = _invoke(runner, backend, config, ["doctor", "--json"])
        assert result.exit_code == 1
        checks = json.loads(result.output)
        integrity = [c for c in checks if c["name"] == "Schema integrity"]
        assert integrity and integrity[0]["level"] == "warning"
        assert "request_params" in integrity[0]["message"]
    finally:
        backend.close()


def test_doctor_repair_heals_missing_schema_column(runner, config, tmp_path):
    """`tj doctor --repair` reconciles the missing columns; a re-run is clean (#55)."""
    _clean_doctor_config(config, tmp_path)
    backend = _duckdb_with_dropped_migration_7(tmp_path)
    config_file = tmp_path / "tokenjam.toml"
    config_file.write_text('version = "1"\n')
    try:
        with patch("tokenjam.cli.cmd_doctor.find_config_file", return_value=config_file):
            repair = _invoke(runner, backend, config, ["doctor", "--repair"])
        assert "Schema reconciled" in repair.output
        cols = {
            r[0]
            for r in backend.conn.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name = 'spans'"
            ).fetchall()
        }
        assert {"request_params", "request_tools"} <= cols
        # A follow-up doctor run now reports the schema as healthy.
        with patch("tokenjam.cli.cmd_doctor.find_config_file", return_value=config_file):
            again = _invoke(runner, backend, config, ["doctor", "--json"])
        checks = json.loads(again.output)
        integrity = [c for c in checks if c["name"] == "Schema integrity"]
        assert integrity and integrity[0]["level"] == "ok"
    finally:
        backend.close()


def test_doctor_mcp_wiring_checks(runner, db, config, tmp_path):
    # Set up a fake home directory and fake cwd
    fake_home = tmp_path / "fake_home"
    fake_home.mkdir()
    fake_cwd = tmp_path / "fake_cwd"
    fake_cwd.mkdir()

    # Stub doctor config to avoid other warning noise
    _clean_doctor_config(config, tmp_path)
    config_file = tmp_path / "tokenjam.toml"
    config_file.write_text('version = "1"\n')

    # Case 1: No files, no executables -> should be level: "info"
    with patch("pathlib.Path.home", return_value=fake_home), \
         patch("pathlib.Path.cwd", return_value=fake_cwd), \
         patch("shutil.which", return_value=None), \
         patch("tokenjam.cli.cmd_doctor.find_config_file", return_value=config_file):
        result = _invoke(runner, db, config, ["doctor", "--json"])
        assert result.exit_code == 0
        checks = json.loads(result.output)
        mcp_checks = [c for c in checks if c["name"] == "MCP wiring"]
        assert mcp_checks and mcp_checks[0]["level"] == "info"

    # Case 2: Claude Code CLI is on PATH but the MCP is NOT registered. Post-#59
    # the MCP is an SDK-only surface, so its absence for a coding agent is the
    # correct state and must NOT warn (it's "info"). Instead the STATUSLINE check
    # warns, because the zero-token statusline isn't wired yet.
    with patch("pathlib.Path.home", return_value=fake_home), \
         patch("pathlib.Path.cwd", return_value=fake_cwd), \
         patch("shutil.which", side_effect=lambda cmd: "/bin/claude" if cmd == "claude" else None), \
         patch("tokenjam.cli.cmd_doctor.find_config_file", return_value=config_file):
        result = _invoke(runner, db, config, ["doctor", "--json"])
        checks = json.loads(result.output)
        mcp_checks = [c for c in checks if c["name"] == "MCP wiring"]
        assert mcp_checks and mcp_checks[0]["level"] == "info"
        # The MCP must never steer a Claude Code user to register it.
        assert "SDK" in mcp_checks[0]["message"]
        statusline_checks = [c for c in checks if c["name"] == "Statusline wiring"]
        assert statusline_checks and statusline_checks[0]["level"] == "warning"
        assert "tj onboard --claude-code" in statusline_checks[0]["message"]

    # Case 3: MCP registered globally in Codex (~/.codex/config.toml) -> should
    # be level "warning" (#94): Codex gets the same +36% quota-tax reasoning as
    # Claude Code (see cmd_onboard.py's Codex path, which actively retires this
    # legacy block), so a green check here would contradict the #59 decision.
    codex_dir = fake_home / ".codex"
    codex_dir.mkdir(parents=True, exist_ok=True)
    codex_config = codex_dir / "config.toml"
    codex_config.write_text("[mcp_servers.tj]\ncommand = 'tj'\nargs = ['mcp']\n")

    with patch("pathlib.Path.home", return_value=fake_home), \
         patch("pathlib.Path.cwd", return_value=fake_cwd), \
         patch("shutil.which", return_value=None), \
         patch("tokenjam.cli.cmd_doctor.find_config_file", return_value=config_file):
        result = _invoke(runner, db, config, ["doctor", "--json"])
        assert result.exit_code == 1
        checks = json.loads(result.output)
        mcp_checks = [c for c in checks if c["name"] == "MCP wiring"]
        assert mcp_checks and mcp_checks[0]["level"] == "warning"
        assert "Codex" in mcp_checks[0]["message"]
        assert "#59" in mcp_checks[0]["message"]

    # Clean up codex file
    codex_config.unlink()
    codex_dir.rmdir()

    # Case 4: MCP registered globally in Claude Code (~/.claude.json) -> should
    # be level "warning" (#94) — a green check for this state contradicts the
    # #59 decision to keep MCP out of the Claude Code loop.
    claude_json = fake_home / ".claude.json"
    claude_json.write_text('{"mcpServers": {"tj": {"command": "tj"}}}')

    with patch("pathlib.Path.home", return_value=fake_home), \
         patch("pathlib.Path.cwd", return_value=fake_cwd), \
         patch("shutil.which", return_value=None), \
         patch("tokenjam.cli.cmd_doctor.find_config_file", return_value=config_file):
        result = _invoke(runner, db, config, ["doctor", "--json"])
        assert result.exit_code == 1
        checks = json.loads(result.output)
        mcp_checks = [c for c in checks if c["name"] == "MCP wiring"]
        assert mcp_checks and mcp_checks[0]["level"] == "warning"
        assert "Claude Code" in mcp_checks[0]["message"]
        assert "claude mcp remove tj --scope user" in mcp_checks[0]["message"]

    # Clean up claude json file
    claude_json.unlink()

    # Case 5: MCP registered locally in project-level .mcp.json -> should be
    # level "warning" (#94) — same quota-tax reasoning applies at project scope.
    project_mcp = fake_cwd / ".mcp.json"
    project_mcp.write_text('{"mcpServers": {"tj": {"command": "tj"}}}')

    with patch("pathlib.Path.home", return_value=fake_home), \
         patch("pathlib.Path.cwd", return_value=fake_cwd), \
         patch("shutil.which", return_value=None), \
         patch("tokenjam.cli.cmd_doctor.find_config_file", return_value=config_file):
        result = _invoke(runner, db, config, ["doctor", "--json"])
        assert result.exit_code == 1
        checks = json.loads(result.output)
        mcp_checks = [c for c in checks if c["name"] == "MCP wiring"]
        assert mcp_checks and mcp_checks[0]["level"] == "warning"
        assert "project scope" in mcp_checks[0]["message"]


def test_doctor_mcp_wiring_message_renders_bracket_literal(runner, db, config, tmp_path):
    """The Codex removal hint contains the literal TOML section header
    `[mcp_servers.tj]`. In human (non-JSON) rendering, `_print_check` feeds
    the message straight into `console.print`, and Rich treats an unescaped
    `[...]` as a markup tag — stripping it instead of printing it (same class
    of bug as #157 / PR #407). The console-rendered check must show the
    bracket text verbatim."""
    fake_home = tmp_path / "fake_home"
    fake_home.mkdir()
    fake_cwd = tmp_path / "fake_cwd"
    fake_cwd.mkdir()

    _clean_doctor_config(config, tmp_path)
    config_file = tmp_path / "tokenjam.toml"
    config_file.write_text('version = "1"\n')

    codex_dir = fake_home / ".codex"
    codex_dir.mkdir(parents=True, exist_ok=True)
    (codex_dir / "config.toml").write_text("[mcp_servers.tj]\ncommand = 'tj'\nargs = ['mcp']\n")

    with patch("pathlib.Path.home", return_value=fake_home), \
         patch("pathlib.Path.cwd", return_value=fake_cwd), \
         patch("shutil.which", return_value=None), \
         patch("tokenjam.cli.cmd_doctor.find_config_file", return_value=config_file):
        result = _invoke(runner, db, config, ["doctor"])

    assert "[mcp_servers.tj]" in result.output


# -- since flag parsing --

def test_since_flag_parses_all_formats(runner, db, config):
    _seed_agent_and_session(db)
    for since_val in ["30m", "1h", "7d", "2026-03-01"]:
        result = _invoke(runner, db, config, ["traces", "--since", since_val, "--json"])
        assert result.exit_code == 0, f"Failed for --since {since_val}: {result.output}"


# -- drift tests --

def _seed_baseline(db, agent_id="test-agent"):
    """Insert a DriftBaseline into the DB."""
    from tokenjam.core.models import DriftBaseline
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


def test_drift_no_baselines_warning_message(runner, db, config):
    result = _invoke(runner, db, config, ["drift"])
    assert result.exit_code == 0
    assert "No drift baselines found." in result.output


def test_drift_with_baseline_no_violations(runner, db, config):
    """Normal session (tokens within threshold) -> exit 0."""
    # Use a baseline without tool sequences so Jaccard doesn't fire
    from tokenjam.core.models import DriftBaseline
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

    with patch("tokenjam.cli.cmd_onboard.find_config_file", return_value=None), \
         patch("tokenjam.cli.cmd_onboard.Path.home", return_value=fake_home), \
         patch("tokenjam.cli.cmd_onboard.click.confirm", return_value=False):
        result = runner.invoke(cli, ["onboard", "--claude-code", "--no-daemon", "--budget", "5.0", "--plan", "max_20x", "--project", "aquanode"])

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

    with patch("tokenjam.cli.cmd_onboard.find_config_file", return_value=None), \
         patch("tokenjam.cli.cmd_onboard.Path.home", return_value=fake_home), \
         patch("tokenjam.cli.cmd_onboard.click.confirm", return_value=False):
        runner.invoke(cli, ["onboard", "--claude-code", "--no-daemon", "--budget", "5.0", "--plan", "max_20x", "--project", "aquanode"])

    data = json.loads(settings_path.read_text())
    # Original top-level key preserved
    assert data.get("theme") == "dark"
    # Original env var preserved
    assert data["env"].get("MY_VAR") == "preserved"
    # New env vars added
    assert data["env"].get("CLAUDE_CODE_ENABLE_TELEMETRY") == "1"


def test_onboard_claude_code_creates_tj_config(runner, tmp_path):
    """tj config is created when none exists."""
    fake_home = tmp_path / "home"
    fake_home.mkdir()

    with patch("tokenjam.cli.cmd_onboard.find_config_file", return_value=None), \
         patch("tokenjam.cli.cmd_onboard.Path.home", return_value=fake_home), \
         patch("tokenjam.cli.cmd_onboard.click.confirm", return_value=False), \
         patch("tokenjam.core.config.write_config") as mock_write:
        runner.invoke(cli, ["onboard", "--claude-code", "--no-daemon", "--budget", "5.0", "--plan", "max_20x", "--project", "aquanode"])

    # write_config should have been called with an TjConfig containing a claude-code-* agent
    assert mock_write.called
    saved_config = mock_write.call_args[0][0]
    assert any(k.startswith("claude-code-") for k in saved_config.agents)


def test_onboard_claude_code_prompts_for_budget(runner, tmp_path):
    """Budget prompt is shown when --budget is not passed."""
    fake_home = tmp_path / "home"
    fake_home.mkdir()

    with patch("tokenjam.cli.cmd_onboard.find_config_file", return_value=None), \
         patch("tokenjam.cli.cmd_onboard.Path.home", return_value=fake_home), \
         patch("tokenjam.cli.cmd_onboard.click.confirm", return_value=False), \
         patch("tokenjam.core.config.write_config") as mock_write:
        result = runner.invoke(cli, ["onboard", "--claude-code", "--no-daemon", "--plan", "max_20x", "--project", "aquanode"], input="7.0\n")

    assert result.exit_code == 0
    assert mock_write.called
    saved_config = mock_write.call_args[0][0]
    agent_id = next(k for k in saved_config.agents if k.startswith("claude-code-"))
    assert saved_config.agents[agent_id].budget.daily_usd == 7.0


def test_onboard_claude_code_project_flag_sets_config_project(runner, tmp_path):
    """--project sets [agents.<id>].project in config and does NOT write
    OTEL_RESOURCE_ATTRIBUTES into project settings (the claude wrapper owns it)."""
    from tokenjam.core.config import load_config

    fake_home = tmp_path / "home"
    fake_home.mkdir()

    with runner.isolated_filesystem() as cwd, \
         patch("tokenjam.cli.cmd_onboard.find_config_file", return_value=None), \
         patch("tokenjam.cli.cmd_onboard.Path.home", return_value=fake_home), \
         patch("tokenjam.cli.cmd_onboard.click.confirm", return_value=False):
        result = runner.invoke(cli, [
            "onboard", "--claude-code", "--no-daemon", "--budget", "5.0",
            "--plan", "max_20x", "--project", "aquanode",
        ])
        assert result.exit_code == 0
        env = json.loads(
            (Path(cwd) / ".claude" / "settings.json").read_text()
        ).get("env", {})
        assert "OTEL_RESOURCE_ATTRIBUTES" not in env

    cfg = load_config(str(fake_home / ".config" / "tj" / "config.toml"))
    agent_id = next(k for k in cfg.agents if k.startswith("claude-code-"))
    assert cfg.agents[agent_id].project == "aquanode"


def test_onboard_claude_code_prompts_for_project_name(runner, tmp_path):
    """Without --project, onboard prompts for a project name and stores it as
    the agent's configured project (not in project settings.json)."""
    from tokenjam.core.config import load_config

    fake_home = tmp_path / "home"
    fake_home.mkdir()

    with runner.isolated_filesystem() as cwd, \
         patch("tokenjam.cli.cmd_onboard.find_config_file", return_value=None), \
         patch("tokenjam.cli.cmd_onboard.Path.home", return_value=fake_home), \
         patch("tokenjam.cli.cmd_onboard.click.confirm", return_value=False):
        # First prompt is the project name; budget is supplied via flag.
        result = runner.invoke(cli, [
            "onboard", "--claude-code", "--no-daemon", "--budget", "5.0",
            "--plan", "max_20x",
        ], input="myproject\n")
        assert result.exit_code == 0
        env = json.loads(
            (Path(cwd) / ".claude" / "settings.json").read_text()
        ).get("env", {})
        assert "OTEL_RESOURCE_ATTRIBUTES" not in env

    cfg = load_config(str(fake_home / ".config" / "tj" / "config.toml"))
    agent_id = next(k for k in cfg.agents if k.startswith("claude-code-"))
    assert cfg.agents[agent_id].project == "myproject"


def test_onboard_claude_code_removes_existing_resource_attrs(runner, tmp_path):
    """A pre-existing env.OTEL_RESOURCE_ATTRIBUTES in project settings is removed
    (migrated); other env keys are preserved."""
    fake_home = tmp_path / "home"
    fake_home.mkdir()

    with runner.isolated_filesystem() as cwd, \
         patch("tokenjam.cli.cmd_onboard.find_config_file", return_value=None), \
         patch("tokenjam.cli.cmd_onboard.Path.home", return_value=fake_home), \
         patch("tokenjam.cli.cmd_onboard.click.confirm", return_value=False):
        proj_settings = Path(cwd) / ".claude" / "settings.json"
        proj_settings.parent.mkdir(parents=True)
        proj_settings.write_text(json.dumps({
            "env": {"OTEL_RESOURCE_ATTRIBUTES": "service.name=old", "KEEP": "yes"}
        }) + "\n")

        result = runner.invoke(cli, [
            "onboard", "--claude-code", "--no-daemon", "--budget", "5.0",
            "--plan", "max_20x", "--project", "aquanode",
        ])
        assert result.exit_code == 0
        env = json.loads(proj_settings.read_text())["env"]
        assert "OTEL_RESOURCE_ATTRIBUTES" not in env   # migrated away
        assert env["KEEP"] == "yes"                    # other keys untouched


def test_onboard_claude_code_resyncs_secret_on_rerun(runner, tmp_path):
    """Re-running --claude-code always writes the current secret, even if OTLP already configured.

    Regression test: previously the guard `if OTEL_EXPORTER_OTLP_ENDPOINT not in global_env`
    silently skipped updating the secret when settings.json already existed, causing 401s
    whenever the TokenJam config was regenerated without re-running onboard --claude-code.
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

    from tokenjam.core.config import AgentConfig, TjConfig, SecurityConfig
    fake_config = TjConfig(
        version="1",
        agents={"claude-code-tokenjam": AgentConfig()},
        security=SecurityConfig(ingest_secret=new_secret),
    )

    with patch("tokenjam.core.config.load_config", return_value=fake_config), \
         patch("tokenjam.core.config.write_config"), \
         patch("tokenjam.cli.cmd_onboard.Path.home", return_value=fake_home), \
         patch("tokenjam.cli.cmd_onboard.click.confirm", return_value=False):
        result = runner.invoke(cli, ["onboard", "--claude-code", "--no-daemon", "--budget", "5.0", "--plan", "max_20x", "--project", "aquanode"])

    assert result.exit_code == 0
    data = json.loads(settings_path.read_text())
    # Secret must be updated to match the current TokenJam config, not left as stale old value
    assert data["env"]["OTEL_EXPORTER_OTLP_HEADERS"] == f"Authorization=Bearer {new_secret}"


def test_onboard_claude_code_preserves_custom_otlp_headers(runner, tmp_path):
    """Re-running --claude-code does NOT overwrite manually customised OTEL_EXPORTER_OTLP_HEADERS.

    Only headers that were previously written by tj (i.e. contain 'Authorization=Bearer')
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

    from tokenjam.core.config import AgentConfig, TjConfig, SecurityConfig
    fake_config = TjConfig(
        version="1",
        agents={"claude-code-tokenjam": AgentConfig()},
        security=SecurityConfig(ingest_secret="newsecret" * 4),
    )

    with patch("tokenjam.core.config.load_config", return_value=fake_config), \
         patch("tokenjam.core.config.write_config"), \
         patch("tokenjam.cli.cmd_onboard.Path.home", return_value=fake_home), \
         patch("tokenjam.cli.cmd_onboard.click.confirm", return_value=False):
        result = runner.invoke(cli, ["onboard", "--claude-code", "--no-daemon", "--budget", "5.0", "--plan", "max_20x", "--project", "aquanode"])

    assert result.exit_code == 0
    data = json.loads(settings_path.read_text())
    # Manually customised header must survive the re-run
    assert data["env"]["OTEL_EXPORTER_OTLP_HEADERS"] == custom_header


def test_onboard_does_not_prompt_for_daemon(runner, tmp_path):
    """Regression: tj onboard should auto-install the daemon without
    prompting. The prompt was removed in v0.1.6 but reappeared.
    See v0.1.7 fix."""
    with patch("tokenjam.cli.cmd_onboard.find_config_file", return_value=None), \
         patch("tokenjam.cli.cmd_onboard._install_daemon", return_value="Daemon installed") as mock_daemon:
        result = runner.invoke(cli, ["onboard", "--budget", "5.0"], input="")
    assert result.exit_code == 0
    # Daemon should be auto-installed (not prompted)
    mock_daemon.assert_called_once()
    # Should NOT contain the interactive prompt text
    assert "Install background daemon" not in result.output


def test_onboard_no_daemon_skips_install(runner, tmp_path):
    """--no-daemon flag should skip daemon installation entirely."""
    with patch("tokenjam.cli.cmd_onboard.find_config_file", return_value=None), \
         patch("tokenjam.cli.cmd_onboard._install_daemon") as mock_daemon:
        result = runner.invoke(cli, ["onboard", "--budget", "5.0", "--no-daemon"])
    assert result.exit_code == 0
    mock_daemon.assert_not_called()


def test_budget_show_displays_defaults(runner, db, config):
    """tj budget with no flags shows current budgets."""
    result = _invoke(runner, db, config, ["budget"])
    assert result.exit_code == 0
    assert "5.00" in result.output  # fixture has daily_usd=5.0


def test_budget_show_json_outputs_valid_budget_rows(runner, db, config):
    """tj budget --json emits machine-readable rows."""
    result = _invoke(runner, db, config, ["budget", "--json"])

    assert result.exit_code == 0
    data = json.loads(result.output)
    test_agent = next(r for r in data if r["agent_id"] == "test-agent")
    assert data[0]["scope"] == "defaults"
    assert test_agent["daily_usd"] == 5.0
    assert test_agent["effective_daily_usd"] == 5.0


def test_budget_set_global_writes_config(runner, db, config, tmp_path):
    """tj budget --daily updates global defaults and writes config."""
    config_file = tmp_path / "config.toml"
    config_file.write_text("")

    with patch("tokenjam.cli.main.load_config", return_value=config), \
         patch("tokenjam.cli.main.open_db", return_value=db), \
         patch("tokenjam.cli.cmd_budget.find_config_file", return_value=str(config_file)), \
         patch("tokenjam.cli.cmd_budget.write_config") as mock_write:
        result = runner.invoke(cli, ["budget", "--daily", "8.0"])

    assert result.exit_code == 0
    assert mock_write.called
    saved_config = mock_write.call_args[0][0]
    assert saved_config.defaults.budget.daily_usd == 8.0


def test_budget_set_json_outputs_updated_values(runner, db, config, tmp_path):
    """tj budget --daily --json emits the updated budget values."""
    config_file = tmp_path / "config.toml"
    config_file.write_text("")

    with patch("tokenjam.cli.main.load_config", return_value=config), \
         patch("tokenjam.cli.main.open_db", return_value=db), \
         patch("tokenjam.cli.cmd_budget.find_config_file", return_value=str(config_file)), \
         patch("tokenjam.cli.cmd_budget.write_config"):
        result = runner.invoke(cli, ["budget", "--daily", "8.0", "--json"])

    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data == {
        "scope": "defaults",
        "agent_id": None,
        "daily_usd": 8.0,
        "session_usd": None,
        "effective_daily_usd": 8.0,
        "effective_session_usd": None,
    }


def test_budget_set_agent_writes_config(runner, db, config, tmp_path):
    """tj budget --agent --daily --session updates per-agent budget."""
    config_file = tmp_path / "config.toml"
    config_file.write_text("")

    with patch("tokenjam.cli.main.load_config", return_value=config), \
         patch("tokenjam.cli.main.open_db", return_value=db), \
         patch("tokenjam.cli.cmd_budget.find_config_file", return_value=str(config_file)), \
         patch("tokenjam.cli.cmd_budget.write_config") as mock_write:
        result = runner.invoke(
            cli, ["budget", "--agent", "test-agent", "--daily", "3.0", "--session", "0.25"]
        )

    assert result.exit_code == 0
    assert mock_write.called
    saved_config = mock_write.call_args[0][0]
    assert saved_config.agents["test-agent"].budget.daily_usd == 3.0
    assert saved_config.agents["test-agent"].budget.session_usd == 0.25


def test_optimize_empty_db_outputs_friendly_message(runner, db, config):
    result = _invoke(runner, db, config, ["optimize"])
    assert result.exit_code == 0
    assert "No usage data found" in result.output


def test_optimize_flags_downgrade_candidate(runner, db, config):
    """A small Opus session in the window should appear as a candidate."""
    from datetime import timedelta
    from tests.factories import make_llm_span
    from tokenjam.utils.time_parse import utcnow

    start = utcnow() - timedelta(days=2)
    span = make_llm_span(
        agent_id="test-agent",
        model="claude-opus-4-7",
        provider="anthropic",
        input_tokens=1000,
        output_tokens=200,
        cost_usd=0.030,
        session_id="s-opus",
        start_time=start,
    )
    db.insert_span(span)

    result = _invoke(runner, db, config, ["optimize"])
    assert result.exit_code == 0
    assert "Model downgrade" in result.output
    # Mandatory caveat must appear in human output
    assert "Candidate-flagging heuristic" in result.output


def test_optimize_json_output_includes_caveat(runner, db, config):
    from datetime import timedelta
    from tests.factories import make_llm_span
    from tokenjam.utils.time_parse import utcnow

    span = make_llm_span(
        agent_id="test-agent", model="claude-opus-4-7", provider="anthropic",
        input_tokens=1000, output_tokens=200, cost_usd=0.030,
        session_id="s", start_time=utcnow() - timedelta(days=1),
    )
    db.insert_span(span)

    result = _invoke(runner, db, config, ["optimize", "--json"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["downgrade"] is not None
    assert "Candidate-flagging heuristic" in data["downgrade"]["caveat"]


def _seed_optimize_window(db, *, plan_tier: str, sessions: int = 5,
                          billing_account: str = "anthropic") -> None:
    """
    Helper: insert N sessions with the given plan_tier plus matching LLM spans.
    Used by the v1.1 plan-tier-aware rendering tests below.
    """
    from datetime import timedelta
    from tests.factories import make_llm_span, make_session
    from tokenjam.utils.time_parse import utcnow

    for i in range(sessions):
        sess = make_session(
            agent_id="test-agent",
            session_id=f"s{i}",
            plan_tier=plan_tier,
            duration_seconds=60.0,
        )
        db.upsert_session(sess)
        span = make_llm_span(
            agent_id="test-agent",
            model="claude-opus-4-7",
            provider="anthropic",
            billing_account=billing_account,
            input_tokens=2_000, output_tokens=200, cost_usd=4.0,
            session_id=f"s{i}",
            start_time=utcnow() - timedelta(days=1),
        )
        db.insert_span(span)


def test_optimize_subscription_renders_implied_api_value(runner, db, config):
    """Subscription users see implied-API-value framing, never dollar 'spend'."""
    _seed_optimize_window(db, plan_tier="max_20x")

    result = _invoke(runner, db, config, ["optimize"])
    assert result.exit_code == 0
    out = result.output
    # Subscription header: plan label + implied API value
    assert "Max 20x plan" in out
    assert "Implied API value" in out
    # Token-share framing in downgrade body (no "$X/mo savings")
    assert "this cycle's tokens" in out or "% of this cycle's tokens" in out
    # The caveat line stays
    assert "Candidate-flagging heuristic" in out
    # The literal phrase "spend (last" must NOT appear in subscription rendering
    assert "spend (last" not in out


def test_optimize_local_suppresses_dollar_figures(runner, db, config):
    """Local-inference users see no dollar figures at all."""
    _seed_optimize_window(db, plan_tier="local", billing_account="local.ollama")

    result = _invoke(runner, db, config, ["optimize"])
    assert result.exit_code == 0
    out = result.output
    assert "Local inference" in out
    assert "no marginal cost" in out
    # Token framing in downgrade body
    if "Model downgrade" in out:
        assert "Relevant for capacity planning" in out


def test_optimize_api_mode_unchanged(runner, db, config):
    """API users see the existing dollar-denominated rendering."""
    _seed_optimize_window(db, plan_tier="api")

    result = _invoke(runner, db, config, ["optimize"])
    assert result.exit_code == 0
    out = result.output
    assert "spend (last" in out  # historical header
    assert "Implied API value" not in out


def test_optimize_json_includes_plan_and_pricing_mode(runner, db, config):
    """JSON output carries plan and pricing_mode fields."""
    _seed_optimize_window(db, plan_tier="max_20x")

    result = _invoke(runner, db, config, ["optimize", "--json"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["plan"] == "max_20x"
    assert data["pricing_mode"] == "subscription"
    # plan_tier_mix carries the raw counts
    assert data["plan_tier_mix"].get("max_20x", 0) > 0
    # For subscription, downgrade payload carries monthly_tokens_freed
    if data.get("downgrade"):
        assert "monthly_tokens_freed" in data["downgrade"]


def test_cost_compare_renders_window_diff(runner, db, config):
    """`tj cost --compare previous` produces a diff report against the prior window."""
    from datetime import timedelta
    from tests.factories import make_llm_span, make_session
    from tokenjam.utils.time_parse import utcnow

    now = utcnow()
    for i in range(3):
        s = make_session(session_id=f"prev-{i}", plan_tier="api")
        db.upsert_session(s)
        db.insert_span(make_llm_span(
            session_id=f"prev-{i}",
            input_tokens=1000, output_tokens=200, cost_usd=0.01,
            start_time=now - timedelta(days=10) + timedelta(hours=i),
        ))
    for i in range(3):
        s = make_session(session_id=f"cur-{i}", plan_tier="api")
        db.upsert_session(s)
        db.insert_span(make_llm_span(
            session_id=f"cur-{i}",
            input_tokens=2000, output_tokens=400, cost_usd=0.05,
            start_time=now - timedelta(days=3) + timedelta(hours=i),
        ))

    result = _invoke(runner, db, config, ["cost", "--since", "7d", "--compare", "previous"])
    assert result.exit_code == 0
    out = result.output
    # Header lines for both windows
    assert "Current" in out
    assert "Previous" in out
    # Cost delta + token delta lines
    assert "Cost delta" in out
    assert "Token delta" in out


def test_cost_compare_json_output(runner, db, config):
    """--compare combined with --json returns structured diff data."""
    from datetime import timedelta
    from tests.factories import make_llm_span, make_session
    from tokenjam.utils.time_parse import utcnow

    now = utcnow()
    s = make_session(session_id="cur", plan_tier="api")
    db.upsert_session(s)
    db.insert_span(make_llm_span(
        session_id="cur",
        input_tokens=1000, output_tokens=200, cost_usd=0.05,
        start_time=now - timedelta(days=3),
    ))

    result = _invoke(runner, db, config, [
        "cost", "--since", "7d", "--compare", "previous", "--json",
    ])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert "current" in data and "previous" in data
    assert "cost_delta_usd" in data
    assert "tokens_delta" in data


def test_cost_compare_invalid_keyword_rejected(runner, db, config):
    """Unknown --compare value raises BadParameter."""
    result = _invoke(runner, db, config, ["cost", "--compare", "yesterday"])
    assert result.exit_code != 0
    assert "Unknown --compare" in result.output


def test_optimize_compare_appends_window_diff(runner, db, config):
    """`tj optimize --compare previous` surfaces the diff alongside findings."""
    _seed_optimize_window(db, plan_tier="api")
    result = _invoke(runner, db, config, ["optimize", "--compare", "previous"])
    assert result.exit_code == 0
    # The diff section header is appended after the regular optimize report.
    assert "Window comparison" in result.output


def test_optimize_budget_projection_from_config(runner, db):
    """Budget configured via [budget.anthropic] should surface a projection."""
    from datetime import timedelta
    from tests.factories import make_llm_span
    from tokenjam.core.config import ProviderBudget
    from tokenjam.utils.time_parse import utcnow

    cfg = TjConfig(
        version="1",
        agents={"test-agent": AgentConfig(budget=BudgetConfig(daily_usd=5.0))},
        budgets={"anthropic": ProviderBudget(usd=10.0, cycle_start_day=1)},
    )
    # Insert spend that exceeds the small budget
    for i in range(5):
        span = make_llm_span(
            agent_id="test-agent", model="claude-opus-4-7", provider="anthropic",
            input_tokens=10_000, output_tokens=1_000, cost_usd=20.0,
            session_id=f"s{i}", start_time=utcnow() - timedelta(days=1),
        )
        db.insert_span(span)

    result = _invoke(runner, db, cfg, ["optimize", "budget-projection"])
    assert result.exit_code == 0
    assert "Budget projection" in result.output
    assert "anthropic" in result.output


def test_budget_set_negative_daily_rejected(runner, db, config, tmp_path):
    """tj budget --daily -5 should error, not silently clear the limit."""
    config_file = tmp_path / "config.toml"
    config_file.write_text("")

    with patch("tokenjam.cli.main.load_config", return_value=config), \
         patch("tokenjam.cli.main.open_db", return_value=db), \
         patch("tokenjam.cli.cmd_budget.find_config_file", return_value=str(config_file)), \
         patch("tokenjam.cli.cmd_budget.write_config") as mock_write:
        result = runner.invoke(cli, ["budget", "--daily", "-5"])

    assert result.exit_code != 0
    assert "non-negative" in result.output.lower()
    mock_write.assert_not_called()


# ---------------------------------------------------------------------------
# tj report --reuse / tj optimize reuse --export-templates  (issue #116)
# ---------------------------------------------------------------------------

def _reuse_config(completions: bool = True):
    from tokenjam.core.config import CaptureConfig
    return TjConfig(version="1", capture=CaptureConfig(completions=completions))


def _seed_reuse_cluster(db, *, count: int = 3, completions: bool = True):
    """Seed `count` sessions sharing a planning skeleton + tool sequence."""
    from datetime import timedelta

    from tokenjam.otel.semconv import GenAIAttributes
    from tests.factories import make_session, make_tool_span

    base = utcnow() - timedelta(days=2)
    for i in range(count):
        sid = f"reuse-{i}"
        db.upsert_session(make_session(agent_id="test-agent", session_id=sid))
        t0 = base + timedelta(minutes=i)
        attrs = (
            {GenAIAttributes.COMPLETION_CONTENT: f"Cut release v0.{i} then run tests"}
            if completions else None
        )
        plan = make_llm_span(
            agent_id="test-agent", session_id=sid, start_time=t0,
            cost_usd=0.20, input_tokens=1000, output_tokens=300,
            extra_attributes=attrs,
        )
        db.insert_span(plan)
        for j, tn in enumerate(["read_file", "run_test"]):
            ts = make_tool_span(tool_name=tn)
            ts.session_id = sid
            ts.start_time = t0 + timedelta(seconds=j + 1)
            db.insert_span(ts)


def test_report_reuse_writes_html_and_sidecars(runner, db, tmp_path, monkeypatch):
    monkeypatch.setenv("TOKENJAM_REPORT_DIR", str(tmp_path))
    _seed_reuse_cluster(db, count=3, completions=True)

    result = _invoke(runner, db, _reuse_config(completions=True),
                     ["report", "--reuse", "--no-open"])
    assert result.exit_code == 0, result.output
    assert "Reuse report written" in result.output
    htmls = list(tmp_path.glob("reuse-*.html"))
    mds = list(tmp_path.glob("reuse-*.md"))
    assert len(htmls) == 1
    assert len(mds) == 1                      # one cluster → one sidecar
    # The skeleton picked up the varying version token as a slot.
    assert "{{slot_1}}" in mds[0].read_text()
    assert "cluster_id:" in mds[0].read_text()


def test_report_reuse_sidecar_is_idempotent(runner, db, tmp_path, monkeypatch):
    monkeypatch.setenv("TOKENJAM_REPORT_DIR", str(tmp_path))
    _seed_reuse_cluster(db, count=3, completions=True)
    cfg = _reuse_config(completions=True)

    _invoke(runner, db, cfg, ["report", "--reuse", "--no-open"])
    _invoke(runner, db, cfg, ["report", "--reuse", "--no-open"])
    # Re-running overwrites the same cluster_id-keyed sidecar, not duplicates.
    assert len(list(tmp_path.glob("reuse-*.md"))) == 1


def test_report_reuse_empty_window_writes_nothing(runner, db, tmp_path, monkeypatch):
    monkeypatch.setenv("TOKENJAM_REPORT_DIR", str(tmp_path))
    result = _invoke(runner, db, _reuse_config(),
                     ["report", "--reuse", "--no-open"])
    assert result.exit_code == 0
    assert "No repeated planning detected" in result.output
    assert list(tmp_path.glob("reuse-*")) == []


def test_report_reuse_capture_off_renders_html_without_sidecar(
    runner, db, tmp_path, monkeypatch
):
    monkeypatch.setenv("TOKENJAM_REPORT_DIR", str(tmp_path))
    _seed_reuse_cluster(db, count=3, completions=False)

    result = _invoke(runner, db, _reuse_config(completions=False),
                     ["report", "--reuse", "--no-open"])
    assert result.exit_code == 0
    assert len(list(tmp_path.glob("reuse-*.html"))) == 1
    assert list(tmp_path.glob("reuse-*.md")) == []   # no skeleton text → no md


def test_optimize_export_templates_writes_markdown(runner, db, tmp_path, monkeypatch):
    monkeypatch.setenv("TOKENJAM_REPORT_DIR", str(tmp_path))
    _seed_reuse_cluster(db, count=3, completions=True)

    result = _invoke(runner, db, _reuse_config(completions=True),
                     ["optimize", "reuse", "--export-templates"])
    assert result.exit_code == 0, result.output
    assert "Reuse skeleton" in result.output
    assert len(list(tmp_path.glob("reuse-*.md"))) == 1
    assert list(tmp_path.glob("reuse-*.html")) == []   # markdown only, no HTML


def test_report_reuse_api_mode_writes_artifacts(runner, db, tmp_path, monkeypatch):
    """#154: with the daemon holding the DB lock, `tj report --reuse` fetches
    the finding + skeleton text from /api/v1/reuse/clusters via ApiBackend and
    still writes the HTML + Markdown — instead of erroring with 'needs direct
    database access'."""
    import asyncio

    import httpx

    from tokenjam.api.app import create_app
    from tokenjam.core.api_backend import ApiBackend
    from tokenjam.core.ingest import IngestPipeline

    monkeypatch.setenv("TOKENJAM_REPORT_DIR", str(tmp_path))
    cfg = _reuse_config(completions=True)
    _seed_reuse_cluster(db, count=3, completions=True)

    # Capture the real endpoint payload (exercises the route handler).
    app = create_app(config=cfg, db=db, ingest_pipeline=IngestPipeline(db=db, config=cfg))

    async def _fetch():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            r = await c.get("/api/v1/reuse/clusters", params={"since": "30d"})
            assert r.status_code == 200, r.text
            return r.json()

    payload = asyncio.run(_fetch())

    # An ApiBackend has no `.conn`, so cmd_report takes the HTTP path. Stub the
    # network call with the captured payload (sync httpx can't drive ASGI).
    backend = ApiBackend("http://test")
    monkeypatch.setattr(backend, "fetch_reuse_clusters", lambda **kw: payload)
    assert getattr(backend, "conn", None) is None

    result = _invoke(runner, backend, cfg, ["report", "--reuse", "--no-open"])
    assert result.exit_code == 0, result.output
    assert "Reuse report written" in result.output
    assert "needs direct database access" not in result.output
    assert len(list(tmp_path.glob("reuse-*.html"))) == 1
    assert len(list(tmp_path.glob("reuse-*.md"))) == 1   # skeleton text → sidecar


# -- otel-resource-attrs tests --

def test_otel_resource_attrs_includes_namespace_for_configured_project(runner):
    """When the repo's agent has a project set, namespace is appended."""
    cfg = TjConfig(
        version="1",
        agents={"claude-code-myrepo": AgentConfig(project="aquanode")},
    )
    with patch("tokenjam.cli.cmd_otel._derive_project_name", return_value="myrepo"), \
         patch("tokenjam.cli.main.load_config", return_value=cfg):
        result = runner.invoke(cli, ["otel-resource-attrs"])

    assert result.exit_code == 0
    assert result.output.strip() == (
        "service.name=claude-code-myrepo,service.namespace=aquanode"
    )


def test_otel_resource_attrs_omits_namespace_without_config(runner):
    """No config / no project => service.name only, single line, nothing else."""
    cfg = TjConfig(version="1")  # no agents configured
    with patch("tokenjam.cli.cmd_otel._derive_project_name", return_value="myrepo"), \
         patch("tokenjam.cli.main.load_config", return_value=cfg):
        result = runner.invoke(cli, ["otel-resource-attrs"])

    assert result.exit_code == 0
    assert result.output.strip() == "service.name=claude-code-myrepo"
    # Single bare line — safe to embed in $(tj otel-resource-attrs).
    assert "service.namespace" not in result.output
    assert result.output.count("\n") == 1


# -- onboard --claude-code per-terminal wrapper tests --

def test_onboard_claude_code_installs_claude_wrapper(runner, tmp_path):
    """--claude-code installs the per-terminal `claude` wrapper into ~/.zshrc."""
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    zshrc = fake_home / ".zshrc"

    with patch("tokenjam.cli.cmd_onboard.find_config_file", return_value=None), \
         patch("tokenjam.cli.cmd_onboard.Path.home", return_value=fake_home), \
         patch("tokenjam.cli.cmd_onboard.click.confirm", return_value=False):
        result = runner.invoke(cli, [
            "onboard", "--claude-code", "--no-daemon", "--budget", "5.0",
            "--plan", "max_20x", "--project", "aquanode",
        ])

    assert result.exit_code == 0
    assert zshrc.exists()
    text = zshrc.read_text()
    assert "# tj per-terminal naming" in text
    assert "claude() {" in text
    # The wrapper sources project attrs from the new utility command and tags
    # a per-terminal instance id, invoking the real binary without recursion.
    assert "tj otel-resource-attrs" in text
    assert "service.instance.id=" in text
    assert "command claude" in text
    assert "--as" in text
    # Close-signal: reports the session ended on exit and on interrupt.
    assert "tj session-end --instance" in text
    assert "trap " in text


def test_onboard_claude_code_wrapper_is_idempotent(runner, tmp_path):
    """Re-running --claude-code does not duplicate the wrapper block."""
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    zshrc = fake_home / ".zshrc"

    with patch("tokenjam.cli.cmd_onboard.find_config_file", return_value=None), \
         patch("tokenjam.cli.cmd_onboard.Path.home", return_value=fake_home), \
         patch("tokenjam.cli.cmd_onboard.click.confirm", return_value=False):
        args = [
            "onboard", "--claude-code", "--no-daemon", "--budget", "5.0",
            "--plan", "max_20x", "--project", "aquanode",
        ]
        first = runner.invoke(cli, args)
        second = runner.invoke(cli, args)

    assert first.exit_code == 0
    assert second.exit_code == 0
    text = zshrc.read_text()
    assert text.count("claude() {") == 1
    assert text.count("# tj per-terminal naming") == 1
    assert text.count("# end tj per-terminal naming") == 1


def test_onboard_claude_code_wrapper_writes_bashrc_when_present(runner, tmp_path):
    """~/.bashrc gets the wrapper only when it already exists."""
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    bashrc = fake_home / ".bashrc"
    bashrc.write_text("# existing bashrc\n")

    with patch("tokenjam.cli.cmd_onboard.find_config_file", return_value=None), \
         patch("tokenjam.cli.cmd_onboard.Path.home", return_value=fake_home), \
         patch("tokenjam.cli.cmd_onboard.click.confirm", return_value=False):
        result = runner.invoke(cli, [
            "onboard", "--claude-code", "--no-daemon", "--budget", "5.0",
            "--plan", "max_20x", "--project", "aquanode",
        ])

    assert result.exit_code == 0
    bashrc_text = bashrc.read_text()
    assert "# existing bashrc" in bashrc_text  # preserved
    assert "claude() {" in bashrc_text


# -- session-end tests --

def test_session_end_posts_to_endpoint(runner):
    """`tj session-end --instance` POSTs to /api/v1/sessions/close with auth."""
    from tokenjam.core.config import ApiConfig, SecurityConfig

    cfg = TjConfig(
        version="1",
        security=SecurityConfig(ingest_secret="sek"),
        api=ApiConfig(host="127.0.0.1", port=7391),
    )
    captured: dict = {}

    class _Resp:
        status = 200

        def read(self):
            return b'{"closed": 1}'

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def fake_urlopen(req, timeout=None):
        captured["url"] = req.full_url
        captured["data"] = req.data
        captured["auth"] = req.headers.get("Authorization")
        return _Resp()

    with patch("tokenjam.cli.main.load_config", return_value=cfg), \
         patch("tokenjam.cli.main.open_db", side_effect=AssertionError("must not open DB")), \
         patch("tokenjam.cli.cmd_session_end.urllib.request.urlopen",
               side_effect=fake_urlopen):
        result = runner.invoke(cli, ["session-end", "--instance", "term-x"])

    assert result.exit_code == 0
    assert captured["url"].endswith("/api/v1/sessions/close")
    assert b"term-x" in captured["data"]
    assert captured["auth"] == "Bearer sek"


def test_session_end_silent_when_daemon_unreachable(runner):
    """Daemon down -> exit 0, nothing printed (must never break the shell)."""
    import urllib.error
    from tokenjam.core.config import SecurityConfig

    cfg = TjConfig(version="1", security=SecurityConfig(ingest_secret="sek"))
    with patch("tokenjam.cli.main.load_config", return_value=cfg), \
         patch("tokenjam.cli.cmd_session_end.urllib.request.urlopen",
               side_effect=urllib.error.URLError("connection refused")):
        result = runner.invoke(cli, ["session-end", "--instance", "term-x"])

    assert result.exit_code == 0
    assert result.output.strip() == ""


def test_session_end_requires_an_id(runner):
    """Neither --instance nor --session -> usage error."""
    cfg = TjConfig(version="1")
    with patch("tokenjam.cli.main.load_config", return_value=cfg):
        result = runner.invoke(cli, ["session-end"])
    assert result.exit_code != 0
