"""Integration tests for `ocw demo` CLI command."""
from __future__ import annotations

import json

from click.testing import CliRunner

from tj.cli.main import cli


def test_demo_list_shows_all_scenarios():
    runner = CliRunner()
    result = runner.invoke(cli, ["demo"])
    assert result.exit_code == 0, result.output
    assert "retry-loop" in result.output
    assert "surprise-cost" in result.output
    assert "hallucination-drift" in result.output


def test_demo_list_shows_descriptions():
    runner = CliRunner()
    result = runner.invoke(cli, ["demo"])
    assert result.exit_code == 0
    assert len(result.output.strip()) > 0


def test_demo_retry_loop_exits_zero():
    runner = CliRunner()
    result = runner.invoke(cli, ["demo", "retry-loop"])
    assert result.exit_code == 0, result.output


def test_demo_surprise_cost_exits_zero():
    runner = CliRunner()
    result = runner.invoke(cli, ["demo", "surprise-cost"])
    assert result.exit_code == 0, result.output


def test_demo_hallucination_drift_exits_zero():
    runner = CliRunner()
    result = runner.invoke(cli, ["demo", "hallucination-drift"])
    assert result.exit_code == 0, result.output


def test_demo_retry_loop_json_has_required_fields():
    runner = CliRunner()
    result = runner.invoke(cli, ["demo", "retry-loop", "--json"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["scenario"] == "retry-loop"
    assert "alerts" in data
    assert "total_cost_usd" in data
    assert "span_count" in data


def test_demo_retry_loop_json_fires_alert():
    runner = CliRunner()
    result = runner.invoke(cli, ["demo", "retry-loop", "--json"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["alert_count"] > 0
    assert "retry_loop" in data["alert_types"]


def test_demo_surprise_cost_json_records_nonzero_cost():
    runner = CliRunner()
    result = runner.invoke(cli, ["demo", "surprise-cost", "--json"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["total_cost_usd"] > 0
    assert "model_breakdown" in data


def test_demo_hallucination_drift_json_fires_drift_alert():
    runner = CliRunner()
    result = runner.invoke(cli, ["demo", "hallucination-drift", "--json"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["alert_count"] > 0
    assert "drift_detected" in data["alert_types"]


def test_demo_nonexistent_scenario_exits_nonzero():
    runner = CliRunner()
    result = runner.invoke(cli, ["demo", "unknown-scenario"])
    assert result.exit_code != 0
    assert "unknown-scenario" in result.output
