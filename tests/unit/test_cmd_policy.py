"""Unit tests for `tj policy list`."""
from __future__ import annotations

import json
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from tokenjam.cli.cmd_policy import (
    PREVIEW_NOTE,
    PolicyRow,
    _channel_summary,
    _collect_rows,
    _drift_overrides_default,
)
from tokenjam.cli.main import cli
from tokenjam.core.config import (
    AgentConfig,
    AlertChannelConfig,
    AlertsConfig,
    BudgetConfig,
    CaptureConfig,
    DefaultsConfig,
    DriftConfig,
    ProviderBudget,
    SensitiveAction,
    TjConfig,
)
from tokenjam.core.db import InMemoryBackend


@pytest.fixture
def runner():
    return CliRunner()


def _empty_config() -> TjConfig:
    return TjConfig(version="1")


def _invoke(runner, config, args):
    db = InMemoryBackend()
    try:
        with patch("tokenjam.cli.main.load_config", return_value=config), \
             patch("tokenjam.cli.main.open_db", return_value=db):
            return runner.invoke(cli, args)
    finally:
        db.close()


# --- _collect_rows ---

def test_empty_config_yields_no_rows_apart_from_default_alerts():
    # Default AlertsConfig has one stdout channel — so the alerts row always
    # shows. defaults.budget unset, no provider budgets, no agents, no capture.
    cfg = _empty_config()
    rows = _collect_rows(cfg)
    policies = [r.policy for r in rows]
    assert "alerts" in policies
    assert "alerts.channels[0]" in policies
    assert "defaults.budget" not in policies
    assert all(not p.startswith("budget.") for p in policies)
    assert all(not p.startswith("agents.") for p in policies)
    assert "capture" not in policies


def test_provider_budget_row_shows_usd_and_plan():
    cfg = _empty_config()
    cfg.budgets["anthropic"] = ProviderBudget(usd=200.0, plan="max_20x")
    rows = _collect_rows(cfg)
    row = next(r for r in rows if r.policy == "budget.anthropic")
    assert "usd=200" in row.setting
    assert "plan=max_20x" in row.setting
    assert row.source == "[budget.anthropic]"


def test_provider_budgets_sorted_by_provider():
    cfg = _empty_config()
    cfg.budgets["openai"] = ProviderBudget(usd=50.0)
    cfg.budgets["anthropic"] = ProviderBudget(usd=200.0)
    rows = _collect_rows(cfg)
    providers = [r.policy for r in rows if r.policy.startswith("budget.")]
    assert providers == ["budget.anthropic", "budget.openai"]


def test_defaults_budget_only_shown_when_set():
    cfg = _empty_config()
    assert not any(r.policy == "defaults.budget" for r in _collect_rows(cfg))

    cfg.defaults = DefaultsConfig(budget=BudgetConfig(daily_usd=5.0))
    rows = _collect_rows(cfg)
    row = next(r for r in rows if r.policy == "defaults.budget")
    assert "daily_usd=5" in row.setting


def test_agent_rows_emitted_only_for_overrides():
    cfg = _empty_config()
    cfg.agents["bare"] = AgentConfig()  # all defaults — no rows
    cfg.agents["full"] = AgentConfig(
        budget=BudgetConfig(daily_usd=10.0),
        sensitive_actions=[
            SensitiveAction(name="email_send"),
            SensitiveAction(name="file_delete", severity="critical"),
        ],
        output_schema="schemas/out.json",
        drift=DriftConfig(token_threshold=3.0),
    )
    rows = _collect_rows(cfg)
    policies = {r.policy for r in rows}
    assert "agents.bare.budget" not in policies
    assert "agents.bare.drift" not in policies
    assert "agents.bare.sensitive_actions" not in policies
    assert "agents.bare.schema" not in policies
    assert "agents.full.budget" in policies
    assert "agents.full.drift" in policies
    assert "agents.full.sensitive_actions" in policies
    assert "agents.full.schema" in policies

    sa_row = next(r for r in rows if r.policy == "agents.full.sensitive_actions")
    assert "email_send" in sa_row.setting
    assert "file_delete" in sa_row.setting


def test_capture_row_only_when_any_flag_true():
    cfg = _empty_config()
    assert not any(r.policy == "capture" for r in _collect_rows(cfg))
    cfg.capture = CaptureConfig(prompts=True)
    row = next(r for r in _collect_rows(cfg) if r.policy == "capture")
    assert "prompts=true" in row.setting
    assert "completions=false" in row.setting


def test_alerts_row_summarises_channel_count_and_flags():
    cfg = _empty_config()
    cfg.alerts = AlertsConfig(
        cooldown_seconds=300,
        include_captured_content=True,
        channels=[
            AlertChannelConfig(type="stdout"),
            AlertChannelConfig(type="file", path="/tmp/alerts.log"),
        ],
    )
    rows = _collect_rows(cfg)
    alerts = next(r for r in rows if r.policy == "alerts")
    assert "cooldown_seconds=300" in alerts.setting
    assert "include_captured_content=true" in alerts.setting
    assert "channels=2" in alerts.setting

    ch1 = next(r for r in rows if r.policy == "alerts.channels[1]")
    assert "type=file" in ch1.setting
    assert "path=/tmp/alerts.log" in ch1.setting


def test_drift_overrides_default_detects_change():
    assert _drift_overrides_default(DriftConfig()) is False
    assert _drift_overrides_default(DriftConfig(token_threshold=3.0)) is True
    assert _drift_overrides_default(DriftConfig(enabled=False)) is True


def test_channel_summary_includes_min_severity():
    summary = _channel_summary(AlertChannelConfig(type="stdout", min_severity="warning"))
    assert "min_severity=warning" in summary


# --- CLI integration ---

def test_policy_list_human_output_renders_table(runner):
    cfg = _empty_config()
    cfg.budgets["anthropic"] = ProviderBudget(usd=200.0, plan="max_20x")
    result = _invoke(runner, cfg, ["policy", "list"])
    assert result.exit_code == 0, result.output
    assert "budget.anthropic" in result.output
    assert "[budget.anthropic]" in result.output
    # Rich wraps the note across lines on narrow terminals; check a stable prefix.
    assert "this is a read-only preview" in result.output


def test_policy_list_json_output_is_valid_json(runner):
    cfg = _empty_config()
    cfg.budgets["anthropic"] = ProviderBudget(usd=200.0, plan="max_20x")
    result = _invoke(runner, cfg, ["--json", "policy", "list"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["note"] == PREVIEW_NOTE
    policies = {p["policy"]: p for p in payload["policies"]}
    assert "budget.anthropic" in policies
    assert policies["budget.anthropic"]["source"] == "[budget.anthropic]"


def test_policy_list_empty_config_still_succeeds(runner):
    # Default alerts/channels rows still appear; just confirm exit 0.
    result = _invoke(runner, _empty_config(), ["policy", "list"])
    assert result.exit_code == 0, result.output
    assert "read-only preview" in result.output


def test_policy_add_is_absent_this_sprint(runner):
    result = _invoke(runner, _empty_config(), ["policy", "add", "foo"])
    assert result.exit_code != 0
    assert "No such command" in result.output or "Usage" in result.output


def test_policy_edit_is_absent_this_sprint(runner):
    result = _invoke(runner, _empty_config(), ["policy", "edit", "foo"])
    assert result.exit_code != 0


def test_policy_list_does_not_require_db(runner):
    # InMemoryBackend.open_db patch still runs, but policy is in no_db_commands
    # so the command must succeed even if the DB layer would have failed.
    with patch("tokenjam.cli.main.load_config", return_value=_empty_config()), \
         patch("tokenjam.cli.main.open_db", side_effect=AssertionError("must not open db")):
        result = runner.invoke(cli, ["policy", "list"])
    assert result.exit_code == 0, result.output


def test_policyrow_to_dict_round_trips():
    row = PolicyRow(policy="a", setting="b", source="c")
    assert row.to_dict() == {"policy": "a", "setting": "b", "source": "c"}
