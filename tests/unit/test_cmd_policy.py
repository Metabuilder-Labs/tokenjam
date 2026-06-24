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

def test_empty_config_yields_default_alerts_and_capture_rows():
    # Default AlertsConfig has one stdout channel — so the alerts row always
    # shows. capture also shows even when all toggles are off (#71 fix —
    # an explicit "off" is still a policy choice worth surfacing).
    # defaults.budget unset, no provider budgets, no agents.
    cfg = _empty_config()
    rows = _collect_rows(cfg)
    policies = [r.policy for r in rows]
    assert "alerts" in policies
    assert "alerts.channels[0]" in policies
    assert "defaults.budget" not in policies
    assert all(not p.startswith("budget.") for p in policies)
    assert all(not p.startswith("agents.") for p in policies)
    assert "capture" in policies


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


def test_capture_row_always_shown():
    # Capture is a policy choice even when all toggles are off (#71 fix).
    cfg = _empty_config()
    row = next(r for r in _collect_rows(cfg) if r.policy == "capture")
    assert "prompts=false" in row.setting
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


# --- #220: enforcement-plane [[policies]] + decisions surface ---

def test_list_shows_engine_policies_with_unvalidated_label():
    from tokenjam.core.config import PolicyConfig
    from tokenjam.cli.cmd_policy import _policy_engine_rows

    rows = _policy_engine_rows([
        PolicyConfig(name="cap", kind="noop", mode="suggest", target_provider="openai"),
    ])
    assert len(rows) == 1
    assert rows[0].policy == "policies.cap"
    assert "kind=noop" in rows[0].setting
    assert "label=unvalidated" in rows[0].setting       # honesty: never implied validated
    assert rows[0].source == "[[policies]][0]"


def test_list_command_renders_engine_policies_and_note(runner):
    from tokenjam.core.config import PolicyConfig
    cfg = TjConfig(version="1",
                   policies=[PolicyConfig(name="cap", kind="noop", mode="suggest")])
    result = _invoke(runner, cfg, ["policy", "list"])
    assert result.exit_code == 0, result.output
    assert "policies.cap" in result.output
    assert "unvalidated" in result.output


def test_list_json_includes_unvalidated_note_when_policies_present(runner):
    from tokenjam.core.config import PolicyConfig
    cfg = TjConfig(version="1", policies=[PolicyConfig(name="cap", kind="noop")])
    result = _invoke(runner, cfg, ["--json", "policy", "list"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert "unvalidated" in payload["unvalidated_note"]
    assert any(r["policy"] == "policies.cap" for r in payload["policies"])


# --- #221: `tj policy decisions` reads persisted decisions + savings from DB ---

_SAMPLE_PROXY = {"decisions": [{
    "ts": "2026-06-24T00:00:00+00:00", "provider": "openai",
    "path": "/v1/chat/completions",
    "policy": {"overall_action": "would_block", "label": "unvalidated",
               "evaluations": [{"policy_name": "blocker"}]},
}], "label": "unvalidated"}


def _seed_decision_db():
    """An InMemoryBackend with one persisted decision + savings entry (#221)."""
    from tokenjam.core.db import InMemoryBackend
    from tokenjam.core.models import PolicyDecisionRecord, SavingsLedgerEntry
    from tokenjam.utils.time_parse import utcnow
    db = InMemoryBackend()
    db.insert_policy_decision(PolicyDecisionRecord(
        decision_id="d1", ts=utcnow(), provider="openai", pricing_mode="api",
        gate_decision="policy", path="/v1/chat/completions", would_action="would_block",
        policy_name="blocker", policy_kind="noop",
        envelope={"overall_action": "would_block", "label": "unvalidated"}))
    db.insert_savings_entry(SavingsLedgerEntry(
        ledger_id="l1", decision_id="d1", ts=utcnow(), provider="openai",
        pricing_mode="api", would_action="would_block",
        estimated_recoverable_usd=0.50, estimate_basis="stub", billing_period="2026-06"))
    return db


def test_decisions_reads_persisted_from_db(runner):
    cfg = TjConfig(version="1")
    db = _seed_decision_db()
    with patch("tokenjam.core.db.open_db", return_value=db):
        result = _invoke(runner, cfg, ["policy", "decisions"])
    assert result.exit_code == 0, result.output
    assert "openai" in result.output
    assert "unvalidated" in result.output
    # The savings meter is shown and is ESTIMATED / RECOVERABLE — never "saved".
    assert "Estimated recoverable" in result.output


def test_decisions_savings_never_says_saved(runner):
    cfg = TjConfig(version="1")
    db = _seed_decision_db()
    with patch("tokenjam.core.db.open_db", return_value=db):
        result = _invoke(runner, cfg, ["--json", "policy", "decisions"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["source"] == "db"
    assert payload["label"] == "unvalidated"
    sv = payload["savings"]
    assert sv["realized"] is False
    assert sv["estimated_recoverable_usd"] == 0.5
    # Honesty: the meter never claims realized savings.
    assert "saved" not in json.dumps(sv).lower().replace("would-have-saved", "")
    assert "not realized" in sv["disclaimer"].lower()


def test_decisions_falls_back_to_proxy_when_db_locked(runner):
    cfg = TjConfig(version="1")
    with patch("tokenjam.core.db.open_db", side_effect=RuntimeError("locked")), \
         patch("tokenjam.cli.cmd_policy._fetch_proxy_json", return_value=_SAMPLE_PROXY):
        result = _invoke(runner, cfg, ["--json", "policy", "decisions"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["source"] == "proxy"
    assert payload["decisions"][0]["would_action"] == "would_block"


def test_decisions_unreachable_when_db_locked_and_no_proxy(runner):
    cfg = TjConfig(version="1")
    with patch("tokenjam.core.db.open_db", side_effect=RuntimeError("locked")), \
         patch("tokenjam.cli.cmd_policy._fetch_proxy_json", return_value=None):
        result = _invoke(runner, cfg, ["policy", "decisions"])
    assert result.exit_code == 0, result.output
    assert "no running proxy reachable" in result.output.lower()


def test_decisions_startup_does_not_open_db(runner):
    # `policy` is in no_db_commands: STARTUP must not open the DB (main.open_db).
    # The command opens its OWN connection lazily, which is allowed.
    cfg = TjConfig(version="1")
    with patch("tokenjam.cli.main.load_config", return_value=cfg), \
         patch("tokenjam.cli.main.open_db", side_effect=AssertionError("startup must not open db")), \
         patch("tokenjam.core.db.open_db", side_effect=RuntimeError("locked")), \
         patch("tokenjam.cli.cmd_policy._fetch_proxy_json", return_value=None):
        result = runner.invoke(cli, ["policy", "decisions"])
    assert result.exit_code == 0, result.output
