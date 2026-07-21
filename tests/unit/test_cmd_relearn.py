"""Unit tests for `tj relearn receipts` (Component G1's read-only CLI verb).

Fully isolated: `storage.path` points under `tmp_path`, so the two ledgers
(`relearn_apply.applied_fixes_path` / `cost_apply.cost_applied_path`) never
touch a real `~/.tj` (mirrors `test_cost_proposals.py` / `test_relearn_apply.py`).
No DB is needed for `receipts` itself, but `tj`'s root group always opens one
before dispatching — patched out the same way `test_cmd_policy.py` does.
"""
from __future__ import annotations

import json
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from tokenjam.cli.main import cli
from tokenjam.core.config import StorageConfig, TjConfig
from tokenjam.core.db import InMemoryBackend
from tokenjam.core.optimize import cost_apply, relearn_apply


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def cfg(tmp_path):
    return TjConfig(version="1", storage=StorageConfig(path=str(tmp_path / "t.duckdb")))


def _invoke(runner, config, args):
    db = InMemoryBackend()
    try:
        with patch("tokenjam.cli.main.load_config", return_value=config), \
             patch("tokenjam.cli.main.open_db", return_value=db):
            return runner.invoke(cli, args)
    finally:
        db.close()


def _write_relearn_ledger(cfg, records):
    path = relearn_apply.applied_fixes_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records), encoding="utf-8")


def _write_cost_ledger(cfg, records):
    path = cost_apply.cost_applied_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records), encoding="utf-8")


def test_receipts_reports_empty_state(runner, cfg):
    result = _invoke(runner, cfg, ["relearn", "receipts"])
    assert result.exit_code == 0
    assert "No fixes verified yet" in result.output


def test_receipts_renders_verified_dollars_and_tokens(runner, cfg):
    _write_relearn_ledger(cfg, [
        {"state": "applied", "rung": 1,
         "verify": {"verdict": "improved", "realized_tokens_saved": 1500}},
    ])
    _write_cost_ledger(cfg, [
        {"state": "applied",
         "verify": {"verdict": "improved", "realized_usd_delta": 0.75,
                    "realized_tokens_delta": 400}},
    ])
    result = _invoke(runner, cfg, ["relearn", "receipts"])
    assert result.exit_code == 0
    assert "$0.75" in result.output
    assert "1,900 tok" in result.output   # 1500 relearn + 400 cost, additive


def test_receipts_shows_regressed_entry_not_hidden(runner, cfg):
    _write_relearn_ledger(cfg, [
        {"state": "applied", "rung": 1,
         "verify": {"verdict": "improved", "realized_tokens_saved": 500}},
        {"state": "applied", "rung": 1, "verify": {"verdict": "regressed"}},
    ])
    _write_cost_ledger(cfg, [
        {"state": "applied", "verify": {"verdict": "regressed", "realized_usd_delta": 0.0}},
    ])
    result = _invoke(runner, cfg, ["relearn", "receipts"])
    assert result.exit_code == 0
    # Two regressed fixes (one relearn, one cost) — both counted, not dropped.
    assert "2 regressed" in result.output


def test_receipts_json_output_matches_summary_shape(runner, cfg):
    _write_relearn_ledger(cfg, [
        {"state": "applied", "rung": 1,
         "verify": {"verdict": "improved", "realized_tokens_saved": 1000}},
    ])
    result = _invoke(runner, cfg, ["relearn", "receipts", "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["relearn_tokens_saved"] == 1000
    assert payload["estimate_confidence"] == "measured"
    assert payload["verified_count"] == 1


# --- the receipts line must not state a bare dollar figure it cannot back --- #
# `verified_saved_usd` prices token deltas at API list rates across ALL traffic
# (plan_tier lives on `sessions`, cost aggregates read `spans`, nothing joins
# them), and it excludes relearn savings entirely because a relearn fix has no
# single model to price against. Printed bare to a subscription user it reads
# as money off a bill they never received.

@pytest.fixture
def sub_cfg(tmp_path):
    from tokenjam.core.config import ProviderBudget
    return TjConfig(
        version="1",
        storage=StorageConfig(path=str(tmp_path / "t.duckdb")),
        budgets={"anthropic": ProviderBudget(plan="max_5x")},
    )


def _ledgers(cfg):
    _write_relearn_ledger(cfg, [
        {"state": "applied", "rung": 1,
         "verify": {"verdict": "improved", "realized_tokens_saved": 1500}},
    ])
    _write_cost_ledger(cfg, [
        {"state": "applied",
         "verify": {"verdict": "improved", "realized_usd_delta": 0.75,
                    "realized_tokens_delta": 400}},
    ])


def test_receipts_leads_with_tokens_on_a_subscription_plan(runner, sub_cfg):
    _ledgers(sub_cfg)
    result = _invoke(runner, sub_cfg, ["relearn", "receipts"])
    assert result.exit_code == 0
    out = " ".join(result.output.split())   # rich wraps; compare unwrapped
    # Tokens are the headline, and the dollar figure never stands alone.
    assert "1,900 tok verified saved to date" in out
    assert "$0.75 verified saved to date" not in out
    # Where the dollars do appear, they carry their basis.
    assert "at API list rates across the cost ledger only" in out
    assert "relearn fixes have no single model to price at" in out


def test_receipts_dollar_line_is_unchanged_on_api_billing(runner, cfg):
    _ledgers(cfg)
    result = _invoke(runner, cfg, ["relearn", "receipts"])
    assert result.exit_code == 0
    out = " ".join(result.output.split())
    assert "$0.75 verified saved to date" in out
    assert "+ 1,900 tok saved" in out
    # No list-price caveat is added for a user whose list price is the bill.
    assert "at API list rates" not in out


def test_receipts_never_claims_dollars_are_api_traffic_only(runner, sub_cfg, cfg):
    for config in (sub_cfg, cfg):
        _ledgers(config)
        result = _invoke(runner, config, ["relearn", "receipts"])
        assert "reflect API traffic only" not in result.output
