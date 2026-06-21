"""Plan-tier framing for the bare `tj cost` table (#175).

The COST column must not show raw dollars to subscription users (who pay a flat
fee), while API users see unchanged dollar figures.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest
from click.testing import CliRunner

from tokenjam.cli.main import cli
from tokenjam.core.config import ApiConfig, ApiAuthConfig, ProviderBudget, TjConfig
from tokenjam.core.db import InMemoryBackend
from tests.factories import make_llm_span, make_session


@pytest.fixture(autouse=True)
def _isolate_home(monkeypatch, tmp_path):
    """Point Path.home() at an empty tmp dir so config_declared_plan's global
    fallback never reads this dev machine's ~/.config/tj/config.toml."""
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)


def _config(plan: str | None) -> TjConfig:
    cfg = TjConfig(version="1", api=ApiConfig(auth=ApiAuthConfig(enabled=False)))
    if plan is not None:
        cfg.budgets["anthropic"] = ProviderBudget(plan=plan)
    return cfg


def _invoke(db, config, args):
    runner = CliRunner()
    with patch("tokenjam.cli.main.load_config", return_value=config), \
         patch("tokenjam.cli.main.open_db", return_value=db):
        return runner.invoke(cli, args)


def _seed(db, *, plan_tier: str, cost_usd: float) -> None:
    """One session at the given plan tier + one costly LLM span in it."""
    sess = make_session(agent_id="a", session_id="s1", plan_tier=plan_tier)
    db.upsert_session(sess)
    db.insert_span(make_llm_span(
        agent_id="a", session_id="s1", model="claude-opus-4-8",
        input_tokens=1000, output_tokens=300, cost_usd=cost_usd,
    ))


@pytest.fixture
def db():
    backend = InMemoryBackend()
    yield backend
    backend.close()


def test_api_plan_shows_raw_dollars(db):
    """API users see the historical raw-dollar COST rendering, unchanged."""
    _seed(db, plan_tier="api", cost_usd=2599.13)
    result = _invoke(db, _config("api"), ["cost", "--since", "30d", "--group-by", "model"])
    assert result.exit_code == 0, result.output
    assert "$2599.13" in result.output          # raw dollars preserved
    assert "% of cycle" not in result.output     # no subscription reframing
    assert "Subscription plan" not in result.output


def test_subscription_plan_suppresses_raw_dollars(db):
    """Max-plan users must NOT see the raw $2599 they never paid; the COST
    column is reframed as a share of the monthly plan, with an honest note."""
    _seed(db, plan_tier="max_5x", cost_usd=2599.13)
    result = _invoke(db, _config("max_5x"), ["cost", "--since", "30d", "--group-by", "model"])
    assert result.exit_code == 0, result.output
    assert "$2599.13" not in result.output       # raw dollars suppressed
    # render_dollar's "X% of cycle" framing (the COST column may wrap it, so
    # match the distinctive token rather than the whole phrase).
    assert "cycle" in result.output
    assert "Subscription plan" in result.output  # honesty note surfaced


def test_unknown_plan_keeps_dollars_with_qualifier(db):
    """Unknown plan tier keeps dollar figures but surfaces the overstate
    qualifier (defensive honesty), per compute_framing's unknown path."""
    _seed(db, plan_tier="unknown", cost_usd=2599.13)
    result = _invoke(db, _config(None), ["cost", "--since", "30d", "--group-by", "model"])
    assert result.exit_code == 0, result.output
    assert "$2599.13" in result.output           # dollars still shown
    assert "may overstate" in result.output       # qualifier surfaced
