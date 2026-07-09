"""Subscription users don't get asked for a USD daily budget (#128).

`_prompt_daily_budget` is gated on the just-resolved plan tier: subscription
tiers (`SUBSCRIPTION_PLAN_TIERS`) and local inference have a $0/day marginal
cost, so the question ("Daily budget in USD...") right after the user just
named a flat-rate plan contradicted the framing discipline `core/framing.py`
already applies everywhere else. `api` (and `unknown`, if reachable) still get
prompted, and `--budget` always wins regardless of tier.
"""
from __future__ import annotations

import pytest
from click.testing import CliRunner

from tokenjam.cli.cmd_onboard import _prompt_daily_budget, cmd_onboard
from tokenjam.otel.semconv import SUBSCRIPTION_PLAN_TIERS

# --- Unit tests: _prompt_daily_budget ---------------------------------------


def _forbid_prompt(monkeypatch):
    """Fail loudly if the daily-budget prompt is reached when it shouldn't
    be — a click.prompt refusal, not a lenient stub."""
    import click as _click

    def _boom(*a, **k):
        raise AssertionError(f"unexpected click.prompt call: {a!r} {k!r}")

    monkeypatch.setattr(_click, "prompt", _boom)


@pytest.mark.parametrize("plan_tier", sorted(SUBSCRIPTION_PLAN_TIERS))
def test_subscription_tiers_skip_prompt_silently(plan_tier, monkeypatch):
    _forbid_prompt(monkeypatch)
    assert _prompt_daily_budget(None, plan_tier) == 0.0


def test_local_tier_skips_prompt_like_subscription(monkeypatch):
    _forbid_prompt(monkeypatch)
    assert _prompt_daily_budget(None, "local") == 0.0


def test_api_tier_still_prompts(monkeypatch):
    import click as _click

    monkeypatch.setattr(_click, "prompt", lambda *a, **k: 42.0)
    assert _prompt_daily_budget(None, "api") == 42.0


def test_unknown_tier_still_prompts(monkeypatch):
    # `unknown` isn't reachable from the current onboard choice menus, but if
    # it ever is, it should behave like `api` — prompt, don't assume $0.
    import click as _click

    monkeypatch.setattr(_click, "prompt", lambda *a, **k: 7.0)
    assert _prompt_daily_budget(None, "unknown") == 7.0


def test_none_tier_still_prompts(monkeypatch):
    # `plan_tier=None` (no resolved tier) must behave like `unknown`, not like
    # a subscription tier — never silently assume $0 for an unresolved plan.
    import click as _click

    monkeypatch.setattr(_click, "prompt", lambda *a, **k: 3.0)
    assert _prompt_daily_budget(None, None) == 3.0


@pytest.mark.parametrize(
    "plan_tier", sorted(SUBSCRIPTION_PLAN_TIERS | {"local", "api", "unknown"}),
)
def test_budget_flag_wins_regardless_of_tier(plan_tier, monkeypatch):
    _forbid_prompt(monkeypatch)
    assert _prompt_daily_budget(5.0, plan_tier) == 5.0


# --- CLI-level: fresh config (no existing ~/.config/tj/config.toml) --------


@pytest.fixture
def _isolated_home(monkeypatch, tmp_path):
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    monkeypatch.setattr(
        "tokenjam.cli.cmd_onboard._stop_serve_for_db_write", lambda: False,
    )
    monkeypatch.setattr(
        "tokenjam.cli.cmd_onboard._finish_onboard_serve", lambda *a, **k: None,
    )
    monkeypatch.setattr(
        "tokenjam.cli.cmd_onboard._try_apply_declared_plans", lambda *a, **k: None,
    )
    monkeypatch.setattr("tokenjam.cli.cmd_onboard.shutil.which", lambda _x: None)
    import tokenjam.core.backfill as backfill_mod
    monkeypatch.setattr(
        backfill_mod, "CLAUDE_CODE_PROJECTS_ROOT", tmp_path / "no-such-claude",
    )


def _invoke(tmp_path, *args, input_text: str = ""):
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        res = runner.invoke(cmd_onboard, ["--no-daemon", *args], input=input_text, obj={})
        cfg = tmp_path / ".config" / "tj" / "config.toml"
        return res, (cfg.read_text() if cfg.exists() else "")


def test_claude_code_fresh_subscription_no_budget_prompt(_isolated_home, tmp_path):
    res, cfg = _invoke(
        tmp_path, "--claude-code", "--project", "testproj", "--plan", "max_5x",
    )
    assert res.exit_code == 0, res.output
    assert "Daily budget in USD" not in res.output
    assert "daily_usd" not in cfg


def test_claude_code_fresh_api_prompts_and_writes_budget(_isolated_home, tmp_path):
    res, cfg = _invoke(
        tmp_path, "--claude-code", "--project", "testproj", "--plan", "api",
        input_text="25\n",
    )
    assert res.exit_code == 0, res.output
    assert "Daily budget in USD" in res.output
    assert "daily_usd = 25" in cfg


def test_claude_code_fresh_budget_flag_skips_prompt_for_api(_isolated_home, tmp_path):
    res, cfg = _invoke(
        tmp_path, "--claude-code", "--project", "testproj", "--plan", "api",
        "--budget", "9",
    )
    assert res.exit_code == 0, res.output
    assert "Daily budget in USD" not in res.output
    assert "daily_usd = 9" in cfg


def test_codex_fresh_subscription_no_budget_prompt(_isolated_home, tmp_path):
    res, cfg = _invoke(tmp_path, "--codex", "--plan", "plus")
    assert res.exit_code == 0, res.output
    assert "Daily budget in USD" not in res.output
    assert "daily_usd" not in cfg


def test_codex_fresh_api_prompts_and_writes_budget(_isolated_home, tmp_path):
    res, cfg = _invoke(tmp_path, "--codex", "--plan", "api", input_text="12\n")
    assert res.exit_code == 0, res.output
    assert "Daily budget in USD" in res.output
    assert "daily_usd = 12" in cfg


# --- CLI-level: existing config, plan tier already resolved -----------------
#
# Regression coverage for the `plan = existing_plan` fallback: when a plan was
# set on a previous onboard and neither --plan nor --reconfigure is passed,
# the plan-tier prompt block is skipped entirely, so `plan` must still be
# bound (from the stored config) before `_prompt_daily_budget` reads it.


def _seed_existing_config(tmp_path, *, provider: str, plan: str):
    cfg_dir = tmp_path / ".config" / "tj"
    cfg_dir.mkdir(parents=True)
    (cfg_dir / "config.toml").write_text(
        f'version = "1"\n'
        f"[security]\ningest_secret = \"{'a' * 64}\"\n"
        f"[budget.{provider}]\nplan = \"{plan}\"\ncycle_start_day = 1\n"
    )


def test_claude_code_existing_subscription_plan_no_reprompt(_isolated_home, tmp_path):
    _seed_existing_config(tmp_path, provider="anthropic", plan="max_20x")
    res, cfg = _invoke(
        tmp_path, "--claude-code", "--project", "testproj",
    )
    assert res.exit_code == 0, res.output
    assert "How do you pay for Claude?" not in res.output
    assert "Daily budget in USD" not in res.output
    assert "daily_usd" not in cfg


def test_codex_existing_subscription_plan_no_reprompt(_isolated_home, tmp_path):
    _seed_existing_config(tmp_path, provider="openai", plan="team")
    res, cfg = _invoke(tmp_path, "--codex")
    assert res.exit_code == 0, res.output
    assert "How do you pay for OpenAI / Codex?" not in res.output
    assert "Daily budget in USD" not in res.output
    assert "daily_usd" not in cfg
