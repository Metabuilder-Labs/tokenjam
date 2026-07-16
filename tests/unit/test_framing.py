"""Unit tests for tokenjam.core.framing (issue #110)."""
from __future__ import annotations

from dataclasses import dataclass

import pytest

from tokenjam.core.framing import (
    DISPLAY_SHOW_DOLLARS,
    DISPLAY_SHOW_DOLLARS_WITH_QUALIFIER,
    DISPLAY_SUPPRESS_SUBSCRIPTION,
    DISPLAY_SUPPRESS_UNKNOWN,
    DISPLAY_TOKENS_ONLY,
    Framing,
    WindowSummary,
    compute_framing,
    config_declared_plan,
    config_declared_plan_labels,
    dominant_persona,
    dominant_plan,
    pricing_mode_for,
    render_dollar,
    render_savings,
)


# --------------------------------------------------------------------------- #
# config stubs
# --------------------------------------------------------------------------- #
@dataclass
class _Budget:
    plan: str | None = None


class _Config:
    def __init__(self, budgets: dict | None = None):
        self.budgets = budgets or {}


@pytest.fixture(autouse=True)
def _isolate_home(monkeypatch, tmp_path):
    """Point Path.home() at an empty tmp dir so config_declared_plan's global
    fallback never reads this dev machine's ~/.config/tj/config.toml."""
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)


# --------------------------------------------------------------------------- #
# pure helpers
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "plan_tier,expected",
    [
        ("local", "local"),
        ("api", "api"),
        ("pro", "subscription"),
        ("max_5x", "subscription"),
        ("max_20x", "subscription"),
        ("plus", "subscription"),
        ("team", "subscription"),
        ("enterprise", "subscription"),
        ("unknown", "unknown"),
        ("garbage", "unknown"),
    ],
)
def test_pricing_mode_for(plan_tier, expected):
    assert pricing_mode_for(plan_tier) == expected


def test_dominant_plan_empty_defaults_to_api():
    assert dominant_plan({}) == "api"


def test_dominant_plan_all_unknown():
    assert dominant_plan({"unknown": 5}) == "unknown"


def test_dominant_plan_most_common_known_wins():
    assert dominant_plan({"api": 2, "max_5x": 7, "unknown": 3}) == "max_5x"


def test_config_declared_plan_from_active_config():
    cfg = _Config({"anthropic": _Budget(plan="max_5x")})
    assert config_declared_plan(cfg) == "max_5x"


def test_config_declared_plan_none_when_unset():
    assert config_declared_plan(_Config()) is None


def test_config_declared_plan_sorted_provider_order():
    cfg = _Config({"openai": _Budget(plan="plus"), "anthropic": _Budget(plan="pro")})
    # 'anthropic' sorts before 'openai'
    assert config_declared_plan(cfg) == "pro"


def test_config_declared_plan_labels_single_provider():
    cfg = _Config({"anthropic": _Budget(plan="api")})
    assert config_declared_plan_labels(cfg) == ["API billing"]


def test_config_declared_plan_labels_multi_provider():
    cfg = _Config({
        "anthropic": _Budget(plan="api"),
        "openai": _Budget(plan="plus"),
    })
    assert config_declared_plan_labels(cfg) == [
        "API billing (anthropic)",
        "ChatGPT Plus (openai)",
    ]


def test_compute_framing_emits_declared_plan_labels():
    cfg = _Config({
        "anthropic": _Budget(plan="api"),
        "openai": _Budget(plan="plus"),
    })
    f = compute_framing(
        cfg,
        WindowSummary(total_cost_usd=10.0, total_tokens=500, sessions=3,
                      plan_tier_mix={"api": 3}),
    )
    assert f.plan_labels == [
        "API billing (anthropic)",
        "ChatGPT Plus (openai)",
    ]


# --------------------------------------------------------------------------- #
# compute_framing — one path per pricing_mode
# --------------------------------------------------------------------------- #
def test_compute_framing_api_clean():
    f = compute_framing(
        _Config(),
        WindowSummary(total_cost_usd=47.0, total_tokens=1000, sessions=10,
                      plan_tier_mix={"api": 10}),
    )
    assert f.pricing_mode == "api"
    assert f.plan_tier == "api"
    assert f.plan_label == "API billing"
    assert f.display_rule == DISPLAY_SHOW_DOLLARS
    assert f.qualifier_text is None
    assert f.api_share_pct == 100.0


def test_compute_framing_api_with_unknown_qualifier():
    f = compute_framing(
        _Config(),
        WindowSummary(total_cost_usd=47.0, total_tokens=1000, sessions=10,
                      plan_tier_mix={"api": 7, "unknown": 3}),
    )
    assert f.pricing_mode == "api"
    assert f.display_rule == DISPLAY_SHOW_DOLLARS_WITH_QUALIFIER
    assert f.qualifier_text is not None
    assert "3 of 10" in f.qualifier_text


def test_compute_framing_subscription():
    f = compute_framing(
        _Config(),
        WindowSummary(total_cost_usd=148.0, total_tokens=2_000_000, sessions=20,
                      plan_tier_mix={"max_5x": 20}),
    )
    assert f.pricing_mode == "subscription"
    assert f.plan_tier == "max_5x"
    assert f.plan_label == "Max 5x plan"
    assert f.plan_monthly_usd == 100.0
    assert f.display_rule == DISPLAY_SUPPRESS_SUBSCRIPTION
    assert f.subscription_share_pct == 100.0


def test_compute_framing_subscription_mixed_window_qualifier():
    f = compute_framing(
        _Config(),
        WindowSummary(total_cost_usd=148.0, total_tokens=2_000_000, sessions=20,
                      plan_tier_mix={"max_5x": 17, "api": 3}),
    )
    assert f.pricing_mode == "subscription"
    assert f.display_rule == DISPLAY_SUPPRESS_SUBSCRIPTION
    assert f.qualifier_text is not None
    assert "subscription-billed" in f.qualifier_text
    assert f.subscription_share_pct == 85.0
    assert f.api_share_pct == 15.0


def test_compute_framing_local():
    f = compute_framing(
        _Config(),
        WindowSummary(total_cost_usd=0.0, total_tokens=5000, sessions=4,
                      plan_tier_mix={"local": 4}),
    )
    assert f.pricing_mode == "local"
    assert f.plan_label == "Local inference"
    assert f.display_rule == DISPLAY_TOKENS_ONLY
    assert "no marginal cost" in (f.qualifier_text or "")


def test_compute_framing_all_unknown_suppressed():
    f = compute_framing(
        _Config(),
        WindowSummary(total_cost_usd=10.0, total_tokens=500, sessions=5,
                      plan_tier_mix={"unknown": 5}),
    )
    assert f.pricing_mode == "unknown"
    assert f.display_rule == DISPLAY_SUPPRESS_UNKNOWN
    assert "claude-code --reconfigure" in (f.qualifier_text or "")


def test_compute_framing_empty_data_falls_back_to_declared_plan():
    # No window data at all (e.g. /api/v1/budget) → use the declared plan.
    cfg = _Config({"anthropic": _Budget(plan="max_20x")})
    f = compute_framing(cfg, WindowSummary())
    assert f.pricing_mode == "subscription"
    assert f.plan_tier == "max_20x"
    assert f.plan_label == "Max 20x plan"


def test_compute_framing_empty_data_no_plan_defaults_api():
    f = compute_framing(_Config(), WindowSummary())
    assert f.pricing_mode == "api"


def test_compute_framing_accepts_dict_window():
    f = compute_framing(
        _Config(),
        {"total_cost_usd": 5.0, "total_tokens": 100, "sessions": 1,
         "plan_tier_mix": {"api": 1}},
    )
    assert f.pricing_mode == "api"


# --------------------------------------------------------------------------- #
# render_dollar / render_savings
# --------------------------------------------------------------------------- #
def test_render_dollar_api():
    f = Framing(pricing_mode="api")
    assert render_dollar(148.0, f) == "$148"
    assert render_dollar(5.94, f) == "$5.94"


def test_render_dollar_subscription_share_of_cycle():
    f = Framing(pricing_mode="subscription", plan_monthly_usd=100.0)
    assert render_dollar(12.4, f) == "12.4% of cycle"


def test_render_dollar_local_dash():
    assert render_dollar(99.0, Framing(pricing_mode="local")) == "—"


def test_render_dollar_none():
    assert render_dollar(None, Framing(pricing_mode="api")) == "—"


def test_render_savings_api_dollars():
    f = Framing(pricing_mode="api")
    assert render_savings(148.0, 999, f) == "$148"


def test_render_savings_subscription_token_share():
    f = Framing(pricing_mode="subscription", window_total_tokens=1000)
    assert render_savings(None, 124, f) == "12.4% of cycle tokens"


def test_render_savings_local_token_count():
    f = Framing(pricing_mode="local")
    assert render_savings(None, 1_200_000, f) == "1.2M tokens"


def test_render_savings_none_dash():
    f = Framing(pricing_mode="api")
    assert render_savings(None, None, f) == "—"


# --------------------------------------------------------------------------- #
# plan_determination_mix — window-independent (#177)
# --------------------------------------------------------------------------- #
def test_plan_determination_mix_ignores_time_window():
    """The framing mix must not be scoped to a window: a 24h-vs-30d split was
    the root cause of window-dependent framing (#177). plan_determination_mix
    passes since=until=None to plan_tier_mix, so it counts every session."""
    from unittest.mock import MagicMock

    from tokenjam.core.framing import plan_determination_mix

    conn = MagicMock()
    conn.execute.return_value.fetchall.return_value = [("max_5x", 3), ("unknown", 40)]
    result = plan_determination_mix(conn, agent_id="agent-x")
    assert result == {"max_5x": 3, "unknown": 40}
    # The SQL is bound only with the agent_id — no started_at window clause.
    sql, params = conn.execute.call_args[0]
    assert "started_at" not in sql
    assert params == ["agent-x"]


def test_framing_to_dict_has_contract_fields():
    f = compute_framing(
        _Config(),
        WindowSummary(total_cost_usd=1.0, total_tokens=1, sessions=1,
                      plan_tier_mix={"api": 1}),
    )
    d = f.to_dict()
    for key in (
        "pricing_mode", "plan_tier", "plan_label", "plan_monthly_usd",
        "subscription_share_pct", "api_share_pct", "display_rule",
        "qualifier_text",
    ):
        assert key in d


# --------------------------------------------------------------------------- #
# agent_persona_mix / dominant_persona — Claude Code vs SDK/API developer (#97)
# --------------------------------------------------------------------------- #
def test_agent_persona_mix_classifies_by_claude_code_prefix():
    from unittest.mock import MagicMock

    from tokenjam.core.framing import agent_persona_mix

    conn = MagicMock()
    conn.execute.return_value.fetchall.return_value = [
        ("claude-code-my-project",), ("claude-code-other",), ("sdk-agent-x",),
    ]
    result = agent_persona_mix(conn, agent_id="agent-x")
    assert result == {"claude_code": 2, "other": 1}
    sql, params = conn.execute.call_args[0]
    assert "agent_id = $" in sql
    assert params == ["agent-x"]


def test_agent_persona_mix_margin_cases_match_is_interactive_coding_agent():
    """agent_persona_mix must agree with alerts.is_interactive_coding_agent
    at the margins: a bare "claude-code" id, and codex ids (previously "other"
    here, sending a pure-codex window to the SDK-facing bench CTA)."""
    from unittest.mock import MagicMock

    from tokenjam.core.framing import agent_persona_mix

    conn = MagicMock()
    conn.execute.return_value.fetchall.return_value = [
        ("claude-code",), ("codex",), ("codex-cli-session",), ("sdk-agent-x",),
    ]
    result = agent_persona_mix(conn)
    assert result == {"claude_code": 3, "other": 1}


def test_agent_persona_mix_empty_when_no_sessions():
    from unittest.mock import MagicMock

    from tokenjam.core.framing import agent_persona_mix

    conn = MagicMock()
    conn.execute.return_value.fetchall.return_value = []
    assert agent_persona_mix(conn) == {"claude_code": 0, "other": 0}


@pytest.mark.parametrize("mix,expected", [
    ({"claude_code": 9, "other": 1}, "claude-code"),
    ({"claude_code": 1, "other": 9}, "sdk"),
    ({"claude_code": 5, "other": 5}, "mixed"),
])
def test_dominant_persona_from_agent_mix(mix, expected):
    assert dominant_persona(mix) == expected


def test_dominant_persona_falls_back_to_declared_plan_when_mix_empty():
    assert dominant_persona({"claude_code": 0, "other": 0}, declared_plan="max_5x") == "claude-code"
    assert dominant_persona({"claude_code": 0, "other": 0}, declared_plan="api") == "sdk"
    assert dominant_persona({"claude_code": 0, "other": 0}, declared_plan=None) == "unknown"
    assert dominant_persona({}, declared_plan="pro") == "claude-code"
