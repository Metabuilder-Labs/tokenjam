"""Unit tests for the Claude Code routing-config export."""
from __future__ import annotations

from tokenjam.core.export.claude_code import render_claude_code_snippet
from tokenjam.core.optimize.types import (
    DowngradeExample,
    DowngradeFinding,
)


def _sample_finding() -> DowngradeFinding:
    return DowngradeFinding(
        candidate_sessions=8,
        total_sessions=17,
        actual_cost_usd=12.34,
        alternative_cost_usd=2.10,
        monthly_savings_usd=42.50,
        percent_of_sessions=47.0,
        examples=[DowngradeExample(
            trace_id="0102030405060708",
            session_id="sess-x",
            model="claude-opus-4-7",
            tool_calls=2,
            duration_seconds=12.3,
            cost_usd=4.0,
        )],
        suggestions={"claude-opus-4-7": "claude-haiku-4-5"},
        candidate_tokens=350_000,
        window_total_tokens=900_000,
        percent_of_tokens=38.9,
        monthly_tokens_in_candidates=1_400_000,
    )


# `estimated_savings_usd_month` is written unconditionally (dollars are always
# legitimate); pricing_mode/plan_tier only echo back as metadata text and no
# longer gate which fields appear, so one representative mode covers all three
# (product decision: tj no longer differentiates export output between
# subscription and API users).

def test_render_api_user_carries_usd_savings():
    snippet = render_claude_code_snippet(
        downgrade=_sample_finding(),
        pricing_mode="api",
        plan_tier="api",
    )
    assert "estimated_savings_usd_month" in snippet
    assert "42.5" in snippet
    # Subscription-only field should NOT appear
    assert "estimated_tokens_freed" not in snippet


def test_render_always_includes_honest_framing_comments():
    """The caveat block must appear in every export, regardless of plan."""
    snippet = render_claude_code_snippet(
        downgrade=_sample_finding(),
        pricing_mode="api",
        plan_tier="api",
    )
    assert "STRUCTURAL HEURISTIC ONLY" in snippet
    assert "TokenJam does not enforce" in snippet
    assert "tokenjam.dev/products/downsize" in snippet


def test_render_with_no_findings_returns_empty_rules():
    """When the analyzer found nothing, the rules array is empty but the block still renders."""
    snippet = render_claude_code_snippet(
        downgrade=None,
        pricing_mode="api",
        plan_tier="api",
    )
    assert '"rules": [' in snippet
    # Empty rules: no suggestion blocks, but caveat is still there
    assert "STRUCTURAL HEURISTIC ONLY" in snippet


def test_render_carries_routing_namespace():
    """Output is namespaced under tokenjam.routing_recommendations."""
    snippet = render_claude_code_snippet(
        downgrade=_sample_finding(),
        pricing_mode="api",
        plan_tier="api",
    )
    assert '"tokenjam"' in snippet
    assert '"routing_recommendations"' in snippet


def test_render_carries_agent_scope_when_provided():
    snippet = render_claude_code_snippet(
        downgrade=_sample_finding(),
        pricing_mode="api",
        plan_tier="api",
        agent_id="my-agent",
    )
    assert "agent_id=my-agent" in snippet


def test_render_includes_plan_tier_metadata():
    snippet = render_claude_code_snippet(
        downgrade=_sample_finding(),
        pricing_mode="subscription",
        plan_tier="max_20x",
    )
    assert '"plan_tier": "max_20x"' in snippet
    assert '"pricing_mode": "subscription"' in snippet
