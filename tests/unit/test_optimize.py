"""Unit tests for the tj optimize analyzers."""
from __future__ import annotations

from datetime import timedelta

import pytest

from tokenjam.core.config import ProviderBudget, TjConfig
from tokenjam.core.db import InMemoryBackend
from tokenjam.core.optimize import (
    DOWNGRADE_CANDIDATES,
    MODEL_DOWNGRADE_CAVEAT,
    _cycle_bounds,
    analyze_model_downgrade,
    build_report,
    project_budget,
    summarize_window,
)
from tokenjam.utils.time_parse import utcnow
from tests.factories import make_llm_span, make_tool_span


@pytest.fixture
def db():
    backend = InMemoryBackend()
    yield backend
    backend.close()


def _insert_small_opus_session(db, start_time=None, session_id="s-small"):
    """Insert one Opus LLM span + 2 tool spans matching the downgrade heuristic."""
    start = start_time or utcnow() - timedelta(days=2)
    llm = make_llm_span(
        agent_id="claude-code-x",
        model="claude-opus-4-7",
        provider="anthropic",
        input_tokens=1000,
        output_tokens=200,
        cost_usd=0.030,
        session_id=session_id,
        start_time=start,
    )
    db.insert_span(llm)
    for i in range(2):
        tool = make_tool_span(
            agent_id="claude-code-x",
            tool_name="Read",
            trace_id=llm.trace_id,
        )
        # Force into the same session for the analyzer query
        tool.session_id = session_id
        tool.start_time = start
        db.insert_span(tool)


def _insert_large_opus_session(db, session_id="s-large"):
    """Session with high tokens — should NOT match heuristic."""
    start = utcnow() - timedelta(days=1)
    llm = make_llm_span(
        agent_id="claude-code-x",
        model="claude-opus-4-7",
        provider="anthropic",
        input_tokens=50_000,
        output_tokens=2_000,
        cost_usd=1.500,
        session_id=session_id,
        start_time=start,
    )
    db.insert_span(llm)


def test_summarize_window_counts_and_costs(db):
    _insert_small_opus_session(db, session_id="a")
    _insert_large_opus_session(db, session_id="b")
    since = utcnow() - timedelta(days=30)
    until = utcnow() + timedelta(hours=1)
    s = summarize_window(db.conn, since, until)
    assert s.sessions == 2
    assert s.total_tokens == 1000 + 200 + 50_000 + 2_000
    assert abs(s.total_cost_usd - (0.030 + 1.500)) < 1e-6


def test_summarize_window_total_includes_cache_read_tokens(db):
    """Window header total must include cache-read tokens, sharing one basis
    with the downsize denominator + the canonical WindowTotals (#33)."""
    start = utcnow() - timedelta(days=2)
    db.insert_span(make_llm_span(
        model="claude-opus-4-7",
        provider="anthropic",
        input_tokens=1000,
        output_tokens=200,
        cache_tokens=5000,
        cost_usd=0.030,
        session_id="cache-heavy",
        start_time=start,
    ))
    since = utcnow() - timedelta(days=30)
    until = utcnow() + timedelta(hours=1)
    s = summarize_window(db.conn, since, until)
    assert s.total_tokens == 1000 + 200 + 5000


def test_summarize_window_total_includes_cache_write_tokens(db):
    """Window header total must also include cache-CREATION tokens, not just
    cache-read, so a cache-write-heavy window's reclaimable-share denominator
    (cmd_optimize._reclaimable_share) isn't inflated."""
    start = utcnow() - timedelta(days=2)
    db.insert_span(make_llm_span(
        model="claude-opus-4-7",
        provider="anthropic",
        input_tokens=1000,
        output_tokens=200,
        cache_tokens=500,
        cache_write_tokens=8000,
        cost_usd=0.030,
        session_id="cache-write-heavy",
        start_time=start,
    ))
    since = utcnow() - timedelta(days=30)
    until = utcnow() + timedelta(hours=1)
    s = summarize_window(db.conn, since, until)
    assert s.total_tokens == 1000 + 200 + 500 + 8000


def test_downgrade_window_total_tokens_includes_cache_write_tokens(db):
    """window_total_tokens (the percent_of_tokens denominator) must include
    cache-creation volume for every session in the window, candidate or not."""
    _insert_small_opus_session(db, session_id="a")
    db.insert_span(make_llm_span(
        model="claude-sonnet-4-6",
        provider="anthropic",
        input_tokens=2000,
        output_tokens=300,
        cache_write_tokens=9000,
        cost_usd=0.10,
        session_id="cache-write-session",
        start_time=utcnow() - timedelta(days=1),
    ))
    since = utcnow() - timedelta(days=30)
    until = utcnow() + timedelta(hours=1)
    finding = analyze_model_downgrade(
        db.conn, since, until, agent_id=None, window_days=30.0,
    )
    assert finding is not None
    assert finding.window_total_tokens == (1000 + 200) + (2000 + 300 + 9000)


def test_downgrade_percent_of_tokens_counts_cache_write_in_numerator(db):
    """percent_of_tokens = candidate_tokens / window_total_tokens must measure
    both sides on the same basis. When every session is a downgrade candidate
    carrying cache-write tokens, the candidate share is ~100% — a numerator that
    omitted cache-write would understate it against the (now cache-write-inclusive)
    denominator."""
    db.insert_span(make_llm_span(
        model="claude-opus-4-7",
        provider="anthropic",
        input_tokens=1000,
        output_tokens=200,
        cache_tokens=500,
        cache_write_tokens=8000,
        cost_usd=0.030,
        session_id="candidate-cache-write",
        start_time=utcnow() - timedelta(days=2),
    ))
    since = utcnow() - timedelta(days=30)
    until = utcnow() + timedelta(hours=1)
    finding = analyze_model_downgrade(
        db.conn, since, until, agent_id=None, window_days=30.0,
    )
    assert finding is not None
    assert finding.candidate_sessions == 1
    expected_tokens = 1000 + 200 + 500 + 8000
    assert finding.candidate_tokens == expected_tokens
    assert finding.window_total_tokens == expected_tokens
    assert finding.percent_of_tokens == 100.0


def test_downgrade_flags_small_opus_but_not_large(db):
    _insert_small_opus_session(db, session_id="a")
    _insert_large_opus_session(db, session_id="b")
    since = utcnow() - timedelta(days=30)
    until = utcnow() + timedelta(hours=1)
    finding = analyze_model_downgrade(
        db.conn, since, until, agent_id=None, window_days=30.0,
    )
    assert finding is not None
    assert finding.candidate_sessions == 1
    assert finding.total_sessions == 2
    assert finding.suggestions.get("claude-opus-4-7") == "claude-haiku-4-5"
    assert finding.bench_command == "tjb run --original anthropic:claude-opus-4-7 --candidate anthropic:claude-haiku-4-5"
    # Caveat must be present as the dataclass default
    assert finding.caveat == MODEL_DOWNGRADE_CAVEAT


def test_downgrade_proposes_candidates_for_fable_and_opus_4_8(db):
    """The downsize analyzer must route Fable (tier above Opus) and Opus 4.8 —
    both previously missing from DOWNGRADE_CANDIDATES — to a cheaper same-family
    model when the session is structurally small."""
    start = utcnow() - timedelta(days=2)
    db.insert_span(make_llm_span(
        model="claude-fable-5", provider="anthropic",
        input_tokens=1_000, output_tokens=200, cost_usd=0.10,
        session_id="fable-s", start_time=start,
    ))
    db.insert_span(make_llm_span(
        model="claude-opus-4-8", provider="anthropic",
        input_tokens=1_000, output_tokens=200, cost_usd=0.05,
        session_id="opus48-s", start_time=start,
    ))
    since = utcnow() - timedelta(days=30)
    until = utcnow() + timedelta(hours=1)
    finding = analyze_model_downgrade(
        db.conn, since, until, agent_id=None, window_days=30.0,
    )
    assert finding is not None
    assert finding.candidate_sessions == 2
    assert finding.suggestions.get("claude-fable-5") == "claude-sonnet-4-6"
    assert finding.suggestions.get("claude-opus-4-8") == "claude-haiku-4-5"


def test_downgrade_returns_none_when_no_candidates(db):
    _insert_large_opus_session(db, session_id="b")
    since = utcnow() - timedelta(days=30)
    until = utcnow() + timedelta(hours=1)
    finding = analyze_model_downgrade(
        db.conn, since, until, agent_id=None, window_days=30.0,
    )
    assert finding is None


def test_project_budget_under_budget(db):
    _insert_small_opus_session(db, session_id="a")
    since = utcnow() - timedelta(days=30)
    until = utcnow() + timedelta(hours=1)
    budget = ProviderBudget(usd=200.0, cycle_start_day=1)
    proj = project_budget(db.conn, "anthropic", budget, since, until)
    assert proj is not None
    assert proj.budget_usd == 200.0
    # Small fixture spend is way under budget
    assert proj.over_budget is False


def test_project_budget_over_budget_signals_overage(db):
    # Insert a big run-rate
    start = utcnow() - timedelta(days=1)
    for i in range(20):
        llm = make_llm_span(
            agent_id="claude-code-x",
            model="claude-opus-4-7",
            provider="anthropic",
            input_tokens=10_000,
            output_tokens=2_000,
            cost_usd=50.0,
            session_id=f"s{i}",
            start_time=start,
        )
        db.insert_span(llm)
    since = utcnow() - timedelta(days=2)
    until = utcnow() + timedelta(hours=1)
    budget = ProviderBudget(usd=200.0, cycle_start_day=1)
    proj = project_budget(db.conn, "anthropic", budget, since, until)
    assert proj is not None
    assert proj.over_budget is True
    assert proj.projected_overage_usd > 0


def test_project_budget_returns_none_for_zero_budget(db):
    since = utcnow() - timedelta(days=30)
    until = utcnow()
    proj = project_budget(
        db.conn, "anthropic", ProviderBudget(usd=None), since, until
    )
    assert proj is None


def test_budget_projection_scoped_to_provider(db):
    # Anthropic spend
    start = utcnow() - timedelta(days=1)
    db.insert_span(make_llm_span(
        provider="anthropic", model="claude-opus-4-7",
        input_tokens=1000, output_tokens=200, cost_usd=10.0,
        session_id="a", start_time=start,
    ))
    # OpenAI spend should NOT count toward the Anthropic budget
    db.insert_span(make_llm_span(
        provider="openai", model="gpt-4o",
        input_tokens=1000, output_tokens=200, cost_usd=5.0,
        session_id="b", start_time=start,
    ))
    since = utcnow() - timedelta(days=2)
    until = utcnow() + timedelta(hours=1)
    proj = project_budget(
        db.conn, "anthropic", ProviderBudget(usd=100.0, cycle_start_day=1),
        since, until,
    )
    assert proj is not None
    assert abs(proj.window_spend_usd - 10.0) < 1e-6


def test_build_report_returns_caveat_in_dict(db):
    _insert_small_opus_session(db, session_id="a")
    cfg = TjConfig(version="1")
    since = utcnow() - timedelta(days=30)
    until = utcnow() + timedelta(hours=1)
    report = build_report(db=db, config=cfg, since=since, until=until)
    assert report.downgrade is not None
    assert report.downgrade.caveat == MODEL_DOWNGRADE_CAVEAT


def test_cycle_bounds_handles_mid_month():
    from datetime import datetime, timezone
    now = datetime(2026, 5, 15, tzinfo=timezone.utc)
    cs, ce = _cycle_bounds(now, start_day=1)
    assert cs.day == 1 and cs.month == 5
    assert ce.day == 1 and ce.month == 6


def test_cycle_bounds_before_start_day_uses_prior_month():
    from datetime import datetime, timezone
    now = datetime(2026, 5, 3, tzinfo=timezone.utc)
    cs, ce = _cycle_bounds(now, start_day=15)
    assert cs.month == 4 and cs.day == 15
    assert ce.month == 5 and ce.day == 15


def test_lookup_downgrade_normalizes_context_tag_and_bedrock():
    """`_lookup_downgrade` must resolve the same id shapes `is_premium_tier`
    accepts — [1m] context tags and Bedrock region/provider prefixes — or the
    audit silently drops premium sessions the subagent analyzer flags."""
    from tokenjam.core.optimize.analyzers.model_downgrade import _lookup_downgrade

    # [1m] context tag (1M-context variant) resolves to the base model's alt.
    assert _lookup_downgrade("anthropic", "claude-fable-5[1m]") == "claude-sonnet-4-6"
    assert _lookup_downgrade("anthropic", "claude-opus-4-8[1m]") == "claude-haiku-4-5"
    # Trailing date still works (regression guard).
    assert _lookup_downgrade("anthropic", "claude-opus-4-8-20260115") == "claude-haiku-4-5"
    # Bedrock-hosted Claude: region/provider prefix + version suffix, provider
    # arrives as a bedrock alias — resolves via the base Anthropic model.
    assert _lookup_downgrade(
        "aws.bedrock", "us-anthropic-claude-opus-4-8-20260115-v1"
    ) == "claude-haiku-4-5"
    # Unrecognised model still returns None (no invented alt).
    assert _lookup_downgrade("anthropic", "gpt-4o") is None


def test_downgrade_candidates_have_pricing_for_alternative():
    # Sanity check: every alternative model is in the pricing table.
    from tokenjam.core.pricing import get_rates
    for provider, mapping in DOWNGRADE_CANDIDATES.items():
        for alt in mapping.values():
            assert get_rates(provider, alt) is not None, (
                f"Pricing missing for {provider}/{alt}"
            )


def test_subscription_json_zeros_monthly_savings_usd(db, monkeypatch, tmp_path):
    """`tj optimize --json` must not leak a dollar `monthly_savings_usd` for
    subscription users — the same suppression the human render applies (#32).

    The renderer reframes downgrade savings as a token share for flat-fee
    plans, so the machine payload must agree: `monthly_savings_usd == 0`.
    """
    import json
    from unittest.mock import patch

    from click.testing import CliRunner

    from tokenjam.cli.main import cli
    from tokenjam.core.config import ApiAuthConfig, ApiConfig, TjConfig
    from tests.factories import make_session

    # Keep config_declared_plan's global fallback off this dev machine.
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)

    # One subscription-tier session whose spans match the downgrade heuristic
    # (small Opus session → projected dollar savings the renderer suppresses).
    db.upsert_session(make_session(
        agent_id="claude-code-x", session_id="s-small", plan_tier="max_5x",
    ))
    _insert_small_opus_session(db, session_id="s-small")

    config = TjConfig(version="1", api=ApiConfig(auth=ApiAuthConfig(enabled=False)))

    runner = CliRunner()
    with patch("tokenjam.cli.main.load_config", return_value=config), \
         patch("tokenjam.cli.main.open_db", return_value=db):
        result = runner.invoke(cli, ["optimize", "--since", "30d", "--json"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["pricing_mode"] == "subscription"
    downgrade = payload["downgrade"]
    assert downgrade is not None
    assert downgrade["monthly_savings_usd"] == 0
