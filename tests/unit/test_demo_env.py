"""Unit tests for DemoEnvironment."""
from __future__ import annotations

from tests.factories import make_llm_span
from tj.utils.ids import new_trace_id


def test_demo_env_creates_successfully():
    from tj.demo.env import DemoEnvironment
    env = DemoEnvironment()
    assert env.db is not None
    assert env.pipeline is not None


def test_demo_env_process_stores_span():
    from tj.demo.env import DemoEnvironment
    env = DemoEnvironment()
    env.process(make_llm_span(agent_id="test-agent"))
    count = env.db.conn.execute("SELECT COUNT(*) FROM spans").fetchone()[0]
    assert count == 1


def test_demo_env_get_alerts_returns_empty_initially():
    from tj.demo.env import DemoEnvironment
    env = DemoEnvironment()
    assert env.get_alerts() == []


def test_demo_env_total_cost_starts_at_zero():
    from tj.demo.env import DemoEnvironment
    env = DemoEnvironment()
    assert env.total_cost_usd() == 0.0


def test_demo_env_cost_increases_after_llm_span():
    from tj.demo.env import DemoEnvironment
    env = DemoEnvironment()
    env.process(make_llm_span(
        agent_id="test-agent",
        provider="anthropic",
        model="claude-haiku-4-5",
        input_tokens=100_000,
        output_tokens=50_000,
    ))
    assert env.total_cost_usd() > 0.0


def test_demo_env_trace_count_matches_injected():
    from tj.demo.env import DemoEnvironment
    from tj.core.models import TraceFilters
    env = DemoEnvironment()
    t1, t2 = new_trace_id(), new_trace_id()
    env.process(make_llm_span(agent_id="test-agent", trace_id=t1))
    env.process(make_llm_span(agent_id="test-agent", trace_id=t2))
    assert len(env.db.get_traces(TraceFilters())) == 2


def test_demo_result_fields():
    from tj.demo.env import DemoEnvironment, DemoResult
    env = DemoEnvironment()
    env.process(make_llm_span(agent_id="test-agent"))
    result = env.build_result("test-agent")
    assert isinstance(result, DemoResult)
    assert result.agent_id == "test-agent"
    assert result.span_count == 1
    assert result.alert_count == 0
    assert result.alert_types == []
    assert isinstance(result.total_cost_usd, float)
    assert result.trace_count == 1
