"""Unit tests for the cache-efficacy analyzer."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from tokenjam.core.config import TjConfig
from tokenjam.core.db import InMemoryBackend
from tokenjam.core.optimize import build_report
from tokenjam.core.optimize.analyzers.cache_efficacy import (
    MIN_INPUT_TOKENS,
    _compute_rows,
)
from tests.factories import make_llm_span


@pytest.fixture
def db():
    backend = InMemoryBackend()
    yield backend
    backend.close()


@pytest.fixture
def config():
    return TjConfig(version="1")


def _seed_spans(db, *, provider, model, input_tokens, cache_tokens,
                count=10, start=None):
    """Helper: insert N spans with the given per-span token counts."""
    start = start or datetime(2026, 5, 10, tzinfo=timezone.utc)
    for i in range(count):
        span = make_llm_span(
            agent_id="test-agent",
            provider=provider,
            model=model,
            billing_account=provider,
            input_tokens=input_tokens,
            output_tokens=200,
            cache_tokens=cache_tokens,
            cost_usd=0.01,
            start_time=start + timedelta(minutes=i),
        )
        db.insert_span(span)


def test_compute_rows_no_data(db):
    """Empty window returns no rows."""
    since = datetime(2026, 5, 1, tzinfo=timezone.utc)
    until = datetime(2026, 5, 30, tzinfo=timezone.utc)
    assert _compute_rows(db.conn, since, until, agent_id=None) == []


def test_anthropic_low_efficacy_flagged(db):
    """Anthropic with large input + low caching gets flagged."""
    _seed_spans(
        db, provider="anthropic", model="claude-sonnet-4-6",
        input_tokens=15_000, cache_tokens=1_000, count=10,
    )
    since = datetime(2026, 5, 1, tzinfo=timezone.utc)
    until = datetime(2026, 5, 30, tzinfo=timezone.utc)
    rows = _compute_rows(db.conn, since, until, agent_id=None)
    assert len(rows) == 1
    r = rows[0]
    assert r.provider == "anthropic"
    assert r.support == "full"
    assert r.input_tokens == 150_000  # 10 spans × 15K
    assert r.cache_tokens == 10_000   # 10 × 1K
    # efficacy = 10k / 160k = 0.0625, below 0.30 threshold
    assert r.efficacy == pytest.approx(0.0625, abs=0.001)
    assert r.flagged is True


def test_anthropic_high_efficacy_not_flagged(db):
    """High caching ratio above threshold isn't flagged."""
    _seed_spans(
        db, provider="anthropic", model="claude-sonnet-4-6",
        input_tokens=5_000, cache_tokens=20_000, count=10,
    )
    since = datetime(2026, 5, 1, tzinfo=timezone.utc)
    until = datetime(2026, 5, 30, tzinfo=timezone.utc)
    rows = _compute_rows(db.conn, since, until, agent_id=None)
    assert len(rows) == 1
    r = rows[0]
    # efficacy = 200k / 250k = 0.80, above 0.30 threshold
    assert r.efficacy == pytest.approx(0.80, abs=0.001)
    assert r.flagged is False


def test_small_input_not_flagged_even_at_low_efficacy(db):
    """Below MIN_INPUT_TOKENS, low efficacy doesn't matter — savings would be trivial."""
    _seed_spans(
        db, provider="anthropic", model="claude-sonnet-4-6",
        input_tokens=1_000, cache_tokens=10, count=10,  # 10K total — below MIN
    )
    since = datetime(2026, 5, 1, tzinfo=timezone.utc)
    until = datetime(2026, 5, 30, tzinfo=timezone.utc)
    rows = _compute_rows(db.conn, since, until, agent_id=None)
    assert len(rows) == 1
    assert rows[0].input_tokens < MIN_INPUT_TOKENS
    assert rows[0].flagged is False


def test_openai_marked_best_effort(db):
    """OpenAI rows carry best_effort support label."""
    _seed_spans(
        db, provider="openai", model="gpt-4o",
        input_tokens=15_000, cache_tokens=0, count=10,
    )
    since = datetime(2026, 5, 1, tzinfo=timezone.utc)
    until = datetime(2026, 5, 30, tzinfo=timezone.utc)
    rows = _compute_rows(db.conn, since, until, agent_id=None)
    assert rows[0].support == "best_effort"
    # Still flagged: best_effort providers should surface low-efficacy too.
    assert rows[0].flagged is True


def test_bedrock_marked_unsupported_not_flagged(db):
    """Bedrock is unsupported in v1 — never flagged even with low efficacy."""
    _seed_spans(
        db, provider="bedrock", model="claude-3",
        input_tokens=15_000, cache_tokens=0, count=10,
    )
    since = datetime(2026, 5, 1, tzinfo=timezone.utc)
    until = datetime(2026, 5, 30, tzinfo=timezone.utc)
    rows = _compute_rows(db.conn, since, until, agent_id=None)
    assert rows[0].support == "unsupported"
    assert rows[0].flagged is False


def test_run_integrates_with_build_report(db, config):
    """The analyzer wires through build_report and surfaces in report.findings."""
    _seed_spans(
        db, provider="anthropic", model="claude-sonnet-4-6",
        input_tokens=15_000, cache_tokens=1_000, count=10,
    )
    since = datetime(2026, 5, 1, tzinfo=timezone.utc)
    until = datetime(2026, 5, 30, tzinfo=timezone.utc)
    report = build_report(
        db=db, config=config, since=since, until=until,
        findings=["cache-efficacy"],
    )
    assert "cache-efficacy" in report.findings
    finding = report.findings["cache-efficacy"]
    assert finding.confidence == "structural"
    assert len(finding.rows) == 1
    assert len(finding.flagged) == 1
