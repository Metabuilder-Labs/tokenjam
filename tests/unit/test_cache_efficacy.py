"""Unit tests for the cache analyzer."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from tokenjam.core.config import OptimizeConfig, TjConfig
from tokenjam.core.db import InMemoryBackend
from tokenjam.core.optimize import build_report
from tokenjam.core.optimize.analyzers.cache_efficacy import (
    EFFICACY_THRESHOLD,
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


def test_compute_rows_min_input_tokens_override(db):
    """`_compute_rows`'s `min_input_tokens` param (what run() threads from
    `[optimize] min_cache_input_tokens`) changes which rows clear the bar,
    using the exact data from test_small_input_not_flagged_even_at_low_efficacy."""
    _seed_spans(
        db, provider="anthropic", model="claude-sonnet-4-6",
        input_tokens=1_000, cache_tokens=10, count=10,  # 10K total — below MIN
    )
    since = datetime(2026, 5, 1, tzinfo=timezone.utc)
    until = datetime(2026, 5, 30, tzinfo=timezone.utc)

    rows = _compute_rows(db.conn, since, until, agent_id=None)
    assert rows[0].flagged is False

    lowered_rows = _compute_rows(
        db.conn, since, until, agent_id=None, min_input_tokens=5_000,
    )
    assert lowered_rows[0].flagged is True


def test_run_reads_thresholds_from_ctx_config(db):
    """The registered run(ctx) entry point reads `ctx.config.optimize`'s
    cache thresholds, not just the module constants directly."""
    _seed_spans(
        db, provider="anthropic", model="claude-sonnet-4-6",
        input_tokens=1_000, cache_tokens=10, count=10,
    )
    since = datetime(2026, 5, 1, tzinfo=timezone.utc)
    until = datetime(2026, 5, 30, tzinfo=timezone.utc)

    default_report = build_report(
        db=db, config=TjConfig(version="1"), since=since, until=until,
        findings=["cache"],
    )
    default_finding = default_report.findings["cache"]
    assert default_finding.flagged == []
    assert default_finding.min_input_tokens == MIN_INPUT_TOKENS
    assert default_finding.efficacy_threshold == EFFICACY_THRESHOLD

    lowered_config = TjConfig(
        version="1", optimize=OptimizeConfig(min_cache_input_tokens=5_000),
    )
    lowered_report = build_report(
        db=db, config=lowered_config, since=since, until=until,
        findings=["cache"],
    )
    lowered_finding = lowered_report.findings["cache"]
    assert len(lowered_finding.flagged) == 1
    assert lowered_finding.min_input_tokens == 5_000


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
        findings=["cache"],
    )
    assert "cache" in report.findings
    finding = report.findings["cache"]
    assert finding.confidence == "structural"
    assert len(finding.rows) == 1
    assert len(finding.flagged) == 1


# --- CLI text-view rendering: remedy line ------------------------------------
# `cache` reports the efficacy ratio but the actual fix (a cache_control
# breakpoint) lives under the separate `cache-recommend` finding -- a user
# reading `cache` alone could miss it entirely. The renderer must point there.

def test_render_cache_flagged_rows_point_to_cache_recommend(capsys):
    from tokenjam.cli.cmd_optimize import _render_cache_efficacy
    from tokenjam.core.optimize.analyzers.cache_efficacy import (
        CacheEfficacyFinding,
        CacheEfficacyRow,
    )

    row = CacheEfficacyRow(
        provider="anthropic", model="claude-sonnet-4-6",
        input_tokens=150_000, cache_tokens=10_000, efficacy=0.06,
        support="full", flagged=True,
    )
    finding = CacheEfficacyFinding(rows=[row], flagged=[row])

    _render_cache_efficacy(finding, pricing_mode="api", marker="①")
    out = capsys.readouterr().out

    assert "tj optimize cache-recommend" in out


def test_render_cache_no_flagged_rows_omits_remedy_pointer(capsys):
    """No flagged rows means nothing to fix -- don't print a remedy pointer
    that implies a problem the table doesn't show."""
    from tokenjam.cli.cmd_optimize import _render_cache_efficacy
    from tokenjam.core.optimize.analyzers.cache_efficacy import (
        CacheEfficacyFinding,
        CacheEfficacyRow,
    )

    row = CacheEfficacyRow(
        provider="anthropic", model="claude-sonnet-4-6",
        input_tokens=150_000, cache_tokens=100_000, efficacy=0.67,
        support="full", flagged=False,
    )
    finding = CacheEfficacyFinding(rows=[row], flagged=[])

    _render_cache_efficacy(finding, pricing_mode="api", marker="①")
    out = capsys.readouterr().out

    assert "tj optimize cache-recommend" not in out
