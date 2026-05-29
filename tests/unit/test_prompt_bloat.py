"""
Unit tests for the trim (Trim) analyzer.

The LLMLingua-2 model is mocked across these tests so CI doesn't have to
download ~110MB and run an actual BERT classifier. The mock returns
hand-crafted (token, score) tuples that exercise each branch of the
region-extraction logic.
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest

from tokenjam.core.config import CaptureConfig, TjConfig
from tokenjam.core.db import InMemoryBackend
from tokenjam.core.optimize import build_report
from tokenjam.core.optimize.analyzers.prompt_bloat import (
    MIN_REGION_LENGTH,
    PromptBloatFinding,
    _regions_from_scores,
)
from tokenjam.otel.semconv import GenAIAttributes
from tests.factories import make_llm_span


@pytest.fixture
def db():
    backend = InMemoryBackend()
    yield backend
    backend.close()


def _config(prompts: bool) -> TjConfig:
    return TjConfig(version="1", capture=CaptureConfig(prompts=prompts))


# -- Pure-function tests --

def test_regions_from_scores_finds_low_significance_span():
    """A long run of low-score tokens becomes a region."""
    # "important" gets high score; the 30-char blob of "x" gets low scores.
    text = "important " + ("x" * 30) + " important"
    scores = [
        ("important", 0.9),
    ] + [("x", 0.1)] * 30 + [
        ("important", 0.9),
    ]
    regions = _regions_from_scores(text, scores)
    assert len(regions) == 1
    r = regions[0]
    assert r.char_length >= MIN_REGION_LENGTH
    assert r.avg_score < 0.40


def test_regions_skips_short_low_significance_spans():
    """Short low-score runs aren't worth flagging."""
    text = "important x y important"
    scores = [("important", 0.9), ("x", 0.1), ("y", 0.1), ("important", 0.9)]
    assert _regions_from_scores(text, scores) == []


def test_regions_empty_input():
    assert _regions_from_scores("", []) == []


def test_regions_all_low_significance():
    """An entirely low-significance prompt becomes one region."""
    text = "x" * 50
    scores = [("x", 0.1)] * 50
    regions = _regions_from_scores(text, scores)
    # The trailing-flush branch should produce a single region covering the rest.
    assert len(regions) == 1


# -- Integration tests via build_report --

def test_disabled_without_capture_prompts(db):
    """Without capture.prompts the analyzer returns a hint, not a model run."""
    config = _config(prompts=False)
    since = datetime(2026, 5, 1, tzinfo=timezone.utc)
    until = datetime(2026, 5, 30, tzinfo=timezone.utc)
    report = build_report(db=db, config=config, since=since, until=until,
                          findings=["trim"])
    finding = report.findings["trim"]
    assert isinstance(finding, PromptBloatFinding)
    assert finding.enabled is False
    assert "capture" in finding.hint.lower()


def test_disabled_when_llmlingua_missing(db, monkeypatch):
    """Without the bloat extra installed, the analyzer surfaces the install hint."""
    # Force the deferred import to fail by inserting a sentinel that raises.
    monkeypatch.setitem(sys.modules, "llmlingua", None)  # blocks import
    config = _config(prompts=True)
    since = datetime(2026, 5, 1, tzinfo=timezone.utc)
    until = datetime(2026, 5, 30, tzinfo=timezone.utc)
    report = build_report(db=db, config=config, since=since, until=until,
                          findings=["trim"])
    finding = report.findings["trim"]
    assert finding.enabled is False
    assert "tokenjam[bloat]" in finding.hint


def _install_fake_llmlingua(monkeypatch, token_scores):
    """
    Inject a fake `llmlingua.PromptCompressor` whose
    `get_distillation_token_scores` returns the given (token, score) list.
    """
    fake_module = MagicMock()

    class FakeCompressor:
        def __init__(self, *args, **kwargs):
            pass

        def get_distillation_token_scores(self, text):
            return token_scores

    fake_module.PromptCompressor = FakeCompressor
    monkeypatch.setitem(sys.modules, "llmlingua", fake_module)


def _seed_prompt(db, *, text: str, count: int = 1, start=None):
    """Seed N captured-prompt spans."""
    start = start or datetime(2026, 5, 10, tzinfo=timezone.utc)
    for i in range(count):
        span = make_llm_span(
            agent_id="test-agent",
            provider="anthropic",
            billing_account="anthropic",
            model="claude-sonnet-4-6",
            input_tokens=int(len(text) / 4),
            cost_usd=0.001,
            start_time=start + timedelta(minutes=i),
            extra_attributes={GenAIAttributes.PROMPT_CONTENT: text},
        )
        db.insert_span(span)


def test_scores_prompts_and_finds_bloat(db, monkeypatch):
    """When the model is available and prompts captured, the analyzer surfaces bloat."""
    text = "important " + ("filler " * 40) + "important"
    scores = (
        [("important", 0.9)]
        + [("filler", 0.1)] * 40
        + [("important", 0.9)]
    )
    _install_fake_llmlingua(monkeypatch, scores)
    _seed_prompt(db, text=text, count=3)
    config = _config(prompts=True)
    since = datetime(2026, 5, 1, tzinfo=timezone.utc)
    until = datetime(2026, 5, 30, tzinfo=timezone.utc)
    report = build_report(db=db, config=config, since=since, until=until,
                          findings=["trim"])
    finding = report.findings["trim"]
    assert finding.enabled is True
    assert finding.prompts_scored == 3
    # Each scored prompt produces one BloatPrompt entry, up to the 10 cap.
    assert len(finding.per_prompt) == 3
    p = finding.per_prompt[0]
    assert p.bloat_chars > 0
    assert p.estimated_token_reduction > 0
    assert len(p.regions) >= 1


def test_skips_short_prompts(db, monkeypatch):
    """Prompts under 200 chars are skipped — no model run, no finding."""
    _install_fake_llmlingua(monkeypatch, [("x", 0.5)])
    _seed_prompt(db, text="too short", count=5)
    config = _config(prompts=True)
    since = datetime(2026, 5, 1, tzinfo=timezone.utc)
    until = datetime(2026, 5, 30, tzinfo=timezone.utc)
    report = build_report(db=db, config=config, since=since, until=until,
                          findings=["trim"])
    finding = report.findings["trim"]
    assert finding.prompts_scored == 0
    assert finding.prompts_skipped == 5
    assert finding.per_prompt == []
