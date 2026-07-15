"""Unit tests for the empirical validation engine (issue #477).

NO real API calls — the provider client is a mock implementing the
``ProviderClient`` protocol. Covers sampling, the token/cost delta computation,
the exact-match quality check, the cost estimate, and the honesty framing.
"""
from __future__ import annotations

import json

from tokenjam.core.db import InMemoryBackend
from tokenjam.core.optimize.validate import (
    VALIDATE_HONESTY_CAVEAT,
    Completion,
    SampledCall,
    collect_downsize_samples,
    estimate_sample_cost,
    exact_match,
    result_to_dict,
    run_validation,
)
from tokenjam.otel.semconv import GenAIAttributes
from tokenjam.utils.time_parse import utcnow
from tests.factories import make_llm_span

from datetime import timedelta


# ---------------------------------------------------------------------------
# Mock provider
# ---------------------------------------------------------------------------


class MockProvider:
    """Records the calls it receives and returns scripted completions keyed by
    model. NO network. Implements the ProviderClient protocol structurally."""

    def __init__(self, responses: dict[str, Completion]):
        self.responses = responses
        self.calls: list[tuple[str, str]] = []

    def complete(self, *, provider, model, messages, max_tokens):  # noqa: ANN001
        self.calls.append((provider, model))
        return self.responses[model]


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


def _seed_captured_downsize_span(db, *, model="claude-opus-4-8",
                                 input_tokens=1000, output_tokens=100,
                                 prompt=None, session_id="s1"):
    """Insert an LLM span that matches the downsize candidate shape AND carries
    captured prompt content, so collect_downsize_samples can select it."""
    if prompt is None:
        prompt = [{"role": "user", "content": "Say hello."}]
    span = make_llm_span(
        model=model,
        provider="anthropic",
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        session_id=session_id,
        extra_attributes={GenAIAttributes.PROMPT_CONTENT: json.dumps(prompt)},
    )
    db.insert_span(span)
    return span


def test_collect_samples_selects_captured_downsize_candidates():
    db = InMemoryBackend()
    _seed_captured_downsize_span(db, session_id="s1")
    _seed_captured_downsize_span(db, session_id="s2")
    since = utcnow() - timedelta(days=1)
    until = utcnow() + timedelta(minutes=1)

    samples = collect_downsize_samples(db.conn, since, until, None, 5)

    assert len(samples) == 2
    s = samples[0]
    assert s.model == "claude-opus-4-8"
    # Opus -> Haiku per the downgrade map.
    assert s.candidate_model == "claude-haiku-4-5"
    assert s.messages == [{"role": "user", "content": "Say hello."}]
    db.close()


def test_collect_samples_skips_spans_without_captured_prompt():
    db = InMemoryBackend()
    # A downsize-shaped span but NO captured prompt content -> unreplayable.
    span = make_llm_span(model="claude-opus-4-8", input_tokens=1000,
                         output_tokens=100, session_id="s1")
    db.insert_span(span)
    since = utcnow() - timedelta(days=1)
    until = utcnow() + timedelta(minutes=1)

    assert collect_downsize_samples(db.conn, since, until, None, 5) == []
    db.close()


def test_collect_samples_skips_non_candidate_models():
    db = InMemoryBackend()
    # Haiku has no cheaper same-family candidate in the downgrade map.
    _seed_captured_downsize_span(db, model="claude-haiku-4-5")
    since = utcnow() - timedelta(days=1)
    until = utcnow() + timedelta(minutes=1)

    assert collect_downsize_samples(db.conn, since, until, None, 5) == []
    db.close()


def test_collect_samples_respects_sample_size_cap():
    db = InMemoryBackend()
    for i in range(5):
        _seed_captured_downsize_span(db, session_id=f"s{i}")
    since = utcnow() - timedelta(days=1)
    until = utcnow() + timedelta(minutes=1)

    assert len(collect_downsize_samples(db.conn, since, until, None, 2)) == 2
    db.close()


def test_collect_samples_skips_large_shape_spans():
    db = InMemoryBackend()
    # Output above the small-shape threshold -> not a downsize candidate.
    _seed_captured_downsize_span(db, output_tokens=5000)
    since = utcnow() - timedelta(days=1)
    until = utcnow() + timedelta(minutes=1)

    assert collect_downsize_samples(db.conn, since, until, None, 5) == []
    db.close()


# ---------------------------------------------------------------------------
# Quality check
# ---------------------------------------------------------------------------


def test_exact_match_normalizes_whitespace_and_case():
    assert exact_match("Hello World", "hello   world\n")
    assert not exact_match("hello", "goodbye")


# ---------------------------------------------------------------------------
# Cost estimate
# ---------------------------------------------------------------------------


def test_estimate_sample_cost_is_positive():
    sample = SampledCall(
        span_id="x", session_id="s1", provider="anthropic",
        model="claude-opus-4-8", candidate_model="claude-haiku-4-5",
        messages=[{"role": "user", "content": "hi"}], max_tokens=256,
        recorded_input_tokens=1000, recorded_output_tokens=100,
    )
    assert estimate_sample_cost([sample]) > 0.0


# ---------------------------------------------------------------------------
# The A/B run — delta + quality aggregation
# ---------------------------------------------------------------------------


def _one_sample():
    return SampledCall(
        span_id="x", session_id="s1", provider="anthropic",
        model="claude-opus-4-8", candidate_model="claude-haiku-4-5",
        messages=[{"role": "user", "content": "hi"}], max_tokens=256,
        recorded_input_tokens=1000, recorded_output_tokens=100,
    )


def test_run_validation_measures_delta_and_quality_preserved():
    provider = MockProvider({
        "claude-opus-4-8": Completion(text="Answer", input_tokens=1000,
                                      output_tokens=100),
        "claude-haiku-4-5": Completion(text="answer\n", input_tokens=1000,
                                       output_tokens=100),
    })
    result = run_validation([_one_sample()], provider)

    assert result.sample_size == 1
    # Same token counts, so token delta is zero here; the cost delta is the win
    # (Haiku is cheaper than Opus).
    assert result.baseline_tokens == 1100
    assert result.candidate_tokens == 1100
    assert result.candidate_cost_usd < result.baseline_cost_usd
    assert result.cost_delta_usd < 0
    # Exact-match survives whitespace/case normalization.
    assert result.quality_preserved == 1
    # Both models were actually re-run.
    assert provider.calls == [
        ("anthropic", "claude-opus-4-8"),
        ("anthropic", "claude-haiku-4-5"),
    ]


def test_run_validation_flags_quality_regression():
    provider = MockProvider({
        "claude-opus-4-8": Completion(text="Correct answer", input_tokens=1000,
                                      output_tokens=100),
        "claude-haiku-4-5": Completion(text="Wrong answer", input_tokens=1000,
                                       output_tokens=90),
    })
    result = run_validation([_one_sample()], provider)

    assert result.quality_preserved == 0
    assert len(result.measurements) == 1
    assert result.measurements[0].quality_preserved is False


def test_run_validation_aggregates_multiple_calls():
    provider = MockProvider({
        "claude-opus-4-8": Completion(text="same", input_tokens=1000,
                                      output_tokens=100),
        "claude-haiku-4-5": Completion(text="same", input_tokens=1000,
                                       output_tokens=50),
    })
    result = run_validation([_one_sample(), _one_sample()], provider)

    assert result.sample_size == 2
    assert result.quality_preserved == 2
    # Candidate produced fewer output tokens -> fewer total tokens.
    assert result.candidate_tokens < result.baseline_tokens
    assert result.tokens_delta < 0


# ---------------------------------------------------------------------------
# Serialization + honesty framing (Rule 14)
# ---------------------------------------------------------------------------


def test_result_to_dict_shape_and_honesty():
    provider = MockProvider({
        "claude-opus-4-8": Completion(text="x", input_tokens=1000,
                                      output_tokens=100),
        "claude-haiku-4-5": Completion(text="x", input_tokens=1000,
                                       output_tokens=100),
    })
    payload = result_to_dict(run_validation([_one_sample()], provider))

    assert payload["sample_size"] == 1
    assert payload["quality_metric"] == "exact_match"
    assert payload["quality_preserved"] == 1
    assert "measured on a sample of 1" in payload["basis"]
    assert payload["caveat"] == VALIDATE_HONESTY_CAVEAT
    # Honesty (Rule 14): never the reserved paid-layer vocabulary.
    blob = json.dumps(payload).lower()
    assert "certified" not in blob
    assert "guaranteed" not in blob


def test_honesty_caveat_never_claims_guarantee():
    low = VALIDATE_HONESTY_CAVEAT.lower()
    assert "certified" not in low
    assert "guaranteed" not in low
    assert "sample" in low
