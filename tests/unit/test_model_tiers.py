"""Unit tests for the shared model-family tier predicate.

This is the single source of truth for "which model families are premium"
consumed by the downsize / quota-audit / subagent right-sizing analyzers, so it
must classify Fable (the tier ABOVE Opus) as premium and tolerate the id shapes
that show up in real telemetry (dates, `[1m]` tags, Bedrock prefixes).
"""
from __future__ import annotations

import pytest

from tokenjam.core.model_tiers import (
    PREMIUM_TIERS,
    is_premium_tier,
    model_tier,
)


@pytest.mark.parametrize(
    "model, tier",
    [
        ("claude-fable-5", "fable"),
        ("claude-opus-4-8", "opus"),
        ("claude-opus-4-7", "opus"),
        ("claude-opus-4", "opus"),
        ("claude-sonnet-4-6", "sonnet"),
        ("claude-sonnet-5", "sonnet"),
        ("claude-haiku-4-5", "haiku"),
        # Tolerate dates, [1m] context tags, and Bedrock provider prefixes.
        ("claude-opus-4-8-20260115", "opus"),
        ("claude-opus-4-8[1m]", "opus"),
        ("claude-fable-5[1m]", "fable"),
        ("us-anthropic-claude-opus-4-1-20250805-v1", "opus"),
        ("global-anthropic-claude-opus-4-5-20251101-v1", "opus"),
    ],
)
def test_model_tier_classifies_known_families(model, tier):
    assert model_tier(model) == tier


@pytest.mark.parametrize("model", ["gpt-4o", "gemini-2-5-pro", "unknown", "", None])
def test_model_tier_none_for_unrecognised(model):
    assert model_tier(model) is None


def test_fable_and_opus_are_premium():
    assert is_premium_tier("claude-fable-5")
    assert is_premium_tier("claude-opus-4-8")
    assert {"fable", "opus"} <= PREMIUM_TIERS


def test_sonnet_and_haiku_are_not_premium():
    assert not is_premium_tier("claude-sonnet-4-6")
    assert not is_premium_tier("claude-haiku-4-5")
    assert not is_premium_tier("gpt-4o")
    assert not is_premium_tier(None)
