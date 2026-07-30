"""Unit tests for the shared model-family tier predicate.

This is the single source of truth for "which model families are premium"
consumed by the downsize / quota-audit / subagent right-sizing analyzers, so it
must classify Fable (the tier ABOVE Opus) as premium and tolerate the id shapes
that show up in real telemetry (dates, `[1m]` tags, Bedrock prefixes).
"""
from __future__ import annotations

import pytest

from tokenjam.core.model_tiers import (
    PREMIUM_TIER_LABEL,
    PREMIUM_TIERS,
    is_premium_tier,
    model_tier,
)
from tokenjam.core.pricing import load_pricing_rows


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


def test_mythos_is_premium():
    """`claude-mythos-5` prices identically to `claude-fable-5`, so a session
    that ran on it burns premium quota exactly as a Fable one does.

    It shipped priced-but-untiered, which meant `is_premium_tier` said False and
    it fell out of every premium-gated flag (subagent's `over_powered`,
    downsize's driver-role case, the quota audit) while still being billed at
    the top rate. Only downsize's tiny-session case reached it, because that one
    gates on `DOWNGRADE_CANDIDATES` membership rather than on the tier."""
    assert is_premium_tier("claude-mythos-5")
    assert model_tier("claude-mythos-5") == "mythos"
    assert "mythos" in PREMIUM_TIERS
    # Same id tolerance the other families get.
    assert model_tier("claude-mythos-5-20260601") == "mythos"
    assert model_tier("claude-mythos-5[1m]") == "mythos"


def test_premium_tier_label_names_every_premium_tier():
    """The label is what the quota audit prints when it says which sessions it
    inspects. A tier the audit DOES inspect but the label omits reads to the
    user as out of scope — which is exactly what happened while Mythos was
    audited-but-unnamed. Derived from PREMIUM_TIERS, not spelled out, so adding
    a premium family cannot leave the copy behind."""
    label = PREMIUM_TIER_LABEL.lower()
    for tier in PREMIUM_TIERS:
        assert tier in label, (
            f"{tier!r} is a premium tier but PREMIUM_TIER_LABEL "
            f"({PREMIUM_TIER_LABEL!r}) does not name it"
        )


def test_sonnet_and_haiku_are_not_premium():
    assert not is_premium_tier("claude-sonnet-4-6")
    assert not is_premium_tier("claude-haiku-4-5")
    assert not is_premium_tier("gpt-4o")
    assert not is_premium_tier(None)


# ---------------------------------------------------------------------------
# Fail-closed guard: priced, therefore tiered
# ---------------------------------------------------------------------------
#
# `claude-mythos-5` landed in pricing/models.toml with a full set of rates but
# no entry in TIER_SUBSTRINGS. Nothing failed: it simply resolved to tier None,
# `is_premium_tier` returned False, and it dropped silently out of every
# premium-gated analyzer while being billed at Fable's rate. The failure mode is
# invisible by construction — a half-added family looks exactly like a
# deliberately-unclassified one — so it needs a test that says so out loud.
#
# SCOPE. The ladder in TIER_SUBSTRINGS is the ANTHROPIC family ladder
# (fable/mythos > opus > sonnet > haiku); it makes no claim about GPT, Gemini,
# Nova or Grok, and `test_model_tier_none_for_unrecognised` pins that those
# correctly return None. So the guard covers every priced model whose id names a
# Claude model — under any provider, since Bedrock and HUD both host Claude
# under their own keys — and fails closed within that scope.
_CLAUDE_ID_MARKER = "claude"

#: Claude-id models that legitimately have no tier. Add here ONLY with a reason;
#: an entry is a claim that no premium gate should ever see this model. Empty
#: today: every Claude id the packaged table prices is a real billable family.
_UNTIERED_CLAUDE_MODELS: frozenset[str] = frozenset()


def _priced_claude_models() -> list[tuple[str, str]]:
    """Every (provider, model) in the packaged table naming a Claude model."""
    return sorted(
        (provider, model)
        for provider, models in load_pricing_rows().items()
        for model in models
        if _CLAUDE_ID_MARKER in model.lower()
    )


def test_the_guard_has_a_corpus_to_guard():
    """A lookup that quietly returned nothing would make the guard below vacuous
    and green forever."""
    assert len(_priced_claude_models()) >= 5


def test_every_priced_model_resolves_to_a_tier():
    """A Claude model with pricing but no tier is a half-added family.

    It is billed at its real rate but invisible to `is_premium_tier`, so every
    premium-gated flag skips it without a word. Adding a model to
    pricing/models.toml is therefore also a commitment to classify it — or to
    declare it exempt in `_UNTIERED_CLAUDE_MODELS`, with a reason."""
    untiered = [
        f"{provider}/{model}"
        for provider, model in _priced_claude_models()
        if model not in _UNTIERED_CLAUDE_MODELS and model_tier(model) is None
    ]
    assert not untiered, (
        "priced but untiered — invisible to every premium-gated analyzer: "
        + ", ".join(untiered)
        + ". Add the family to model_tiers.TIER_SUBSTRINGS (and to PREMIUM_TIERS "
        "if it is a premium family), or list it in _UNTIERED_CLAUDE_MODELS with "
        "a reason."
    )


def test_exemptions_are_live_not_stale():
    """An exemption for a model no longer in the table is a stale claim that
    would silently re-open the hole if the id ever came back differently."""
    priced = {model for _provider, model in _priced_claude_models()}
    assert _UNTIERED_CLAUDE_MODELS <= priced, (
        "exemption(s) name models the pricing table no longer carries: "
        f"{sorted(_UNTIERED_CLAUDE_MODELS - priced)}"
    )
