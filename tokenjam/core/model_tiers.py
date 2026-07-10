"""Provider-agnostic model-family tier classification.

One place that knows which Anthropic model families are *premium* (worth a
quota audit / right-sizing flag) and how the tiers rank. The optimize analyzers
consume this instead of each hardcoding an ``"opus"`` substring — so when a new
family launches above the current top (as **Fable** launched above Opus), it is
a one-line edit here, not a grep-and-patch across every analyzer.

Membership is decided by a lowercased-substring match on the model id, which
tolerates version suffixes, ``YYYYMMDD`` dates, provider prefixes (Bedrock's
``us-anthropic-…`` / ``global-anthropic-…``), and ``[1m]`` context tags — e.g.
``us-anthropic-claude-opus-4-8-20260115[1m]`` all resolve to ``"opus"``. This
mirrors the tolerance the pricing layer already relies on and is the same
matching the analyzers used before this module existed.
"""
from __future__ import annotations

# Model-family tiers, most capable first. Fable sits ABOVE Opus. Each entry is
# ``(substring, tier)``; the first substring found in the (lowercased) model id
# wins. No model id contains two family names, so ordering only matters as a
# tie-break that never fires in practice — but keeping it capability-ordered
# documents the ladder for the downgrade suggestions below.
TIER_SUBSTRINGS: tuple[tuple[str, str], ...] = (
    ("fable", "fable"),
    ("opus", "opus"),
    ("sonnet", "sonnet"),
    ("haiku", "haiku"),
)

# Tiers whose sessions are worth a premium-quota audit / right-sizing flag.
# Extend this (and TIER_SUBSTRINGS) when a new premium family launches — that is
# the single edit that teaches every consumer about the new tier.
PREMIUM_TIERS: frozenset[str] = frozenset({"fable", "opus"})

# Human-facing label for the premium tier, for copy that names what is audited.
PREMIUM_TIER_LABEL = "Opus/Fable"


def model_tier(model: str | None) -> str | None:
    """Classify a model id into its family tier, or ``None`` if unrecognised."""
    normalised = (model or "").lower()
    for substring, tier in TIER_SUBSTRINGS:
        if substring in normalised:
            return tier
    return None


def is_premium_tier(model: str | None) -> bool:
    """True when the model belongs to a premium tier (Fable or Opus today)."""
    return model_tier(model) in PREMIUM_TIERS
