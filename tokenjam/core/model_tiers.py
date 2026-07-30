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
    ("mythos", "mythos"),
    ("opus", "opus"),
    ("sonnet", "sonnet"),
    ("haiku", "haiku"),
)

# Tiers whose sessions are worth a premium-quota audit / right-sizing flag.
# Extend this (and TIER_SUBSTRINGS) when a new premium family launches — that is
# the single edit that teaches every consumer about the new tier.
#
# Mythos sits alongside Fable: `claude-mythos-5` carries the SAME published
# rates as `claude-fable-5` in pricing/models.toml, so a session that ran on it
# burns premium quota exactly as a Fable one does. It was priced but untiered
# for a while, which made it invisible to every premium-gated flag while still
# being billed at the top rate — see `test_every_priced_model_resolves_to_a_tier`
# for the guard that now stops the next family landing half-added.
PREMIUM_TIERS: frozenset[str] = frozenset({"fable", "mythos", "opus"})

# Human-facing label for the premium tier, for copy that names what is audited.
# Must name every tier in PREMIUM_TIERS: it appears in "the quota audit only
# inspects premium-tier (…) sessions", and a tier the audit DOES inspect but the
# label omits reads to the user as out of scope. Pinned by
# `test_premium_tier_label_names_every_premium_tier`.
PREMIUM_TIER_LABEL = "Opus/Fable/Mythos"


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
