"""The pricing-mode gate — the non-negotiable substrate invariant (#219).

The FIRST step in the proxy's decision path resolves the session's pricing mode.
**Subscription traffic — and ``unknown`` as a fail-safe — is forwarded
UNMODIFIED (observe-only, never a policy decision)**, because intercepting
subscription-plan traffic is outside provider terms of service. ``local`` is
treated the same way (not usage-billed). ONLY ``api``/usage-billed traffic
reaches the policy path.

This module is intentionally pure — it imports only the existing pricing-mode
logic (``core/framing.provider_pricing_mode`` → ``pricing_mode_for`` →
``SUBSCRIPTION_PLAN_TIERS`` in ``otel/semconv``) and carries NO HTTP dependency,
so the invariant stays inspectable and is unit-tested directly.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from tokenjam.core.framing import provider_pricing_mode

# The two decision paths a request can take.
OBSERVE_ONLY = "observe_only"  # forward UNMODIFIED; never a policy decision
POLICY = "policy"              # api/usage-billed → policy path (suggest-mode stub here)

# Providers the proxy understands. Anything else resolves to "unknown" → observe.
KNOWN_PROVIDERS = frozenset({"anthropic", "openai"})


@dataclass(frozen=True)
class GateDecision:
    """The result of the pricing-mode gate for one request — fully inspectable.

    ``path`` is the contract the rest of the proxy honors: ``OBSERVE_ONLY``
    requests are forwarded unmodified and never reach policy evaluation;
    ``POLICY`` requests reach the (suggest-mode, no-op) policy path. In suggest
    mode BOTH are forwarded unmodified — the distinction is which requests are
    *eligible* for enforcement once it lands (#220).
    """
    provider:     str
    plan_tier:    str | None
    pricing_mode: str          # api | subscription | local | unknown
    path:         str          # OBSERVE_ONLY | POLICY
    reason:       str
    killswitch:   bool = False

    @property
    def observe_only(self) -> bool:
        return self.path == OBSERVE_ONLY


def classify(config: Any, provider: str, *, killswitch: bool = False) -> GateDecision:
    """Resolve the pricing mode FIRST, then decide the path — the invariant.

    - ``killswitch`` → always ``OBSERVE_ONLY`` (pass-through-everything).
    - ``api`` (usage-billed) → ``POLICY`` (the ONLY path to policy evaluation).
    - subscription / local / unknown → ``OBSERVE_ONLY`` (TOS + fail-safe).

    The pricing mode is read from the existing declared-plan logic, never
    re-derived here.
    """
    plan_tier, pricing_mode = provider_pricing_mode(config, provider)

    if killswitch:
        return GateDecision(
            provider=provider, plan_tier=plan_tier, pricing_mode=pricing_mode,
            path=OBSERVE_ONLY, reason="killswitch_passthrough", killswitch=True,
        )

    # api/usage-billed is the ONLY traffic that may reach the policy path.
    if pricing_mode == "api":
        return GateDecision(
            provider=provider, plan_tier=plan_tier, pricing_mode=pricing_mode,
            path=POLICY, reason="api_usage_billed",
        )

    # Subscription (TOS), local (not usage-billed), and unknown (fail-safe) are
    # all forwarded unmodified and observed only — never a policy decision.
    return GateDecision(
        provider=provider, plan_tier=plan_tier, pricing_mode=pricing_mode,
        path=OBSERVE_ONLY, reason=f"{pricing_mode}_passthrough",
    )
