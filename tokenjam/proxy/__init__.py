"""Optional enforcement-plane proxy (suggest mode only) — issue #219.

An in-process listener inside ``tj serve`` (default port 7392) that sits between
an agent and its LLM provider, speaking the Anthropic (``/v1/messages``) and
OpenAI (``/v1/chat/completions``) APIs. It ships in SUGGEST MODE ONLY: it
records what a policy *would* do and enforces nothing.

The package is layered so the safety-critical decision logic carries no HTTP
dependency and is trivially unit-testable:

- :mod:`tokenjam.proxy.gate` — the pricing-mode gate invariant (pure).
- :mod:`tokenjam.proxy.observer` — observe-only decision recorder.
- :mod:`tokenjam.proxy.app` — the Starlette forwarding app (HTTP).
- :mod:`tokenjam.proxy.server` — build/run the proxy listener inside ``tj serve``.
- :mod:`tokenjam.proxy.wiring` — base-URL env wiring + orphan detection.

Pass-through is sacred: any error in classification / policy / metering forwards
the request unmodified. The proxy holds no keys — client credentials pass
through from the caller.
"""
from __future__ import annotations

from tokenjam.proxy.gate import (
    OBSERVE_ONLY,
    POLICY,
    GateDecision,
    classify,
)

__all__ = ["OBSERVE_ONLY", "POLICY", "GateDecision", "classify"]
