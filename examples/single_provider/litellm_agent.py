"""
LiteLLM multi-provider agent with OCW observability.

Demonstrates patch_litellm() routing calls to multiple LLM providers through
LiteLLM's unified interface. Each call is automatically attributed to the
correct provider in tj spans.

Requirements:
    pip install litellm tokenjam

Environment:
    OPENAI_API_KEY     — required (for OpenAI calls)
    ANTHROPIC_API_KEY  — required (for Anthropic calls)

Usage:
    python examples/single_provider/litellm_agent.py
"""
from __future__ import annotations

import os
import sys

REQUIRED_KEYS = ["OPENAI_API_KEY", "ANTHROPIC_API_KEY"]
missing = [k for k in REQUIRED_KEYS if not os.environ.get(k)]
if missing:
    print(f"ERROR: Missing environment variables: {', '.join(missing)}")
    for k in missing:
        print(f"  export {k}=...")
    sys.exit(1)

import litellm  # noqa: E402

from tokenjam.sdk import watch  # noqa: E402
from tokenjam.sdk.integrations.litellm import patch_litellm  # noqa: E402

# Patch litellm BEFORE making any calls.
# This single patch covers all providers that litellm routes to.
# If you also have patch_openai() or patch_anthropic() active, the litellm
# patch takes priority — no double-counted spans.
patch_litellm()


# ---------------------------------------------------------------------------
# Agent entry point
# ---------------------------------------------------------------------------

@watch(agent_id="litellm-multi-provider")
def run() -> None:
    # Call 1 — routed to OpenAI
    print("Calling OpenAI via LiteLLM...")
    response = litellm.completion(
        model="openai/gpt-4o-mini",
        messages=[{"role": "user", "content": "What is 2 + 2?"}],
        max_tokens=50,
    )
    print(f"  OpenAI: {response.choices[0].message.content.strip()}")

    # Call 2 — routed to Anthropic
    print("Calling Anthropic via LiteLLM...")
    response = litellm.completion(
        model="anthropic/claude-haiku-4-5",
        messages=[{"role": "user", "content": "What is the capital of Japan?"}],
        max_tokens=50,
    )
    print(f"  Anthropic: {response.choices[0].message.content.strip()}")

    # Call 3 — another OpenAI call to show cost accumulation
    print("Calling OpenAI again via LiteLLM...")
    response = litellm.completion(
        model="openai/gpt-4o-mini",
        messages=[
            {"role": "user", "content": "Write a haiku about observability."},
        ],
        max_tokens=100,
    )
    print(f"  OpenAI: {response.choices[0].message.content.strip()}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run()

    print("\n--- OCW Observation ---")
    print("All 3 calls routed through LiteLLM appear as separate spans,")
    print("each attributed to the correct provider (openai / anthropic).")
    print()
    print("  tj traces                        # see the trace with all 3 spans")
    print("  tj cost --since 1h               # cost breakdown by provider")
    print("  tj status --agent litellm-multi-provider")
