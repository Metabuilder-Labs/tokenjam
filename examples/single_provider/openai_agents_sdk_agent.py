"""
OpenAI Agents SDK multi-agent example with OCW observability.

Demonstrates a triage/specialist handoff pattern using the OpenAI Agents SDK.
The triage agent inspects the user query and delegates to a specialist agent
for a detailed answer.

This integration works differently from other providers: it configures the
Agents SDK's native OTel support to export traces to `tj serve` via OTLP.
You must have `tj serve` running before executing this script.

Requirements:
    pip install openai-agents httpx tokenjam

Environment:
    OPENAI_API_KEY  — required

Prerequisites:
    tj serve must be running:  tj serve --port 8787

Usage:
    python examples/single_provider/openai_agents_sdk_agent.py
"""
from __future__ import annotations

import asyncio
import os
import sys

if not os.environ.get("OPENAI_API_KEY"):
    print("ERROR: OPENAI_API_KEY environment variable is required.")
    print("  export OPENAI_API_KEY=sk-...")
    sys.exit(1)

# Check that tj serve is reachable before proceeding.
try:
    import httpx

    resp = httpx.get("http://127.0.0.1:8787/api/v1/traces", timeout=2)
    if resp.status_code >= 500:
        raise ConnectionError(f"tj serve returned {resp.status_code}")
except Exception:
    print("ERROR: Cannot reach tj serve at http://127.0.0.1:8787")
    print()
    print("This example requires a running tj serve instance because the")
    print("OpenAI Agents SDK exports traces via OTLP HTTP to the local server.")
    print()
    print("Start it in another terminal:")
    print("  tj serve --port 8787")
    sys.exit(1)

from agents import Agent, Runner  # noqa: E402

from tokenjam.sdk import watch  # noqa: E402
from tokenjam.sdk.integrations.openai_agents_sdk import patch_openai_agents  # noqa: E402

# Configure the Agents SDK's native OTel to export to tj serve.
# NOTE: patch_openai_agents() does NOT call ensure_initialised() — it sets up
# OTLP export to tj serve instead. We still use @watch() for session tracking.
patch_openai_agents()

# ---------------------------------------------------------------------------
# Agent definitions
# ---------------------------------------------------------------------------

coding_specialist = Agent(
    name="coding_specialist",
    instructions=(
        "You are a coding expert. When asked about programming topics, give "
        "clear, concise explanations with short code examples when helpful. "
        "Keep answers under 200 words."
    ),
    model="gpt-4o-mini",
)

science_specialist = Agent(
    name="science_specialist",
    instructions=(
        "You are a science expert. When asked about scientific topics, give "
        "clear, concise explanations suitable for a general audience. "
        "Keep answers under 200 words."
    ),
    model="gpt-4o-mini",
)

triage_agent = Agent(
    name="triage_agent",
    instructions=(
        "You are a triage agent. Examine the user's question and hand off to "
        "the most appropriate specialist:\n"
        "- For programming, software, or coding questions -> coding_specialist\n"
        "- For science, physics, biology, or chemistry questions -> science_specialist\n"
        "Do NOT answer the question yourself. Always hand off to a specialist."
    ),
    model="gpt-4o-mini",
    handoffs=[coding_specialist, science_specialist],
)

# ---------------------------------------------------------------------------
# Agent entry point
# ---------------------------------------------------------------------------


@watch(agent_id="openai-agents-sdk-demo")
def run() -> str:
    query = "How does a Python decorator work under the hood?"
    print(f"User query: {query}\n")

    result = asyncio.run(
        Runner.run(triage_agent, input=query)
    )

    return result.final_output


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    result = run()
    print(f"\nAgent response:\n{result}")

    print("\n--- OCW Observation ---")
    print("Session and agent handoff spans have been recorded.")
    print("Run 'tj status --agent openai-agents-sdk-demo' to view telemetry.")
    print("Run 'tj traces --agent openai-agents-sdk-demo' to see the trace.")
