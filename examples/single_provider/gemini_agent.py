"""
Google Gemini summarization agent with TokenJam observability.

Demonstrates the Gemini provider path: passes a multi-paragraph text to
gemini-2.0-flash and asks for a concise summary. All LLM calls are captured
by tj via the Gemini integration patch.

Requirements:
    pip install google-generativeai tokenjam

Environment:
    GOOGLE_API_KEY or GEMINI_API_KEY  — required (either one works)

Usage:
    python examples/single_provider/gemini_agent.py
"""
from __future__ import annotations

import os
import sys

# Check for API key before importing the SDK (avoids confusing errors).
api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
if not api_key:
    print("ERROR: GOOGLE_API_KEY or GEMINI_API_KEY environment variable is required.")
    print("  export GOOGLE_API_KEY=AI...")
    sys.exit(1)

# Make sure google.generativeai picks up the key from the standard env var.
os.environ.setdefault("GOOGLE_API_KEY", api_key)

import google.generativeai as genai  # noqa: E402

from tokenjam.sdk import watch  # noqa: E402
from tokenjam.sdk.integrations.gemini import patch_gemini  # noqa: E402

# Monkey-patch the Gemini client BEFORE creating any model instances.
patch_gemini()

# ---------------------------------------------------------------------------
# Source text for summarization
# ---------------------------------------------------------------------------

SOURCE_TEXT = """\
AI agents represent a significant evolution in how we interact with large language
models. Rather than issuing single prompts and receiving single responses, agents
operate in loops: they observe their environment, reason about what to do next,
take an action (such as calling a tool or querying an API), and then feed the
result back into the model for further reasoning.

This loop enables capabilities that were previously impossible with simple
prompt-response interactions. Agents can break complex tasks into subtasks,
maintain working memory across many steps, recover from errors by re-planning,
and coordinate with other agents to tackle problems that require diverse
expertise.

The tooling ecosystem around AI agents has grown rapidly. Frameworks like
LangChain, LangGraph, CrewAI, and AutoGen provide abstractions for defining
agents, tools, and orchestration patterns. Meanwhile, observability platforms
are emerging to help developers understand what their agents are actually doing
at runtime: which tools are being called, how many tokens are consumed, where
latency bottlenecks occur, and whether the agent is drifting from expected
behavior.

Local-first observability is particularly valuable during development. By
capturing telemetry to a local database instead of a cloud service, developers
can iterate quickly without worrying about data privacy, network latency, or
subscription costs. Once the agent is ready for production, the same telemetry
pipeline can be reconfigured to export to a remote backend.

The OpenTelemetry standard provides a natural foundation for agent observability.
OTel's span model maps cleanly to agent actions: each LLM call, tool invocation,
and planning step becomes a span in a trace. Semantic conventions for generative
AI (such as token counts, model names, and provider identifiers) give structure
to the data, enabling consistent dashboards and alerts across different providers
and frameworks.\
"""

# ---------------------------------------------------------------------------
# Agent entry point
# ---------------------------------------------------------------------------


@watch(agent_id="gemini-summarizer")
def run() -> str:
    model = genai.GenerativeModel("gemini-2.0-flash")
    prompt = (
        "Summarize the following text in exactly two sentences. "
        "Be concise and capture the key ideas.\n\n"
        f"{SOURCE_TEXT}"
    )
    response = model.generate_content(prompt)
    return response.text


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    result = run()
    print(f"\nSummary:\n{result}")

    print("\n--- TokenJam Observation ---")
    print("Session and LLM spans have been recorded.")
    print("Run 'tj status --agent gemini-summarizer' to view telemetry.")
    print("Run 'tj cost --agent gemini-summarizer' to see token costs.")
