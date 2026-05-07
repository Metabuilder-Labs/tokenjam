"""
Provider Router Agent — routes tasks to the best LLM provider.

Demonstrates multi-provider observability with ocw: each routing decision
and LLM call appears as a span in a single trace, enabling cost comparison
across providers.

Extra deps:
    pip install anthropic openai google-generativeai

Required env vars:
    ANTHROPIC_API_KEY, OPENAI_API_KEY, GOOGLE_API_KEY (or GEMINI_API_KEY)
"""
from __future__ import annotations

import os
import sys

from tj.sdk import watch
from tj.sdk.agent import record_tool_call
from tj.sdk.integrations.anthropic import patch_anthropic
from tj.sdk.integrations.gemini import patch_gemini
from tj.sdk.integrations.openai import patch_openai

# ---------------------------------------------------------------------------
# Env-var gate
# ---------------------------------------------------------------------------
REQUIRED_KEYS = ["ANTHROPIC_API_KEY", "OPENAI_API_KEY"]
GOOGLE_KEY = os.environ.get("GOOGLE_API_KEY") or os.environ.get(
    "GEMINI_API_KEY"
)
missing = [k for k in REQUIRED_KEYS if not os.environ.get(k)]
if not GOOGLE_KEY:
    missing.append("GOOGLE_API_KEY or GEMINI_API_KEY")
if missing:
    sys.exit(f"Missing env vars: {', '.join(missing)}")

# ---------------------------------------------------------------------------
# Activate provider patches BEFORE creating any clients
# ---------------------------------------------------------------------------
patch_openai()
patch_anthropic()
patch_gemini()


# ---------------------------------------------------------------------------
# Provider helpers
# ---------------------------------------------------------------------------
def route(task_type: str) -> str:
    """Pick the best provider for a given task type."""
    mapping = {
        "factual": "gemini",
        "code": "anthropic",
        "creative": "openai",
    }
    return mapping.get(task_type, "openai")


def ask_gemini(prompt: str) -> str:
    """Send a prompt to Gemini Flash."""
    import google.generativeai as genai  # type: ignore[import-untyped]

    genai.configure(api_key=GOOGLE_KEY)
    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(prompt)
    return response.text


def ask_claude(prompt: str) -> str:
    """Send a prompt to Claude."""
    import anthropic

    client = anthropic.Anthropic()
    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text


def ask_openai(prompt: str) -> str:
    """Send a prompt to GPT-4o."""
    import openai

    client = openai.OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1024,
    )
    return response.choices[0].message.content or ""


PROVIDERS = {
    "gemini": ask_gemini,
    "anthropic": ask_claude,
    "openai": ask_openai,
}


# ---------------------------------------------------------------------------
# Main agent
# ---------------------------------------------------------------------------
@watch(agent_id="router-agent")
def main() -> None:
    tasks = [
        ("factual", "What is the capital of France?"),
        ("code", "Write a Python function to find prime numbers"),
        ("creative", "Write a short poem about debugging"),
    ]

    for task_type, prompt in tasks:
        provider = route(task_type)

        # Record the routing decision as a tool call span
        record_tool_call(
            "route",
            tool_input={"task_type": task_type},
            tool_output={"provider": provider},
        )

        print(f"\n[{task_type}] Routed to: {provider}")
        print(f"  Prompt: {prompt}")

        handler = PROVIDERS[provider]
        response = handler(prompt)
        print(f"  Response: {response[:200]}...")


if __name__ == "__main__":
    main()

# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------
# After running this script, inspect the trace and cost breakdown:
#
#   ocw traces --since 5m
#       -> Shows a single trace with spans for each provider call
#
#   ocw trace <trace-id>
#       -> Waterfall view: session span -> route tool calls + LLM calls
#
#   ocw cost --since 1h
#       -> Compare cost across gemini, anthropic, and openai in one session
#
#   ocw tools
#       -> Shows the "route" tool call count and timing
