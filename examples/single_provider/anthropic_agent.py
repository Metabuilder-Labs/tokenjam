"""
Anthropic tool-use agent with OCW observability.

Demonstrates the full Anthropic tool-use loop: message -> tool_use -> tool_result -> final
response, with each tool invocation recorded via tj's record_tool_call().

Requirements:
    pip install anthropic tokenjam

Environment:
    ANTHROPIC_API_KEY  — required

Usage:
    python examples/single_provider/anthropic_agent.py
"""
from __future__ import annotations

import json
import os
import sys

import anthropic

from tokenjam.sdk import watch
from tokenjam.sdk.agent import record_tool_call
from tokenjam.sdk.integrations.anthropic import patch_anthropic

if not os.environ.get("ANTHROPIC_API_KEY"):
    print("ERROR: ANTHROPIC_API_KEY environment variable is required.")
    print("  export ANTHROPIC_API_KEY=sk-ant-...")
    sys.exit(1)

# Monkey-patch the Anthropic client BEFORE creating any instances.
patch_anthropic()

# ---------------------------------------------------------------------------
# Tool definitions (Anthropic format)
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "name": "calculator",
        "description": "Evaluate a mathematical expression and return the result.",
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Math expression to evaluate, e.g. '42 * 17'",
                },
            },
            "required": ["expression"],
        },
    },
    {
        "name": "get_weather",
        "description": "Get the current weather for a given city.",
        "input_schema": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "City name, e.g. 'San Francisco'",
                },
            },
            "required": ["city"],
        },
    },
]


# ---------------------------------------------------------------------------
# Local tool implementations
# ---------------------------------------------------------------------------


def calculator(expression: str) -> dict:
    """Safely evaluate a math expression."""
    try:
        result = eval(expression, {"__builtins__": {}}, {})  # noqa: S307
        return {"result": float(result)}
    except Exception as exc:
        return {"error": str(exc)}


def get_weather(city: str) -> dict:
    """Return stub weather data for demo purposes."""
    return {
        "city": city,
        "temperature_f": 62,
        "condition": "Partly cloudy",
        "humidity_pct": 73,
    }


TOOL_DISPATCH = {
    "calculator": calculator,
    "get_weather": get_weather,
}


# ---------------------------------------------------------------------------
# Agent entry point
# ---------------------------------------------------------------------------


@watch(agent_id="anthropic-tool-agent")
def run() -> str:
    client = anthropic.Anthropic()
    messages: list[dict] = [
        {
            "role": "user",
            "content": (
                "What's the weather in San Francisco right now, "
                "and what is 42 * 17?"
            ),
        },
    ]

    # Step 1 — initial request (model will request tool calls)
    response = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=1024,
        tools=TOOLS,
        messages=messages,
    )

    # Step 2 — process tool_use blocks and collect results
    while response.stop_reason == "tool_use":
        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue

            tool_name = block.name
            tool_input = block.input
            func = TOOL_DISPATCH.get(tool_name)
            if func is None:
                tool_output = {"error": f"Unknown tool: {tool_name}"}
            else:
                tool_output = func(**tool_input)

            # Record the tool call in tj for observability
            record_tool_call(
                tool_name=tool_name,
                tool_input=tool_input,
                tool_output=tool_output,
            )

            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": json.dumps(tool_output),
                }
            )

        # Send tool results back to the model
        messages.append({"role": "assistant", "content": response.content})
        messages.append({"role": "user", "content": tool_results})

        response = client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=1024,
            tools=TOOLS,
            messages=messages,
        )

    # Step 3 — extract final text response
    final_text = ""
    for block in response.content:
        if hasattr(block, "text"):
            final_text += block.text
    return final_text


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    result = run()
    print(f"\nAgent response:\n{result}")

    print("\n--- OCW Observation ---")
    print("Session and tool spans have been recorded.")
    print("Run 'tj status --agent anthropic-tool-agent' to view telemetry.")
    print("Run 'tj tools --agent anthropic-tool-agent' to see tool call stats.")
