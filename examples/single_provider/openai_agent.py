"""
OpenAI tool-use agent with streaming and OCW observability.

Demonstrates the OpenAI function-calling loop with a stub tool, then streams
the final response token by token. All LLM calls and tool invocations are
captured by tj.

Requirements:
    pip install openai tokenjam

Environment:
    OPENAI_API_KEY  — required

Usage:
    python examples/single_provider/openai_agent.py
"""
from __future__ import annotations

import json
import os
import sys

import openai

from tokenjam.sdk import watch
from tokenjam.sdk.agent import record_tool_call
from tokenjam.sdk.integrations.openai import patch_openai

if not os.environ.get("OPENAI_API_KEY"):
    print("ERROR: OPENAI_API_KEY environment variable is required.")
    print("  export OPENAI_API_KEY=sk-...")
    sys.exit(1)

# Monkey-patch the OpenAI client BEFORE creating any instances.
patch_openai()

# ---------------------------------------------------------------------------
# Tool definition (OpenAI function-calling format)
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Search the web for current information on a topic.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query.",
                    },
                },
                "required": ["query"],
            },
        },
    },
]


# ---------------------------------------------------------------------------
# Local tool implementation
# ---------------------------------------------------------------------------


def search_web(query: str) -> dict:
    """Return stub search results for demo purposes."""
    return {
        "results": [
            {
                "title": f"Top result for: {query}",
                "snippet": (
                    "AI agents are autonomous software systems that use LLMs "
                    "to plan and execute multi-step tasks."
                ),
                "url": "https://example.com/ai-agents-overview",
            },
            {
                "title": f"Recent news: {query}",
                "snippet": (
                    "The latest frameworks include LangGraph, CrewAI, and "
                    "OpenAI's Agents SDK for building agentic workflows."
                ),
                "url": "https://example.com/agent-frameworks",
            },
        ],
    }


TOOL_DISPATCH = {
    "search_web": search_web,
}


# ---------------------------------------------------------------------------
# Agent entry point
# ---------------------------------------------------------------------------


@watch(agent_id="openai-tool-agent")
def run() -> str:
    client = openai.OpenAI()
    messages: list[dict] = [
        {
            "role": "user",
            "content": "What are the latest AI agent frameworks available today?",
        },
    ]

    # Step 1 — initial request (model may request tool calls)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        tools=TOOLS,
    )
    choice = response.choices[0]

    # Step 2 — handle tool calls if present
    if choice.finish_reason == "tool_calls" and choice.message.tool_calls:
        # Append the assistant message with tool calls
        messages.append(choice.message.model_dump())

        for tool_call in choice.message.tool_calls:
            func_name = tool_call.function.name
            func_args = json.loads(tool_call.function.arguments)
            func = TOOL_DISPATCH.get(func_name)

            if func is None:
                tool_output = {"error": f"Unknown tool: {func_name}"}
            else:
                tool_output = func(**func_args)

            # Record the tool call in tj
            record_tool_call(
                tool_name=func_name,
                tool_input=func_args,
                tool_output=tool_output,
            )

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(tool_output),
                }
            )

    # Step 3 — stream the final response
    print("\nAgent response (streamed):")
    stream = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        stream=True,
    )

    collected: list[str] = []
    for chunk in stream:
        delta = chunk.choices[0].delta if chunk.choices else None
        if delta and delta.content:
            print(delta.content, end="", flush=True)
            collected.append(delta.content)
    print()  # newline after streaming

    return "".join(collected)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    result = run()

    print("\n--- OCW Observation ---")
    print("Session, LLM, and tool spans have been recorded.")
    print("Run 'tj status --agent openai-tool-agent' to view telemetry.")
    print("Run 'tj tools --agent openai-tool-agent' to see tool call stats.")
