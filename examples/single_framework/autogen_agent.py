"""
AutoGen agent example with OCW observability.

Creates two ConversableAgent debaters that argue for and against a position.
OCW patches ConversableAgent.generate_reply and .initiate_chat for span capture.

Extra deps: pip install pyautogen
Run:        python examples/single_framework/autogen_agent.py
"""
import os
import sys

if not os.environ.get("OPENAI_API_KEY"):
    sys.exit(
        "OPENAI_API_KEY not set.\n"
        "Export it before running: export OPENAI_API_KEY=sk-..."
    )

from autogen import ConversableAgent  # noqa: E402

from tj.sdk import watch, patch_autogen  # noqa: E402

# Patch AutoGen BEFORE creating agents
patch_autogen()

TOPIC = "AI agents should always have observability built in"


@watch(agent_id="autogen-demo")
def main():
    llm_config = {
        "model": "gpt-4o-mini",
        "api_key": os.environ["OPENAI_API_KEY"],
    }

    debater_for = ConversableAgent(
        name="debater_for",
        system_message=(
            f"You argue FOR the position: '{TOPIC}'. "
            "Keep responses to 2-3 sentences. Be persuasive but concise."
        ),
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    debater_against = ConversableAgent(
        name="debater_against",
        system_message=(
            f"You argue AGAINST the position: '{TOPIC}'. "
            "Keep responses to 2-3 sentences. Be persuasive but concise."
        ),
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    print(f"Debate topic: {TOPIC}\n")
    print("Starting 3-turn debate...\n")

    # initiate_chat is patched by OCW to create a span
    chat_result = debater_for.initiate_chat(
        debater_against,
        message=(
            f"I'll start: {TOPIC}. "
            "Every AI agent in production needs observability -- without it, "
            "you're flying blind when things go wrong."
        ),
        max_turns=3,
    )

    print("\n--- Chat History ---")
    for msg in chat_result.chat_history:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        print(f"[{role}] {content}\n")

    # --- Observation ---
    print("--- OCW Observation ---")
    print("AutoGen integration captured spans for:")
    print("  - Chat initiation via ConversableAgent.initiate_chat")
    print("  - Reply generation via ConversableAgent.generate_reply")
    print("Run 'ocw traces' to see the captured telemetry.")


if __name__ == "__main__":
    main()
