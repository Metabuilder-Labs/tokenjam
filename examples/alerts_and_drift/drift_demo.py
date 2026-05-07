"""
Behavioral Drift Detection Demo

Phase 1: Builds a baseline from 12 normal sessions (same agent, consistent
         behavior).
Phase 2: Runs 1 anomalous session with 5x token usage and different tool calls.

Shows how ocw detects statistical drift and fires DRIFT_DETECTED alerts.

No API keys required — uses simulated instrumentation.

Drift detection is on by default for any observed agent (see DriftConfig
defaults), so this demo needs no extra config — it just runs.
"""
from __future__ import annotations

import time

from tj.sdk.agent import AgentSession, record_llm_call, record_tool_call
from tj.utils.ids import new_uuid


# ---------------------------------------------------------------------------
# Session runners
# ---------------------------------------------------------------------------

def run_normal_session(session_num: int) -> None:
    """Run a single baseline session with consistent, predictable behavior."""
    with AgentSession(agent_id="drift-demo", conversation_id=new_uuid()):
        # 3 LLM calls with stable token counts
        for _ in range(3):
            record_llm_call(
                model="claude-haiku-4-5",
                provider="anthropic",
                input_tokens=200,
                output_tokens=100,
            )

        # 2 tool calls -- always the same tools in the same order
        record_tool_call(
            "search",
            tool_input={"query": f"topic for session {session_num}"},
            tool_output={"results": ["result_a", "result_b"]},
        )
        record_tool_call(
            "summarize",
            tool_input={"text": "Some content to summarize."},
            tool_output={"summary": "A brief summary."},
        )


def run_anomalous_session() -> None:
    """Run a single anomalous session -- 5x tokens, different tools."""
    with AgentSession(agent_id="drift-demo", conversation_id=new_uuid()):
        # 8 LLM calls with 5x token counts
        for _ in range(8):
            record_llm_call(
                model="claude-haiku-4-5",
                provider="anthropic",
                input_tokens=1000,
                output_tokens=500,
            )

        # 5 tool calls -- completely different tool names
        record_tool_call(
            "fetch_data",
            tool_input={"url": "https://example.com/data"},
            tool_output={"rows": 1500},
        )
        record_tool_call(
            "parse_html",
            tool_input={"html": "<div>...</div>"},
            tool_output={"elements": 42},
        )
        record_tool_call(
            "extract_entities",
            tool_input={"text": "Long document text..."},
            tool_output={"entities": ["Entity1", "Entity2", "Entity3"]},
        )
        record_tool_call(
            "classify",
            tool_input={"text": "Classify this content."},
            tool_output={"label": "finance", "confidence": 0.92},
        )
        record_tool_call(
            "store_results",
            tool_input={"key": "analysis_001", "value": "..."},
            tool_output={"stored": True},
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from tj.sdk.bootstrap import ensure_initialised
    ensure_initialised()

    print("=" * 60)
    print("OCW Behavioral Drift Detection Demo")
    print("=" * 60)

    # Phase 1 -- build baseline
    print("\nPhase 1: Building baseline (12 normal sessions)...")
    for i in range(1, 13):
        run_normal_session(i)
        print(f"  Session {i:2d}/12 complete")
        time.sleep(0.05)  # Distinct timestamps between sessions

    # Phase 2 -- anomalous session
    print("\nPhase 2: Running anomalous session...")
    run_anomalous_session()
    print("  Anomalous session complete")

    print("\n" + "=" * 60)
    print("What to observe:")
    print("=" * 60)
    print(
        "The anomalous session used 5x the normal token count and\n"
        "a completely different set of tool calls. If drift detection\n"
        "is enabled in your ocw.toml, a DRIFT_DETECTED alert should\n"
        "fire when the anomalous session ends.\n"
        "\n"
        "Run these commands to inspect:\n"
        "\n"
        "  ocw alerts                          # see drift alerts\n"
        "  ocw status --agent drift-demo       # agent overview\n"
        "  ocw traces                          # list all 13 traces\n"
    )
