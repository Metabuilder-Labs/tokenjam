"""
Budget Breach Alert Demo

Simulates an agent that exceeds a very low budget ($0.05 daily, $0.02 session).
Shows how ocw tracks costs and fires budget alerts.

No API keys required — uses simulated instrumentation.

The demo seeds its own [agents.budget-demo] block in the active ocw
config on startup, so it works out of the box on a fresh `ocw onboard`.
The injected config is equivalent to:

    [agents.budget-demo.budget]
    daily_usd = 0.05
    session_usd = 0.02
"""
from __future__ import annotations

import time

from tj.sdk.agent import watch, record_llm_call, record_tool_call


# ---------------------------------------------------------------------------
# Pricing reference (from pricing/models.toml for claude-sonnet-4-20250514):
#   input  = $3.00 / MTok  -> $0.000003 per token
#   output = $15.00 / MTok -> $0.000015 per token
#
# With 1000 input + 500 output tokens per call:
#   cost = 1000*3e-6 + 500*15e-6 = 0.003 + 0.0075 = $0.0105 per call
#
# Session budget of $0.02 should be breached after ~2 calls.
# Daily budget of $0.05 should be breached after ~5 calls.
# ---------------------------------------------------------------------------

@watch(agent_id="budget-demo")
def run_expensive_agent() -> None:
    """Simulate an agent that blows through its budget."""

    cumulative_cost = 0.0

    for i in range(1, 11):
        # Escalate token counts each iteration to accelerate spending
        input_tokens = 1000 + (i * 200)
        output_tokens = 500 + (i * 100)

        # Estimate cost (using claude-sonnet-4-20250514 rates)
        est_cost = (input_tokens * 3e-6) + (output_tokens * 15e-6)
        cumulative_cost += est_cost

        print(f"  Call {i:2d}: {input_tokens:5d} in / {output_tokens:4d} out "
              f"  ~${est_cost:.4f}  (cumulative ~${cumulative_cost:.4f})")

        record_llm_call(
            model="claude-sonnet-4-20250514",
            provider="anthropic",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

        # Occasional tool call to make the session look realistic
        if i % 3 == 0:
            record_tool_call(
                "web_search",
                tool_input={"query": f"research topic {i}"},
                tool_output={"results": [f"result_{i}_a", f"result_{i}_b"]},
            )

        time.sleep(0.1)  # Small gap so spans get distinct timestamps

    print(f"\n  Estimated total spend: ${cumulative_cost:.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from examples.alerts_and_drift._shared import ensure_demo_agent_config
    ensure_demo_agent_config(
        "budget-demo",
        {"budget": {"daily_usd": 0.05, "session_usd": 0.02}},
    )

    from tj.sdk.bootstrap import ensure_initialised
    ensure_initialised()

    print("=" * 60)
    print("OCW Budget Breach Alert Demo")
    print("=" * 60)
    print(
        "\nMaking 10 LLM calls with escalating token counts.\n"
        "Session budget ($0.02) should breach after ~2 calls.\n"
        "Daily budget  ($0.05) should breach after ~5 calls.\n"
    )

    run_expensive_agent()

    print("\n" + "=" * 60)
    print("What to observe:")
    print("=" * 60)
    print(
        "If your ocw.toml has the budget config shown at the top of\n"
        "this file, ocw should have fired budget alerts.\n"
        "\n"
        "Run these commands to inspect:\n"
        "\n"
        "  ocw cost --since 1h         # cost breakdown for the last hour\n"
        "  ocw alerts                  # see budget-breach alerts\n"
        "  ocw status                  # agent overview with cost totals\n"
    )
