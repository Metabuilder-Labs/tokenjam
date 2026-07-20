"""
Record an Outcome Demo

Runs a small "support agent" session (a couple of LLM + tool calls) and then
attaches a business OUTCOME to it with `record_outcome()` — one line instead of
hand-POSTing an OTLP event.

No API keys required — uses simulated instrumentation.

What record_outcome does: it emits the emerging gen_ai outcome event (OTel
semconv issue #2665) as a span alongside your telemetry. TokenJam Cloud's ROI
backend turns declared value ÷ measured cost into an ROI figure. The OSS SDK
here only EMITS the event — there is no local ROI compute.

Honesty note: `value_usd` is OPTIONAL and SELF-REPORTED. It is a value YOU
declare for the outcome; TokenJam does not measure or verify it.
"""
from __future__ import annotations

from tokenjam.sdk import record_outcome
from tokenjam.sdk.agent import AgentSession, record_llm_call, record_tool_call
from tokenjam.utils.ids import new_uuid


def run_support_session() -> None:
    """A tiny support workflow that resolves a ticket, then records the outcome."""
    with AgentSession(agent_id="support-agent", conversation_id=new_uuid()):
        record_llm_call(
            model="claude-haiku-4-5",
            provider="anthropic",
            input_tokens=350,
            output_tokens=120,
        )
        record_tool_call(
            tool_name="lookup_order",
            tool_input={"order_id": "A-1001"},
            tool_output={"status": "shipped"},
        )
        record_llm_call(
            model="claude-haiku-4-5",
            provider="anthropic",
            input_tokens=180,
            output_tokens=90,
        )

        # Attach the business outcome. Inside an active AgentSession the session
        # is inherited automatically — no need to pass workflow_id/session_id.
        record_outcome(
            "ticket_resolved",
            success=True,
            value_usd=25.00,  # self-reported: what resolving this ticket is worth
        )


if __name__ == "__main__":
    run_support_session()
    print("Recorded a 'ticket_resolved' outcome for the support-agent session.")
    print("Inspect the spans with:  tj traces --since 1h")
    print("ROI compute is a TokenJam Cloud feature — the SDK only emits the event.")
