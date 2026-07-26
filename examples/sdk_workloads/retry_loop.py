"""Waste shape: retry loop.

A tool call fails validation, and the agent retries with the EXACT same
arguments instead of adjusting; a common real bug shape (a coupon code
that will never validate, a malformed ID that will never resolve, a stuck
policy) that burns tool-call volume (and often a follow-up LLM call per
retry) without making progress.

This is NOT one of the `tj optimize` cost analyzers; it is caught by
`AlertEngine._check_retry_loop` (`tokenjam/core/alerts.py`): fires when the
SAME tool name with the SAME argument signature appears 4+ times in the
last 6 `gen_ai.tool.call` spans of a session. The signature comes from
`TjAttributes.TOOL_ARG_SIG`, computed at ingest from the tool's input, so
this workload deliberately passes byte-identical arguments on every retry
(a real "stuck" agent, not one exploring different fixes).

Kept cheap: one real LLM call frames the task, five identical local tool
calls do the actual "retrying" (no LLM round-trip needed to reproduce the
bug shape), one final LLM call frames the give-up/escalate step.
"""
from __future__ import annotations

from tokenjam.sdk.agent import record_tool_call, watch
from tokenjam.sdk.attribution import attribution

from _shared import guarded_chat_completion, run_workload_main, set_default_environment

set_default_environment("sdk-workload-demo")

AGENT_ID = "sdk-workload-retry-loop"
TENANT_ID = "initech-ops"
FEATURE = "order-submission"
PROMPT_TEMPLATE_ID = "order-retry-agent"
PROMPT_TEMPLATE_VERSION = "v1"

NUM_RETRIES = 5

# Fixed, deterministic tool arguments; the whole point is that these never
# change between attempts, which is what makes it a genuine retry loop
# rather than normal iterative problem-solving.
STUCK_ORDER_ARGS = {"order_id": "ORD-1042", "coupon": "SAVE20-EXPIRED"}

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "submit_order",
            "description": "Submit an order with a coupon code applied.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string"},
                    "coupon": {"type": "string"},
                },
                "required": ["order_id", "coupon"],
            },
        },
    },
]


def submit_order(order_id: str, coupon: str) -> dict:
    """Deterministic stub: this coupon is always expired. Never succeeds;
    models the real-world bug where the caller never adjusts its input."""
    return {
        "order_id": order_id,
        "status": "rejected",
        "error": "coupon_expired",
        "message": f"Coupon {coupon!r} expired before order {order_id} was submitted.",
    }


@watch(agent_id=AGENT_ID, tenant_id=TENANT_ID, feature=FEATURE)
def run(client, guard, dry_run: bool, model: str) -> None:
    with attribution(
        prompt_template_id=PROMPT_TEMPLATE_ID,
        prompt_template_version=PROMPT_TEMPLATE_VERSION,
    ):
        guarded_chat_completion(
            client,
            guard,
            dry_run=dry_run,
            model=model,
            messages=[
                {"role": "system", "content": "You are an order-submission agent."},
                {
                    "role": "user",
                    "content": (
                        f"Submit order {STUCK_ORDER_ARGS['order_id']} with coupon "
                        f"{STUCK_ORDER_ARGS['coupon']!r}."
                    ),
                },
            ],
            tools=TOOLS,
            force_tool_call={"name": "submit_order", "arguments": STUCK_ORDER_ARGS},
            max_tokens=30,
        )

        for attempt in range(1, NUM_RETRIES + 1):
            result = submit_order(**STUCK_ORDER_ARGS)
            record_tool_call(
                tool_name="submit_order",
                tool_input=STUCK_ORDER_ARGS,
                tool_output=result,
                error=result["error"],
            )
            print(f"  attempt {attempt}/{NUM_RETRIES}: {result['message']}")

        guarded_chat_completion(
            client,
            guard,
            dry_run=dry_run,
            model=model,
            messages=[
                {"role": "system", "content": "You are an order-submission agent."},
                {
                    "role": "user",
                    "content": (
                        f"Order {STUCK_ORDER_ARGS['order_id']} failed "
                        f"{NUM_RETRIES} times with the same coupon error. "
                        "Escalate to a human in one short sentence."
                    ),
                },
            ],
            max_tokens=30,
        )


def _main(client, guard, dry_run, model, _args) -> None:
    run(client, guard, dry_run, model)


if __name__ == "__main__":
    run_workload_main(
        "Retry-loop workload; targets the AlertEngine RETRY_LOOP alert.",
        _main,
    )
