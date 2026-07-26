"""Waste shape: streaming with an early client disconnect.

This is the one workload that demonstrates a real gap in tokenjam's OWN
SDK rather than a waste shape in the AGENT's behavior; worth building
because it is, per the brief that asked for this corpus, "the failure
mode no competitor surfaces."

`tokenjam/sdk/integrations/openai.py`'s `_StreamWrapper.__iter__` only
sets `INPUT_TOKENS`/`OUTPUT_TOKENS` on the span from the LAST streamed
chunk (the one carrying `usage`, when `stream_options={"include_usage":
True}` is requested). If the consumer stops reading the stream before
that final chunk arrives; a client disconnect, a timeout, a `break` out
of the loop once enough of the answer was shown to a user, a cancelled
UI request. Python raises `GeneratorExit` inside the wrapper's `finally`
block, which ends the span with `status=UNSET` and NO token counts at
all. The call really happened and really cost money; tokenjam silently
records a $0, 0-token span for it. No alert, no analyzer, nothing in this
codebase currently flags a zero-usage LLM span as suspicious; it just
disappears from every cost view.

Real-run path: uses the REAL `patch_openai()`-patched client, issues a
streaming request, and deterministically triggers this exact code path by
calling `iterator.close()` (not just `break`, which relies on CPython's
refcounting timing; an explicit `.close()` reproduces "the consumer
stopped reading" portably) partway through the stream, before the
usage-bearing final chunk.

Dry-run path: cannot exercise `_StreamWrapper` at all (there is no real
`openai` client to patch), so it does not pretend to. It calls
`record_llm_call(..., output_tokens=0)` directly to reproduce the same
SPAN SHAPE the real bug produces; useful for testing the harness/report
path against this corpus without spending money, but it is a simulation
of the shape, not a second exercise of the actual bug. Say so plainly
rather than let a reader assume both paths prove the same thing.
"""
from __future__ import annotations

from tokenjam.sdk.agent import record_llm_call, watch
from tokenjam.sdk.attribution import attribution

from _shared import (
    estimate_messages_tokens,
    run_workload_main,
    set_default_environment,
)

set_default_environment("sdk-workload-demo")

AGENT_ID = "sdk-workload-streaming-disconnect"
TENANT_ID = "wayne-stream"
FEATURE = "live-answer"
PROMPT_TEMPLATE_ID = "stream-answer"
PROMPT_TEMPLATE_VERSION = "v1"

# How many streamed chunks to consume before simulating the disconnect:
# well before OpenAI's final usage-only chunk on any non-trivial answer.
BREAK_AFTER_CHUNKS = 3
MAX_TOKENS = 300

MESSAGES = [
    {"role": "system", "content": "You are a helpful assistant. Answer in 3-4 sentences."},
    {"role": "user", "content": "Explain why streaming responses improve perceived latency."},
]


def _run_real(client, guard, model: str) -> None:
    from tokenjam.core.cost import calculate_cost

    estimated_input = estimate_messages_tokens(MESSAGES)
    estimated_cost = calculate_cost("openai", model, estimated_input, MAX_TOKENS)
    guard.check_before_call(estimated_cost)

    stream = client.chat.completions.create(
        model=model,
        messages=MESSAGES,
        max_tokens=MAX_TOKENS,
        temperature=0,
        seed=42,
        stream=True,
        stream_options={"include_usage": True},
    )
    # Real usage is never learned on this path; that is the bug. Record the
    # worst-case estimate as spent, conservatively, rather than under-count.
    guard.record_actual(estimated_cost)

    iterator = iter(stream)
    chunks_seen = 0
    try:
        for chunk in iterator:
            if chunk.choices and chunk.choices[0].delta.content:
                print(chunk.choices[0].delta.content, end="", flush=True)
            chunks_seen += 1
            if chunks_seen >= BREAK_AFTER_CHUNKS:
                break
    finally:
        # Deterministically simulate "the consumer stopped reading"; the
        # generator's `finally` (in _StreamWrapper.__iter__) runs right now,
        # ending the span with whatever usage it has seen so far (none).
        iterator.close()
    print(f"\n  [disconnected after {chunks_seen} chunk(s), before the usage-bearing final chunk]")


def _run_simulated(model: str) -> None:
    estimated_input = estimate_messages_tokens(MESSAGES)
    print(
        "  [dry-run] simulating the SAME span shape the real disconnect bug "
        "produces (record_llm_call with output_tokens=0); this does not "
        "exercise tokenjam/sdk/integrations/openai.py's _StreamWrapper "
        "itself, see module docstring."
    )
    record_llm_call(
        model=model,
        provider="openai",
        input_tokens=estimated_input,
        output_tokens=0,
    )


@watch(agent_id=AGENT_ID, tenant_id=TENANT_ID, feature=FEATURE)
def run(client, guard, dry_run: bool, model: str) -> None:
    with attribution(
        prompt_template_id=PROMPT_TEMPLATE_ID,
        prompt_template_version=PROMPT_TEMPLATE_VERSION,
    ):
        if dry_run:
            _run_simulated(model)
        else:
            _run_real(client, guard, model)


def _main(client, guard, dry_run, model, _args) -> None:
    run(client, guard, dry_run, model)


if __name__ == "__main__":
    run_workload_main(
        "Streaming early-disconnect workload; demonstrates a lost-usage SDK gap; no analyzer covers it yet.",
        _main,
    )
