"""Waste shape: growing context / re-send.

Simulates the classic stateless chat-completions loop: the client has no
server-side conversation state, so every turn re-sends the ENTIRE
accumulated message history (system prompt + every prior user/assistant
turn) just to add one new turn. Prompt size grows monotonically within a
session and most of what's billed on turn N was already billed on turns
1..N-1.

Targets the `resend` analyzer (`tokenjam/core/optimize/analyzers/
context_resend.py`): the token-weighted repeat share across sessions,
gated on `MIN_SESSIONS_FOR_SIGNAL = 3` sessions and `MIN_TURNS_FOR_SIGNAL
= 6` total LLM turns in the window before the aggregate means anything.
This workload runs 3 sessions x 4 turns each (12 turns total) to clear
both gates on a single cheap invocation.

Per-turn additions are small and fixed (not random), so the token growth
curve is identical every run; required for a later before/after replay
(e.g. re-run after switching this loop to send only a rolling summary
instead of full history, and diff actual token usage).
"""
from __future__ import annotations

from tokenjam.sdk.agent import watch
from tokenjam.sdk.attribution import attribution

from _shared import guarded_chat_completion, run_workload_main, set_default_environment

set_default_environment("sdk-workload-demo")

AGENT_ID = "sdk-workload-growing-context"
TENANT_ID = "globex-chat"
FEATURE = "customer-chat"
PROMPT_TEMPLATE_ID = "chat-loop-system-prompt"
PROMPT_TEMPLATE_VERSION = "v2"

NUM_SESSIONS = 3
TURNS_PER_SESSION = 4

SYSTEM_PROMPT = (
    "You are Globex's customer chat assistant. Answer in one short sentence. "
    "Stay consistent with everything said earlier in this conversation."
)

# Fixed, deterministic per-turn user messages (no randomness) so token growth
# is identical across runs; required for the before/after replay use case.
_TURNS = [
    "Hi, I'm looking at your Pro plan. What does it include?",
    "Does that include priority support?",
    "And what's the annual discount if I pay upfront?",
    "Great, can you summarize everything you just told me?",
]


def _run_session(client, guard, dry_run: bool, model: str, session_index: int) -> None:
    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn_index in range(TURNS_PER_SESSION):
        messages.append({"role": "user", "content": _TURNS[turn_index % len(_TURNS)]})
        # The entire accumulated `messages` list is re-sent every turn; this
        # IS the waste shape; a stateless chat-completions API structurally
        # requires it unless the caller does its own compaction/summarization.
        response = guarded_chat_completion(
            client,
            guard,
            dry_run=dry_run,
            model=model,
            messages=messages,
            max_tokens=60,
        )
        reply = response.choices[0].message.content
        messages.append({"role": "assistant", "content": reply})
        print(
            f"  session {session_index + 1}/{NUM_SESSIONS} turn "
            f"{turn_index + 1}/{TURNS_PER_SESSION} "
            f"(prompt now carries {len(messages)} messages)"
        )


@watch(agent_id=AGENT_ID, tenant_id=TENANT_ID, feature=FEATURE)
def run_one_session(client, guard, dry_run: bool, model: str, session_index: int) -> None:
    with attribution(
        prompt_template_id=PROMPT_TEMPLATE_ID,
        prompt_template_version=PROMPT_TEMPLATE_VERSION,
    ):
        _run_session(client, guard, dry_run, model, session_index)


def run(client, guard, dry_run: bool, model: str) -> None:
    for session_index in range(NUM_SESSIONS):
        run_one_session(client, guard, dry_run, model, session_index)


def _main(client, guard, dry_run, model, _args) -> None:
    run(client, guard, dry_run, model)


if __name__ == "__main__":
    run_workload_main(
        "Growing-context re-send workload; targets the `resend` analyzer.",
        _main,
    )
