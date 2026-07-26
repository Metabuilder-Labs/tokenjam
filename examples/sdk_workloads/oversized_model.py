"""Waste shape: oversized model for a trivial subtask.

Uses a premium-tier model (`gpt-4o` by default) for tasks that need none
of its extra capability; one-word spellchecks and yes/no classifications.
Targets the `downsize` analyzer's SECONDARY "tiny-session" case
(`tokenjam/core/optimize/analyzers/model_downgrade.py`): a session with
input_tokens < 5,000, output_tokens < 500, and tool_calls <= 5 on a model
that has a known cheaper same-family alternative
(`DOWNGRADE_CANDIDATES["openai"]["gpt-4o"] == "gpt-4o-mini"`).

A SINGLE qualifying session is enough; the analyzer emits a finding as
soon as `candidate_sessions >= 1` (see model_downgrade.py: it returns None
only when there are zero candidates AND no driver-role sessions). This
workload runs three trivial calls in one session for a slightly more
convincing corpus, but even one would clear the gate.

`--model` here defaults to the intentionally-oversized `gpt-4o` (the whole
point of this workload); override it to see the finding disappear once the
"right-sized" model is already in use.
"""
from __future__ import annotations

from tokenjam.sdk.agent import watch
from tokenjam.sdk.attribution import attribution

from _shared import guarded_chat_completion, run_workload_main, set_default_environment

set_default_environment("sdk-workload-demo")

AGENT_ID = "sdk-workload-oversized-model"
TENANT_ID = "stark-faq"
FEATURE = "spellcheck"
PROMPT_TEMPLATE_ID = "trivial-qa"
PROMPT_TEMPLATE_VERSION = "v1"

TRIVIAL_QUESTIONS = [
    "Is the word 'cat' spelled correctly? Answer yes or no.",
    "Is 'recieve' spelled correctly? Answer yes or no.",
    "Is the following a question: 'What time is it?' Answer yes or no.",
]


@watch(agent_id=AGENT_ID, tenant_id=TENANT_ID, feature=FEATURE)
def run(client, guard, dry_run: bool, model: str) -> None:
    with attribution(
        prompt_template_id=PROMPT_TEMPLATE_ID,
        prompt_template_version=PROMPT_TEMPLATE_VERSION,
    ):
        for i, question in enumerate(TRIVIAL_QUESTIONS):
            response = guarded_chat_completion(
                client,
                guard,
                dry_run=dry_run,
                model=model,
                messages=[
                    {"role": "system", "content": "Answer in one word: yes or no."},
                    {"role": "user", "content": question},
                ],
                max_tokens=4,
            )
            answer = response.choices[0].message.content
            print(f"  Q{i + 1}: {question!r} -> {answer!r}")


def _main(client, guard, dry_run, model, _args) -> None:
    run(client, guard, dry_run, model)


if __name__ == "__main__":
    run_workload_main(
        "Oversized-model workload; targets the `downsize` analyzer.",
        _main,
        default_model="gpt-4o",
    )
