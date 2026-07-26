"""Waste shape: repeated stable prefix.

Every call in this workload shares one long, byte-identical system prompt
(a ~5,000-token support policy) with only a short per-ticket question
varying. This is the shape Anthropic-style explicit prompt caching
(`cache_control` breakpoints) exists to fix, and the shape the `cache`
analyzer (`tokenjam/core/optimize/analyzers/cache_efficacy.py`) measures:
current caching efficacy per (provider, model), flagged when the
cache-read share of input volume falls below 30% over >=100,000 input
tokens across >=20 calls.

Sized to clear both gates cheaply: 25 calls x ~5,000 prefix tokens =
~125,000 input tokens on a mini-class model, a few cents total.

Which analyzer actually fires, and why the answer is a real gap rather
than a clean "no caching" story:

  - `cache-recommend` (cache_recommend.py) is Anthropic-only in v1; it
    skips every non-Anthropic span outright, so it never fires here no
    matter how stable the prefix is or how many times it repeats.
  - `cache` (cache_efficacy.py) is "best-effort" for OpenAI per its own
    docstring, but tokenjam's OpenAI integration
    (tokenjam/sdk/integrations/openai.py) never reads
    `response.usage.prompt_tokens_details.cached_tokens`; it only sets
    INPUT_TOKENS/OUTPUT_TOKENS. So cache_tokens is always 0/unset for
    every OpenAI span tokenjam has ever captured, and this analyzer's
    "efficacy" reads as 0% regardless of whether OpenAI's own automatic
    prompt caching was actually hitting. The card will fire, but it is
    measuring an instrumentation gap, not a real caching problem; worth
    knowing before trusting this analyzer's OpenAI numbers.
"""
from __future__ import annotations

from tokenjam.sdk.agent import watch
from tokenjam.sdk.attribution import attribution

from _shared import guarded_chat_completion, run_workload_main, set_default_environment

set_default_environment("sdk-workload-demo")

AGENT_ID = "sdk-workload-repeated-prefix"
TENANT_ID = "acme-support"
FEATURE = "ticket-triage"
PROMPT_TEMPLATE_ID = "triage-system-prompt"
PROMPT_TEMPLATE_VERSION = "v1"

NUM_CALLS = 25

_POLICY_PARAGRAPH = (
    "You are a support-ticket triage assistant for Acme Corp. Classify each "
    "incoming ticket into exactly one category: billing, technical, account, "
    "or other. Always respond with the category name only, lowercase, no "
    "punctuation, no explanation. Never invent a category outside the four "
    "listed. If a ticket mentions a refund, chargeback, invoice, or payment "
    "method, it is billing. If it mentions an error message, a crash, a "
    "broken feature, or a bug, it is technical. If it mentions login, "
    "password, two-factor auth, or profile settings, it is account. "
)
# ~500 chars per paragraph x 40 repeats ~= 20,000 chars ~= 5,000 tokens
# (CHARS_PER_TOKEN=4 heuristic, matching the rest of tokenjam's codebase).
SYSTEM_PREFIX = _POLICY_PARAGRAPH * 40

_SAMPLE_TICKETS = [
    "My card was charged twice for the same invoice.",
    "The app crashes every time I open the dashboard.",
    "I can't log in, it says my password is wrong.",
    "Where do I update my billing address?",
    "The export button throws a 500 error.",
]


@watch(agent_id=AGENT_ID, tenant_id=TENANT_ID, feature=FEATURE)
def run(client, guard, dry_run: bool, model: str) -> None:
    with attribution(
        prompt_template_id=PROMPT_TEMPLATE_ID,
        prompt_template_version=PROMPT_TEMPLATE_VERSION,
    ):
        for i in range(NUM_CALLS):
            ticket = _SAMPLE_TICKETS[i % len(_SAMPLE_TICKETS)]
            messages = [
                {"role": "system", "content": SYSTEM_PREFIX},
                {"role": "user", "content": f"Ticket #{i + 1}: {ticket}"},
            ]
            response = guarded_chat_completion(
                client,
                guard,
                dry_run=dry_run,
                model=model,
                messages=messages,
                max_tokens=8,
            )
            category = response.choices[0].message.content
            print(f"  ticket {i + 1}/{NUM_CALLS}: {ticket!r} -> {category!r}")


def _main(client, guard, dry_run, model, _args) -> None:
    run(client, guard, dry_run, model)


if __name__ == "__main__":
    run_workload_main(
        "Repeated stable prefix workload; targets the `cache` analyzer.",
        _main,
    )
