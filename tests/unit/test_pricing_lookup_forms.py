"""Every model-name form a real agent run emits must resolve to a real rate.

A benchmark replay of nine real agent runs (21,562 LLM calls) found the cost
figures wrong by 5-30x for seven of the nine model strings its per-model cost
table breaks out. Nothing was wrong with the arithmetic — the lookup simply
never found the row. Two structural gaps produced all seven:

  * the date-suffix stripper only understood Anthropic's compact `-YYYYMMDD`
    form, so OpenAI's dashed `-YYYY-MM-DD` (`gpt-4o-2024-08-06`) never fell
    back to its bare table entry even though that entry existed; and
  * nothing stripped a routing prefix, so `anthropic/claude-opus-4.1` (the form
    LiteLLM and OpenRouter emit) never reached the table at all.

Both failures are silent in the UI — one log warning per process, then a
plausible-looking dollar figure computed at the flat default rate. This module
pins each lookup form independently, and pins the real model strings against
the rates they must resolve to. `BENCHMARK_CORPUS_MODELS` states below exactly
which of those strings the replay measured and which it did not — the list is
longer than nine and the difference is provenance, not a bigger corpus.
"""

from __future__ import annotations

import pytest

from tokenjam.core.pricing import (
    DEFAULT_INPUT_PER_MTOK,
    classify_pricing_source,
    get_rates,
)


# ── Individual lookup forms ────────────────────────────────────────────────
# One case per structural transform, so a regression names the transform that
# broke rather than just "some model got cheaper".

@pytest.mark.parametrize(
    "provider,model,expected_input,expected_kind",
    [
        # Exact — the form that always worked.
        ("anthropic", "claude-opus-4-1", 15.00, "exact"),
        # Compact date suffix (Anthropic).
        ("anthropic", "claude-opus-4-20250514", 15.00, "date_stripped"),
        # Dashed date suffix (OpenAI) — the gap.
        ("openai", "gpt-4o-2024-08-06", 2.50, "date_stripped"),
        ("openai", "gpt-4o-2024-11-20", 2.50, "date_stripped"),
        ("openai", "gpt-4.1-2025-04-14", 2.00, "date_stripped"),
        # Provider routing prefix — the other gap.
        ("anthropic", "anthropic/claude-opus-4-1", 15.00, "provider_prefix"),
        # Nested routing prefix, as OpenRouter emits it.
        ("anthropic", "openrouter/anthropic/claude-opus-4-1", 15.00, "provider_prefix"),
        # Dotted version segments (`claude-opus-4.1` vs the table's `-4-1`).
        ("anthropic", "claude-opus-4.1", 15.00, "version_dots"),
        # Prefix AND dots together — the real HAL corpus form.
        ("anthropic", "anthropic/claude-opus-4.1", 15.00, "provider_prefix"),
        ("anthropic", "anthropic/claude-3.7-sonnet", 3.00, "provider_prefix"),
        # Context tag, still working alongside the new transforms.
        ("anthropic", "claude-opus-4-1[1m]", 15.00, "context_tag"),
    ],
)
def test_each_lookup_form_resolves(provider, model, expected_input, expected_kind):
    rates = get_rates(provider, model)
    assert rates is not None, f"{provider}/{model} did not resolve"
    assert rates.input_per_mtok == expected_input
    assert classify_pricing_source(provider, model) == expected_kind


def test_an_unknown_model_still_reports_the_default_fallback():
    """The fallback must stay reachable — a wider lookup that resolves
    everything to *something* would hide the signal instead of fixing it."""
    assert get_rates("anthropic", "claude-not-a-real-model") is None
    assert classify_pricing_source("anthropic", "claude-not-a-real-model") == (
        "default_fallback"
    )


def test_a_routing_prefix_does_not_invent_a_match_for_an_unknown_model():
    assert get_rates("openai", "someproxy/gpt-not-real") is None


# ── The benchmark corpus ───────────────────────────────────────────────────
# The `gen_ai.request.model` strings this module pins, each to the input rate
# it must price at. If any regresses to the default rate the cost dashboard is
# wrong again by the same 5-30x.
#
# Provenance, stated exactly because a list that misdescribes its own corpus is
# the defect class this codebase treats as first-order:
#
#   * The first NINE entries are the nine model strings the replay's per-model
#     cost table breaks out — the nine whose rates were checked against HAL's
#     own ground truth, and the population "seven of nine were mispriced"
#     counts. That is where the module docstring's "nine" comes from.
#   * `anthropic/claude-sonnet-4-5-20250929` is a real tenth corpus string,
#     carried by a handful of calls in the corebench_hard run (`summary.usage`
#     keys the same underlying model both bare and provider-prefixed within one
#     run). It is below the table's reporting threshold, not absent from the
#     corpus, and tj must still price it.
#   * The two `claude-haiku-4-5` strings come from the SEEACT run, which was
#     probed for cache usage but EXCLUDED from the nine-run corpus (its export
#     has no per-call data left to inspect). They are pinned because they are
#     real strings tj has to price, not because the replay measured them.

# The nine model strings the replay's per-model cost table breaks out.
_TABULATED_CORPUS_MODELS = [
    ("anthropic", "anthropic/claude-opus-4.1", 15.00),
    ("anthropic", "claude-opus-4-20250514", 15.00),
    ("anthropic", "claude-3-7-sonnet-20250219", 3.00),
    ("anthropic", "anthropic/claude-3.7-sonnet", 3.00),
    ("anthropic", "claude-sonnet-4-5-20250929", 3.00),
    ("openai", "gpt-4.1-2025-04-14", 2.00),
    ("openai", "gpt-4o-2024-08-06", 2.50),
    ("openai", "gpt-4o-2024-11-20", 2.50),
    ("google", "gemini-2.0-flash", 0.10),
]

# Below the table's reporting threshold, but present in the corpus.
_LOW_VOLUME_CORPUS_MODELS = [
    ("anthropic", "anthropic/claude-sonnet-4-5-20250929", 3.00),
]

# From the tenth run, probed but excluded from the measured corpus.
_EXCLUDED_RUN_MODELS = [
    ("anthropic", "claude-haiku-4-5-20251001", 1.00),
    ("anthropic", "anthropic/claude-haiku-4.5", 1.00),
]

BENCHMARK_CORPUS_MODELS = [
    *_TABULATED_CORPUS_MODELS,
    *_LOW_VOLUME_CORPUS_MODELS,
    *_EXCLUDED_RUN_MODELS,
]


def test_the_tabulated_corpus_is_the_nine_the_replay_measured():
    """Pins the count the module docstring narrates, so the two cannot drift:
    the replay's per-model table has nine rows, and any entry added beyond it
    has to go in one of the explicitly-labelled extra lists instead of
    silently inflating the population the "seven of nine" claim rests on."""
    assert len(_TABULATED_CORPUS_MODELS) == 9


@pytest.mark.parametrize("provider,model,expected_input", BENCHMARK_CORPUS_MODELS)
def test_benchmark_corpus_model_prices_at_its_real_rate(provider, model, expected_input):
    rates = get_rates(provider, model)
    assert rates is not None, (
        f"{provider}/{model} fell through to the default rate — the cost "
        f"dashboard would be silently wrong for every call using it"
    )
    assert rates.input_per_mtok == expected_input


@pytest.mark.parametrize("provider,model,_expected", BENCHMARK_CORPUS_MODELS)
def test_no_benchmark_corpus_model_uses_the_default_fallback(provider, model, _expected):
    source = classify_pricing_source(provider, model)
    assert source != "default_fallback"
    rates = get_rates(provider, model)
    assert rates is not None
    assert rates.input_per_mtok != DEFAULT_INPUT_PER_MTOK or source != "default_fallback"


# ── Backfilled table entries ───────────────────────────────────────────────
# These pin what `models.toml` says, which is all a test can do — a test
# asserting its own inputs cannot establish that a rate is right. The auditable
# provenance for each of the three lives next to its entry in
# `tokenjam/pricing/models.toml`, including which one is flagged UNVERIFIED
# because its provider retired it off every reachable published page.

@pytest.mark.parametrize(
    "provider,model,input_rate,output_rate",
    [
        ("anthropic", "claude-3-7-sonnet", 3.00, 15.00),
        ("openai", "gpt-4.1", 2.00, 8.00),
        ("google", "gemini-2-0-flash", 0.10, 0.40),
    ],
)
def test_backfilled_models_carry_their_published_rates(
    provider, model, input_rate, output_rate,
):
    rates = get_rates(provider, model)
    assert rates is not None
    assert rates.input_per_mtok == input_rate
    assert rates.output_per_mtok == output_rate
