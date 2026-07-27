"""Critical Rule 28 price-band helpers, shared by every analyzer's rate tests.

Rule 28 asks that a ``past_overspend_usd / past_overspend_tokens`` pair divide
out to a $/MTok somebody actually charges. The assertion is on a RATE, never a
hardcoded dollar figure: a hardcoded number passes happily while both fields
drift together, whereas a basis mismatch throws the implied rate orders of
magnitude out of any real price band, so only the rate can catch it.

WHY THIS IS A SHARED MODULE, AND NOT A HELPER PER TEST FILE
-----------------------------------------------------------
The band has to be DERIVED from the pricing table at test time — the table
carries a rate-history axis, so a literal would rot into a false green the next
time a rate row lands. But deriving it is not enough on its own: the derivation
has to ask the table the SAME question the analyzer answered.

That is the trap these helpers exist to close. The analyzers used to price with
a bare ``get_rates(provider, model)`` — no ``at=``, i.e. today's rate — and the
band assertions read the table the same way, which is the only reason they
agreed. They did not agree because both were right; they agreed because both
were wrong in the same direction. The analyzers now price each span at its own
timestamp (see :mod:`tokenjam.core.optimize.span_pricing`), so these read the
table through that same module, at the window the test actually seeded.

A concrete failure this prevents: ``claude-sonnet-5`` carries an introductory
rate through 2026-08-31 and a higher standard rate from 2026-09-01. A test that
seeds spans in the past and asserts against the bare "rate now" passes today and
starts failing on 2026-09-01 for no reason connected to the code under test.
:func:`rate_for_window` refuses to hand back a single rate for a window that
straddles a change, so that class of bug surfaces as a clear message instead of
a mystery red.
"""
from __future__ import annotations

from datetime import datetime

from tokenjam.core.optimize.span_pricing import rates_in_window
from tokenjam.core.pricing import ModelRates


def implied_rate(usd: float, tokens: int) -> float:
    """The $/MTok a (dollars, tokens) aggregate pair divides out to."""
    return usd / tokens * 1_000_000


def rate_for_window(
    provider: str, model: str, since: datetime, until: datetime,
) -> ModelRates:
    """THE rate for this model over the test's window, for a tight assertion.

    Most tests want one exact rate to check a term against. This gives it, and
    refuses when the window straddles a rate change — at which point there is no
    single rate the analyzer's output corresponds to, and the test must assert
    against :func:`price_band` instead of pretending otherwise.
    """
    rates = rates_in_window(provider, model, since, until)
    assert rates, f"{provider}/{model} has no rate in the pricing table"
    assert len(rates) == 1, (
        f"{provider}/{model} changes rate inside {since:%Y-%m-%d}..{until:%Y-%m-%d} "
        f"({len(rates)} rates apply). The analyzer prices each span at its own "
        "timestamp, so its total is a blend and no single rate equals it — "
        "assert against price_band(), or move the test window inside one rate era."
    )
    return rates[0]


def price_band(
    models: list[tuple[str, str]], since: datetime, until: datetime,
) -> tuple[float, float]:
    """(floor, ceiling) of any per-token rate a finding over this window could
    legitimately imply: no term may be cheaper than the cheapest cache read
    available to it, and none may exceed the priciest fresh input token.

    Widened across every rate in effect anywhere in the window, not just the one
    at some chosen instant — because a window straddling a price change produces
    totals blended across both, and a band drawn at one instant would reject the
    correct answer.
    """
    priced = [
        r
        for provider, model in models
        for r in rates_in_window(provider, model, since, until)
    ]
    assert priced, "corpus has no priced model — the band would be vacuous"
    return (
        min(r.cache_read_per_mtok for r in priced),
        max(r.input_per_mtok for r in priced),
    )
