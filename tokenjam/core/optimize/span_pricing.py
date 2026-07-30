"""Time-correct pricing for the optimize analyzers: one pricer, one convention.

Every analyzer figure that puts a dollar sign on past traffic resolves its rate
through this module. Two things live here and nowhere else.


THE TIME AXIS
-------------
``pricing/models.toml`` carries a rate HISTORY: a model can have several rates,
each valid from a date. :func:`tokenjam.core.pricing.get_rates` therefore takes
an ``at=`` instant, and **defaults to now** when it is omitted.

That default is right for a live call and wrong for every analyzer, all of which
look backwards. An analyzer that omits ``at=`` prices a 30-day window entirely at
today's list price, so day-1 traffic is repriced at the day-30 rate. The damage
is not a rounding error, it is a fabricated trend: a price CUT reprices last
month's spend downward and surfaces as a spend reduction that never happened,
attributed to whatever the user changed in between.

The functions here make the instant a REQUIRED keyword argument. There is no
call shape that silently prices the past at today's rate.


THE AGGREGATION CONVENTION
--------------------------
Stated once, here, because it is the kind of decision each analyzer would
otherwise settle differently and none would write down:

    **Price each span at its OWN timestamp, then sum the dollars.
    Never price an aggregate at a single date.**

Concretely: a per-cluster / per-session / per-agent rollup sums money, never
tokens-then-rate. Summing tokens first and applying one rate forces a choice of
which instant to price the bucket at — window start, window end, "now" — and
every choice is wrong for most of the spans in it. Summing money asks no such
question: each span was billed at one rate, that rate is knowable, and addition
is exact.

The consequence to expect, and to accept: for a window straddling a rate change
there is no single "the rate" the totals correspond to. A test or a UI string
that needs a rate for such a window must take the whole BAND the window's rates
span, which is what :func:`rates_in_window` is for.

A caller that genuinely holds no per-span timestamp — an already-aggregated SQL
row — must widen its query to carry one rather than reach for a stand-in date.
Usually that is one line: group by ``SPAN_UTC_DAY_SQL`` as well and select
``MIN(start_time)`` per bucket.

**Never invent an instant.** No window midpoint, no ``until``, no
"representative" date. An approximated ``at=`` is worse than an honest "now",
because it reads as principled while being arbitrary. Where a real per-span (or
per-bucket) instant genuinely cannot be had, price at NOW and say so at the call
site — a stated limitation someone can find and fix beats a plausible-looking
number nobody questions.

There is exactly one such site, and it is labelled: ``cache_efficacy
.estimate_cache_recoverable``, whose row is a whole-window aggregate that is
also the UI's display row, so splitting it per rate era is a change with its own
blast radius rather than a pricing fix. Everything else here takes a real
instant.


AGGREGATES THAT MULTIPLY A TOTAL BY A RATE
------------------------------------------
Several analyzers do not price a span at a time. They compute a group total and
multiply it by one rate component — ``over_tokens x output_per_mtok``,
``tax_tokens x input_per_mtok``, ``shifted x (input - cache_read)``. Rewriting
each into a per-span loop would churn a lot of tested arithmetic to reach a
number that is, for a linear formula, identical:

    sum_i(tokens_i x rate_i) == (sum_i tokens_i) x weighted_avg(rate_i)

when the average is weighted by those same ``tokens_i``. So :func:`blended_rates`
gives the group ONE ``ModelRates`` whose every component is that volume-weighted
average over the group's own timestamps. It drops into the existing ``rates.X``
expressions unchanged and satisfies the convention exactly, because it IS
price-per-span-then-sum, factored.

Where the formula is not linear — a ``max(0.0, rate_a - rate_b)`` clamp that
could fire for some eras and not others — clamping the blend is not identical to
blending the clamps. Every such clamp in the analyzers guards an inversion (a
cache read pricier than fresh input) that no real model in the table exhibits,
so the two agree; the note is here so the next reader does not have to re-derive
whether it matters.
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import fields
from datetime import datetime

from tokenjam.core.cost import calculate_cost
from tokenjam.core.pricing import (
    STANDARD_VARIANT,
    ModelRates,
    get_rates,
    get_rates_in_range,
)

#: Provider string used when a span's provider column is null. Matches what the
#: analyzers passed before this module existed, so lookups behave identically —
#: `get_rates` still finds the model through a model-keyed user override.
UNKNOWN_PROVIDER = "unknown"

#: The numeric rate components of :class:`ModelRates`, i.e. the fields a blend
#: averages. Derived from the dataclass rather than listed, so a new rate class
#: is blended the day it is added instead of being silently dropped.
_RATE_FIELDS: tuple[str, ...] = tuple(
    f.name for f in fields(ModelRates) if f.type is not bool and f.name.endswith("_per_mtok")
)


#: SQL expression bucketing a span into its UTC calendar day, for an aggregate
#: query that must stay priceable. Every ``valid_from`` in the pricing table is a
#: DATE, so a rate can only change at UTC midnight — which makes a UTC-day bucket
#: the coarsest grouping guaranteed never to straddle a rate change. Group by
#: this alongside (provider, model) and select ``MIN(start_time)`` as the
#: bucket's instant, and a rolled-up query satisfies the per-span convention
#: without being rewritten span by span. Pinned by
#: `test_rate_boundaries_are_utc_midnight`, which is what makes the guarantee
#: real rather than a comment.
#:
#: The explicit ``AT TIME ZONE 'UTC'`` is load-bearing: casting a TIMESTAMPTZ to
#: DATE uses the session timezone, so on a non-UTC host the "day" would be a
#: local day, which straddles UTC midnight by the offset.
SPAN_UTC_DAY_SQL = "CAST(start_time AT TIME ZONE 'UTC' AS DATE)"


def span_instant(when: datetime | None, *, window_start: datetime) -> datetime:
    """The instant to price a span at, given its recorded time may be absent.

    Real spans always carry a start time — every analyzer loader selects it and
    filters the window on it, so ``None`` is unreachable through the live query
    paths. It stays typed as optional for the serialized round-trip that omits
    it, so the pricing call sites need one agreed answer rather than five.

    The answer is the WINDOW START, never "now". Both are guesses, but the
    window start is a guess inside the range the span provably falls in, while
    "now" is the one instant it provably does not — and "now" is exactly the
    default that made every analyzer reprice its whole window at today's list
    price. Dropping the span instead would be worse again: it would quietly
    shrink an observed-cost figure rather than approximate it.
    """
    return when if when is not None else window_start


def rates_at(
    provider: str | None,
    model: str,
    at: datetime,
    *,
    variant: str = STANDARD_VARIANT,
) -> ModelRates | None:
    """The rate that actually billed a span recorded at ``at``, or ``None``.

    Use this when an analyzer needs a rate COMPONENT rather than a total — a
    cache-read rate on its own, or the input-rate gap between two models. Those
    cannot go through :func:`price_span`, which prices a whole token mix; this
    is the sanctioned way to reach one, and it takes the same required instant
    so it cannot drift from the totals around it.

    ``at`` is positional and required by design: see the module docstring.
    """
    return get_rates(provider or UNKNOWN_PROVIDER, model, at=at, variant=variant)


def price_span(
    provider: str | None,
    model: str,
    *,
    at: datetime,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cache_read_tokens: int = 0,
    cache_write_tokens: int = 0,
    variant: str = STANDARD_VARIANT,
) -> float | None:
    """Cost of one span's token mix at the rate in effect when it ran.

    ``None`` when the model has no rate at that instant — the caller then
    contributes nothing rather than inventing a number from the default flat
    rate, which is the established analyzer contract (a figure we cannot price
    is a figure we do not claim).

    Routes through :func:`tokenjam.core.cost.calculate_cost`, the ONE place that
    prices all four token classes. Hand-rolling the arithmetic a second time is
    exactly how ``model_downgrade._alt_unit_cost`` silently dropped cache-write
    from its alternative side while the observed side included it.
    """
    resolved = rates_at(provider, model, at, variant=variant)
    if resolved is None:
        return None
    return calculate_cost(
        provider or UNKNOWN_PROVIDER,
        model,
        input_tokens,
        output_tokens,
        cache_read_tokens=cache_read_tokens,
        cache_write_tokens=cache_write_tokens,
        at=at,
        variant=variant,
    )


def blended_rates(
    provider: str | None,
    model: str,
    dated_volumes: Iterable[tuple[datetime | None, float]],
    *,
    variant: str = STANDARD_VARIANT,
) -> ModelRates | None:
    """The one rate to use for a GROUP of spans priced as ``total x rate``.

    ``dated_volumes`` is ``(when the span ran, how much volume it contributed)``
    — the same token count the caller is about to multiply the rate by, so the
    weighting matches the arithmetic. Every component of the returned
    ``ModelRates`` is the volume-weighted average of that component across the
    group's own timestamps; see the module docstring for why that equals
    price-per-span-then-sum for a linear figure.

    When the group does not straddle a rate change — every group, on every
    window that fits between two price moves — this returns that single rate
    exactly, which is why adopting it is a no-op on today's numbers.

    ``None`` when the model has no rate at all, or when the group carries no
    positive volume to weight by (a rate for nothing is not a rate). Entries
    with a missing timestamp or non-positive volume are skipped rather than
    silently priced at "now": a span we cannot place in time must not drag the
    group's rate toward today's.
    """
    weighted: dict[str, float] = {}
    total = 0.0
    cache_read_specified = True
    cache_write_specified = True
    for when, volume in dated_volumes:
        if when is None or volume <= 0:
            continue
        rates = rates_at(provider, model, when, variant=variant)
        if rates is None:
            return None
        total += volume
        for field_name in _RATE_FIELDS:
            weighted[field_name] = (
                weighted.get(field_name, 0.0) + getattr(rates, field_name) * volume
            )
        cache_read_specified = cache_read_specified and rates.cache_read_specified
        cache_write_specified = cache_write_specified and rates.cache_write_specified

    if total <= 0:
        return None
    return ModelRates(
        **{name: weighted[name] / total for name in _RATE_FIELDS},
        # A blend is only as quoted as its least-quoted term: if ANY era's cache
        # rate was a guess, the average is partly guessed, and a consumer
        # deciding whether it may show this as a real rate must hear that.
        cache_read_specified=cache_read_specified,
        cache_write_specified=cache_write_specified,
    )


def rates_in_window(
    provider: str | None,
    model: str,
    since: datetime,
    until: datetime,
    *,
    variant: str = STANDARD_VARIANT,
) -> tuple[ModelRates, ...]:
    """Every rate that could legitimately have billed this model in the window.

    One entry when the price never moved (the common case), more when it did.
    The counterpart to the aggregation convention above: because each span is
    priced at its own timestamp, a window's totals are a blend, and anything
    checking those totals against "a price somebody actually charges" has to
    accept the whole band rather than one instant's rate.

    This is the reason it is exported from the pricer rather than reimplemented
    by each consumer: the Critical Rule 28 band assertions read the table
    through here, so they keep asking exactly the question the analyzers answer.
    Empty when the model is not in the table at all.
    """
    return get_rates_in_range(
        provider or UNKNOWN_PROVIDER, model, since, until, variant=variant,
    )
