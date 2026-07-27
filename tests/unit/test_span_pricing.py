"""The analyzers' pricer: the time axis, and the aggregation convention.

`get_rates` defaults its `at=` to now, which is right for a live call and wrong
for every analyzer, all of which look backwards. These pin the module that
removes that default — `tokenjam.core.optimize.span_pricing` — and the pricing
primitive it rests on, `pricing.get_rates_in_range`.

The load-bearing case throughout is a window that STRADDLES a rate change. On
today's packaged table that case barely exists, which is exactly why it needs a
synthetic table: a test that only ever runs inside one rate era cannot tell a
correct pricer from one that ignores time entirely.
"""
from __future__ import annotations

from datetime import datetime, time, timezone
from pathlib import Path

import pytest

from tokenjam.core import pricing
from tokenjam.core.optimize.span_pricing import (
    SPAN_UTC_DAY_SQL,
    blended_rates,
    price_span,
    rates_at,
    rates_in_window,
    span_instant,
)
from tokenjam.core.pricing import as_utc, get_rates_in_range, load_pricing_rows

UTC = timezone.utc

# A model whose price DOUBLES on 2026-09-01 — a change big enough that pricing
# the wrong side of it can never be mistaken for a rounding difference.
RATE_HISTORY_TOML = """
[testprovider.stepped]
input_per_mtok = 2.00
output_per_mtok = 10.00
cache_read_per_mtok = 0.20
cache_write_per_mtok = 2.50

[[testprovider.stepped.rates]]
valid_from = 2026-09-01
input_per_mtok = 4.00
output_per_mtok = 20.00
cache_read_per_mtok = 0.40
cache_write_per_mtok = 5.00
"""

BOUNDARY = datetime(2026, 9, 1, tzinfo=UTC)
BEFORE = datetime(2026, 8, 15, 12, 0, tzinfo=UTC)
AFTER = datetime(2026, 9, 15, 12, 0, tzinfo=UTC)


@pytest.fixture(autouse=True)
def _clean_pricing_cache(monkeypatch, tmp_path):
    monkeypatch.delenv(pricing.USER_PRICING_ENV, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    pricing.clear_pricing_cache()
    yield
    pricing.clear_pricing_cache()


@pytest.fixture
def stepped(monkeypatch, tmp_path):
    """A pricing table carrying one model whose rate steps up at BOUNDARY."""
    path: Path = tmp_path / "pricing.toml"
    path.write_text(RATE_HISTORY_TOML, encoding="utf-8")
    monkeypatch.setenv(pricing.USER_PRICING_ENV, str(path))
    pricing.clear_pricing_cache()
    return ("testprovider", "stepped")


# --------------------------------------------------------------------------
# rates_at / price_span: the instant is required, and it is honoured
# --------------------------------------------------------------------------

def test_a_span_prices_at_the_rate_that_billed_it_not_the_rate_now(stepped):
    """The whole defect in one assertion: two spans with identical tokens on
    opposite sides of a price change must NOT cost the same."""
    provider, model = stepped
    before = price_span(provider, model, at=BEFORE, input_tokens=1_000_000)
    after = price_span(provider, model, at=AFTER, input_tokens=1_000_000)
    assert before == pytest.approx(2.00)
    assert after == pytest.approx(4.00)


def test_rates_at_resolves_each_side_of_the_boundary(stepped):
    provider, model = stepped
    assert rates_at(provider, model, BEFORE).input_per_mtok == 2.00
    assert rates_at(provider, model, AFTER).input_per_mtok == 4.00
    # The boundary instant itself belongs to the NEW rate — `valid_from` is
    # inclusive, so a span at exactly midnight bills at the rate taking effect.
    assert rates_at(provider, model, BOUNDARY).input_per_mtok == 4.00


def test_an_unpriced_model_yields_nothing_rather_than_a_default(stepped):
    """A figure we cannot price is a figure we do not claim — never the flat
    default rate `calculate_cost` falls back to for a live call."""
    assert rates_at("testprovider", "no-such-model", BEFORE) is None
    assert price_span("testprovider", "no-such-model", at=BEFORE,
                      input_tokens=1_000_000) is None


def test_a_null_provider_still_resolves_through_the_shared_unknown_key(stepped):
    """Analyzers pass `span.provider`, which is nullable. The pricer must
    normalise it the same way the analyzers did before it existed, or a null
    provider would start raising instead of looking up."""
    assert rates_at(None, "no-such-model", BEFORE) is None


# --------------------------------------------------------------------------
# span_instant: the fallback is the window, never "now"
# --------------------------------------------------------------------------

def test_span_instant_passes_a_real_timestamp_through():
    assert span_instant(BEFORE, window_start=AFTER) == BEFORE


def test_span_instant_falls_back_to_the_window_start_not_to_now():
    """Both are guesses, but the window start is a guess inside the range the
    span provably falls in — and "now" is the default that caused the bug."""
    assert span_instant(None, window_start=BEFORE) == BEFORE


# --------------------------------------------------------------------------
# blended_rates: the aggregate form of price-each-span-then-sum
# --------------------------------------------------------------------------

def test_a_group_inside_one_rate_era_blends_to_that_exact_rate(stepped):
    """The reason adopting this is a no-op on today's numbers: with no rate
    change inside the group, the blend IS the single rate."""
    provider, model = stepped
    blended = blended_rates(provider, model, [(BEFORE, 100), (BEFORE, 900)])
    assert blended.input_per_mtok == pytest.approx(2.00)
    assert blended.cache_read_per_mtok == pytest.approx(0.20)


def test_a_group_straddling_a_change_blends_by_volume(stepped):
    provider, model = stepped
    half = blended_rates(provider, model, [(BEFORE, 500), (AFTER, 500)])
    assert half.input_per_mtok == pytest.approx(3.00)
    # Weighted, not averaged: 90% of the volume on the old rate stays near it.
    skewed = blended_rates(provider, model, [(BEFORE, 900), (AFTER, 100)])
    assert skewed.input_per_mtok == pytest.approx(2.20)


def test_the_blend_equals_pricing_each_span_and_summing(stepped):
    """The identity the convention rests on. If this ever fails, the module
    docstring's claim that the blend IS price-per-span-then-sum is false."""
    provider, model = stepped
    volumes = [(BEFORE, 300_000.0), (AFTER, 700_000.0), (BEFORE, 250_000.0)]
    per_span = sum(
        price_span(provider, model, at=when, input_tokens=int(vol))
        for when, vol in volumes
    )
    total_tokens = sum(vol for _when, vol in volumes)
    blended = blended_rates(provider, model, volumes)
    factored = total_tokens / 1_000_000 * blended.input_per_mtok
    assert factored == pytest.approx(per_span, rel=1e-9)


def test_a_group_with_no_volume_has_no_rate(stepped):
    """A rate for nothing is not a rate — returning one would let a zero-volume
    group contribute a term to a weighted average it has no claim on."""
    provider, model = stepped
    assert blended_rates(provider, model, []) is None
    assert blended_rates(provider, model, [(BEFORE, 0)]) is None


def test_an_undated_member_is_skipped_not_priced_at_now(stepped):
    """A span we cannot place in time must not drag the group's rate toward
    today's — which is the failure this whole module exists to prevent."""
    provider, model = stepped
    blended = blended_rates(provider, model, [(BEFORE, 100), (None, 900)])
    assert blended.input_per_mtok == pytest.approx(2.00)


def test_a_blend_is_only_as_quoted_as_its_least_quoted_term(stepped):
    """`cache_read_specified` distinguishes a real quoted rate from a stand-in.
    A blend that mixed one of each and claimed "specified" would let a guess be
    presented as a price somebody charges."""
    provider, model = stepped
    blended = blended_rates(provider, model, [(BEFORE, 100), (AFTER, 100)])
    assert blended.cache_read_specified is True


# --------------------------------------------------------------------------
# rates_in_window / get_rates_in_range: the whole band, not one instant
# --------------------------------------------------------------------------

def test_a_window_inside_one_era_reports_exactly_one_rate(stepped):
    provider, model = stepped
    rates = rates_in_window(provider, model, BEFORE, BOUNDARY)
    assert [r.input_per_mtok for r in rates] == [2.00]


def test_a_window_straddling_a_change_reports_both_rates_in_order(stepped):
    provider, model = stepped
    rates = rates_in_window(provider, model, BEFORE, AFTER)
    assert [r.input_per_mtok for r in rates] == [2.00, 4.00]


def test_a_window_wholly_after_the_change_reports_only_the_new_rate(stepped):
    provider, model = stepped
    rates = rates_in_window(provider, model, AFTER, datetime(2026, 10, 1, tzinfo=UTC))
    assert [r.input_per_mtok for r in rates] == [4.00]


def test_an_unknown_model_reports_no_rates_rather_than_raising(stepped):
    assert rates_in_window("testprovider", "no-such-model", BEFORE, AFTER) == ()
    assert get_rates_in_range("nope", "nope", BEFORE, AFTER) == ()


def test_a_boundary_exactly_on_the_window_edge_is_not_double_counted(stepped):
    """A window starting exactly at the change sees one rate, not two — the
    boundary is not "inside" it."""
    provider, model = stepped
    rates = rates_in_window(provider, model, BOUNDARY, AFTER)
    assert [r.input_per_mtok for r in rates] == [4.00]


# --------------------------------------------------------------------------
# The premise the UTC-day SQL bucket rests on
# --------------------------------------------------------------------------

def test_rate_boundaries_are_utc_midnight():
    """`SPAN_UTC_DAY_SQL` groups aggregate queries by UTC day and prices each
    bucket at one rate. That is only sound while every rate change lands on a
    UTC midnight — true today because `valid_from` is a TOML DATE. If a row
    ever carries a time-of-day, a day bucket starts straddling a change and
    every aggregate query using this constant has to get finer."""
    dated = [
        (f"{provider}/{model}", row.valid_from)
        for provider, models in load_pricing_rows().items()
        for model, rows in models.items()
        for row in rows
        if row.valid_from is not None
    ]
    assert dated, (
        "no dated rate row in the table at all — this guard would be vacuous, "
        "and so would every rate-history assertion that relies on it"
    )
    offenders = [
        f"{name} @ {when.isoformat()}"
        for name, when in dated
        if as_utc(when).timetz() != time(0, 0, tzinfo=UTC)
    ]
    assert not offenders, (
        "rate change(s) not on a UTC midnight, so a UTC-day bucket now "
        f"straddles a rate change: {offenders}. Every query grouping by "
        "SPAN_UTC_DAY_SQL needs a finer bucket."
    )


def test_the_day_bucket_sql_pins_utc_explicitly():
    """Casting a TIMESTAMPTZ to DATE uses the SESSION timezone, so on a
    non-UTC host the "day" would be a local day — which straddles UTC midnight
    by the offset, defeating the guarantee above."""
    assert "AT TIME ZONE 'UTC'" in SPAN_UTC_DAY_SQL


# --------------------------------------------------------------------------
# The required-instant rule, and its one labelled exception
# --------------------------------------------------------------------------

def test_only_the_labelled_call_site_prices_at_now():
    """No analyzer may quietly reacquire a bare `get_rates`.

    The defect this whole module addresses was invisible precisely because
    `get_rates(provider, model)` is a perfectly ordinary-looking call whose
    `at=None` default silently means "today". A new one would reintroduce it
    with nothing to notice.

    `cache_efficacy.estimate_cache_recoverable` is the sanctioned exception: its
    row is a whole-window aggregate that is also the UI's display row, so it
    prices at now and says so in its docstring. If you are adding a second
    exception, the bar is a comment explaining why a real instant cannot be had
    — not an entry in this list.
    """
    import re
    from pathlib import Path

    import tokenjam.core.optimize as optimize_pkg

    allowed = {("cache_efficacy.py", "estimate_cache_recoverable")}
    root = Path(optimize_pkg.__file__).parent

    offenders: list[str] = []
    for path in sorted(root.rglob("*.py")):
        if path.name == "span_pricing.py":  # defines the wrappers
            continue
        source = path.read_text(encoding="utf-8")
        enclosing = "<module>"
        for lineno, line in enumerate(source.splitlines(), 1):
            if m := re.match(r"\s*def\s+(\w+)", line):
                enclosing = m.group(1)
            code = line.split("#", 1)[0]
            if re.search(r"\bget_rates\s*\(", code) and "at=" not in code:
                if (path.name, enclosing) not in allowed:
                    offenders.append(f"{path.name}:{lineno} in {enclosing}()")

    assert not offenders, (
        "bare get_rates (no at=) under core/optimize — these price PAST traffic "
        f"at today's rate: {offenders}. Route through span_pricing, which makes "
        "the instant a required argument."
    )


def test_the_sanctioned_exception_still_states_its_limitation():
    """The exception is only acceptable while it is labelled. A silent one is
    the original bug wearing a test's approval."""
    from tokenjam.core.optimize.analyzers.cache_efficacy import (
        estimate_cache_recoverable,
    )

    doc = estimate_cache_recoverable.__doc__ or ""
    assert "KNOWN LIMITATION" in doc
    assert "today" in doc.lower()
