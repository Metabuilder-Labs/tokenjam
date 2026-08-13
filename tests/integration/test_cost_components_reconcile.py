"""`/cost` and `/cost/components` publish the same window total.

`/cost`'s `total_cost_usd` is `SUM(spans.cost_usd)` — priced ONCE at ingest by
`CostEngine.process_span`, at each span's own `start_time`, at the variant that
actually billed it, and with a guaranteed non-null fallback for a model missing
from the pricing table.

`/cost/components` re-prices the same window from raw tokens. Until this file
existed it had NO test coverage at all, and it diverged from the stored total in
two concrete ways:

  * it never passed a variant, so every span re-priced at `STANDARD_VARIANT` —
    an Anthropic `speed="fast"` call bills at double the standard rate, so the
    two endpoints named totals differing by that whole premium;
  * it `continue`d past a `None` from `rates_at`, contributing exactly `$0` for
    any model absent from the pricing table, while the same traffic contributed
    real fallback dollars to `/cost`.

Both are published as "the cost of this window". These tests pin that they
agree, in exactly those two regimes plus the plain one.

Reconciliation is asserted to a tolerance, not exactly: `calculate_cost` rounds
each span to 8 dp while the component path sums a bucket's tokens before
pricing. The tolerance is far tighter than either divergence above, both of
which are whole-multiple errors.
"""
from __future__ import annotations

from datetime import timedelta

import httpx
import pytest

from tokenjam.api.app import create_app
from tokenjam.core.config import TjConfig
from tokenjam.core.cost import CostEngine
from tokenjam.core.db import InMemoryBackend
from tokenjam.core.ingest import IngestPipeline
from tokenjam.utils.time_parse import utcnow
from tests.factories import make_llm_span, make_session

#: Absolute dollars the two totals may differ by. Per-span 8-dp rounding over a
#: handful of spans is bounded by ~1e-7; a wrong variant or a dropped unpriced
#: model is off by 100% of that traffic's cost.
_ROUNDING_TOLERANCE_USD = 1e-6


def _app(db, config):
    return create_app(
        config=config, db=db, ingest_pipeline=IngestPipeline(db=db, config=config),
    )


def _store(db, span):
    """Insert a span and price it exactly as ingest does.

    `CostEngine.process_span` IS the stored-cost path (it is the post-ingest
    hook), so driving it directly is what makes `spans.cost_usd` here the same
    figure a real install would carry — a hand-supplied `cost_usd=` would prove
    nothing about the pricing convention this file is testing.
    """
    db.insert_span(span)
    CostEngine(db).process_span(span)
    return span


async def _totals(db, cfg):
    transport = httpx.ASGITransport(app=_app(db, cfg))
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
        cost = (await c.get("/api/v1/cost?since=30d")).json()
        comps = (await c.get("/api/v1/cost/components?since=30d")).json()
    return cost, comps


@pytest.mark.asyncio
async def test_the_component_split_sums_to_the_window_total():
    """The baseline: ordinary standard-variant, fully-priced traffic."""
    db = InMemoryBackend()
    cfg = TjConfig(version="1")
    now = utcnow()
    for i in range(5):
        db.upsert_session(make_session(session_id=f"s{i}"))
        _store(db, make_llm_span(
            session_id=f"s{i}", provider="anthropic", model="claude-sonnet-5",
            input_tokens=10_000, output_tokens=500,
            cache_tokens=4_000, cache_write_tokens=1_000,
            start_time=now - timedelta(days=i),
        ))
    try:
        cost, comps = await _totals(db, cfg)
    finally:
        db.close()

    assert cost["total_cost_usd"] > 0, "fixture produced no priced traffic"
    assert comps["total_cost_usd"] == pytest.approx(
        cost["total_cost_usd"], abs=_ROUNDING_TOLERANCE_USD,
    )
    assert comps["total_tokens"] == cost["total_tokens"]


@pytest.mark.asyncio
async def test_a_fast_variant_span_is_priced_at_its_own_variant_in_both_totals():
    """`speed="fast"` bills the SAME model id at a premium rate.

    The component path took no variant argument at all, so it always resolved
    `STANDARD_VARIANT`. The pricing table's fast row for `claude-opus-5` is
    exactly double the standard one, so the two published totals differed by a
    factor of two on this traffic — which is why the second assertion here
    checks the total is genuinely above the standard-priced figure rather than
    only that the two endpoints agree. Two endpoints can agree by both being
    wrong.
    """
    db = InMemoryBackend()
    cfg = TjConfig(version="1")
    now = utcnow()
    db.upsert_session(make_session(session_id="fast"))
    _store(db, make_llm_span(
        session_id="fast", provider="anthropic", model="claude-opus-5",
        input_tokens=1_000_000, output_tokens=0,
        request_params={"speed": "fast"},
        start_time=now - timedelta(days=1),
    ))
    try:
        cost, comps = await _totals(db, cfg)
    finally:
        db.close()

    assert comps["total_cost_usd"] == pytest.approx(
        cost["total_cost_usd"], abs=_ROUNDING_TOLERANCE_USD,
    )
    # 1M input tokens: $5.00 at the standard rate, $10.00 at the fast rate.
    # Read from the table rather than hard-coded, so a repricing moves the
    # test with the product instead of failing it.
    from tokenjam.core.pricing import get_rates

    standard = get_rates("anthropic", "claude-opus-5", at=now - timedelta(days=1))
    fast = get_rates(
        "anthropic", "claude-opus-5", at=now - timedelta(days=1), variant="fast",
    )
    assert fast is not None and standard is not None
    assert fast.input_per_mtok > standard.input_per_mtok, (
        "fixture assumption broken: the fast row no longer costs more than the "
        "standard one, so this test cannot detect a variant being ignored"
    )
    assert comps["total_cost_usd"] == pytest.approx(
        fast.input_per_mtok, abs=_ROUNDING_TOLERANCE_USD,
    )
    assert comps["components"][0]["key"] == "input"
    assert comps["components"][0]["cost_usd"] == pytest.approx(
        fast.input_per_mtok, abs=_ROUNDING_TOLERANCE_USD,
    )


@pytest.mark.asyncio
async def test_a_model_missing_from_the_pricing_table_contributes_to_both_totals():
    """An unpriced model used to contribute fallback dollars to `/cost` and a
    silent `$0` to `/cost/components`.

    A zero here is the worst possible placeholder: the component bars would show
    a window as costing nothing while the cost view showed real money for the
    same spans, and nothing on either surface named the difference.
    """
    db = InMemoryBackend()
    cfg = TjConfig(version="1")
    now = utcnow()
    db.upsert_session(make_session(session_id="unpriced"))
    _store(db, make_llm_span(
        session_id="unpriced", provider="anthropic",
        model="totally-not-in-the-pricing-table",
        input_tokens=500_000, output_tokens=100_000,
        cache_tokens=50_000, cache_write_tokens=20_000,
        start_time=now - timedelta(days=1),
    ))
    try:
        cost, comps = await _totals(db, cfg)
    finally:
        db.close()

    assert cost["total_cost_usd"] > 0, (
        "fixture assumption broken: the stored path is supposed to price an "
        "unknown model at the default rate, so there is a figure to reconcile"
    )
    assert comps["total_cost_usd"] > 0, "the component split dropped an unpriced model"
    assert comps["total_cost_usd"] == pytest.approx(
        cost["total_cost_usd"], abs=_ROUNDING_TOLERANCE_USD,
    )
    # And the estimate is DISCLOSED as one, on the components payload too — the
    # figure now exists there, so the surface has to be able to tell a reader it
    # is a default-rate estimate rather than a quoted price.
    coverage = comps["pricing_coverage"]
    assert coverage["measured"] is True
    assert coverage["unpriced_call_count"] == 1
    assert coverage["note"], "an unpriced model must not be published silently"
    assert coverage == cost["pricing_coverage"], (
        "the two endpoints must publish one coverage derivation, not two"
    )


@pytest.mark.asyncio
async def test_a_mixed_window_of_all_three_regimes_reconciles():
    """Standard, fast-variant and unpriced traffic in ONE window.

    Each of the fixes above is exact in isolation; this is the case a real
    corpus actually presents, and the one where a per-bucket grouping bug (the
    variant folded into the wrong GROUP BY key, say) would show up.
    """
    db = InMemoryBackend()
    cfg = TjConfig(version="1")
    now = utcnow()
    db.upsert_session(make_session(session_id="mixed"))
    _store(db, make_llm_span(
        session_id="mixed", provider="anthropic", model="claude-opus-5",
        input_tokens=200_000, output_tokens=10_000,
        start_time=now - timedelta(days=1),
    ))
    _store(db, make_llm_span(
        session_id="mixed", provider="anthropic", model="claude-opus-5",
        input_tokens=300_000, output_tokens=20_000,
        request_params={"speed": "fast"},
        start_time=now - timedelta(days=1),
    ))
    _store(db, make_llm_span(
        session_id="mixed", provider="openai", model="gpt-not-a-real-model",
        input_tokens=100_000, output_tokens=5_000,
        billing_account="openai",
        start_time=now - timedelta(days=2),
    ))
    try:
        cost, comps = await _totals(db, cfg)
    finally:
        db.close()

    assert comps["total_cost_usd"] == pytest.approx(
        cost["total_cost_usd"], abs=_ROUNDING_TOLERANCE_USD,
    )
    assert comps["total_tokens"] == cost["total_tokens"]
    # The same model id appears at two variants in one day-bucket; if the
    # grouping collapsed them the standard span would be repriced as fast (or
    # vice versa) and the totals above would part.
    assert comps["pricing_coverage"]["unpriced_call_count"] == 1


@pytest.mark.asyncio
async def test_capture_off_traffic_prices_as_standard():
    """`request_params` is NULL whenever full-request capture is off, which is
    the default. The variant expression must read that as `standard` rather than
    NULL — a NULL variant would miss every pricing row and send the whole window
    down the default-rate fallback.
    """
    db = InMemoryBackend()
    cfg = TjConfig(version="1")
    now = utcnow()
    db.upsert_session(make_session(session_id="plain"))
    _store(db, make_llm_span(
        session_id="plain", provider="anthropic", model="claude-opus-5",
        input_tokens=1_000_000, output_tokens=0,
        request_params=None,
        start_time=now - timedelta(days=1),
    ))
    try:
        cost, comps = await _totals(db, cfg)
    finally:
        db.close()

    from tokenjam.core.pricing import get_rates

    standard = get_rates("anthropic", "claude-opus-5", at=now - timedelta(days=1))
    assert standard is not None
    assert comps["total_cost_usd"] == pytest.approx(
        standard.input_per_mtok, abs=_ROUNDING_TOLERANCE_USD,
    )
    assert comps["total_cost_usd"] == pytest.approx(
        cost["total_cost_usd"], abs=_ROUNDING_TOLERANCE_USD,
    )
    assert comps["pricing_coverage"]["unpriced_call_count"] == 0
