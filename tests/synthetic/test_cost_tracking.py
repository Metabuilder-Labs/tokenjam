"""Synthetic tests for CostEngine against the real InMemoryBackend.

Previously these ran against a hand-rolled `FakeDB` stub that exposed a raw
`.conn` — which only worked because CostEngine reached into `db.conn` directly.
Issue #309 moved those writes behind the StorageBackend protocol
(`update_span_cost` / `increment_session_cost`), so the cost path now exercises
the same backend production uses. Span cost is read back through
`get_trace_spans`; the session total through `get_session().total_cost_usd`
(the stored column, distinct from `get_session_cost`, which sums span costs).
"""
from __future__ import annotations

import pytest

from tokenjam.core.cost import CostEngine
from tokenjam.core.db import InMemoryBackend
from tests.factories import make_llm_span, make_session


@pytest.fixture
def db():
    backend = InMemoryBackend()
    yield backend
    backend.close()


@pytest.fixture
def engine(db: InMemoryBackend) -> CostEngine:
    return CostEngine(db)


def _span_cost(db: InMemoryBackend, span) -> float | None:
    """Read a single span's persisted cost_usd back via the protocol."""
    for s in db.get_trace_spans(span.trace_id):
        if s.span_id == span.span_id:
            return s.cost_usd
    return None


def _session_total(db: InMemoryBackend, session_id: str) -> float | None:
    session = db.get_session(session_id)
    return session.total_cost_usd if session else None


def test_cost_engine_updates_span_cost_in_db(db: InMemoryBackend, engine: CostEngine) -> None:
    span = make_llm_span(
        provider="anthropic", model="claude-haiku-4-5",
        input_tokens=1000, output_tokens=200,
    )
    db.insert_span(span)

    engine.process_span(span)

    db_cost = _span_cost(db, span)
    assert db_cost is not None
    assert db_cost == pytest.approx(0.002)
    assert span.cost_usd == pytest.approx(0.002)


def test_cost_engine_updates_session_total_cost(db: InMemoryBackend, engine: CostEngine) -> None:
    session = make_session(total_cost_usd=None)
    db.upsert_session(session)

    span = make_llm_span(
        provider="anthropic", model="claude-haiku-4-5",
        input_tokens=1000, output_tokens=200,
        session_id=session.session_id,
    )
    db.insert_span(span)

    engine.process_span(span)

    session_cost = _session_total(db, session.session_id)
    assert session_cost is not None
    assert session_cost == pytest.approx(0.002)


def test_cost_engine_accumulates_across_multiple_spans(
    db: InMemoryBackend, engine: CostEngine,
) -> None:
    session = make_session(total_cost_usd=None)
    db.upsert_session(session)

    for _ in range(3):
        span = make_llm_span(
            provider="anthropic", model="claude-haiku-4-5",
            input_tokens=1000, output_tokens=200,
            session_id=session.session_id,
        )
        db.insert_span(span)
        engine.process_span(span)

    session_cost = _session_total(db, session.session_id)
    assert session_cost is not None
    # 3 * 0.002 = 0.006
    assert session_cost == pytest.approx(0.006)


def test_cost_engine_prices_cache_write_tokens(
    db: InMemoryBackend, engine: CostEngine,
) -> None:
    # claude-haiku-4-5 rates: input=1.00, output=5.00, cache_read=0.10,
    # cache_write=1.25 per MTok. The cache-WRITE tokens must be billed at the
    # cache_write rate — previously they were dropped and never priced.
    span = make_llm_span(
        provider="anthropic", model="claude-haiku-4-5",
        input_tokens=1000, output_tokens=200,
        cache_tokens=5000,         # reads  -> 5000/1e6 * 0.10 = 0.0005
        cache_write_tokens=10000,  # writes -> 10000/1e6 * 1.25 = 0.0125
    )
    db.insert_span(span)

    engine.process_span(span)

    # 0.001 (in) + 0.001 (out) + 0.0005 (read) + 0.0125 (write) = 0.015
    assert span.cost_usd == pytest.approx(0.015)
    assert _span_cost(db, span) == pytest.approx(0.015)


def test_cost_engine_no_op_when_tokens_missing(db: InMemoryBackend, engine: CostEngine) -> None:
    span = make_llm_span(
        provider="anthropic", model="claude-haiku-4-5",
        input_tokens=0, output_tokens=0,
    )
    db.insert_span(span)

    engine.process_span(span)

    assert _span_cost(db, span) is None
    assert span.cost_usd is None


def test_cost_engine_costs_cache_only_span(db: InMemoryBackend, engine: CostEngine) -> None:
    # A span with no new input/output but cache-read tokens (a cache hit) still
    # costs the cache-read rate and must be recorded, not dropped as a no-op.
    # claude-haiku-4-5: cache_read=0.10 per MTok.
    span = make_llm_span(
        provider="anthropic", model="claude-haiku-4-5",
        input_tokens=0, output_tokens=0, cache_tokens=1_000_000,
    )
    db.insert_span(span)

    engine.process_span(span)

    assert _span_cost(db, span) == pytest.approx(0.10)
    assert span.cost_usd == pytest.approx(0.10)


def test_cost_engine_cache_only_span_updates_session_total(
    db: InMemoryBackend, engine: CostEngine,
) -> None:
    # The cache-only span's cost must also accumulate into the session total,
    # not just the span row — dropping it previously under-reported the session.
    session = make_session(total_cost_usd=None)
    db.upsert_session(session)

    span = make_llm_span(
        provider="anthropic", model="claude-haiku-4-5",
        input_tokens=0, output_tokens=0, cache_tokens=1_000_000,
        session_id=session.session_id,
    )
    db.insert_span(span)

    engine.process_span(span)

    assert _session_total(db, session.session_id) == pytest.approx(0.10)


def test_cost_engine_costs_cache_write_span(db: InMemoryBackend, engine: CostEngine) -> None:
    # A span whose only tokens are cache-CREATION (cache write) must be costed at
    # the cache-write rate, not dropped as a no-op and not charged the read rate.
    # claude-haiku-4-5: cache_write=1.25 per MTok.
    span = make_llm_span(
        provider="anthropic", model="claude-haiku-4-5",
        input_tokens=0, output_tokens=0, cache_write_tokens=1_000_000,
    )
    db.insert_span(span)

    engine.process_span(span)

    assert _span_cost(db, span) == pytest.approx(1.25)
    assert span.cost_usd == pytest.approx(1.25)


def test_cost_engine_costs_cache_read_and_write_together(
    db: InMemoryBackend, engine: CostEngine,
) -> None:
    # Read and write cache tokens are priced at different rates and must both be
    # charged. claude-haiku-4-5: cache_read=0.10, cache_write=1.25 per MTok.
    span = make_llm_span(
        provider="anthropic", model="claude-haiku-4-5",
        input_tokens=0, output_tokens=0,
        cache_tokens=1_000_000, cache_write_tokens=1_000_000,
    )
    db.insert_span(span)

    engine.process_span(span)

    assert _span_cost(db, span) == pytest.approx(1.35)
    assert span.cost_usd == pytest.approx(1.35)


def test_cost_engine_cache_write_span_updates_session_total(
    db: InMemoryBackend, engine: CostEngine,
) -> None:
    # The cache-write span's cost must also accumulate into the session total.
    session = make_session(total_cost_usd=None)
    db.upsert_session(session)

    span = make_llm_span(
        provider="anthropic", model="claude-haiku-4-5",
        input_tokens=0, output_tokens=0, cache_write_tokens=1_000_000,
        session_id=session.session_id,
    )
    db.insert_span(span)

    engine.process_span(span)

    assert _session_total(db, session.session_id) == pytest.approx(1.25)


def test_cost_engine_cache_write_pre_priced_does_not_double_count_session(
    db: InMemoryBackend, engine: CostEngine,
) -> None:
    # A pre-priced span (cost_usd already set, e.g. from the parser) has its
    # session cost handled by ingest's _build_or_update_session. process_span
    # must still recompute the span cost but must NOT re-add to the session
    # total, or cache-write spend would be double-counted.
    session = make_session(total_cost_usd=5.0)
    db.upsert_session(session)

    span = make_llm_span(
        provider="anthropic", model="claude-haiku-4-5",
        input_tokens=0, output_tokens=0, cache_write_tokens=1_000_000,
        session_id=session.session_id, cost_usd=1.25,
    )
    db.insert_span(span)

    engine.process_span(span)

    # Span cost recomputed, session total left untouched (no double-count).
    assert _span_cost(db, span) == pytest.approx(1.25)
    assert _session_total(db, session.session_id) == pytest.approx(5.0)


def test_cost_engine_no_op_when_provider_missing(db: InMemoryBackend, engine: CostEngine) -> None:
    span = make_llm_span(input_tokens=1000, output_tokens=200)
    span.provider = None
    db.insert_span(span)

    engine.process_span(span)

    assert _span_cost(db, span) is None
