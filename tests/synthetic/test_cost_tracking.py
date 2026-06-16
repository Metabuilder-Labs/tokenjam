"""Synthetic tests for CostEngine with an in-memory DuckDB backend."""
from __future__ import annotations

import duckdb
import pytest

from tokenjam.core.cost import CostEngine
from tests.factories import make_llm_span, make_session


class FakeDB:
    """Minimal DB stub with just enough schema for CostEngine tests."""

    def __init__(self) -> None:
        self.conn = duckdb.connect(":memory:")
        self.conn.execute("""
            CREATE TABLE spans (
                span_id TEXT PRIMARY KEY,
                cost_usd DOUBLE
            )
        """)
        self.conn.execute("""
            CREATE TABLE sessions (
                session_id TEXT PRIMARY KEY,
                total_cost_usd DOUBLE
            )
        """)

    def insert_span_stub(self, span_id: str) -> None:
        self.conn.execute(
            "INSERT INTO spans (span_id, cost_usd) VALUES (?, NULL)",
            [span_id],
        )

    def insert_session_stub(self, session_id: str) -> None:
        self.conn.execute(
            "INSERT INTO sessions (session_id, total_cost_usd) VALUES (?, NULL)",
            [session_id],
        )

    def get_span_cost(self, span_id: str) -> float | None:
        row = self.conn.execute(
            "SELECT cost_usd FROM spans WHERE span_id = ?", [span_id]
        ).fetchone()
        return row[0] if row else None

    def get_session_cost(self, session_id: str) -> float | None:
        row = self.conn.execute(
            "SELECT total_cost_usd FROM sessions WHERE session_id = ?", [session_id]
        ).fetchone()
        return row[0] if row else None


@pytest.fixture
def fake_db() -> FakeDB:
    return FakeDB()


@pytest.fixture
def engine(fake_db: FakeDB) -> CostEngine:
    return CostEngine(fake_db)


def test_cost_engine_updates_span_cost_in_db(fake_db: FakeDB, engine: CostEngine) -> None:
    span = make_llm_span(
        provider="anthropic", model="claude-haiku-4-5",
        input_tokens=1000, output_tokens=200,
    )
    fake_db.insert_span_stub(span.span_id)

    engine.process_span(span)

    db_cost = fake_db.get_span_cost(span.span_id)
    assert db_cost is not None
    assert db_cost == pytest.approx(0.002)
    assert span.cost_usd == pytest.approx(0.002)


def test_cost_engine_updates_session_total_cost(fake_db: FakeDB, engine: CostEngine) -> None:
    session = make_session()
    fake_db.insert_session_stub(session.session_id)

    span = make_llm_span(
        provider="anthropic", model="claude-haiku-4-5",
        input_tokens=1000, output_tokens=200,
        session_id=session.session_id,
    )
    fake_db.insert_span_stub(span.span_id)

    engine.process_span(span)

    session_cost = fake_db.get_session_cost(session.session_id)
    assert session_cost is not None
    assert session_cost == pytest.approx(0.002)


def test_cost_engine_accumulates_across_multiple_spans(
    fake_db: FakeDB, engine: CostEngine,
) -> None:
    session = make_session()
    fake_db.insert_session_stub(session.session_id)

    for _ in range(3):
        span = make_llm_span(
            provider="anthropic", model="claude-haiku-4-5",
            input_tokens=1000, output_tokens=200,
            session_id=session.session_id,
        )
        fake_db.insert_span_stub(span.span_id)
        engine.process_span(span)

    session_cost = fake_db.get_session_cost(session.session_id)
    assert session_cost is not None
    # 3 * 0.002 = 0.006
    assert session_cost == pytest.approx(0.006)


def test_cost_engine_prices_cache_write_tokens(
    fake_db: FakeDB, engine: CostEngine,
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
    fake_db.insert_span_stub(span.span_id)

    engine.process_span(span)

    # 0.001 (in) + 0.001 (out) + 0.0005 (read) + 0.0125 (write) = 0.015
    assert span.cost_usd == pytest.approx(0.015)
    assert fake_db.get_span_cost(span.span_id) == pytest.approx(0.015)


def test_cost_engine_no_op_when_tokens_missing(fake_db: FakeDB, engine: CostEngine) -> None:
    span = make_llm_span(
        provider="anthropic", model="claude-haiku-4-5",
        input_tokens=0, output_tokens=0,
    )
    fake_db.insert_span_stub(span.span_id)

    engine.process_span(span)

    db_cost = fake_db.get_span_cost(span.span_id)
    assert db_cost is None
    assert span.cost_usd is None


def test_cost_engine_costs_cache_only_span(fake_db: FakeDB, engine: CostEngine) -> None:
    # A span with no new input/output but cache-read tokens (a cache hit) still
    # costs the cache-read rate and must be recorded, not dropped as a no-op.
    # claude-haiku-4-5: cache_read=0.10 per MTok.
    span = make_llm_span(
        provider="anthropic", model="claude-haiku-4-5",
        input_tokens=0, output_tokens=0, cache_tokens=1_000_000,
    )
    fake_db.insert_span_stub(span.span_id)

    engine.process_span(span)

    db_cost = fake_db.get_span_cost(span.span_id)
    assert db_cost == pytest.approx(0.10)
    assert span.cost_usd == pytest.approx(0.10)


def test_cost_engine_cache_only_span_updates_session_total(
    fake_db: FakeDB, engine: CostEngine,
) -> None:
    # The cache-only span's cost must also accumulate into the session total,
    # not just the span row — dropping it previously under-reported the session.
    session = make_session()
    fake_db.insert_session_stub(session.session_id)

    span = make_llm_span(
        provider="anthropic", model="claude-haiku-4-5",
        input_tokens=0, output_tokens=0, cache_tokens=1_000_000,
        session_id=session.session_id,
    )
    fake_db.insert_span_stub(span.span_id)

    engine.process_span(span)

    session_cost = fake_db.get_session_cost(session.session_id)
    assert session_cost == pytest.approx(0.10)


def test_cost_engine_costs_cache_write_span(fake_db: FakeDB, engine: CostEngine) -> None:
    # A span whose only tokens are cache-CREATION (cache write) must be costed at
    # the cache-write rate, not dropped as a no-op and not charged the read rate.
    # claude-haiku-4-5: cache_write=1.25 per MTok.
    span = make_llm_span(
        provider="anthropic", model="claude-haiku-4-5",
        input_tokens=0, output_tokens=0, cache_write_tokens=1_000_000,
    )
    fake_db.insert_span_stub(span.span_id)

    engine.process_span(span)

    db_cost = fake_db.get_span_cost(span.span_id)
    assert db_cost == pytest.approx(1.25)
    assert span.cost_usd == pytest.approx(1.25)


def test_cost_engine_costs_cache_read_and_write_together(
    fake_db: FakeDB, engine: CostEngine,
) -> None:
    # Read and write cache tokens are priced at different rates and must both be
    # charged. claude-haiku-4-5: cache_read=0.10, cache_write=1.25 per MTok.
    span = make_llm_span(
        provider="anthropic", model="claude-haiku-4-5",
        input_tokens=0, output_tokens=0,
        cache_tokens=1_000_000, cache_write_tokens=1_000_000,
    )
    fake_db.insert_span_stub(span.span_id)

    engine.process_span(span)

    db_cost = fake_db.get_span_cost(span.span_id)
    assert db_cost == pytest.approx(1.35)
    assert span.cost_usd == pytest.approx(1.35)


def test_cost_engine_cache_write_span_updates_session_total(
    fake_db: FakeDB, engine: CostEngine,
) -> None:
    # The cache-write span's cost must also accumulate into the session total.
    session = make_session()
    fake_db.insert_session_stub(session.session_id)

    span = make_llm_span(
        provider="anthropic", model="claude-haiku-4-5",
        input_tokens=0, output_tokens=0, cache_write_tokens=1_000_000,
        session_id=session.session_id,
    )
    fake_db.insert_span_stub(span.span_id)

    engine.process_span(span)

    session_cost = fake_db.get_session_cost(session.session_id)
    assert session_cost == pytest.approx(1.25)


def test_cost_engine_cache_write_pre_priced_does_not_double_count_session(
    fake_db: FakeDB, engine: CostEngine,
) -> None:
    # A pre-priced span (cost_usd already set, e.g. from the parser) has its
    # session cost handled by ingest's _build_or_update_session. process_span
    # must still recompute the span cost but must NOT re-add to the session
    # total, or cache-write spend would be double-counted.
    session = make_session()
    fake_db.conn.execute(
        "INSERT INTO sessions (session_id, total_cost_usd) VALUES (?, ?)",
        [session.session_id, 5.0],
    )

    span = make_llm_span(
        provider="anthropic", model="claude-haiku-4-5",
        input_tokens=0, output_tokens=0, cache_write_tokens=1_000_000,
        session_id=session.session_id, cost_usd=1.25,
    )
    fake_db.insert_span_stub(span.span_id)

    engine.process_span(span)

    # Span cost recomputed, session total left untouched (no double-count).
    assert fake_db.get_span_cost(span.span_id) == pytest.approx(1.25)
    assert fake_db.get_session_cost(session.session_id) == pytest.approx(5.0)


def test_cost_engine_no_op_when_provider_missing(fake_db: FakeDB, engine: CostEngine) -> None:
    span = make_llm_span(input_tokens=1000, output_tokens=200)
    span.provider = None
    fake_db.insert_span_stub(span.span_id)

    engine.process_span(span)

    db_cost = fake_db.get_span_cost(span.span_id)
    assert db_cost is None
