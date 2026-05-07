"""Synthetic tests for CostEngine with an in-memory DuckDB backend."""
from __future__ import annotations

import duckdb
import pytest

from tj.core.cost import CostEngine
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
    assert db_cost == pytest.approx(0.0016)
    assert span.cost_usd == pytest.approx(0.0016)


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
    assert session_cost == pytest.approx(0.0016)


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
    # 3 * 0.0016 = 0.0048
    assert session_cost == pytest.approx(0.0048)


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


def test_cost_engine_no_op_when_provider_missing(fake_db: FakeDB, engine: CostEngine) -> None:
    span = make_llm_span(input_tokens=1000, output_tokens=200)
    span.provider = None
    fake_db.insert_span_stub(span.span_id)

    engine.process_span(span)

    db_cost = fake_db.get_span_cost(span.span_id)
    assert db_cost is None
