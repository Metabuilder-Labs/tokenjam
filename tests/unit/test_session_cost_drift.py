"""`sessions.total_cost_usd` must agree with `SUM(spans.cost_usd)`.

Two figures the UI can show side by side — a session card's own total and a
total derived from spans — so a gap between them is a published number that
excludes rows it should include. `recompute_session_totals_from_spans` already
documents the span sum as the source of truth; `session_cost_drift` is the
read-only detector that makes a stale session row visible instead of leaving a
user to notice a percentage discrepancy between two screens.
"""
from __future__ import annotations

import pytest

from tests.factories import make_llm_span, make_session
from tokenjam.core.config import StorageConfig
from tokenjam.core.db import (
    SESSION_COST_DRIFT_TOLERANCE_USD,
    DuckDBBackend,
    session_cost_drift,
)


@pytest.fixture
def backend(tmp_path):
    db = DuckDBBackend(StorageConfig(path=str(tmp_path / "t.duckdb")))
    yield db
    db.close()


def _session(db, session_id: str, total) -> None:
    db.upsert_session(make_session(session_id=session_id, total_cost_usd=total))


def _span(db, span_id: str, session_id: str, cost) -> None:
    span = make_llm_span(session_id=session_id, cost_usd=cost)
    span.span_id = span_id
    db.insert_span(span)


def test_agreeing_session_reports_no_drift(backend):
    _session(backend, "s1", 3.0)
    _span(backend, "sp1", "s1", 1.0)
    _span(backend, "sp2", "s1", 2.0)

    count, total, worst = session_cost_drift(backend.conn)

    assert (count, total, worst) == (0, 0.0, [])


def test_session_holding_a_stale_total_is_reported_with_both_figures(backend):
    # The shape the live corpus showed: the session kept an upstream-supplied
    # total while its span rows were rewritten with tj's own pricing.
    _session(backend, "s1", 177.14)
    _span(backend, "sp1", "s1", 569.52)

    count, total, worst = session_cost_drift(backend.conn)

    assert count == 1
    assert total == pytest.approx(392.38, abs=0.01)
    assert worst == [("s1", pytest.approx(177.14), pytest.approx(569.52))]


def test_null_total_cost_is_not_drift_when_no_span_carries_a_cost(backend):
    """A NULL total is the correct value for a session with nothing to price.

    Sessions made only of tool/marker spans (or LLM calls that arrived with no
    usage attached) have no priced span at all, and `SUM` over an all-NULL
    column is itself NULL. NULL and 0.0 are the same "no priced spans"
    statement here, so neither side may be reported as disagreeing with the
    other — otherwise every such session shows up as a false positive.
    """
    _session(backend, "null-side", None)
    _span(backend, "sp1", "null-side", None)
    _span(backend, "sp2", "null-side", None)
    _session(backend, "zero-side", 0.0)
    _span(backend, "sp3", "zero-side", None)

    count, _total, worst = session_cost_drift(backend.conn)

    assert (count, worst) == (0, [])


def test_null_total_IS_drift_when_a_span_was_priced(backend):
    # The other half: a NULL total on a session whose spans do carry cost is a
    # genuinely missing number, not an honest "nothing to price".
    _session(backend, "s1", None)
    _span(backend, "sp1", "s1", 4.25)

    count, total, _worst = session_cost_drift(backend.conn)

    assert count == 1
    assert total == pytest.approx(4.25)


def test_float_residue_below_tolerance_is_not_drift(backend):
    _session(backend, "s1", 1.0)
    _span(backend, "sp1", "s1", 1.0 + SESSION_COST_DRIFT_TOLERANCE_USD / 2)

    assert session_cost_drift(backend.conn)[0] == 0


def test_recompute_clears_the_drift_it_reports(backend):
    _session(backend, "s1", 177.14)
    _span(backend, "sp1", "s1", 569.52)
    assert session_cost_drift(backend.conn)[0] == 1

    backend.recompute_session_totals_from_spans(["s1"])

    assert session_cost_drift(backend.conn)[0] == 0


def test_doctor_reports_drift_and_offers_the_repair(backend):
    from tokenjam.cli.cmd_doctor import _check_cost_integrity

    _session(backend, "s1", 177.14)
    _span(backend, "sp1", "s1", 569.52)

    check = _check_cost_integrity(backend)

    assert check["level"] == "warning"
    assert check["repair_action"] == "heal_session_costs"
    # Both figures named, so the user can see which screen is stale.
    assert "177.14" in check["message"] and "569.52" in check["message"]


def test_doctor_is_ok_on_a_healthy_store(backend):
    from tokenjam.cli.cmd_doctor import _check_cost_integrity

    _session(backend, "s1", 3.0)
    _span(backend, "sp1", "s1", 3.0)
    _session(backend, "s2", None)
    _span(backend, "sp2", "s2", None)

    check = _check_cost_integrity(backend)

    assert check["level"] == "ok"
    assert "repair_action" not in check
