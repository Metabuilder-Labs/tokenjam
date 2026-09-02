"""Integration tests that demonstrate why --dist loadscope (not --dist load) is required.

Two isolation properties under test:

1. OTel module-level span collector
   Each xdist worker imports the module fresh, so _TestSpanCollector is a
   different object per process. Under --dist load, step_3 may land on a
   different worker than step_1/step_2, whose _collector is empty, so the
   cumulative-span assertion fails. Under --dist loadscope all three steps
   run on the same worker.

2. DuckDB single-writer concurrency
   Under --dist load, parametrized cases can run concurrently on different
   workers; each opens its own in-memory connection so writes are invisible
   across processes and the final row-count assertion can fail. Under
   --dist loadscope all six parametrized cases run sequentially on one worker,
   sharing the same in-memory backend via the module-scoped shared_db fixture.

CI / pytest configuration note
-------------------------------
These tests actively assert the --dist loadscope requirement.  If you run them
without ``-p xdist`` or with a different ``--dist`` strategy (e.g. ``--dist load``
or ``--dist no``), expect up to four failures that look like product regressions
but are actually a config mismatch.  The canonical invocation used in CI is::

    pytest -n auto --dist loadscope tests/integration/test_xdist_isolation.py

If you need to remove or change the ``--dist loadscope`` flag in CI, update or
remove this module first.
"""
from __future__ import annotations

import time
from typing import Sequence

import pytest
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter, SpanExportResult

from tokenjam.core.db import InMemoryBackend


# ---------------------------------------------------------------------------
# 1. OTel module-level collector
# ---------------------------------------------------------------------------

class _TestSpanCollector(SpanExporter):
    def __init__(self) -> None:
        self.collected_spans: list[ReadableSpan] = []

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        self.collected_spans.extend(spans)
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        pass


_collector = _TestSpanCollector()
_provider = TracerProvider()
_provider.add_span_processor(SimpleSpanProcessor(_collector))
_tracer = _provider.get_tracer("tokenjam.integration.isolation")


def test_otel_step_1_records_span() -> None:
    with _tracer.start_as_current_span("step_1_operation"):
        pass
    assert "step_1_operation" in [s.name for s in _collector.collected_spans]


def test_otel_step_2_records_span() -> None:
    with _tracer.start_as_current_span("step_2_operation"):
        pass
    assert "step_2_operation" in [s.name for s in _collector.collected_spans]


def test_otel_step_3_cumulative_spans_in_worker() -> None:
    """Key test: span_1 and span_2 are only visible here if all three steps
    ran on the same worker process (--dist loadscope). Under --dist load this
    worker's _collector may be empty for step_1/step_2."""
    with _tracer.start_as_current_span("step_3_operation"):
        pass
    span_names = [s.name for s in _collector.collected_spans]
    assert "step_1_operation" in span_names, (
        f"step_1_operation not found — step_3 ran on a different worker. "
        f"Spans visible here: {span_names}"
    )
    assert "step_2_operation" in span_names, (
        f"step_2_operation not found — step_3 ran on a different worker. "
        f"Spans visible here: {span_names}"
    )
    assert "step_3_operation" in span_names


# ---------------------------------------------------------------------------
# 2. DuckDB single-writer concurrency
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def shared_db():
    """Module-scoped in-memory backend shared by all DuckDB tests in this module.

    Sharing a single connection ensures writes from test_duckdb_concurrent_writer_lock
    are visible to test_duckdb_all_writes_completed. Under --dist loadscope all
    tests in the module run on the same worker, so the fixture is created once
    and torn down after the last test.
    """
    backend = InMemoryBackend()
    backend.conn.execute(
        "CREATE TABLE IF NOT EXISTS integration_events (step VARCHAR, ts DOUBLE)"
    )
    yield backend
    backend.close()


def _write_with_held_connection(db, step_name: str) -> None:
    """Inserts a row into integration_events using the provided backend."""
    db.conn.execute("INSERT INTO integration_events VALUES (?, ?)", [step_name, time.time()])


@pytest.mark.parametrize("step_num", range(1, 7))
def test_duckdb_concurrent_writer_lock(step_num: int, shared_db) -> None:
    """Under --dist load multiple workers call _write_with_held_connection()
    concurrently; because each worker has its own in-memory connection, writes
    are invisible across processes and the row-count guard below can fail.
    Under --dist loadscope all six cases run sequentially on one worker."""
    _write_with_held_connection(shared_db, f"step_{step_num}")


def test_duckdb_all_writes_completed(shared_db) -> None:
    """Guard: after all parametrized steps finish, at least 6 rows must exist.
    Under --dist load this may fail if dispatched to a worker that never wrote.
    Under --dist loadscope it always runs last on the same worker."""
    count = shared_db.conn.execute("SELECT count(*) FROM integration_events").fetchone()[0]
    assert count >= 6, f"Expected at least 6 rows, got {count}"
