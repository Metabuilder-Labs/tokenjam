"""Integration tests that demonstrate why --dist loadscope (not --dist load) is required.

Two isolation properties under test:

1. OTel module-level span collector
   Each xdist worker imports the module fresh, so _TestSpanCollector is a
   different object per process. Under --dist load, step_3 may land on a
   different worker than step_1/step_2, whose _collector is empty, so the
   cumulative-span assertion fails. Under --dist loadscope all three steps
   run on the same worker.

2. DuckDB single-writer file lock
   DuckDB allows at most one writer per file at a time. Under --dist load,
   parametrized cases can run concurrently on different workers; each opens
   the shared file and holds the write connection for 250 ms, so concurrent
   workers hit duckdb.IOException. Under --dist loadscope all six parametrized
   cases run sequentially on one worker.
"""
from __future__ import annotations

import os
import tempfile
import time
from typing import Sequence

import pytest
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter, SpanExportResult

from tokenjam.core.config import StorageConfig
from tokenjam.core.db import DuckDBBackend
from tokenjam.core.db import InMemoryBackend


@pytest.fixture
def db():
    backend = InMemoryBackend()
    yield backend
    backend.close()

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
# 2. DuckDB single-writer lock
#
# Fixed path so all xdist workers target the same file simultaneously.
# mkdtemp() at module-import time would give each worker a different directory.
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def shared_db():
    """Module-scoped backend shared by all DuckDB tests in this module.

    A single in-memory DuckDB connection is shared so that writes from
    test_duckdb_concurrent_writer_lock are visible to test_duckdb_all_writes_completed.
    Under a file-based backend with concurrent xdist workers this fixture would
    instead open the same on-disk file, which is where the lock-contention
    described in the module docstring would manifest.
    """
    backend = InMemoryBackend()
    backend.conn.execute(
        "CREATE TABLE IF NOT EXISTS integration_events (step VARCHAR, ts DOUBLE)"
    )
    yield backend
    backend.close()


@pytest.fixture(scope="module", autouse=True)
def _init_shared_db(shared_db) -> None:
    """Ensure the shared schema is initialised before any test in this module runs."""
    pass  # schema creation is handled inside shared_db


def _write_with_held_connection(db, step_name: str) -> None:
    """Inserts a row into integration_events using the provided backend.

    With a file-based DuckDB backend, holding the write connection for real work
    while a second worker opens the same file would produce:
        IOException: Could not set lock on file "...": Conflicting lock is held
    Under --dist loadscope all parametrized cases run sequentially on one worker,
    so the lock is never contested.
    """
    db.conn.execute("INSERT INTO integration_events VALUES (?, ?)", [step_name, time.time()])


@pytest.mark.parametrize("step_num", range(1, 7))
def test_duckdb_concurrent_writer_lock(step_num: int, shared_db) -> None:
    """Under --dist load multiple workers call _write_with_held_connection()
    concurrently and one of them raises IOException on the locked file.
    Under --dist loadscope all six cases run sequentially on one worker."""
    _write_with_held_connection(shared_db, f"step_{step_num}")


def test_duckdb_all_writes_completed(shared_db) -> None:
    """Guard: after all parametrized steps finish, at least 6 rows must exist.
    Under --dist load this may also fail if dispatched before the writes finish
    or on a worker that never wrote. Under --dist loadscope it always runs last
    on the same worker."""
    count = shared_db.conn.execute("SELECT count(*) FROM integration_events").fetchone()[0]
    assert count >= 6, f"Expected at least 6 rows, got {count}"