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
        time.sleep(0.05)
    assert "step_1_operation" in [s.name for s in _collector.collected_spans]


def test_otel_step_2_records_span() -> None:
    with _tracer.start_as_current_span("step_2_operation"):
        time.sleep(0.05)
    assert "step_2_operation" in [s.name for s in _collector.collected_spans]


def test_otel_step_3_cumulative_spans_in_worker() -> None:
    """Key test: span_1 and span_2 are only visible here if all three steps
    ran on the same worker process (--dist loadscope). Under --dist load this
    worker's _collector may be empty for step_1/step_2."""
    with _tracer.start_as_current_span("step_3_operation"):
        time.sleep(0.05)
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

_SHARED_DB_PATH = os.path.join(
    tempfile.gettempdir(), "tokenjam_xdist_isolation_primary.duckdb"
)


@pytest.fixture(scope="module", autouse=True)
def _init_shared_db() -> None:
    """Create the shared schema once per worker. Retries on IOException because
    multiple workers may race to init simultaneously under --dist load."""
    for attempt in range(10):
        try:
            db = DuckDBBackend(StorageConfig(path=_SHARED_DB_PATH))
            db.conn.execute(
                "CREATE TABLE IF NOT EXISTS integration_events (step VARCHAR, ts DOUBLE)"
            )
            db.close()
            return
        except Exception:  # noqa: BLE001
            time.sleep(0.05 * (attempt + 1))
    raise RuntimeError(f"Could not initialise {_SHARED_DB_PATH} after 10 retries")


def _write_with_held_connection(step_name: str) -> None:
    """Opens a DuckDBBackend write connection, holds it for 250 ms (simulating
    real work), inserts a row, then closes. Two concurrent workers calling this
    both open the same file while the other holds the write lock, producing:
        IOException: Could not set lock on file "...": Conflicting lock is held
    """
    db = DuckDBBackend(StorageConfig(path=_SHARED_DB_PATH))
    time.sleep(0.25)
    db.conn.execute("INSERT INTO integration_events VALUES (?, ?)", [step_name, time.time()])
    db.close()


@pytest.mark.parametrize("step_num", range(1, 7))
def test_duckdb_concurrent_writer_lock(step_num: int) -> None:
    """Under --dist load multiple workers call _write_with_held_connection()
    concurrently and one of them raises IOException on the locked file.
    Under --dist loadscope all six cases run sequentially on one worker."""
    _write_with_held_connection(f"step_{step_num}")


def test_duckdb_all_writes_completed() -> None:
    """Guard: after all parametrized steps finish, at least 6 rows must exist.
    Under --dist load this may also fail if dispatched before the writes finish
    or on a worker that never wrote. Under --dist loadscope it always runs last
    on the same worker."""
    db = DuckDBBackend(StorageConfig(path=_SHARED_DB_PATH))
    count = db.conn.execute("SELECT count(*) FROM integration_events").fetchone()[0]
    db.close()
    assert count >= 6, f"Expected at least 6 rows, got {count}"
