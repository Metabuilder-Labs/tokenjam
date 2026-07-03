"""Concurrency + shutdown-safety tests for the async post-ingest hook worker.

These exercise the `[alerts] async_hooks = true` path against a *real*
`DuckDBBackend` (not `InMemoryBackend`, whose dict writes are GIL-protected and
so can't reproduce a DuckDB write-write conflict). They cover the review
blockers:

  * Blocker 2 — concurrent writes: the ingest (main) thread writes spans/cost
    while the `TjHookWorker` thread writes alerts. Asserts no lost writes and no
    conflict errors surface.
  * Blocker 1 — shutdown drains the queue: `flush()` + `close()` must process
    every queued span before the worker exits.
  * Blocker 3 — bounded queue overflow drops the oldest span and *logs* it.
  * Default-off: a config without `async_hooks` runs hooks synchronously with no
    worker thread (identical to the historical pipeline).
"""

from __future__ import annotations

import logging
import threading

from tokenjam.core.config import StorageConfig, TjConfig
from tokenjam.core.db import DuckDBBackend
from tokenjam.core.ingest import HOOK_QUEUE_MAXSIZE, IngestPipeline
from tokenjam.core.models import Alert, AlertType, Severity
from tokenjam.utils.ids import new_uuid
from tokenjam.utils.time_parse import utcnow
from tests.factories import make_llm_span


def _duckdb(tmp_path) -> DuckDBBackend:
    return DuckDBBackend(StorageConfig(path=str(tmp_path / "concurrency.duckdb")))


class _AlertWritingEngine:
    """Stand-in AlertEngine whose evaluate() persists one alert per span.

    This forces the worker thread to WRITE to the DB on every span, which is the
    exact cross-thread write contention the real AlertEngine can produce when it
    fires. Using a real engine would need synthetic threshold-tripping spans;
    this is deterministic and hits the same `db.insert_alert` write path.
    """

    def __init__(self, db: DuckDBBackend) -> None:
        self.db = db

    def evaluate(self, span) -> None:
        self.db.insert_alert(
            Alert(
                alert_id=new_uuid(),
                fired_at=utcnow(),
                type=AlertType.TOKEN_ANOMALY,
                severity=Severity.INFO,
                title="synthetic",
                detail={"span_id": span.span_id},
                agent_id=span.agent_id,
                span_id=span.span_id,
            )
        )


def test_async_hooks_concurrent_writes_no_lost_writes(tmp_path):
    """Main thread inserts spans while the hook worker inserts alerts.

    Under async hooks the two threads write different tables of the same DuckDB
    database concurrently. With the write lock in place, every span is written
    and every span produces exactly one alert — no write-write conflict aborts a
    write, and none is silently swallowed by the hook error handler.
    """
    db = _duckdb(tmp_path)
    config = TjConfig(version="1")
    config.alerts.async_hooks = True

    pipeline = IngestPipeline(
        db=db,
        config=config,
        alert_engine=_AlertWritingEngine(db),
    )

    n_spans = 400
    try:
        for _ in range(n_spans):
            # process() writes the span (main thread) and enqueues the alert hook
            # (worker thread) — the two writes race on the same database.
            pipeline.process(make_llm_span())
        # Drain the hook queue: every enqueued alert must be persisted.
        pipeline.flush()
    finally:
        pipeline.close()

    span_count = db.conn.execute("SELECT COUNT(*) FROM spans").fetchone()[0]
    alert_count = db.conn.execute("SELECT COUNT(*) FROM alerts").fetchone()[0]

    assert span_count == n_spans, "an ingest-thread span write was lost"
    assert alert_count == n_spans, "a worker-thread alert write was lost/dropped"


def test_async_hooks_concurrent_two_producers(tmp_path):
    """Multiple ingest threads + the single hook worker all writing at once.

    Stress the write lock with several producer threads calling process()
    concurrently (each writes spans on its own DuckDB cursor) while the worker
    writes alerts. Asserts the totals reconcile with no lost writes.
    """
    db = _duckdb(tmp_path)
    config = TjConfig(version="1")
    config.alerts.async_hooks = True

    pipeline = IngestPipeline(
        db=db,
        config=config,
        alert_engine=_AlertWritingEngine(db),
    )

    per_thread = 150
    n_threads = 4
    errors: list[BaseException] = []

    def producer() -> None:
        try:
            for _ in range(per_thread):
                pipeline.process(make_llm_span())
        except BaseException as exc:  # pragma: no cover - failure path
            errors.append(exc)

    threads = [threading.Thread(target=producer) for _ in range(n_threads)]
    try:
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        pipeline.flush()
    finally:
        pipeline.close()

    assert not errors, f"producer thread raised: {errors!r}"
    total = per_thread * n_threads
    span_count = db.conn.execute("SELECT COUNT(*) FROM spans").fetchone()[0]
    alert_count = db.conn.execute("SELECT COUNT(*) FROM alerts").fetchone()[0]
    assert span_count == total
    assert alert_count == total


def test_close_drains_queued_hooks(tmp_path):
    """close() must process queued spans before the worker exits (blocker 1).

    A slow hook lets us queue several spans that are still pending when we shut
    down. close() (flush + sentinel drain) must run every one, not bail on the
    shutdown event and drop them.
    """
    db = _duckdb(tmp_path)
    config = TjConfig(version="1")
    config.alerts.async_hooks = True

    processed: list[str] = []
    gate = threading.Event()

    class _SlowEngine:
        def evaluate(self, span) -> None:
            # Block the very first hook until the gate opens, so the rest pile up
            # in the queue and are only drained during close().
            gate.wait(timeout=5.0)
            processed.append(span.span_id)

    pipeline = IngestPipeline(db=db, config=config, alert_engine=_SlowEngine())

    spans = [make_llm_span() for _ in range(20)]
    for span in spans:
        pipeline.process(span)

    # Release the worker, then flush + close — everything queued must be drained.
    gate.set()
    pipeline.flush()
    pipeline.close()

    assert len(processed) == len(spans), "close() dropped queued hooks on shutdown"
    assert set(processed) == {s.span_id for s in spans}


def test_flush_returns_when_worker_not_started(tmp_path):
    """flush() must not hang when async hooks were never enabled/started."""
    db = _duckdb(tmp_path)
    config = TjConfig(version="1")  # async_hooks default False
    pipeline = IngestPipeline(db=db, config=config)
    # No worker, no queue — flush()/close() are no-ops that return promptly.
    pipeline.flush()
    pipeline.close()
    assert pipeline._hook_queue is None
    assert pipeline._hook_thread is None


def test_bounded_queue_drops_oldest_and_logs(tmp_path, caplog):
    """Overflow drops the oldest queued span and logs it — never silent (blocker 3)."""
    db = _duckdb(tmp_path)
    config = TjConfig(version="1")
    config.alerts.async_hooks = True

    release = threading.Event()

    class _BlockingEngine:
        def evaluate(self, span) -> None:
            # Hold the worker so the queue fills up and overflow kicks in.
            release.wait(timeout=10.0)

    pipeline = IngestPipeline(db=db, config=config, alert_engine=_BlockingEngine())

    # Start the worker and let it pick up (and block on) the first span.
    pipeline.process(make_llm_span())
    # Fill the queue to capacity while the worker is blocked, then overflow it.
    overflow_by = 5
    with caplog.at_level(logging.WARNING, logger="tokenjam.ingest"):
        for _ in range(HOOK_QUEUE_MAXSIZE + overflow_by):
            pipeline.process(make_llm_span())

        assert pipeline._hook_dropped > 0, "overflow did not drop any span"
        assert any(
            "queue full" in r.message or "queue still full" in r.message
            for r in caplog.records
        ), "queue overflow was not logged"

    release.set()
    pipeline.close()


def test_default_off_is_synchronous(tmp_path):
    """A config without async_hooks runs hooks inline — no worker thread."""
    db = _duckdb(tmp_path)
    config = TjConfig(version="1")  # async_hooks defaults False

    ran: list[str] = []

    class _Engine:
        def evaluate(self, span) -> None:
            # If this ran on a worker thread it would be a different thread name.
            ran.append(threading.current_thread().name)

    pipeline = IngestPipeline(db=db, config=config, alert_engine=_Engine())
    pipeline.process(make_llm_span())

    # Ran synchronously on the caller (MainThread here), and no worker was spun up.
    assert ran == [threading.current_thread().name]
    assert pipeline._hook_queue is None
    assert pipeline._hook_thread is None
