from __future__ import annotations

import threading
from unittest.mock import MagicMock, patch

import duckdb

from tokenjam.core.config import _parse, TjConfig
from tokenjam.core.db import InMemoryBackend
from tokenjam.core.ingest import IngestPipeline
from tests.factories import make_tool_span


def test_async_hooks_config_parsing():
    # Test default
    config = _parse({})
    assert config.alerts.async_hooks is False

    # Test explicit True
    config = _parse({"alerts": {"async_hooks": True}})
    assert config.alerts.async_hooks is True

    # Test explicit False
    config = _parse({"alerts": {"async_hooks": False}})
    assert config.alerts.async_hooks is False


def test_sync_hooks_execution():
    config = TjConfig(version="1")
    config.alerts.async_hooks = False

    db = InMemoryBackend()
    cost_mock = MagicMock()
    alert_mock = MagicMock()
    schema_mock = MagicMock()

    pipeline = IngestPipeline(
        db=db,
        config=config,
        cost_engine=cost_mock,
        alert_engine=alert_mock,
        schema_validator=schema_mock,
    )

    span = make_tool_span()
    pipeline.process(span)

    # In sync mode, all hooks run immediately
    cost_mock.process_span.assert_called_once_with(span)
    alert_mock.evaluate.assert_called_once_with(span)
    schema_mock.validate.assert_called_once_with(span)

    # Thread and queue should not be initialized
    assert pipeline._hook_queue is None
    assert pipeline._hook_thread is None


def test_async_hooks_execution():
    config = TjConfig(version="1")
    config.alerts.async_hooks = True

    db = InMemoryBackend()
    cost_mock = MagicMock()
    alert_mock = MagicMock()
    schema_mock = MagicMock()

    started = threading.Event()
    release = threading.Event()

    def block_span_evaluation(_span):
        started.set()
        assert release.wait(timeout=5)

    alert_mock.evaluate.side_effect = block_span_evaluation

    pipeline = IngestPipeline(
        db=db,
        config=config,
        cost_engine=cost_mock,
        alert_engine=alert_mock,
        schema_validator=schema_mock,
    )

    try:
        span = make_tool_span()
        pipeline.process(span)

        # CostEngine is synchronous and must be called immediately.
        cost_mock.process_span.assert_called_once_with(span)

        # Hold the worker inside the first advisory hook. This makes the
        # deferred-vs-inline assertion deterministic instead of racing the
        # worker after process() returns.
        assert started.wait(timeout=5)
        alert_mock.evaluate.assert_called_once_with(span)
        schema_mock.validate.assert_not_called()

        # Queue and thread should be initialized.
        assert pipeline._hook_queue is not None
        assert pipeline._hook_thread is not None

        release.set()
        pipeline.flush()

        # After flush, the deferred hooks must have executed.
        schema_mock.validate.assert_called_once_with(span)
    finally:
        release.set()
        pipeline.close()
    assert pipeline._hook_thread is None


def test_async_session_progress_is_deferred_with_advisory_hooks():
    config = TjConfig(version="1")
    config.alerts.async_hooks = True

    db = InMemoryBackend()
    alert_mock = MagicMock()
    started = threading.Event()
    release = threading.Event()

    def block_span_evaluation(_span):
        started.set()
        assert release.wait(timeout=5)

    alert_mock.evaluate.side_effect = block_span_evaluation
    pipeline = IngestPipeline(db=db, config=config, alert_engine=alert_mock)

    try:
        span = make_tool_span(session_id="async-session")
        pipeline.process(span)

        # Hold the advisory worker after per-span evaluation. Session progress
        # must not run on the ingest thread before the worker is released.
        assert started.wait(timeout=5)
        alert_mock.evaluate_session_progress.assert_not_called()

        release.set()
        pipeline.flush()
        alert_mock.evaluate_session_progress.assert_called_once()
    finally:
        release.set()
        pipeline.close()


def test_session_progress_fatal_storage_error_uses_fatal_handler():
    config = TjConfig(version="1")
    db = InMemoryBackend()
    alert_mock = MagicMock()
    pipeline = IngestPipeline(db=db, config=config, alert_engine=alert_mock)
    db.get_session = MagicMock(side_effect=duckdb.FatalException("FATAL Error: broken"))

    with patch("tokenjam.core.db.handle_if_fatal", return_value=True) as handler:
        pipeline._evaluate_session_progress(make_tool_span(session_id="fatal-session"))

    handler.assert_called_once()
    alert_mock.evaluate_session_progress.assert_not_called()


def test_async_hooks_error_tolerance(caplog):
    config = TjConfig(version="1")
    config.alerts.async_hooks = True

    db = InMemoryBackend()
    cost_mock = MagicMock()
    
    # Mock alert engine to throw exception
    alert_mock = MagicMock()
    alert_mock.evaluate.side_effect = Exception("Boom!")
    
    schema_mock = MagicMock()

    pipeline = IngestPipeline(
        db=db,
        config=config,
        cost_engine=cost_mock,
        alert_engine=alert_mock,
        schema_validator=schema_mock,
    )

    span = make_tool_span()
    pipeline.process(span)
    
    # Wait for execution
    pipeline.flush()

    # Boom exception should be swallowed and logged, and not crash pipeline/thread
    # Also schema validator should still run even if alert engine failed
    schema_mock.validate.assert_called_once_with(span)
    
    assert any("AlertEngine hook failed: Boom!" in record.message for record in caplog.records)

    pipeline.close()
