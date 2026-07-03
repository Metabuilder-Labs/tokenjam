from __future__ import annotations

from unittest.mock import MagicMock

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

    pipeline = IngestPipeline(
        db=db,
        config=config,
        cost_engine=cost_mock,
        alert_engine=alert_mock,
        schema_validator=schema_mock,
    )

    span = make_tool_span()
    pipeline.process(span)

    # CostEngine is synchronous and must be called immediately
    cost_mock.process_span.assert_called_once_with(span)

    # AlertEngine and SchemaValidator should NOT be called yet (deferred)
    alert_mock.evaluate.assert_not_called()
    schema_mock.validate.assert_not_called()

    # Queue and thread should be initialized
    assert pipeline._hook_queue is not None
    assert pipeline._hook_thread is not None

    # Wait for background queue to process (flushing)
    pipeline.flush()

    # After flush, the deferred hooks must have executed
    alert_mock.evaluate.assert_called_once_with(span)
    schema_mock.validate.assert_called_once_with(span)

    # Cleanup pipeline thread
    pipeline.close()
    assert pipeline._hook_thread is None


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
