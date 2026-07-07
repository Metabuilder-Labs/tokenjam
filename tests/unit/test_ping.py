"""`tj ping` — SDK test-span emitter with local proof-of-life (#80).

These tests exercise the command end-to-end through the real @watch()/
record_llm_call path against a live in-process TracerProvider, with only the
bootstrap delivery-mode stubbed (so no real daemon/DB is required).
"""
from __future__ import annotations

import pytest
from click.testing import CliRunner
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider

from tokenjam.cli.cmd_ping import _ProofExporter, cmd_ping
from tokenjam.cli.main import cli


@pytest.fixture(autouse=True)
def _real_provider():
    """Ensure a real SDK TracerProvider is the global one so the proof exporter
    (a SimpleSpanProcessor) can attach and capture the emitted span. OTel only
    honors the first set_tracer_provider per process, but any previously-set SDK
    provider is equally fine — both expose add_span_processor."""
    provider = trace.get_tracer_provider()
    if not hasattr(provider, "add_span_processor"):
        trace.set_tracer_provider(TracerProvider())


@pytest.fixture(autouse=True)
def _stub_bootstrap(monkeypatch):
    """Don't touch a real config/DB/daemon during ping tests."""
    monkeypatch.setattr("tokenjam.sdk.bootstrap.ensure_initialised", lambda: None)


def _run(mode: str, monkeypatch, extra_args=None):
    monkeypatch.setattr("tokenjam.sdk.bootstrap.get_mode", lambda: mode)
    return CliRunner().invoke(cli, ["ping", *(extra_args or [])])


def test_proof_exporter_records_span_names():
    exporter = _ProofExporter()
    span = type("S", (), {"name": "llm_call"})()
    exporter.export([span])
    assert exporter.captured == ["llm_call"]


def test_ping_http_mode_confirms_interception_and_delivery(monkeypatch):
    result = _run("http", monkeypatch)
    assert result.exit_code == 0, result.output
    assert "intercepted a test span" in result.output
    assert "tj-ping-test" in result.output
    assert "tj serve" in result.output


def test_ping_direct_mode_reports_local_db(monkeypatch):
    result = _run("direct", monkeypatch)
    assert result.exit_code == 0, result.output
    assert "intercepted a test span" in result.output
    assert "local database" in result.output


def test_ping_failed_mode_warns_and_exits_nonzero(monkeypatch):
    result = _run("failed", monkeypatch)
    assert result.exit_code == 1, result.output
    assert "Could not reach" in result.output


def test_ping_json_output_is_machine_readable(monkeypatch):
    import json

    result = _run("http", monkeypatch, extra_args=["--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["delivery_mode"] == "http"
    assert payload["model"] == "tj-ping-test"
    assert payload["intercepted"] is True


def test_ping_honors_custom_agent_id(monkeypatch):
    result = _run("http", monkeypatch, extra_args=["--agent", "my-real-agent"])
    assert result.exit_code == 0, result.output
    assert "my-real-agent" in result.output
