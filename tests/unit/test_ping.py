"""`tj ping` — SDK test-span emitter with local proof-of-life (#80).

These tests exercise the command end-to-end through the real @watch()/
record_llm_call path against a live in-process TracerProvider, with only the
bootstrap delivery-mode and daemon confirmation stubbed (so no real daemon/DB
is required).
"""
from __future__ import annotations

import pytest
from click.testing import CliRunner
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider

from tokenjam.cli.cmd_ping import _ProofExporter
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


def _run(mode: str, monkeypatch, extra_args=None, confirm=(True, None)):
    """Run `tj ping` with bootstrap mode stubbed and, for HTTP mode, the
    daemon-confirmation poll stubbed to `confirm` (confirmed, error) so no
    real daemon/DB is needed."""
    monkeypatch.setattr("tokenjam.sdk.bootstrap.get_mode", lambda: mode)
    monkeypatch.setattr(
        "tokenjam.cli.cmd_ping._confirm_delivery", lambda *a, **k: confirm
    )
    return CliRunner().invoke(cli, ["ping", *(extra_args or [])])


def test_proof_exporter_records_span_names():
    exporter = _ProofExporter()
    span = type("S", (), {"name": "llm_call"})()
    exporter.export([span])
    assert exporter.captured == ["llm_call"]


def test_ping_http_mode_confirmed_delivery_exits_zero(monkeypatch):
    result = _run("http", monkeypatch, confirm=(True, None))
    assert result.exit_code == 0, result.output
    assert "intercepted a test span" in result.output
    assert "tj-ping-test" in result.output
    assert "confirmed received" in result.output


def test_ping_http_mode_unconfirmed_delivery_exits_nonzero(monkeypatch):
    """force_flush() succeeding is not enough — HTTP mode must also confirm
    the daemon actually stored the span (#410 review)."""
    result = _run("http", monkeypatch, confirm=(False, None))
    assert result.exit_code == 1, result.output
    assert "not confirmed received" in result.output
    assert "tj serve" in result.output


def test_ping_http_mode_confirm_error_surfaces_reason(monkeypatch):
    result = _run(
        "http", monkeypatch, confirm=(False, "the database is locked (is tj serve running?)")
    )
    assert result.exit_code == 1, result.output
    assert "not confirmed received" in result.output
    assert "database is locked" in result.output


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

    result = _run("http", monkeypatch, extra_args=["--json"], confirm=(True, None))
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["delivery_mode"] == "http"
    assert payload["model"] == "tj-ping-test"
    assert payload["intercepted"] is True
    assert payload["confirmed"] is True


def test_ping_json_output_reports_unconfirmed(monkeypatch):
    import json

    result = _run("http", monkeypatch, extra_args=["--json"], confirm=(False, None))
    assert result.exit_code == 1, result.output
    payload = json.loads(result.output)
    assert payload["confirmed"] is False


def test_ping_honors_custom_agent_id(monkeypatch):
    result = _run("http", monkeypatch, extra_args=["--agent", "my-real-agent"])
    assert result.exit_code == 0, result.output
    assert "my-real-agent" in result.output


def test_ping_passes_agent_id_and_emission_time_to_confirm(monkeypatch):
    """The confirmation poll must scope to this ping's own agent_id/since so it
    can't match a stale span from an earlier run."""
    captured = {}

    def _fake_confirm(config, agent_id, since):
        captured["agent_id"] = agent_id
        captured["since"] = since
        return True, None

    monkeypatch.setattr("tokenjam.sdk.bootstrap.get_mode", lambda: "http")
    monkeypatch.setattr("tokenjam.cli.cmd_ping._confirm_delivery", _fake_confirm)

    result = CliRunner().invoke(cli, ["ping", "--agent", "my-real-agent"])

    assert result.exit_code == 0, result.output
    assert captured["agent_id"] == "my-real-agent"
    assert captured["since"] is not None
