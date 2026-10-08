"""`TjHttpExporter` Authorization-header construction.

Regression guard for the `tj ping` footgun where an empty ingest secret
produced ``Authorization: Bearer `` — an illegal HTTP header value (leading/
trailing whitespace is forbidden by RFC 9110). httpx/h11 raise
``LocalProtocolError: Illegal header value b'Bearer '`` at send time, so the
OTLP span export failed outright and `tj ping` reported "emitted … but not
confirmed received".

Contract enforced here:
- non-empty secret  -> ``Authorization: Bearer <secret>`` (Critical Rule 4)
- empty / missing   -> no ``Authorization`` header at all (never ``Bearer ``)
"""
from __future__ import annotations

import pytest

from tokenjam.sdk.http_exporter import TjHttpExporter

ENDPOINT = "http://127.0.0.1:7391/api/v1/spans"


def test_header_is_well_formed_bearer_when_secret_present() -> None:
    secret = "0af032cb1234567890abcdef"
    exporter = TjHttpExporter(ENDPOINT, secret)

    assert exporter._headers["Authorization"] == f"Bearer {secret}"
    # Well-formed: exactly one space, no trailing/leading whitespace in value.
    value = exporter._headers["Authorization"]
    assert value == value.strip()
    assert value.startswith("Bearer ")
    assert value[len("Bearer "):] == secret


@pytest.mark.parametrize("empty_secret", ["", None, " ", "   ", "\t", "\n"])
def test_no_authorization_header_when_secret_missing(empty_secret) -> None:
    # ``None`` can slip through if a caller passes an unset config field, and a
    # whitespace-only secret is treated as absent (#431) — otherwise
    # ``Bearer  `` (a stray space) is the same illegal header value.
    exporter = TjHttpExporter(ENDPOINT, empty_secret)  # type: ignore[arg-type]

    # Never emit an empty / whitespace Bearer — omit the header entirely.
    assert "Authorization" not in exporter._headers
    assert all(not v.startswith("Bearer ") for v in exporter._headers.values())
    assert exporter._headers.get("Content-Type") == "application/json"


# ---------------------------------------------------------------------------
# service.name derivation from gen_ai.agent.id (PR: fix http_exporter)
# ---------------------------------------------------------------------------

from unittest.mock import patch, MagicMock
from opentelemetry.sdk.trace import TracerProvider, ReadableSpan
from opentelemetry.sdk.resources import Resource
from tokenjam.otel.semconv import GenAIAttributes


def _make_span(agent_id: str | None = None) -> ReadableSpan:
    """Build a minimal ended ReadableSpan with optional gen_ai.agent.id."""
    provider = TracerProvider(resource=Resource.create({}))
    tracer = provider.get_tracer("test")
    attrs = {}
    if agent_id is not None:
        attrs[GenAIAttributes.AGENT_ID] = agent_id
    span = tracer.start_span("gen_ai.llm.call", attributes=attrs)
    span.end()
    return span  # type: ignore[return-value]


def test_export_uses_agent_id_as_service_name() -> None:
    """gen_ai.agent.id on the span becomes service.name on the wire."""
    exporter = TjHttpExporter(ENDPOINT, "secret")
    captured: list[dict] = []

    def fake_post(url, *, json, headers, timeout):
        captured.append(json)
        resp = MagicMock()
        resp.status_code = 200
        return resp

    with patch("tokenjam.sdk.http_exporter.httpx.post", side_effect=fake_post):
        exporter.export([_make_span(agent_id="my-agent")])

    assert len(captured) == 1
    resource_spans = captured[0]["resourceSpans"]
    assert len(resource_spans) == 1
    attrs = {a["key"]: a["value"]["stringValue"]
             for a in resource_spans[0]["resource"]["attributes"]}
    assert attrs["service.name"] == "my-agent"


def test_export_falls_back_to_tokenjam_when_no_agent_id() -> None:
    """Spans without gen_ai.agent.id fall back to service.name=tokenjam."""
    exporter = TjHttpExporter(ENDPOINT, "secret")
    captured: list[dict] = []

    def fake_post(url, *, json, headers, timeout):
        captured.append(json)
        resp = MagicMock()
        resp.status_code = 200
        return resp

    with patch("tokenjam.sdk.http_exporter.httpx.post", side_effect=fake_post):
        exporter.export([_make_span(agent_id=None)])

    resource_spans = captured[0]["resourceSpans"]
    attrs = {a["key"]: a["value"]["stringValue"]
             for a in resource_spans[0]["resource"]["attributes"]}
    assert attrs["service.name"] == "tokenjam"


def test_export_groups_spans_by_agent_id() -> None:
    """Spans from different agents are sent in separate resourceSpans entries."""
    exporter = TjHttpExporter(ENDPOINT, "secret")
    captured: list[dict] = []

    def fake_post(url, *, json, headers, timeout):
        captured.append(json)
        resp = MagicMock()
        resp.status_code = 200
        return resp

    spans = [
        _make_span(agent_id="agent-a"),
        _make_span(agent_id="agent-b"),
        _make_span(agent_id="agent-a"),
    ]

    with patch("tokenjam.sdk.http_exporter.httpx.post", side_effect=fake_post):
        exporter.export(spans)

    resource_spans = captured[0]["resourceSpans"]
    service_names = {
        next(a["value"]["stringValue"] for a in rs["resource"]["attributes"]
             if a["key"] == "service.name")
        for rs in resource_spans
    }
    assert service_names == {"agent-a", "agent-b"}
    # agent-a has 2 spans, agent-b has 1
    counts = {
        next(a["value"]["stringValue"] for a in rs["resource"]["attributes"]
             if a["key"] == "service.name"): len(rs["scopeSpans"][0]["spans"])
        for rs in resource_spans
    }
    assert counts["agent-a"] == 2
    assert counts["agent-b"] == 1
