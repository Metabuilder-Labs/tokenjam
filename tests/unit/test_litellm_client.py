"""Unit tests for TokenJamClient.emit_litellm_span (LiteLLM named-callback path)."""
from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import patch

import httpx
import pytest

from tokenjam.otel.semconv import GenAIAttributes, TjAttributes
from tokenjam.sdk.client import TokenJamClient, _build_litellm_span


def _attrs_to_dict(otlp_attrs: list[dict]) -> dict[str, object]:
    out: dict[str, object] = {}
    for a in otlp_attrs:
        v = a["value"]
        if "stringValue" in v:
            out[a["key"]] = v["stringValue"]
        elif "intValue" in v:
            out[a["key"]] = int(v["intValue"])
        elif "doubleValue" in v:
            out[a["key"]] = float(v["doubleValue"])
        elif "boolValue" in v:
            out[a["key"]] = v["boolValue"]
    return out


def _basic_response(provider: str = "openai", prompt: int = 10, completion: int = 5) -> object:
    return SimpleNamespace(
        usage=SimpleNamespace(prompt_tokens=prompt, completion_tokens=completion),
        _hidden_params={"custom_llm_provider": provider, "response_cost": 0.00042},
    )


# --- _build_litellm_span ---------------------------------------------------


def test_build_span_success_with_usage_and_cost():
    kwargs = {
        "model": "openai/gpt-4o-mini",
        "metadata": {"tj_agent_id": "agent-a", "tj_session_id": "sess-1"},
    }
    response = _basic_response()
    start = datetime(2026, 5, 12, tzinfo=timezone.utc)
    end = datetime(2026, 5, 12, 0, 0, 1, tzinfo=timezone.utc)

    span = _build_litellm_span(kwargs, response, start, end, success=True)

    assert span["name"] == GenAIAttributes.SPAN_LLM_CALL
    assert span["status"] == {"code": 1}
    assert len(span["traceId"]) == 32
    assert len(span["spanId"]) == 16

    attrs = _attrs_to_dict(span["attributes"])
    assert attrs[GenAIAttributes.REQUEST_MODEL] == "gpt-4o-mini"  # prefix stripped
    assert attrs[GenAIAttributes.PROVIDER_NAME] == "openai"
    assert attrs[GenAIAttributes.INPUT_TOKENS] == 10
    assert attrs[GenAIAttributes.OUTPUT_TOKENS] == 5
    assert attrs[TjAttributes.COST_USD] == pytest.approx(0.00042)
    assert attrs[GenAIAttributes.AGENT_ID] == "agent-a"
    assert attrs[GenAIAttributes.CONVERSATION_ID] == "sess-1"


def test_build_span_failure_records_error_status():
    kwargs = {"model": "anthropic/claude-3-5-sonnet"}
    err = RuntimeError("rate limited")
    start = datetime.now(timezone.utc)
    end = datetime.now(timezone.utc)

    span = _build_litellm_span(kwargs, err, start, end, success=False)
    assert span["status"]["code"] == 2
    assert span["status"]["message"] == "rate limited"


def test_build_span_falls_back_to_model_prefix_for_provider():
    # No _hidden_params on response — provider inferred from model prefix.
    kwargs = {"model": "anthropic/claude-3-5-sonnet"}
    response = SimpleNamespace(usage=None)
    span = _build_litellm_span(
        kwargs, response,
        datetime.now(timezone.utc), datetime.now(timezone.utc),
        success=True,
    )
    attrs = _attrs_to_dict(span["attributes"])
    assert attrs[GenAIAttributes.PROVIDER_NAME] == "anthropic"
    assert attrs[GenAIAttributes.REQUEST_MODEL] == "claude-3-5-sonnet"


def test_build_span_handles_dict_usage():
    kwargs = {"model": "gpt-4o"}
    response = {
        "usage": {
            "prompt_tokens": 7,
            "completion_tokens": 3,
            "cache_read_input_tokens": 100,
            "cache_creation_input_tokens": 50,
        },
        "_hidden_params": {"custom_llm_provider": "openai"},
    }
    span = _build_litellm_span(
        kwargs, response,
        datetime.now(timezone.utc), datetime.now(timezone.utc),
        success=True,
    )
    attrs = _attrs_to_dict(span["attributes"])
    assert attrs[GenAIAttributes.INPUT_TOKENS] == 7
    assert attrs[GenAIAttributes.OUTPUT_TOKENS] == 3
    assert attrs[GenAIAttributes.CACHE_READ_TOKENS] == 100
    assert attrs[GenAIAttributes.CACHE_CREATE_TOKENS] == 50


def test_build_span_missing_model_defaults_to_unknown():
    span = _build_litellm_span(
        {}, SimpleNamespace(usage=None),
        datetime.now(timezone.utc), datetime.now(timezone.utc),
        success=True,
    )
    attrs = _attrs_to_dict(span["attributes"])
    assert attrs[GenAIAttributes.REQUEST_MODEL] == "unknown"
    # Unresolvable model -> 'unknown' provider, never the bogus 'litellm' (#194).
    assert attrs[GenAIAttributes.PROVIDER_NAME] == "unknown"


# --- TokenJamClient HTTP behavior ------------------------------------------


def test_client_appends_api_v1_spans_to_base_endpoint():
    c = TokenJamClient(endpoint="http://localhost:7391")
    assert c._endpoint == "http://localhost:7391/api/v1/spans"

    c2 = TokenJamClient(endpoint="http://localhost:7391/api/v1/spans")
    assert c2._endpoint == "http://localhost:7391/api/v1/spans"

    c3 = TokenJamClient(endpoint="http://localhost:7391/")
    assert c3._endpoint == "http://localhost:7391/api/v1/spans"


def test_client_posts_otlp_payload_with_bearer_when_secret_set():
    c = TokenJamClient(endpoint="http://localhost:7391", ingest_secret="s3cret")

    captured: dict[str, object] = {}

    def fake_post(url, json, headers, timeout):
        captured["url"] = url
        captured["json"] = json
        captured["headers"] = headers
        return httpx.Response(200)

    with patch("tokenjam.sdk.client.httpx.post", side_effect=fake_post):
        c.emit_litellm_span(
            kwargs={"model": "openai/gpt-4o-mini"},
            response_obj=_basic_response(),
            start_time=datetime.now(timezone.utc),
            end_time=datetime.now(timezone.utc),
            success=True,
        )

    assert captured["url"] == "http://localhost:7391/api/v1/spans"
    assert captured["headers"]["Authorization"] == "Bearer s3cret"
    body = captured["json"]
    assert "resourceSpans" in body
    assert body["resourceSpans"][0]["scopeSpans"][0]["spans"][0]["name"] == GenAIAttributes.SPAN_LLM_CALL


def test_client_omits_authorization_when_no_secret():
    c = TokenJamClient(endpoint="http://localhost:7391")
    captured: dict[str, object] = {}

    def fake_post(url, json, headers, timeout):
        captured["headers"] = headers
        return httpx.Response(200)

    with patch("tokenjam.sdk.client.httpx.post", side_effect=fake_post):
        c.emit_litellm_span(
            kwargs={"model": "gpt-4o-mini"},
            response_obj=_basic_response(),
            start_time=datetime.now(timezone.utc),
            end_time=datetime.now(timezone.utc),
            success=True,
        )

    assert "Authorization" not in captured["headers"]


def test_client_swallows_connection_errors():
    c = TokenJamClient(endpoint="http://localhost:7391")
    with patch(
        "tokenjam.sdk.client.httpx.post",
        side_effect=httpx.ConnectError("boom"),
    ):
        # Must not raise.
        c.emit_litellm_span(
            kwargs={"model": "gpt-4o-mini"},
            response_obj=_basic_response(),
            start_time=datetime.now(timezone.utc),
            end_time=datetime.now(timezone.utc),
            success=True,
        )


def test_client_swallows_build_errors():
    """A malformed response should not propagate; the event is dropped."""
    c = TokenJamClient(endpoint="http://localhost:7391")
    with patch(
        "tokenjam.sdk.client._build_litellm_span",
        side_effect=ValueError("bad payload"),
    ):
        c.emit_litellm_span(
            kwargs={}, response_obj=None,
            start_time=datetime.now(timezone.utc),
            end_time=datetime.now(timezone.utc),
            success=True,
        )
