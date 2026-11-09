"""Unit tests for the Langfuse ingest adapter."""
from __future__ import annotations

import base64
import json
from datetime import datetime, timezone
from pathlib import Path

import httpx
import pytest

from tokenjam.core.db import InMemoryBackend
from tokenjam.core.ingest_adapters.langfuse import (
    _fetch_url,
    _model_to_provider,
    _observation_to_span,
    ingest_langfuse,
)


FIXTURE_PATH = Path(__file__).parent.parent / "fixtures" / "langfuse_real_response.json"


@pytest.fixture
def db():
    backend = InMemoryBackend()
    yield backend
    backend.close()


# -- Pure-function tests --

def test_model_to_provider_anthropic():
    assert _model_to_provider("claude-sonnet-4-6") == ("anthropic", "anthropic")
    assert _model_to_provider("claude-opus-4-7-20260301") == ("anthropic", "anthropic")


def test_model_to_provider_openai():
    assert _model_to_provider("gpt-4o-mini") == ("openai", "openai")
    assert _model_to_provider("o3") == ("openai", "openai")
    assert _model_to_provider("o4-mini") == ("openai", "openai")


def test_model_to_provider_google():
    assert _model_to_provider("gemini-2-5-flash") == ("google", "google")


def test_model_to_provider_unknown_returns_none():
    assert _model_to_provider("some-random-model") == (None, None)
    assert _model_to_provider(None) == (None, None)


def test_observation_to_span_generation():
    obs = {
        "id": "obs-1",
        "traceId": "trace-1",
        "type": "GENERATION",
        "name": "chat",
        "startTime": "2026-04-01T10:00:00.000Z",
        "endTime": "2026-04-01T10:00:01.500Z",
        "model": "claude-sonnet-4-6",
        "usage": {"input": 1000, "output": 200},
        "calculatedTotalCost": 0.003,
        "sessionId": "sess-1",
    }
    span = _observation_to_span(obs)
    assert span is not None
    assert span.model == "claude-sonnet-4-6"
    assert span.provider == "anthropic"
    assert span.billing_account == "anthropic"
    assert span.input_tokens == 1000
    assert span.output_tokens == 200
    assert span.cost_usd == 0.003
    assert span.duration_ms == 1500.0
    assert span.conversation_id == "sess-1"
    assert span.request_type == "completion"


def test_observation_to_span_deterministic_ids():
    """Same (traceId, id) always produces the same TokenJam span_id."""
    obs = {
        "id": "obs-x", "traceId": "trace-x", "type": "GENERATION",
        "startTime": "2026-04-01T10:00:00Z", "endTime": "2026-04-01T10:00:01Z",
        "model": "claude-haiku-4-5",
    }
    span_a = _observation_to_span(obs)
    span_b = _observation_to_span(obs)
    assert span_a is not None and span_b is not None
    assert span_a.span_id == span_b.span_id
    assert span_a.trace_id == span_b.trace_id


def test_observation_to_span_missing_required_fields():
    """Observations without id, traceId, or startTime return None."""
    assert _observation_to_span({"id": "x"}) is None  # missing traceId
    assert _observation_to_span({"traceId": "x"}) is None  # missing id
    assert _observation_to_span({
        "id": "x", "traceId": "y",
    }) is None  # missing startTime


# -- Integration tests against the fixture --

def test_ingest_fixture_writes_spans(db):
    """The committed fixture parses cleanly and writes 4 spans."""
    result = ingest_langfuse(db, source_file=str(FIXTURE_PATH))
    assert result["observations_read"] == 4
    # 4 observations: 3 GENERATIONs + 1 SPAN. All produce spans (the SPAN has
    # a startTime so it parses).
    assert result["spans_written"] == 4
    assert result["spans_skipped"] == 0


def test_ingest_fixture_is_idempotent(db):
    """Re-running on the same fixture is a no-op (deterministic span IDs)."""
    first = ingest_langfuse(db, source_file=str(FIXTURE_PATH))
    second = ingest_langfuse(db, source_file=str(FIXTURE_PATH))
    assert first["spans_written"] == 4
    assert second["spans_written"] == 0
    assert second["spans_skipped"] == 4


def test_ingest_fixture_billing_accounts(db):
    """billing_account is derived from the model name per observation."""
    ingest_langfuse(db, source_file=str(FIXTURE_PATH))
    rows = db.conn.execute(
        "SELECT model, billing_account FROM spans WHERE model IS NOT NULL "
        "ORDER BY start_time"
    ).fetchall()
    by_model = {r[0]: r[1] for r in rows}
    assert by_model["claude-sonnet-4-6"] == "anthropic"
    assert by_model["gpt-4o-mini"] == "openai"
    assert by_model["gemini-2-5-flash"] == "google"


def test_ingest_since_filter(db, tmp_path):
    """--since filters out observations older than the cutoff."""
    # Use the fixture data spanning 2026-04-01 to 2026-04-02
    since = datetime(2026, 4, 2, 0, 0, tzinfo=timezone.utc)
    result = ingest_langfuse(db, source_file=str(FIXTURE_PATH), since=since)
    # Only the gemini-2-5-flash observation on 2026-04-02 survives the filter.
    assert result["spans_written"] == 1


def test_ingest_handles_bare_list_envelope(db, tmp_path):
    """A top-level JSON array (no {data: ...} wrapper) is accepted."""
    bare = [{
        "id": "obs-A", "traceId": "trace-A", "type": "GENERATION",
        "startTime": "2026-04-01T10:00:00Z", "endTime": "2026-04-01T10:00:01Z",
        "model": "claude-haiku-4-5",
        "usage": {"input": 100, "output": 20},
        "calculatedTotalCost": 0.0001,
    }]
    path = tmp_path / "bare.json"
    path.write_text(json.dumps(bare))
    result = ingest_langfuse(db, source_file=str(path))
    assert result["observations_read"] == 1
    assert result["spans_written"] == 1


def test_ingest_rejects_both_sources(db):
    """Passing both --source-url and --source-file raises."""
    with pytest.raises(ValueError, match="exactly one"):
        ingest_langfuse(db, source_url="http://x", source_file="/y")


def test_ingest_rejects_neither_source(db):
    """Passing neither raises."""
    with pytest.raises(ValueError, match="exactly one"):
        ingest_langfuse(db)


# -- Cache token threading (issue #93) --

def test_observation_to_span_threads_cache_write_tokens():
    """Cache-creation tokens (modern `usageDetails.input_cache_creation` key)
    flow through to `NormalizedSpan.cache_write_tokens`. Without this, the
    higher-rate cache-write cost was never charged on the Langfuse backfill
    path."""
    obs = {
        "id": "obs-cw",
        "traceId": "trace-cw",
        "type": "GENERATION",
        "startTime": "2026-04-01T10:00:00Z",
        "endTime": "2026-04-01T10:00:01Z",
        "model": "claude-haiku-4-5",
        "usageDetails": {
            "input": 100,
            "output": 20,
            "input_cache_read": 800,
            "input_cache_creation": 1500,
        },
        "calculatedTotalCost": 0.001,
    }
    span = _observation_to_span(obs)
    assert span is not None
    assert span.input_tokens == 100
    assert span.cache_tokens == 800
    assert span.cache_write_tokens == 1500


def test_observation_to_span_threads_cache_write_camelcase_key():
    """The legacy camelCase variant (`cacheCreationInputTokens`) is also
    recognised, mirroring the cacheReadInputTokens fallback."""
    obs = {
        "id": "obs-cw2",
        "traceId": "trace-cw2",
        "type": "GENERATION",
        "startTime": "2026-04-01T10:00:00Z",
        "endTime": "2026-04-01T10:00:01Z",
        "model": "claude-haiku-4-5",
        "usageDetails": {
            "cacheReadInputTokens": 800,
            "cacheCreationInputTokens": 1500,
        },
    }
    span = _observation_to_span(obs)
    assert span is not None
    assert span.cache_tokens == 800
    assert span.cache_write_tokens == 1500


def test_observation_to_span_cache_write_absent_leaves_field_none():
    """When the source doesn't carry cache-creation, the field stays None —
    not silently zero, so downstream can tell 'no data' from 'genuinely 0'."""
    obs = {
        "id": "obs-no-cw",
        "traceId": "trace-no-cw",
        "type": "GENERATION",
        "startTime": "2026-04-01T10:00:00Z",
        "endTime": "2026-04-01T10:00:01Z",
        "model": "claude-haiku-4-5",
        "usageDetails": {"input": 100, "output": 20},
    }
    span = _observation_to_span(obs)
    assert span is not None
    assert span.cache_write_tokens is None


def test_ingest_ndjson_input(db, tmp_path):
    """NDJSON files (whether compact or with double-quote starters) are parsed successfully."""
    ndjson_content = (
        '{"id": "obs-1", "traceId": "trace-1", "type": "GENERATION", '
        '"startTime": "2026-04-01T10:00:00.000Z", "endTime": "2026-04-01T10:00:01.500Z", '
        '"model": "claude-sonnet-4-6", "usage": {"input": 1000, "output": 200}}\n'
        '{"id": "obs-2", "traceId": "trace-1", "type": "GENERATION", '
        '"startTime": "2026-04-01T10:00:02.000Z", "endTime": "2026-04-01T10:00:03.500Z", '
        '"model": "claude-sonnet-4-6", "usage": {"input": 1000, "output": 200}}\n'
    )
    path = tmp_path / "dump.ndjson"
    path.write_text(ndjson_content)
    result = ingest_langfuse(db, source_file=str(path))
    assert result["observations_read"] == 2
    assert result["spans_written"] == 2


def test_ingest_malformed_ndjson_raises_valueerror(db, tmp_path):
    """A malformed line in an NDJSON file raises a descriptive ValueError."""
    path = tmp_path / "malformed.ndjson"
    path.write_text(
        '{"id": "obs-1", "traceId": "trace-1", "type": "GENERATION", "startTime": "2026-04-01T10:00:00.000Z"}\n{bad json}\n'
    )
    with pytest.raises(ValueError) as exc:
        ingest_langfuse(db, source_file=str(path))
    assert "Failed to parse NDJSON in" in str(exc.value)
    assert "at line 2" in str(exc.value)


# -- Live API fetch (mocked transport; no network) --

def _client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler), timeout=5.0)


def _v2_obs(i: int) -> dict:
    return {
        "id": f"obs-{i}",
        "traceId": f"trace-{i}",
        "type": "GENERATION",
        "name": "call",
        "startTime": "2026-11-01T10:00:00.000Z",
        "endTime": "2026-11-01T10:00:02.000Z",
        "model": "claude-sonnet-4-6",
        "usageDetails": {"input": 11, "output": 7, "total": 18},
        "totalCost": 0.0123,
    }


def test_fetch_url_uses_basic_auth_for_a_key_pair():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["auth"] = request.headers.get("authorization")
        return httpx.Response(200, json={"data": [], "meta": {}})

    _fetch_url("https://cloud.langfuse.com", "pk-lf-abc:sk-lf-def", None, client=_client(handler))
    expected = base64.b64encode(b"pk-lf-abc:sk-lf-def").decode()
    assert seen["auth"] == f"Basic {expected}"


def test_fetch_url_keeps_bearer_for_a_plain_key():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["auth"] = request.headers.get("authorization")
        return httpx.Response(200, json={"data": [], "meta": {}})

    _fetch_url("https://cloud.langfuse.com", "some-token", None, client=_client(handler))
    assert seen["auth"] == "Bearer some-token"


def test_fetch_url_reads_v2_observations_and_follows_cursors_through_empty_pages():
    pages = {
        None: {"data": [_v2_obs(1), _v2_obs(2)], "meta": {"cursor": "c1"}},
        # a page may be empty while a cursor is present: the loop must keep going
        "c1": {"data": [], "meta": {"cursor": "c2", "comment": "keep going"}},
        "c2": {"data": [_v2_obs(3)], "meta": {}},
    }
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        assert request.url.path == "/api/public/v2/observations"
        return httpx.Response(200, json=pages[request.url.params.get("cursor")])

    got = _fetch_url("https://cloud.langfuse.com", "pk:sk", None, client=_client(handler))
    assert [o["id"] for o in got] == ["obs-1", "obs-2", "obs-3"]
    assert len(requests) == 3
    first = requests[0].url.params
    assert "page" not in first
    assert int(first["limit"]) <= 1000
    assert {"core", "basic", "model", "usage"} <= set(first["fields"].split(","))


def test_fetch_url_sends_the_since_bound():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["from"] = request.url.params.get("fromStartTime")
        return httpx.Response(200, json={"data": [], "meta": {}})

    since = datetime(2026, 10, 1, tzinfo=timezone.utc)
    _fetch_url("https://cloud.langfuse.com", "pk:sk", since, client=_client(handler))
    assert seen["from"] == since.isoformat()


def test_fetch_url_accepts_a_url_that_already_names_the_old_endpoint():
    paths = []

    def handler(request: httpx.Request) -> httpx.Response:
        paths.append(request.url.path)
        return httpx.Response(200, json={"data": [], "meta": {}})

    _fetch_url("https://lf.example.com/api/public/observations", "pk:sk", None, client=_client(handler))
    assert paths == ["/api/public/v2/observations"]


def test_fetch_url_falls_back_to_v1_when_the_server_has_no_v2():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append((request.url.path, request.url.params.get("page")))
        if request.url.path == "/api/public/v2/observations":
            return httpx.Response(404, json={"message": "not found"})
        page = int(request.url.params["page"])
        data = [_v2_obs(page)]
        return httpx.Response(200, json={"data": data, "meta": {"totalPages": 2}})

    got = _fetch_url("https://self-hosted.example.com", "pk:sk", None, client=_client(handler))
    assert [o["id"] for o in got] == ["obs-1", "obs-2"]
    assert calls[0][0] == "/api/public/v2/observations"
    assert [c for c in calls[1:]] == [("/api/public/observations", "1"), ("/api/public/observations", "2")]


def test_fetch_url_does_not_hide_an_auth_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"message": "unauthorized"})

    with pytest.raises(httpx.HTTPStatusError):
        _fetch_url("https://cloud.langfuse.com", "pk:sk", None, client=_client(handler))


def test_v2_observation_maps_to_a_span():
    span = _observation_to_span(_v2_obs(1))
    assert span is not None
    assert (span.input_tokens, span.output_tokens) == (11, 7)
    assert span.cost_usd == pytest.approx(0.0123)
    assert span.model == "claude-sonnet-4-6"
