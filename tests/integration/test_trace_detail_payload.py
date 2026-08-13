"""GET /api/v1/traces/{id} must not ship ~1 GB for a big fan-out trace (#653).

A 45,384-span trace returned every span with its full captured-content
`attributes` dict — a 951 MB JSON payload the browser could neither fetch nor
render, so the detail pane hung forever. The Lens waterfall now requests the
capped + attribute-free payload via ?attributes=false; captured content for a
single span is fetched lazily via GET /traces/{id}/spans/{span_id}.

The DEFAULT response (no param) keeps FULL attributes so
`ApiBackend.get_trace_spans` and every existing complete-span consumer are
unchanged (#659 P1-1).
"""
from __future__ import annotations

import json

import httpx
import pytest

from tokenjam.api.app import create_app
from tokenjam.api.routes.traces import TRACE_SPAN_CAP
from tokenjam.core.config import TjConfig
from tokenjam.core.db import InMemoryBackend
from tokenjam.core.ingest import IngestPipeline
from tests.factories import make_llm_span

# A big captured-content blob so the OLD behaviour (ship every span's
# attributes) would produce a huge payload — this is what the fix removes.
_BIG_CONTENT = "x" * 4000


def _app(db, config):
    return create_app(config=config, db=db, ingest_pipeline=IngestPipeline(db=db, config=config))


def _seed_large_trace(db, n_spans: int, trace_id: str = "big-trace") -> None:
    for i in range(n_spans):
        db.insert_span(make_llm_span(
            trace_id=trace_id,
            span_id=f"span-{i:06d}",
            cost_usd=float(i),  # ascending so the last spans are the costliest
            extra_attributes={"gen_ai.prompt.content": _BIG_CONTENT, "idx": i},
        ))


@pytest.mark.asyncio
async def test_large_trace_is_capped_attribute_free_and_small():
    db = InMemoryBackend()
    cfg = TjConfig(version="1")
    n = 3000
    _seed_large_trace(db, n)
    transport = httpx.ASGITransport(app=_app(db, cfg))
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
        resp = await c.get("/api/v1/traces/big-trace?attributes=false")
    assert resp.status_code == 200
    body = resp.json()

    # True total is disclosed, the returned list is capped, and it says so.
    assert body["span_count"] == n
    assert body["truncated"] is True
    # At most the cap, plus up to the 5 costliest spans that are always kept.
    assert body["returned_count"] <= TRACE_SPAN_CAP + 5
    assert len(body["spans"]) == body["returned_count"]

    # No span in the waterfall payload carries the heavy `attributes` dict.
    assert all("attributes" not in s for s in body["spans"])
    # But it still tells the UI which spans HAVE attributes to fetch lazily.
    assert all("has_attributes" in s for s in body["spans"])

    # The payload is small — the whole point of the fix. The old shape would be
    # ~3000 * 4KB = many MB; capped + attribute-free it must be well under 5 MB.
    payload_bytes = len(resp.content)
    assert payload_bytes < 5 * 1024 * 1024, f"payload too big: {payload_bytes} bytes"

    # The costliest spans are always included even when truncated (they're the
    # last-indexed spans here), so the "jump to costliest" badges resolve.
    returned_ids = {s["span_id"] for s in body["spans"]}
    assert set(body["top_cost_span_ids"]).issubset(returned_ids)
    assert len(body["top_cost_span_ids"]) == 5


@pytest.mark.asyncio
async def test_small_trace_is_not_truncated():
    db = InMemoryBackend()
    cfg = TjConfig(version="1")
    _seed_large_trace(db, 3, trace_id="small")
    transport = httpx.ASGITransport(app=_app(db, cfg))
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
        resp = await c.get("/api/v1/traces/small?attributes=false")
    body = resp.json()
    assert body["span_count"] == 3
    assert body["truncated"] is False
    assert body["returned_count"] == 3
    # Still attribute-free in the light (opt-in) waterfall payload.
    assert body["attributes_included"] is False
    assert all("attributes" not in s for s in body["spans"])


@pytest.mark.asyncio
async def test_default_trace_payload_carries_full_attributes():
    """#659 P1-1: the DEFAULT /traces/{id} (no param) keeps full attributes so
    ApiBackend.get_trace_spans and other complete-span consumers are unchanged.
    The attribute-free payload is OPT-IN via ?attributes=false."""
    db = InMemoryBackend()
    cfg = TjConfig(version="1")
    _seed_large_trace(db, 3, trace_id="full")
    transport = httpx.ASGITransport(app=_app(db, cfg))
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
        resp = await c.get("/api/v1/traces/full")
    assert resp.status_code == 200
    body = resp.json()
    assert body["attributes_included"] is True
    # Every span carries its captured content by default.
    assert all("attributes" in s for s in body["spans"])
    assert all(
        s["attributes"]["gen_ai.prompt.content"] == _BIG_CONTENT for s in body["spans"]
    )


@pytest.mark.asyncio
async def test_single_span_endpoint_returns_attributes():
    db = InMemoryBackend()
    cfg = TjConfig(version="1")
    _seed_large_trace(db, 10, trace_id="tr")
    transport = httpx.ASGITransport(app=_app(db, cfg))
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
        resp = await c.get("/api/v1/traces/tr/spans/span-000003")
    assert resp.status_code == 200
    span = resp.json()
    assert span["span_id"] == "span-000003"
    # The lazy endpoint DOES carry the captured content.
    assert span["attributes"]["gen_ai.prompt.content"] == _BIG_CONTENT


@pytest.mark.asyncio
async def test_single_span_endpoint_404_for_unknown_span():
    db = InMemoryBackend()
    cfg = TjConfig(version="1")
    _seed_large_trace(db, 5, trace_id="tr")
    transport = httpx.ASGITransport(app=_app(db, cfg))
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
        resp = await c.get("/api/v1/traces/tr/spans/does-not-exist")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_single_span_endpoint_does_targeted_fetch_not_full_trace():
    """#659 P1-2: expanding a span must NOT load the whole trace. The route uses
    get_span (a WHERE span_id=? lookup); it must not fall back to get_trace_spans
    (which deserializes every span's attributes) when get_span exists."""
    db = InMemoryBackend()
    cfg = TjConfig(version="1")
    _seed_large_trace(db, 50, trace_id="tr")

    calls = {"get_span": 0, "get_trace_spans": 0}
    real_get_span = db.get_span
    real_get_trace_spans = db.get_trace_spans

    def spy_get_span(trace_id, span_id):
        calls["get_span"] += 1
        return real_get_span(trace_id, span_id)

    def spy_get_trace_spans(trace_id):
        calls["get_trace_spans"] += 1
        return real_get_trace_spans(trace_id)

    db.get_span = spy_get_span  # type: ignore[method-assign]
    db.get_trace_spans = spy_get_trace_spans  # type: ignore[method-assign]

    transport = httpx.ASGITransport(app=_app(db, cfg))
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
        resp = await c.get("/api/v1/traces/tr/spans/span-000010")
    assert resp.status_code == 200
    assert calls["get_span"] == 1
    assert calls["get_trace_spans"] == 0, "expand must not scan the whole trace"


def test_duckdb_get_span_targeted_and_404():
    """#659 P1-2: DuckDBBackend.get_span returns the one row or None."""
    from tokenjam.core.config import StorageConfig
    from tokenjam.core.db import DuckDBBackend
    import tempfile
    import os

    fd, path = tempfile.mkstemp(suffix=".duckdb")
    os.close(fd)
    os.unlink(path)
    db = DuckDBBackend(StorageConfig(path=path))
    try:
        _seed_large_trace(db, 5, trace_id="tr")
        span = db.get_span("tr", "span-000002")
        assert span is not None and span.span_id == "span-000002"
        assert span.attributes["gen_ai.prompt.content"] == _BIG_CONTENT
        assert db.get_span("tr", "nope") is None
        # Scoped to the trace: a real span id under the wrong trace is a miss.
        assert db.get_span("other", "span-000002") is None
    finally:
        db.close()
        if os.path.exists(path):
            os.unlink(path)
