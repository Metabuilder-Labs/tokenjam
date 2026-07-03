"""Tests for session renames (#306): the DB overlay helpers, the
POST /api/v1/sessions/{id}/label endpoint, and its precedence + surfacing in
/status.

A rename is a dashboard action persisted to the `session_labels` table
(migration 15) and overlaid onto the /status tile/archive label — beating the
OTel service.instance.id but NOT a config `[session_labels]` entry.
"""
from __future__ import annotations

import httpx
import pytest

from tokenjam.api.app import create_app
from tokenjam.core.config import (
    ApiAuthConfig,
    ApiConfig,
    SecurityConfig,
    TjConfig,
)
from tokenjam.core.db import (
    InMemoryBackend,
    delete_session_label,
    get_session_labels,
    set_session_label,
)
from tokenjam.core.ingest import IngestPipeline
from tests.factories import make_llm_span


INGEST_SECRET = "test-secret-token"


# ── DB helpers: set / get / delete round-trip ──────────────────────────────

def test_label_helpers_round_trip():
    db = InMemoryBackend()
    try:
        # Empty to start.
        assert get_session_labels(db.conn) == {}

        set_session_label(db, "sid-1", "alpha")
        set_session_label(db, "sid-2", "beta")
        assert get_session_labels(db.conn) == {"sid-1": "alpha", "sid-2": "beta"}

        # Re-labeling upserts (no duplicate PK error, new value wins).
        set_session_label(db, "sid-1", "alpha-renamed")
        assert get_session_labels(db.conn)["sid-1"] == "alpha-renamed"

        # Delete removes just that row; idempotent second delete is a no-op.
        delete_session_label(db, "sid-1")
        delete_session_label(db, "sid-1")
        assert get_session_labels(db.conn) == {"sid-2": "beta"}
    finally:
        db.close()


def test_get_session_labels_guards_none_conn():
    assert get_session_labels(None) == {}


# ── Endpoint + /status surfacing ───────────────────────────────────────────

def _app(config, db):
    pipeline = IngestPipeline(db=db, config=config)
    # Two active coding sessions the /status tiles will surface.
    pipeline.process(make_llm_span(
        agent_id="claude-code", session_id="sess-a", conversation_id="a",
        service_instance_id="ttys001"))
    pipeline.process(make_llm_span(
        agent_id="claude-code", session_id="sess-b", conversation_id="b",
        service_instance_id="ttys002"))
    return create_app(config=config, db=db, ingest_pipeline=pipeline)


def _plain_config():
    return TjConfig(
        version="1",
        security=SecurityConfig(ingest_secret=INGEST_SECRET),
        api=ApiConfig(auth=ApiAuthConfig(enabled=False)),
    )


async def _status_labels(client) -> dict:
    resp = await client.get("/api/v1/status")
    assert resp.status_code == 200
    return {a["session_id"]: a["label"] for a in resp.json()["agents"]}


@pytest.mark.asyncio
async def test_post_label_surfaces_in_status():
    db = InMemoryBackend()
    try:
        app = _app(_plain_config(), db)
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            # Default label is the instance id (ttys001).
            assert (await _status_labels(c))["sess-a"] == "ttys001"

            resp = await c.post(
                "/api/v1/sessions/sess-a/label",
                json={"label": "my-renamed-session"},
            )
            assert resp.status_code == 200
            assert resp.json() == {
                "session_id": "sess-a", "label": "my-renamed-session"
            }

            labels = await _status_labels(c)
            assert labels["sess-a"] == "my-renamed-session"  # rename beats ttys id
            assert labels["sess-b"] == "ttys002"             # untouched
    finally:
        db.close()


@pytest.mark.asyncio
async def test_post_empty_label_clears_rename():
    db = InMemoryBackend()
    try:
        app = _app(_plain_config(), db)
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            await c.post("/api/v1/sessions/sess-a/label", json={"label": "temp"})
            assert (await _status_labels(c))["sess-a"] == "temp"

            # Empty (whitespace-only) label clears -> reverts to the ttys id.
            resp = await c.post(
                "/api/v1/sessions/sess-a/label", json={"label": "   "}
            )
            assert resp.status_code == 200
            assert resp.json() == {"session_id": "sess-a", "label": None}
            assert (await _status_labels(c))["sess-a"] == "ttys001"
    finally:
        db.close()


@pytest.mark.asyncio
async def test_config_label_wins_over_db_rename():
    db = InMemoryBackend()
    try:
        config = TjConfig(
            version="1",
            security=SecurityConfig(ingest_secret=INGEST_SECRET),
            api=ApiConfig(auth=ApiAuthConfig(enabled=False)),
            session_labels={"sess-a": "config-wins"},
        )
        app = _app(config, db)
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.post(
                "/api/v1/sessions/sess-a/label", json={"label": "db-rename"}
            )
            assert resp.status_code == 200
            # DB row is written, but the config entry still wins in /status.
            assert get_session_labels(db.conn)["sess-a"] == "db-rename"
            labels = await _status_labels(c)
            assert labels["sess-a"] == "config-wins"
            # A session with no config entry uses the DB rename.
            await c.post("/api/v1/sessions/sess-b/label", json={"label": "b-rename"})
            assert (await _status_labels(c))["sess-b"] == "b-rename"
    finally:
        db.close()


@pytest.mark.asyncio
async def test_label_stripped_and_truncated():
    db = InMemoryBackend()
    try:
        app = _app(_plain_config(), db)
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.post(
                "/api/v1/sessions/sess-a/label",
                json={"label": "  " + "x" * 200 + "  "},
            )
            assert resp.status_code == 200
            stored = resp.json()["label"]
            assert stored == "x" * 120           # stripped + truncated to 120
            assert get_session_labels(db.conn)["sess-a"] == "x" * 120
    finally:
        db.close()


@pytest.mark.asyncio
async def test_post_label_rejects_non_dict_body():
    db = InMemoryBackend()
    try:
        app = _app(_plain_config(), db)
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.post("/api/v1/sessions/sess-a/label", json=["not", "a", "dict"])
            assert resp.status_code == 400
    finally:
        db.close()
