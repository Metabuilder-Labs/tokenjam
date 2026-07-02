"""Tests for the close-the-loop API routes (#53): annotations, expectations, and
the fix-history ledger over httpx ASGITransport — the same surface the Lens
"Loop" tab drives.
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
from tokenjam.core.db import InMemoryBackend


def _config():
    return TjConfig(
        version="1",
        security=SecurityConfig(ingest_secret="test-secret-token"),
        api=ApiConfig(auth=ApiAuthConfig(enabled=False)),
    )


def _client(db):
    app = create_app(config=_config(), db=db, ingest_pipeline=None)
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://test")


@pytest.mark.asyncio
async def test_annotation_create_and_list():
    db = InMemoryBackend()
    try:
        async with _client(db) as c:
            resp = await c.post(
                "/api/v1/sessions/sid-1/annotations",
                json={"note": "weird output", "verdict": "bad"},
            )
            assert resp.status_code == 201
            assert resp.json()["verdict"] == "bad"

            listed = await c.get("/api/v1/sessions/sid-1/annotations")
            assert listed.status_code == 200
            body = listed.json()
            assert body["count"] == 1
            assert body["annotations"][0]["note"] == "weird output"
    finally:
        db.close()


@pytest.mark.asyncio
async def test_annotation_missing_note_is_400():
    db = InMemoryBackend()
    try:
        async with _client(db) as c:
            resp = await c.post("/api/v1/sessions/sid-1/annotations", json={"verdict": "bad"})
            assert resp.status_code == 400
    finally:
        db.close()


@pytest.mark.asyncio
async def test_annotation_bad_verdict_is_400():
    db = InMemoryBackend()
    try:
        async with _client(db) as c:
            resp = await c.post(
                "/api/v1/sessions/sid-1/annotations",
                json={"note": "x", "verdict": "awful"},
            )
            assert resp.status_code == 400
    finally:
        db.close()


@pytest.mark.asyncio
async def test_full_loop_promote_record_history():
    db = InMemoryBackend()
    try:
        async with _client(db) as c:
            # Promote a bad run into an expectation.
            promote = await c.post(
                "/api/v1/expectations",
                json={
                    "name": "no retry loop",
                    "description": "must not retry 4x",
                    "origin_session_id": "sid-1",
                    "agent_id": "claude-code",
                },
            )
            assert promote.status_code == 201
            eid = promote.json()["expectation_id"]

            # It shows up scoped to the origin run (Lens "from this run" list).
            scoped = await c.get("/api/v1/expectations", params={"session_id": "sid-1"})
            assert scoped.status_code == 200
            assert scoped.json()["count"] == 1

            # Record a regress then a pass on later reruns.
            r1 = await c.post(
                f"/api/v1/expectations/{eid}/runs",
                json={"outcome": "regress", "session_id": "sid-2"},
            )
            assert r1.status_code == 201
            r2 = await c.post(
                f"/api/v1/expectations/{eid}/runs",
                json={"outcome": "pass", "session_id": "sid-3", "note": "fixed"},
            )
            assert r2.status_code == 201

            # History reads newest-first.
            detail = await c.get(f"/api/v1/expectations/{eid}")
            assert detail.status_code == 200
            payload = detail.json()
            assert payload["run_count"] == 2
            assert [r["outcome"] for r in payload["runs"]] == ["pass", "regress"]
    finally:
        db.close()


@pytest.mark.asyncio
async def test_record_bad_outcome_is_400():
    db = InMemoryBackend()
    try:
        async with _client(db) as c:
            exp = await c.post("/api/v1/expectations", json={"name": "case"})
            eid = exp.json()["expectation_id"]
            resp = await c.post(
                f"/api/v1/expectations/{eid}/runs", json={"outcome": "meh"}
            )
            assert resp.status_code == 400
    finally:
        db.close()


@pytest.mark.asyncio
async def test_record_unknown_expectation_is_404():
    db = InMemoryBackend()
    try:
        async with _client(db) as c:
            resp = await c.post(
                "/api/v1/expectations/ghost/runs", json={"outcome": "pass"}
            )
            assert resp.status_code == 404
    finally:
        db.close()


@pytest.mark.asyncio
async def test_get_unknown_expectation_is_404():
    db = InMemoryBackend()
    try:
        async with _client(db) as c:
            resp = await c.get("/api/v1/expectations/ghost")
            assert resp.status_code == 404
    finally:
        db.close()
