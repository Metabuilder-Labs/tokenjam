"""Security-guard tests for the self-improve loop's pothole write endpoints
(api/routes/pothole.py) — the PR-reviewer must-fix items:

  1. Mutating endpoints (apply/enable/disable/revert/refresh) require the
     always-on local write token, independent of ``api.auth.enabled``.
  2. ``/apply`` refuses a ``target_path`` outside the user's home directory
     (defense-in-depth allowlist) and — for rung 1 — a target that isn't an
     allowlisted note file (see test_pothole_apply.py for the pothole_apply
     unit-level version of that same guard).

Everything here talks through the real ASGI app (no mocks on the write path)
so the guards are proven at the route, not just in the core module.
"""
from __future__ import annotations

import httpx
import pytest

from tokenjam.api.app import create_app
from tokenjam.core.config import ApiAuthConfig, ApiConfig, StorageConfig, TjConfig
from tokenjam.core.db import InMemoryBackend
from tokenjam.core.ingest import IngestPipeline
from tokenjam.core.optimize import pothole_apply as pa


@pytest.fixture
def db():
    backend = InMemoryBackend()
    yield backend
    backend.close()


@pytest.fixture
def config(tmp_path):
    # api.auth.enabled=False (the default) — proves the write-token guard is
    # NOT contingent on this flag (must-fix #1's core claim).
    return TjConfig(
        version="1",
        api=ApiConfig(auth=ApiAuthConfig(enabled=False)),
        storage=StorageConfig(path=str(tmp_path / "telemetry.duckdb")),
    )


@pytest.fixture
def app(config, db):
    pipeline = IngestPipeline(db=db, config=config)
    return create_app(config=config, db=db, ingest_pipeline=pipeline)


@pytest.fixture
def client(app):
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")


def _apply_body(target_path: str, *, rung: int = 1, go: bool = True) -> dict:
    return {
        "signature": "cwd_confusion", "family_key": "cwd_confusion",
        "title": "cwd / relative-path confusion", "proposed_fix": "fix it",
        "rung": rung, "scope": "project", "target_path": target_path, "go": go,
    }


# --- must-fix #1: write endpoints require the local write token, always -------

@pytest.mark.parametrize("method,path,body", [
    ("post", "/api/v1/pothole/refresh", None),
    ("post", "/api/v1/pothole/apply", _apply_body("/tmp/whatever.md")),
    ("post", "/api/v1/pothole/some-fix-id/enable", {"confirm": True}),
    ("post", "/api/v1/pothole/some-fix-id/disable", {}),
    ("post", "/api/v1/pothole/some-fix-id/revert", {}),
])
async def test_write_endpoints_refuse_unauthenticated_even_with_global_auth_disabled(
    client, method, path, body,
):
    """No X-TJ-Local-Token header at all -> 401, even though config.api.auth.
    enabled is False (the require_api_key dependency would no-op)."""
    r = await getattr(client, method)(path, json=body)
    assert r.status_code == 401


async def test_write_endpoint_refuses_wrong_token(client):
    r = await client.post(
        "/api/v1/pothole/refresh", headers={"X-TJ-Local-Token": "not-the-real-token"},
    )
    assert r.status_code == 401


async def test_write_endpoint_succeeds_with_the_real_local_token(app, client):
    token = app.state.pothole_write_token
    r = await client.post("/api/v1/pothole/refresh", headers={"X-TJ-Local-Token": token})
    assert r.status_code == 200
    assert r.json()["status"] in ("started", "already_running")


async def test_write_endpoint_refuses_cross_origin_even_with_valid_token(app, client):
    """A correct token from a cross-origin Origin (the browser-CSRF shape) is
    still refused — the same-origin check is a real, independent gate."""
    token = app.state.pothole_write_token
    r = await client.post(
        "/api/v1/pothole/refresh",
        headers={"X-TJ-Local-Token": token, "Origin": "http://evil.example.com"},
    )
    assert r.status_code == 403


async def test_write_endpoint_allows_same_origin_request_with_token(app, client):
    token = app.state.pothole_write_token
    r = await client.post(
        "/api/v1/pothole/refresh",
        headers={"X-TJ-Local-Token": token, "Origin": "http://test"},
    )
    assert r.status_code == 200


async def test_ui_html_carries_the_write_token_meta_tag_unconditionally(app, client):
    """The same-origin UI must be able to read the token off the served page
    even though api.auth.enabled is False (must-fix #1's UI-still-works half)."""
    token = app.state.pothole_write_token
    r = await client.get("/")
    assert r.status_code == 200
    assert f'<meta name="tj-write-token" content="{token}">' in r.text


async def test_read_endpoints_do_not_require_the_write_token(client):
    """GET /proposals and GET /applied stay on the (optional) api-key gate
    only — no regression for the read surface."""
    r = await client.get("/api/v1/pothole/proposals")
    assert r.status_code == 200
    r2 = await client.get("/api/v1/pothole/applied")
    assert r2.status_code == 200


# --- must-fix #1 (defense-in-depth): home-anchored target_path allowlist ------

async def test_apply_refuses_target_outside_home(app, client, monkeypatch, tmp_path):
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setattr(pa.Path, "home", classmethod(lambda cls: fake_home))
    token = app.state.pothole_write_token

    outside = tmp_path / "outside" / "CLAUDE.md"
    outside.parent.mkdir()
    r = await client.post(
        "/api/v1/pothole/apply", json=_apply_body(str(outside)),
        headers={"X-TJ-Local-Token": token},
    )
    assert r.status_code == 403
    assert "outside the allowed root" in r.json()["detail"]
    assert not outside.exists()


async def test_apply_allows_target_inside_home(app, client, monkeypatch, tmp_path):
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setattr(pa.Path, "home", classmethod(lambda cls: fake_home))
    token = app.state.pothole_write_token

    inside = fake_home / "CLAUDE.md"
    inside.write_text("# Repo\n", encoding="utf-8")
    r = await client.post(
        "/api/v1/pothole/apply", json=_apply_body(str(inside)),
        headers={"X-TJ-Local-Token": token},
    )
    assert r.status_code == 200
    assert r.json()["dry_run"] is False


# --- must-fix #2 (routed through the API): rung-1 note target allowlist -------

async def test_apply_note_route_refuses_non_markdown_target(app, client, monkeypatch, tmp_path):
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setattr(pa.Path, "home", classmethod(lambda cls: fake_home))
    token = app.state.pothole_write_token

    target = fake_home / "evil.py"
    target.write_text("print('do not touch me')\n", encoding="utf-8")
    r = await client.post(
        "/api/v1/pothole/apply", json=_apply_body(str(target)),
        headers={"X-TJ-Local-Token": token},
    )
    assert r.status_code == 409
    assert "not an allowlisted note target" in r.json()["detail"]
    assert target.read_text() == "print('do not touch me')\n"
