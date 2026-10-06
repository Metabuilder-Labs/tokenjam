"""`/` re-reads ui/index.html when it changes on disk (issue #777).

create_app() used to read index.html once at startup, so editing it against a
running `tj serve` silently served the stale copy. The cached string is now
refreshed whenever the file's mtime changes.
"""
from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient

import tokenjam.api.app as app_module
from tokenjam.api.app import create_app
from tokenjam.core.config import ApiAuthConfig, ApiConfig, TjConfig
from tokenjam.core.db import InMemoryBackend
from tokenjam.core.ingest import build_default_pipeline


@pytest.fixture
def db():
    backend = InMemoryBackend()
    yield backend
    backend.close()


def _client(tmp_path, monkeypatch, db, *, auth_key: str | None = None) -> TestClient:
    monkeypatch.setattr(app_module, "_UI_DIR", tmp_path)
    auth = ApiAuthConfig(enabled=bool(auth_key), api_key=auth_key)
    config = TjConfig(version="1", api=ApiConfig(auth=auth))
    app = create_app(config=config, db=db, ingest_pipeline=build_default_pipeline(db, config))
    return TestClient(app)


def _write(path, text: str, mtime_ns: int) -> None:
    path.write_text(text)
    # Set mtime explicitly: two writes inside one filesystem timestamp tick
    # would otherwise look unchanged.
    os.utime(path, ns=(mtime_ns, mtime_ns))


def test_edit_to_index_html_is_served_without_restart(tmp_path, monkeypatch, db):
    index = tmp_path / "index.html"
    _write(index, "<html><head></head><body>v1</body></html>", 1_000_000_000)

    with _client(tmp_path, monkeypatch, db) as client:
        assert "v1" in client.get("/").text

        _write(index, "<html><head></head><body>v2</body></html>", 2_000_000_000)

        for path in ("/", "/ui/overview"):
            body = client.get(path).text
            assert "v2" in body
            assert "v1" not in body


def test_meta_injection_still_applies_to_reread_html(tmp_path, monkeypatch, db):
    index = tmp_path / "index.html"
    _write(index, "<html><head></head><body>v1</body></html>", 1_000_000_000)

    with _client(tmp_path, monkeypatch, db, auth_key="secret-key") as client:
        _write(index, "<html><head></head><body>v2</body></html>", 2_000_000_000)
        body = client.get("/").text

    assert "v2" in body
    assert '<meta name="tj-api-key" content="secret-key">' in body
    assert 'name="tj-write-token"' in body


def test_missing_index_html_keeps_serving_last_good_copy(tmp_path, monkeypatch, db):
    index = tmp_path / "index.html"
    _write(index, "<html><head></head><body>v1</body></html>", 1_000_000_000)

    with _client(tmp_path, monkeypatch, db) as client:
        index.unlink()
        resp = client.get("/")

    assert resp.status_code == 200
    assert "v1" in resp.text
