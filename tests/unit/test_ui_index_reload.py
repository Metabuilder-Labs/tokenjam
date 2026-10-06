"""`/` re-reads ui/index.html when it changes on disk (issue #777).

create_app() used to read index.html once at startup, so editing it against a
running `tj serve` silently served the stale copy. The cached string is now
refreshed whenever the file's mtime changes.
"""
from __future__ import annotations

import os
from pathlib import Path

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


def test_undecodable_index_html_keeps_last_good_copy_then_recovers(tmp_path, monkeypatch, db):
    index = tmp_path / "index.html"
    _write(index, "<html><head></head><body>v1</body></html>", 1_000_000_000)

    with _client(tmp_path, monkeypatch, db) as client:
        # A save caught mid-character leaves invalid UTF-8 on disk.
        index.write_bytes(b"<html><head></head><body>\xe2\x82")
        os.utime(index, ns=(2_000_000_000, 2_000_000_000))
        resp = client.get("/")
        assert resp.status_code == 200
        assert "v1" in resp.text

        # Once the file is fixed, the next request picks it up.
        _write(index, "<html><head></head><body>v2</body></html>", 3_000_000_000)
        assert "v2" in client.get("/").text


def test_index_html_is_read_as_utf8_regardless_of_locale(tmp_path, monkeypatch, db):
    """The read must pin its encoding, or a non-UTF-8 locale serves a blank page.

    ui/index.html carries ~3KB of non-ASCII, and `Path.read_text()` with no
    `encoding=` resolves to the locale default. Under LC_ALL=C that raises
    UnicodeDecodeError, which `_load_index_html` catches, leaving the initial
    empty cache in place: HTTP 200, zero-length body, empty console.

    This asserts the call rather than the symptom. A behavioural version would
    have to change the process-wide locale, which is unsafe under `-n auto` and
    resolves differently across the Python versions we support, so it would
    prove less while breaking more.
    """
    index = tmp_path / "index.html"
    _write(index, "<html><head></head><body>v1</body></html>", 1_000_000_000)

    seen: list[str | None] = []
    real_read_text = Path.read_text

    def spy(self, *args, **kwargs):
        if self.name == "index.html":
            seen.append(kwargs.get("encoding"))
        return real_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", spy)
    with _client(tmp_path, monkeypatch, db) as client:
        client.get("/")

    assert seen, "index.html was never read"
    assert set(seen) == {"utf-8"}, f"index.html read without an explicit utf-8 encoding: {seen}"


def test_non_ascii_index_html_survives_the_round_trip(tmp_path, monkeypatch, db):
    """The companion to the pin above: the bytes the real file contains work."""
    index = tmp_path / "index.html"
    marker = "\u2014 \u00b7 \u2192 \u2713"  # the dash/bullet/arrow/tick class ui/index.html uses
    index.write_text(f"<html><head></head><body>{marker}</body></html>", encoding="utf-8")
    os.utime(index, ns=(1_000_000_000, 1_000_000_000))

    with _client(tmp_path, monkeypatch, db) as client:
        body = client.get("/").text

    assert marker in body
