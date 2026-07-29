"""Integration tests for GET /api/v1/context-audit (the context-audit page's
one backend read). Talks through the real ASGI app so the route wiring and
the response shape are proven end to end, not just the core scan function.
"""
from __future__ import annotations

import json

import httpx
import pytest

from tokenjam.api.app import create_app
from tokenjam.api.routes import context_audit as route
from tokenjam.core.config import ApiAuthConfig, ApiConfig, StorageConfig, TjConfig
from tokenjam.core.db import InMemoryBackend
from tokenjam.core.ingest import IngestPipeline
from tokenjam.core.summarize import context_audit as ca


@pytest.fixture(autouse=True)
def _reset_cache():
    """The route caches its result at module scope — clear it around every
    test so one test's scan can never leak into another's assertions."""
    route._cache["result"] = None
    route._cache["at"] = 0.0
    yield
    route._cache["result"] = None
    route._cache["at"] = 0.0


@pytest.fixture
def config(tmp_path):
    return TjConfig(
        version="1",
        api=ApiConfig(auth=ApiAuthConfig(enabled=False)),
        storage=StorageConfig(path=str(tmp_path / "telemetry.duckdb")),
    )


@pytest.fixture
def app(config):
    db = InMemoryBackend()
    pipeline = IngestPipeline(db=db, config=config)
    return create_app(config=config, db=db, ingest_pipeline=pipeline)


@pytest.fixture
def client(app):
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")


@pytest.fixture
def fake_home(tmp_path, monkeypatch):
    """Redirect every home-derived path the scanner reads to a scratch
    directory, so the route test never touches the real operator's
    ~/.claude — same discipline `sandbox`/scratch isolation applies to any
    scan of real user state (root anti-pattern 26b)."""
    home = tmp_path / "home"
    claude_dir = home / ".claude"
    (claude_dir / "rules").mkdir(parents=True)
    (claude_dir / "rules" / "style.md").write_text("Be terse.\n")
    (claude_dir / "CLAUDE.md").write_text("# global rules\n")
    settings = {"enabledPlugins": {}}
    (claude_dir / "settings.json").write_text(json.dumps(settings))

    monkeypatch.setattr(ca, "SETTINGS_PATH", claude_dir / "settings.json")
    monkeypatch.setattr(ca, "PLUGINS_DIR", claude_dir / "plugins")
    monkeypatch.setattr(ca, "INSTALLED_PLUGINS_FILE", claude_dir / "plugins" / "installed_plugins.json")
    monkeypatch.setattr(ca, "_global_paths", lambda: [
        claude_dir / "CLAUDE.md", claude_dir / "rules" / "style.md",
    ])

    def _fake_scan_global(claude_dir_arg=None):
        return ca.ScopeAudit(
            scope=ca.GLOBAL_SCOPE,
            class1=(
                ca.Row(str(claude_dir / "CLAUDE.md"), "session start", 15, "every turn", ca.CLASS_1, ca.GLOBAL_SCOPE),
                ca.Row(str(claude_dir / "rules" / "style.md"), "session start", 10, "every turn", ca.CLASS_1, ca.GLOBAL_SCOPE),
            ),
        )
    monkeypatch.setattr(ca, "scan_global", _fake_scan_global)
    # No project roots for this test — isolate the route test from whatever
    # real transcript corpus happens to exist on the machine running CI.
    monkeypatch.setattr(route, "_resolved_project_roots", lambda config: [])
    return home


async def test_context_audit_route_returns_global_and_project_shape(client, fake_home):
    r = await client.get("/api/v1/context-audit")
    assert r.status_code == 200
    d = r.json()

    assert d["global"]["class1_total_chars"] == 25
    assert len(d["global"]["class1"]) == 2
    assert d["projects"] == []
    assert d["plugins_enabled"] == 0
    assert d["last_scanned_at"]


async def test_second_request_is_served_from_cache(client, fake_home, monkeypatch):
    calls = {"n": 0}
    real_compute = route._compute

    def counting_compute(config):
        calls["n"] += 1
        return real_compute(config)
    monkeypatch.setattr(route, "_compute", counting_compute)

    r1 = await client.get("/api/v1/context-audit")
    r2 = await client.get("/api/v1/context-audit")
    assert r1.status_code == r2.status_code == 200
    assert calls["n"] == 1, "a second request within the TTL must not re-scan"


async def test_refresh_flag_forces_a_rescan(client, fake_home, monkeypatch):
    calls = {"n": 0}
    real_compute = route._compute

    def counting_compute(config):
        calls["n"] += 1
        return real_compute(config)
    monkeypatch.setattr(route, "_compute", counting_compute)

    await client.get("/api/v1/context-audit")
    await client.get("/api/v1/context-audit", params={"refresh": "true"})
    assert calls["n"] == 2
