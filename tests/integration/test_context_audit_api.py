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


# --------------------------------------------------------------------------- #
# Remove / restore. These move real files, so the round trip and the refusals
# are proven through the app, guard headers and all.
# --------------------------------------------------------------------------- #

@pytest.fixture
def removable_home(tmp_path, monkeypatch, fake_home):
    """`fake_home` with the global CLAUDE.md declared removable, plus the
    quarantine store redirected into the same scratch home so the test can
    never write to the operator's real one."""
    from tokenjam.core.summarize import context_quarantine as cq

    claude_dir = fake_home / ".claude"
    target = claude_dir / "CLAUDE.md"

    def _scan(claude_dir_arg=None):
        return ca.ScopeAudit(
            scope=ca.GLOBAL_SCOPE,
            class1=(ca.Row(str(target), "harness auto-load", 15, "every turn",
                           ca.CLASS_1, ca.GLOBAL_SCOPE, **ca._file_removal(target)),),
        )
    monkeypatch.setattr(ca, "scan_global", _scan)
    monkeypatch.setattr(cq, "quarantine_root", lambda home=None: fake_home / ".claude-context-trash")
    return fake_home


def _write_headers(app):
    return {"X-TJ-Local-Token": app.state.relearn_write_token}


async def test_remove_then_restore_round_trips_the_file(client, app, removable_home):
    target = removable_home / ".claude" / "CLAUDE.md"
    before = target.read_text()

    audit = (await client.get("/api/v1/context-audit")).json()
    row_id = audit["global"]["class1"][0]["row_id"]
    assert row_id

    r = await client.post("/api/v1/context-audit/remove", json={"row_id": row_id},
                          headers=_write_headers(app))
    assert r.status_code == 200
    assert not target.exists()

    listed = (await client.get("/api/v1/context-audit/removals")).json()["removals"]
    assert len(listed) == 1 and listed[0]["restorable"] is True

    r = await client.post("/api/v1/context-audit/restore",
                          json={"record_id": listed[0]["record_id"]},
                          headers=_write_headers(app))
    assert r.status_code == 200
    assert target.read_text() == before


async def test_remove_without_the_local_write_token_is_refused(client, removable_home):
    target = removable_home / ".claude" / "CLAUDE.md"
    audit = (await client.get("/api/v1/context-audit")).json()
    row_id = audit["global"]["class1"][0]["row_id"]

    r = await client.post("/api/v1/context-audit/remove", json={"row_id": row_id})

    assert r.status_code == 401
    assert target.exists(), "a refused request must not have touched the file"


async def test_remove_refuses_a_row_id_the_audit_never_reported(client, app, removable_home):
    """The endpoint takes a handle, never a path — an id the scan did not
    produce resolves to nothing and removes nothing."""
    await client.get("/api/v1/context-audit")

    r = await client.post("/api/v1/context-audit/remove",
                          json={"row_id": "not-a-real-row-id"}, headers=_write_headers(app))

    assert r.status_code == 404
    assert (removable_home / ".claude" / "CLAUDE.md").exists()


async def test_removing_a_file_that_vanished_since_the_scan_is_a_conflict(client, app, removable_home):
    target = removable_home / ".claude" / "CLAUDE.md"
    audit = (await client.get("/api/v1/context-audit")).json()
    row_id = audit["global"]["class1"][0]["row_id"]
    target.unlink()

    r = await client.post("/api/v1/context-audit/remove", json={"row_id": row_id},
                          headers=_write_headers(app))

    assert r.status_code == 409
    assert "no longer exists" in r.json()["detail"]
