"""Security-guard tests for the self-improve loop's relearn write endpoints
(api/routes/relearn.py) — the PR-reviewer must-fix items:

  1. Mutating endpoints (apply/enable/disable/revert/refresh) require the
     always-on local write token, independent of ``api.auth.enabled``.
  2. ``/apply`` refuses a ``target_path`` outside the user's home directory
     (defense-in-depth allowlist) and — for rung 1 — a target that isn't an
     allowlisted note file (see test_relearn_apply.py for the relearn_apply
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
from tokenjam.core.optimize import relearn_apply as pa


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


def _apply_body(target_path: str, *, proposal_id: str = "rp_unused000000", go: bool = True) -> dict:
    """An apply request names a STORED proposal; the cluster content itself is
    never accepted from the caller (see the F2 tests at the bottom)."""
    return {
        "proposal_id": proposal_id, "scope": "project",
        "target_path": target_path, "go": go,
    }


@pytest.fixture
def stored_proposal(config) -> str:
    """Persist a detector finding the way a real recompute does, and hand back
    the proposal ID the write endpoints will accept."""
    from tokenjam.core.optimize import relearn_proposals, relearn_store
    from tokenjam.core.optimize.analyzers.relearn import RelearnCluster, RelearnFinding

    cluster = RelearnCluster(
        signature="cwd_confusion", family_key="cwd_confusion",
        title="cwd / relative-path confusion", sessions=5, occurrences=9,
        repos=["demo"], rung=1, scope="project",
        proposed_fix="Verify an absolute cwd before a relative Read.",
    )
    relearn_store.write_cache(RelearnFinding(clusters=[cluster]), config=config)
    return relearn_proposals.list_proposals(config)[0]["proposal_id"]


# --- must-fix #1: write endpoints require the local write token, always -------

@pytest.mark.parametrize("method,path,body", [
    ("post", "/api/v1/relearn/refresh", None),
    ("post", "/api/v1/relearn/apply", _apply_body("/tmp/whatever.md")),
    ("post", "/api/v1/relearn/some-fix-id/enable", {"confirm": True}),
    ("post", "/api/v1/relearn/some-fix-id/disable", {}),
    ("post", "/api/v1/relearn/some-fix-id/revert", {}),
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
        "/api/v1/relearn/refresh", headers={"X-TJ-Local-Token": "not-the-real-token"},
    )
    assert r.status_code == 401


async def test_write_endpoint_succeeds_with_the_real_local_token(app, client):
    token = app.state.relearn_write_token
    r = await client.post("/api/v1/relearn/refresh", headers={"X-TJ-Local-Token": token})
    assert r.status_code == 200
    assert r.json()["status"] in ("started", "already_running")


async def test_write_endpoint_refuses_cross_origin_even_with_valid_token(app, client):
    """A correct token from a cross-origin Origin (the browser-CSRF shape) is
    still refused — the same-origin check is a real, independent gate."""
    token = app.state.relearn_write_token
    r = await client.post(
        "/api/v1/relearn/refresh",
        headers={"X-TJ-Local-Token": token, "Origin": "http://evil.example.com"},
    )
    assert r.status_code == 403


async def test_write_endpoint_allows_same_origin_request_with_token(app, client):
    token = app.state.relearn_write_token
    r = await client.post(
        "/api/v1/relearn/refresh",
        headers={"X-TJ-Local-Token": token, "Origin": "http://test"},
    )
    assert r.status_code == 200


async def test_ui_html_carries_the_write_token_meta_tag_unconditionally(app, client):
    """The same-origin UI must be able to read the token off the served page
    even though api.auth.enabled is False (must-fix #1's UI-still-works half)."""
    token = app.state.relearn_write_token
    r = await client.get("/")
    assert r.status_code == 200
    assert f'<meta name="tj-write-token" content="{token}">' in r.text


async def test_read_endpoints_do_not_require_the_write_token(client):
    """GET /proposals and GET /applied stay on the (optional) api-key gate
    only — no regression for the read surface."""
    r = await client.get("/api/v1/relearn/proposals")
    assert r.status_code == 200
    r2 = await client.get("/api/v1/relearn/applied")
    assert r2.status_code == 200


# --- must-fix #1 (defense-in-depth): home-anchored target_path allowlist ------

async def test_apply_refuses_target_outside_home(app, client, monkeypatch, tmp_path):
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setattr(pa.Path, "home", classmethod(lambda cls: fake_home))
    token = app.state.relearn_write_token

    outside = tmp_path / "outside" / "CLAUDE.md"
    outside.parent.mkdir()
    r = await client.post(
        "/api/v1/relearn/apply", json=_apply_body(str(outside)),
        headers={"X-TJ-Local-Token": token},
    )
    assert r.status_code == 403
    assert "outside the allowed root" in r.json()["detail"]
    assert not outside.exists()


async def test_apply_allows_target_inside_home(app, client, monkeypatch, tmp_path, stored_proposal):
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setattr(pa.Path, "home", classmethod(lambda cls: fake_home))
    token = app.state.relearn_write_token

    inside = fake_home / "CLAUDE.md"
    inside.write_text("# Repo\n", encoding="utf-8")
    r = await client.post(
        "/api/v1/relearn/apply", json=_apply_body(str(inside), proposal_id=stored_proposal),
        headers={"X-TJ-Local-Token": token},
    )
    assert r.status_code == 200
    assert r.json()["dry_run"] is False


# --- must-fix #2 (routed through the API): rung-1 note target allowlist -------

async def test_apply_note_route_refuses_non_markdown_target(
    app, client, monkeypatch, tmp_path, stored_proposal,
):
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setattr(pa.Path, "home", classmethod(lambda cls: fake_home))
    token = app.state.relearn_write_token

    target = fake_home / "evil.py"
    target.write_text("print('do not touch me')\n", encoding="utf-8")
    r = await client.post(
        "/api/v1/relearn/apply", json=_apply_body(str(target), proposal_id=stored_proposal),
        headers={"X-TJ-Local-Token": token},
    )
    assert r.status_code == 409
    assert "not an allowlisted note target" in r.json()["detail"]
    assert target.read_text() == "print('do not touch me')\n"


# --- model-routing apply kinds: the write also opens a priced verify window ---

_AGENT_FILE = """---
name: explore
model: claude-opus-4-8
---

Body.
"""


@pytest.fixture
def stored_cost_proposal(config) -> str:
    """A stored proposal whose card is a cost finding, so the model-routing
    apply below names a proposal the detector actually produced."""
    from tokenjam.core.optimize import relearn_proposals, relearn_store
    from tokenjam.core.optimize.analyzers.relearn import RelearnCluster, RelearnFinding

    cluster = RelearnCluster(
        signature="cost:subagent:explore", family_key="cost:subagent:explore",
        title="Over-powered subagent explore", sessions=4, occurrences=4,
        repos=["demo"], rung=1, scope="project",
        proposed_fix="Route explore to the cheaper same-family model.",
    )
    relearn_store.write_cache(RelearnFinding(clusters=[cluster]), config=config)
    return relearn_proposals.list_proposals(config)[0]["proposal_id"]


async def test_agent_model_apply_opens_a_cost_verify_window(
    app, client, config, monkeypatch, tmp_path, stored_cost_proposal,
):
    """A subagent model write is a cost fix with a file to edit, so the same
    approval that rewrites the frontmatter must also start the exposure window
    the priced receipt is measured over."""
    from tokenjam.core.optimize import cost_apply

    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setattr(pa.Path, "home", classmethod(lambda cls: fake_home))
    token = app.state.relearn_write_token

    target = fake_home / "workspace" / ".claude" / "agents" / "explore.md"
    target.parent.mkdir(parents=True)
    target.write_text(_AGENT_FILE, encoding="utf-8")

    body = {
        "proposal_id": stored_cost_proposal,
        "scope": "project", "target_path": str(target), "go": True,
        "apply_kind": "agent_model", "agent_name": "explore",
        "current_model": "claude-opus-4-8", "proposed_model": "claude-haiku-4-5",
        "analyzer": "subagent",
    }
    r = await client.post(
        "/api/v1/relearn/apply", json=body, headers={"X-TJ-Local-Token": token},
    )

    assert r.status_code == 200
    payload = r.json()
    assert payload["dry_run"] is False
    assert "model: claude-haiku-4-5" in target.read_text()
    # The cost ledger carries the marker the dollar delta is measured from.
    marker = payload["cost_marker"]
    assert marker["analyzer"] == "subagent"
    assert marker["target_key"]["models"] == ["claude-opus-4-8"]
    assert marker["applied_at"]
    assert [rec["signature"] for rec in cost_apply.list_applied(config)] == [
        "cost:subagent:explore",
    ]


async def test_rung_ladder_apply_opens_no_cost_window(
    app, client, config, monkeypatch, tmp_path, stored_proposal,
):
    """A plain note fix has no priced metric, so it must not create a cost
    marker whose delta nothing can measure."""
    from tokenjam.core.optimize import cost_apply

    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setattr(pa.Path, "home", classmethod(lambda cls: fake_home))
    token = app.state.relearn_write_token

    inside = fake_home / "CLAUDE.md"
    inside.write_text("# Repo\n", encoding="utf-8")
    r = await client.post(
        "/api/v1/relearn/apply", json=_apply_body(str(inside), proposal_id=stored_proposal),
        headers={"X-TJ-Local-Token": token},
    )
    assert r.status_code == 200
    assert "cost_marker" not in r.json()
    assert cost_apply.list_applied(config) == []


# --- F2: apply accepts a STORED proposal ID and nothing else ------------------

async def test_apply_refuses_an_unstored_proposal_id(app, client, monkeypatch, tmp_path):
    """The integrity hole: before this, any authenticated local caller could
    hand-build a cluster and have it written. Now an ID the detector never
    produced has no way in."""
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setattr(pa.Path, "home", classmethod(lambda cls: fake_home))
    target = fake_home / "CLAUDE.md"
    target.write_text("# Repo\n", encoding="utf-8")

    r = await client.post(
        "/api/v1/relearn/apply",
        json=_apply_body(str(target), proposal_id="rp_000000000000"),
        headers={"X-TJ-Local-Token": app.state.relearn_write_token},
    )
    assert r.status_code == 404
    assert "no stored proposal" in r.json()["detail"]
    assert target.read_text() == "# Repo\n"


async def test_apply_rejects_a_client_constructed_cluster_payload(
    app, client, monkeypatch, tmp_path, stored_proposal,
):
    """A caller that posts cluster content alongside a valid ID is refused
    outright (422) rather than having its payload silently ignored: whatever
    the human reviewed is what gets written, and the caller is told so."""
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setattr(pa.Path, "home", classmethod(lambda cls: fake_home))
    target = fake_home / "CLAUDE.md"
    target.write_text("# Repo\n", encoding="utf-8")

    body = _apply_body(str(target), proposal_id=stored_proposal)
    body.update({"signature": "attacker", "rung": 3, "title": "not from the detector",
                 "proposed_fix": "rm -rf /"})
    r = await client.post(
        "/api/v1/relearn/apply", json=body,
        headers={"X-TJ-Local-Token": app.state.relearn_write_token},
    )
    assert r.status_code == 422
    assert target.read_text() == "# Repo\n"


async def test_apply_writes_the_stored_content_not_the_requested_content(
    app, client, monkeypatch, tmp_path, stored_proposal,
):
    """End to end: the note that lands on disk carries the DETECTOR's title
    and fix text, sourced from the stored proposal."""
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setattr(pa.Path, "home", classmethod(lambda cls: fake_home))
    target = fake_home / "CLAUDE.md"
    target.write_text("# Repo\n", encoding="utf-8")

    r = await client.post(
        "/api/v1/relearn/apply", json=_apply_body(str(target), proposal_id=stored_proposal),
        headers={"X-TJ-Local-Token": app.state.relearn_write_token},
    )
    assert r.status_code == 200
    written = target.read_text()
    assert "cwd / relative-path confusion" in written
    assert "Verify an absolute cwd before a relative Read." in written


async def test_stored_proposals_are_listed_with_their_ids(client, stored_proposal):
    r = await client.get("/api/v1/relearn/proposals")
    assert r.status_code == 200
    clusters = r.json()["finding"]["clusters"]
    assert [c["proposal_id"] for c in clusters] == [stored_proposal]
