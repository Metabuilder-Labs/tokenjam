"""Integration tests for the cost-proposal Review-inbox endpoints
(api/routes/relearn.py). Talks through the real ASGI app so the read/write
surface + the advise-only marker round-trip are proven at the route.

Isolated: InMemoryBackend + a tmp storage path; nothing touches a real store.
"""
from __future__ import annotations

from datetime import timedelta

import httpx
import pytest

from tokenjam.api.app import create_app
from tokenjam.core.config import ApiAuthConfig, ApiConfig, StorageConfig, TjConfig
from tokenjam.core.db import InMemoryBackend
from tokenjam.core.ingest import IngestPipeline
from tokenjam.otel.semconv import GenAIAttributes
from tokenjam.utils.time_parse import utcnow
from tests.factories import make_llm_span, make_session, make_tool_span


@pytest.fixture
def db():
    backend = InMemoryBackend()
    yield backend
    backend.close()


@pytest.fixture
def config(tmp_path):
    return TjConfig(
        version="1",
        api=ApiConfig(auth=ApiAuthConfig(enabled=False)),
        storage=StorageConfig(path=str(tmp_path / "telemetry.duckdb")),
    )


@pytest.fixture
def app(config, db):
    # Seed low-cache-efficacy spans so the `cache` analyzer flags a row and the
    # refresh produces at least one cost proposal.
    now = utcnow()
    for i in range(12):
        db.insert_span(make_llm_span(
            agent_id="svc-a", provider="anthropic", model="claude-sonnet-5",
            billing_account="anthropic", input_tokens=15_000, output_tokens=200,
            cache_tokens=400, session_id=f"s-{i}",
            start_time=now - timedelta(days=2, minutes=i),
        ))
    pipeline = IngestPipeline(db=db, config=config)
    return create_app(config=config, db=db, ingest_pipeline=pipeline)


@pytest.fixture
def client(app):
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")


async def test_cost_proposals_never_run_before_refresh(client):
    r = await client.get("/api/v1/relearn/cost-proposals")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "never_run"
    assert body["proposals"] == []


async def test_cost_refresh_requires_write_token(client):
    r = await client.post("/api/v1/relearn/cost-proposals/refresh")
    assert r.status_code == 401


async def test_cost_refresh_then_proposals_listed(app, client):
    token = app.state.relearn_write_token
    r = await client.post(
        "/api/v1/relearn/cost-proposals/refresh",
        headers={"X-TJ-Local-Token": token},
    )
    assert r.status_code == 200
    assert r.json()["status"] == "ready"

    r2 = await client.get("/api/v1/relearn/cost-proposals")
    body = r2.json()
    assert body["status"] == "ready"
    assert len(body["proposals"]) >= 1
    cache_props = [p for p in body["proposals"] if p["analyzer"] == "cache"]
    assert cache_props, body["proposals"]
    prop = cache_props[0]
    assert prop["kind"] == "cost"
    assert prop["advise_only"] is True


async def test_mark_cost_applied_round_trip(app, client):
    token = app.state.relearn_write_token
    await client.post(
        "/api/v1/relearn/cost-proposals/refresh",
        headers={"X-TJ-Local-Token": token},
    )
    proposals = (await client.get("/api/v1/relearn/cost-proposals")).json()["proposals"]
    prop = next(p for p in proposals if p["analyzer"] == "cache")
    body = {"proposal_id": prop["proposal_id"]}

    # Mark applied (the marker) — requires the write token.
    unauth = await client.post("/api/v1/relearn/cost-proposals/apply", json=body)
    assert unauth.status_code == 401

    r = await client.post(
        "/api/v1/relearn/cost-proposals/apply", json=body,
        headers={"X-TJ-Local-Token": token},
    )
    assert r.status_code == 200
    rec = r.json()
    assert rec["state"] == "applied"
    assert rec["applied_at"]
    # The ledger carries the STORED estimate, not anything a caller named.
    assert rec["signature"] == prop["signature"]
    assert rec["estimated_recoverable_usd"] == prop["estimated_recoverable_usd"]

    applied = (await client.get("/api/v1/relearn/cost-applied")).json()
    assert len(applied["applied"]) == 1

    # Revert flips it back.
    rev = await client.post(
        f"/api/v1/relearn/cost-applied/{rec['id']}/revert",
        headers={"X-TJ-Local-Token": token},
    )
    assert rev.status_code == 200
    assert rev.json()["state"] == "reverted"


# --- the marker's numbers come from the STORE, never from the caller -------- #

async def test_mark_applied_refuses_a_caller_supplied_estimate(app, client, config):
    """The cost ledger is what the "verified saved" receipts are measured from,
    so a caller must not be able to seed it with a number the detector never
    produced. A valid proposal_id carrying its own estimate is refused outright
    rather than having the extra field quietly ignored."""
    from tokenjam.core.optimize import cost_apply

    token = app.state.relearn_write_token
    hdr = {"X-TJ-Local-Token": token}
    await client.post("/api/v1/relearn/cost-proposals/refresh", headers=hdr)
    proposals = (await client.get("/api/v1/relearn/cost-proposals")).json()["proposals"]
    prop = next(p for p in proposals if p["analyzer"] == "cache")

    r = await client.post(
        "/api/v1/relearn/cost-proposals/apply",
        json={"proposal_id": prop["proposal_id"], "estimated_recoverable_usd": 9999.0},
        headers=hdr,
    )
    assert r.status_code == 422
    assert cost_apply.list_applied(config) == []   # nothing recorded at all


async def test_mark_applied_refuses_an_unstored_proposal_id(app, client, config):
    """An ID the detector never produced has no way into the ledger."""
    from tokenjam.core.optimize import cost_apply

    hdr = {"X-TJ-Local-Token": app.state.relearn_write_token}
    await client.post("/api/v1/relearn/cost-proposals/refresh", headers=hdr)
    r = await client.post(
        "/api/v1/relearn/cost-proposals/apply",
        json={"proposal_id": "rp_000000000000"}, headers=hdr,
    )
    assert r.status_code == 404
    assert "no stored cost proposal" in r.json()["detail"]
    assert cost_apply.list_applied(config) == []


async def test_apply_workspace_refuses_a_caller_supplied_proposed_fix(
    app, client, monkeypatch, tmp_path,
):
    """The note text is the stored proposal's, not the request's: a caller-named
    proposed_fix would be arbitrary text written into the user's CLAUDE.md under
    a reviewed proposal's name."""
    from tokenjam.core.optimize import relearn_apply as pa

    home = tmp_path / "home"
    (home / "proj").mkdir(parents=True)
    monkeypatch.setattr(pa.Path, "home", classmethod(lambda cls: home))
    target = home / "proj" / "CLAUDE.md"

    hdr = {"X-TJ-Local-Token": app.state.relearn_write_token}
    await client.post("/api/v1/relearn/cost-proposals/refresh", headers=hdr)
    proposals = (await client.get("/api/v1/relearn/cost-proposals")).json()["proposals"]
    prop = proposals[0]

    r = await client.post(
        "/api/v1/relearn/cost-proposals/apply-workspace",
        json={
            "proposal_id": prop["proposal_id"], "target_path": str(target),
            "go": True, "proposed_fix": "rm -rf everything",
        },
        headers=hdr,
    )
    assert r.status_code == 422
    assert not target.exists()


async def test_cost_apply_workspace_writes_note_and_records_marker(app, client, db, monkeypatch, tmp_path):
    """A CC-origin subagent proposal routes a reversible rung-1 note through the
    existing relearn apply path, then records the cost marker for delta-verify."""
    from tokenjam.core.optimize import relearn_apply as pa

    # over_powered subagent fan-out on a premium model, in-window.
    now = utcnow()
    for i in range(4):
        db.insert_span(make_llm_span(
            agent_id="claude-code-x", provider="anthropic", model="claude-opus-4-8",
            billing_account="anthropic", input_tokens=60_000, output_tokens=400,
            cost_usd=0.5, session_id="s1", sub_agent_id=f"sa{i}",
            start_time=now - timedelta(days=2, minutes=i),
        ))
    # Home-anchored target allowlist: point Path.home() at tmp so the CLAUDE.md is "inside".
    home = tmp_path / "home"
    (home / "proj").mkdir(parents=True)
    monkeypatch.setattr(pa.Path, "home", classmethod(lambda cls: home))
    target = home / "proj" / "CLAUDE.md"

    token = app.state.relearn_write_token
    hdr = {"X-TJ-Local-Token": token}
    await client.post("/api/v1/relearn/cost-proposals/refresh", headers=hdr)
    proposals = (await client.get("/api/v1/relearn/cost-proposals")).json()["proposals"]
    sub = next(p for p in proposals if p["analyzer"] == "subagent")
    assert sub["apply_capable"] is True
    assert sub["advise_only"] is False

    body = {"proposal_id": sub["proposal_id"], "target_path": str(target)}

    # Dry-run: a diff, nothing written, no cost marker.
    dry = await client.post(
        "/api/v1/relearn/cost-proposals/apply-workspace",
        json={**body, "go": False}, headers=hdr,
    )
    assert dry.status_code == 200
    assert dry.json()["applied"]["dry_run"] is True
    assert not target.exists()

    # Write: the note lands, reversibly, and a cost marker is recorded.
    wrote = await client.post(
        "/api/v1/relearn/cost-proposals/apply-workspace",
        json={**body, "go": True}, headers=hdr,
    )
    assert wrote.status_code == 200
    assert target.exists()
    assert "tokenjam" in target.read_text(encoding="utf-8")
    assert wrote.json()["cost_record"] is not None

    applied = (await client.get("/api/v1/relearn/cost-applied")).json()
    assert any(r["analyzer"] == "subagent" for r in applied["applied"])


async def test_cost_apply_workspace_writes_skill_for_script_and_reverts(
    app, client, db, monkeypatch, tmp_path,
):
    """`apply-workspace` is generic across analyzers, not special-cased to
    `subagent`: a `script` proposal (rung 2 skill note, not rung 1) routes
    through the SAME path, writes, and reverts cleanly."""
    from tokenjam.core.optimize import relearn_apply as pa

    # A deterministic tool-call cluster: >=20 sessions running the identical
    # single-tool structure, which is what MIN_CLUSTER_INSTANCES flags.
    # `agent_id="claude-code-x"` so the window's dominant persona resolves to
    # "claude-code" — the rung-2 skill write this test exercises is only
    # offered for that persona (SDK/unknown windows get a snippet instead;
    # see `cost_proposals._persona_gated_write_fields`).
    base = utcnow() - timedelta(days=2)
    for i in range(20):
        sid = f"det-{i}"
        db.upsert_session(make_session(
            agent_id="claude-code-x", session_id=sid, plan_tier="api",
            duration_seconds=10.0, total_cost_usd=0.02,
        ))
        span = make_tool_span(agent_id="claude-code-x", tool_name="bash")
        span.session_id = sid
        span.start_time = base + timedelta(minutes=i)
        span.attributes = {GenAIAttributes.TOOL_INPUT: {"command": "git pull"}}
        db.insert_span(span)

    home = tmp_path / "home"
    (home / "proj").mkdir(parents=True)
    monkeypatch.setattr(pa.Path, "home", classmethod(lambda cls: home))
    target = home / "proj" / ".claude" / "skills" / "det-pattern" / "SKILL.md"

    token = app.state.relearn_write_token
    hdr = {"X-TJ-Local-Token": token}
    await client.post("/api/v1/relearn/cost-proposals/refresh", headers=hdr)
    proposals = (await client.get("/api/v1/relearn/cost-proposals")).json()["proposals"]
    script_props = [p for p in proposals if p["analyzer"] == "script"]
    assert script_props, proposals
    prop = script_props[0]
    assert prop["apply_capable"] is True
    assert prop["advise_only"] is False
    assert prop["rung"] == 2

    body = {"proposal_id": prop["proposal_id"], "target_path": str(target)}

    # Dry-run: a diff, nothing written yet.
    dry = await client.post(
        "/api/v1/relearn/cost-proposals/apply-workspace",
        json={**body, "go": False}, headers=hdr,
    )
    assert dry.status_code == 200
    assert dry.json()["applied"]["dry_run"] is True
    assert not target.exists()

    # Write: the skill note lands, reversibly.
    wrote = await client.post(
        "/api/v1/relearn/cost-proposals/apply-workspace",
        json={**body, "go": True}, headers=hdr,
    )
    assert wrote.status_code == 200
    assert target.exists()
    content = target.read_text(encoding="utf-8")
    assert "tokenjam:relearn:" in content
    assert wrote.json()["cost_record"] is not None
    fix_id = wrote.json()["applied"]["record"]["id"]

    applied = (await client.get("/api/v1/relearn/cost-applied")).json()
    assert any(r["analyzer"] == "script" for r in applied["applied"])

    # Revert: the freshly-created skill file is removed (a "created", not
    # "restored", backup) — the round trip a workspace write must guarantee.
    revert = await client.post(
        f"/api/v1/relearn/{fix_id}/revert", headers=hdr,
    )
    assert revert.status_code == 200
    assert revert.json()["state"] == "reverted"
    assert not target.exists()


# --- the cost-ledger surfaces carry the plan-tier framing ------------------- #
# `total_realized_usd` can only ever count the API-billed slice of a corpus.
# Without the framing block the UI has no way to know that, and it renders a
# small, misleading dollar figure as the headline while the (complete) token
# figure is demoted. The payload therefore carries the same `framing` block
# every other cost surface emits.

async def test_cost_applied_payload_carries_plan_tier_framing(client):
    r = await client.get("/api/v1/relearn/cost-applied")
    assert r.status_code == 200
    framing = r.json()["framing"]
    assert framing["display_rule"]
    assert "pricing_mode" in framing


async def test_cost_proposals_payload_carries_plan_tier_framing(client):
    """The estimated-recoverable tile picks its unit from the same server-side
    decision as the measured tile beside it, so the two never disagree."""
    r = await client.get("/api/v1/relearn/cost-proposals")
    assert r.status_code == 200
    payload = r.json()
    assert payload["framing"]["display_rule"]
    # The rollup carries both units plus the coverage the tile has to quote.
    rollup = payload["rollup"]
    assert "estimated_recoverable_tokens" in rollup
    assert "token_proposal_count" in rollup
    assert "deduplicated_proposal_count" in rollup
