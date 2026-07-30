"""Integration tests for the cost-proposal Review-inbox endpoints
(api/routes/relearn.py). Talks through the real ASGI app so the read/write
surface + the advise-only marker round-trip are proven at the route.

Isolated: InMemoryBackend + a tmp storage path; nothing touches a real store.
"""
from __future__ import annotations

from datetime import timedelta

import httpx
import pytest

from tokenjam.core.rulewrite.kinds import DELIVERY_SKILL

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
    assert rec["past_overspend_usd"] == prop["past_overspend_usd"]

    applied = (await client.get("/api/v1/relearn/cost-applied")).json()
    assert len(applied["applied"]) == 1

    # Revert flips it back.
    rev = await client.post(
        f"/api/v1/relearn/cost-applied/{rec['id']}/revert",
        headers={"X-TJ-Local-Token": token},
    )
    assert rev.status_code == 200
    assert rev.json()["state"] == "reverted"


# --- an applied proposal loses its OFFER and keeps its FIGURE ---------------- #

async def test_an_applied_proposal_is_withdrawn_on_the_PAYLOAD(app, client):
    """Verified WITHOUT consulting a second endpoint — that is the whole point.

    THE DEFECT. This route computed a filtered ``open_proposals`` for its rollup
    and then returned the UNFILTERED ``proposals`` list, and the rows carried no
    ``applied``/``applied_at`` field at all. So an already-applied proposal went
    out advertising ``apply_capable: true`` with nothing on it to say otherwise.
    Only the browser was safe, and only because it independently re-fetches
    ``/relearn/cost-applied`` and filters client-side — meaning the CLI,
    ``--json``, an export and every future surface saw an offer to re-apply
    something already done, and had to know to cross-reference a second endpoint
    to avoid it. Measured on a real corpus: the ONE apply-capable row in the
    whole inbox was a fix that had already been applied.

    Critical Rule 32: the offer is withdrawn, the figure is kept, and the row
    stays listed CARRYING an applied state.
    """
    token = app.state.relearn_write_token
    hdr = {"X-TJ-Local-Token": token}
    await client.post("/api/v1/relearn/cost-proposals/refresh", headers=hdr)
    before = (await client.get("/api/v1/relearn/cost-proposals")).json()["proposals"]
    prop = next(p for p in before if p["analyzer"] == "cache")
    assert prop["applied"] is False, "every row carries the field, open ones included"
    assert prop["applied_at"] is None

    applied = await client.post(
        "/api/v1/relearn/cost-proposals/apply",
        json={"proposal_id": prop["proposal_id"]}, headers=hdr,
    )
    assert applied.status_code == 200

    after = (await client.get("/api/v1/relearn/cost-proposals")).json()["proposals"]
    row = next(p for p in after if p["signature"] == prop["signature"])
    assert row["applied"] is True
    assert row["applied_at"], "an applied row must carry WHEN"
    assert row.get("apply_capable") is not True, (
        "an applied fix is still being offered for application"
    )
    # The figure is what the behaviour ALREADY cost. Applying the fix afterwards
    # does not un-spend the money, and a gate that edits a past figure is the
    # "this was free" conflation Critical Rule 32 exists to stop.
    assert row["past_overspend_usd"] == prop["past_overspend_usd"]


async def test_a_reverted_proposal_is_offered_again_on_the_payload(app, client):
    """A revert is the user saying the fix is no longer in place."""
    token = app.state.relearn_write_token
    hdr = {"X-TJ-Local-Token": token}
    await client.post("/api/v1/relearn/cost-proposals/refresh", headers=hdr)
    proposals = (await client.get("/api/v1/relearn/cost-proposals")).json()["proposals"]
    prop = next(p for p in proposals if p["analyzer"] == "cache")
    rec = (await client.post(
        "/api/v1/relearn/cost-proposals/apply",
        json={"proposal_id": prop["proposal_id"]}, headers=hdr,
    )).json()
    await client.post(
        f"/api/v1/relearn/cost-applied/{rec['id']}/revert", headers=hdr,
    )

    after = (await client.get("/api/v1/relearn/cost-proposals")).json()["proposals"]
    row = next(p for p in after if p["signature"] == prop["signature"])
    assert row["applied"] is False
    assert row["applied_at"] is None


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
        json={"proposal_id": prop["proposal_id"], "past_overspend_usd": 9999.0},
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
    """A CC-origin subagent proposal routes a reversible CLAUDE.md rule through the
    existing relearn apply path, then records the cost marker for delta-verify."""
    from tokenjam.core.optimize import relearn_apply as pa

    # over_powered subagent fan-out on a premium model, in-window. Sized past
    # the $5 write floor (`write_budget.MIN_NET_WRITE_USD`): this test asserts
    # the card is APPLY-CAPABLE, and the budget declines a permanent write for
    # a couple of dollars, so a cent-scale seed no longer reaches that path.
    now = utcnow()
    for i in range(4):
        db.insert_span(make_llm_span(
            agent_id="claude-code-x", provider="anthropic", model="claude-opus-4-8",
            billing_account="anthropic", input_tokens=60_000, output_tokens=400,
            cost_usd=15.0, session_id="s1", sub_agent_id=f"sa{i}",
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


async def test_cost_apply_workspace_writes_skill_for_a_skill_proposal_and_reverts(
    app, client, config, db, monkeypatch, tmp_path,
):
    """`apply-workspace` is generic across analyzers, not special-cased to
    `subagent`: a skill-note proposal routes through the SAME path,
    writes, and reverts cleanly.

    Two facts are pinned here. First, a deterministic tool-call cluster on a
    claude-code window produces NO `script` card any more: the analyzer has no
    fix that survives for that persona and is skipped before it runs. Second,
    the generic skill write/revert route still works, exercised by seeding the
    proposal the adapter builds directly into the store."""
    from tokenjam.core.optimize import relearn_apply as pa, relearn_store
    from tokenjam.core.optimize.analyzers.workflow_restructure import (
        WorkflowCluster,
        WorkflowRestructureFinding,
    )
    from tokenjam.core.optimize.cost_proposals import _script_to_proposals

    # A deterministic tool-call cluster: >=20 sessions running the identical
    # single-tool structure, which is what MIN_CLUSTER_INSTANCES flags.
    # `agent_id="claude-code-x"` so the window's dominant persona resolves to
    # "claude-code".
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
    assert [p for p in proposals if p["analyzer"] == "script"] == []

    seeded = _script_to_proposals(
        WorkflowRestructureFinding(
            clusters=[WorkflowCluster(
                signature=[{"tool": "bash", "args": ["command_string"]}], instances=25,
                avg_cost_usd=0.02, avg_duration_seconds=1.5, example_session_id="det-0",
                avg_tokens=500, total_cost_usd=0.5, total_tokens=12_500,
                example_session_ids=["det-0", "det-1", "det-2"],
            )],
            sessions_examined=25, degraded=False,
            past_overspend_usd=0.5, past_overspend_tokens=12_500,
            estimate_basis="script basis",
        ),
        persona="claude-code",
    )
    relearn_store.write_cost_proposals(seeded, config=config)

    proposals = (await client.get("/api/v1/relearn/cost-proposals")).json()["proposals"]
    script_props = [p for p in proposals if p["analyzer"] == "script"]
    assert script_props, proposals
    prop = script_props[0]
    assert prop["apply_capable"] is True
    assert prop["advise_only"] is False
    assert prop["delivery"] == DELIVERY_SKILL

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
    """The headline tile picks its unit from the same server-side decision as
    the measured tile beside it, so the two never disagree."""
    r = await client.get("/api/v1/relearn/cost-proposals")
    assert r.status_code == 200
    payload = r.json()
    assert payload["framing"]["display_rule"]
    # ONE aggregate block, carrying both units plus the coverage the tile must
    # quote. There is deliberately no second `rollup` key beside it.
    assert "rollup" not in payload
    block = payload["past_overspend"]
    assert "past_overspend_tokens" in block
    assert "token_proposal_count" in block
    assert "deduplicated_proposal_count" in block


# --- the past-overspend headline ships on the payload ---------------------- #
# The gap this closes: the backend priced the observed figure correctly, wrote
# an honesty basis for it, and handed it to a dashboard that never rendered it.
# Both leading surfaces (the Dashboard hero and the Review inbox headline) read
# THIS block, so it has to be on the payload of the endpoint they already call.

async def test_cost_proposals_payload_carries_the_past_overspend_block(app, client):
    hdr = {"X-TJ-Local-Token": app.state.relearn_write_token}
    refresh = await client.post("/api/v1/relearn/cost-proposals/refresh", headers=hdr)
    assert refresh.status_code == 200

    payload = (await client.get("/api/v1/relearn/cost-proposals")).json()
    block = payload["past_overspend"]
    assert block["past_overspend_usd"] >= 0
    assert "window_days" in block
    assert block["basis"]
    assert block["disclosure"]

    # ONE dollar total on the block. A second key, `observed_cost_usd`, used to
    # ship here under a disclosure calling the headline a subset of it; on live
    # data it covered 2 of the headline's 13 proposals, so the claim was false as
    # published. Key and disclosure are deleted — every figure this block
    # publishes now covers the same set of proposals.
    assert {k for k in block if k.endswith("_usd")} == {"past_overspend_usd"}
    assert "cost_disclosure" not in block
    # No retired per-analyzer dollar field, and no pace to project at.
    assert "projection_ratio" not in block
    assert not [k for k in block if "monthly" in k or "projected" in k]

    # Every proposal carries its own past-tense figure, so a card never has to
    # re-derive its own headline.
    for proposal in payload["proposals"]:
        assert "past_overspend_usd" in proposal
        assert "past_overspend_basis" in proposal
        assert "estimated_recoverable_usd" not in proposal
        assert "estimated_monthly_usd" not in proposal


async def test_register_source_path_previews_then_applies_a_model_swap(
    client, app, config, tmp_path, monkeypatch,
):
    """The eleven-row gap, closed end to end.

    Every live ``downsize`` model-swap proposal carried a real fix snippet and
    ``apply_capable: false``, for exactly one reason: nobody had told tokenjam
    where those agents' source lives, and it refuses to scan a filesystem looking
    (``config.AgentConfig.source_path`` is opt-in, never inferred). So the row's
    only offer was a copy box and a "Mark applied" that recorded the user doing it
    by hand. This proves the row can now ask once and then write.

    Also proves the two properties the ask must not cost us: the preview writes
    NOTHING (not the file, not the config), and the write reads its target out of
    the config it just registered rather than out of the request body.
    """
    import pathlib as pa
    import subprocess

    from tokenjam.core.config import AgentConfig, write_config
    from tokenjam.core.optimize.analyzers.downsize_agents import build_agent_price_rows
    from tokenjam.core.optimize.cost_proposals import _downsize_agent_proposals
    from tokenjam.core.optimize.types import DowngradeFinding
    from tokenjam.core.optimize import relearn_store

    home = tmp_path / "home"
    repo = home / "svc-a"
    repo.mkdir(parents=True)
    monkeypatch.setattr(pa.Path, "home", classmethod(lambda cls: home))
    source = repo / "agent.py"
    source.write_text('MODEL = "claude-opus-4-8"\n', encoding="utf-8")
    for args in (["init", "-q"], ["add", "-A"]):
        subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)
    subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "seed"],
        cwd=repo, check=True, capture_output=True,
    )

    # The config must have a real path on disk, because registration WRITES to it.
    config.config_path = tmp_path / "tj.toml"
    write_config(config, config.config_path)

    rows = build_agent_price_rows([{
        "session_id": "s1", "agent_id": "svc-a", "provider": "anthropic",
        "model": "claude-opus-4-8", "alt_model": "claude-haiku-4-5",
        "input_tokens": 100_000, "output_tokens": 20_000,
        "cache_tokens": 500_000, "cache_write_tokens": 40_000,
        "started_at": utcnow() - timedelta(days=1),
    }], 30.0)
    finding = DowngradeFinding(
        candidate_sessions=1, total_sessions=1, actual_cost_usd=9.0,
        alternative_cost_usd=1.0, monthly_savings_usd=8.0, percent_of_sessions=100.0,
        examples=[], suggestions={"claude-opus-4-8": "claude-haiku-4-5"},
    )
    finding.per_agent = rows
    # Built with the agent UNREGISTERED, which is the state the eleven rows were in.
    relearn_store.write_cost_proposals(
        _downsize_agent_proposals(finding, config, persona="sdk"), config=config,
    )

    hdr = {"X-TJ-Local-Token": app.state.relearn_write_token}
    proposals = (await client.get("/api/v1/relearn/cost-proposals")).json()["proposals"]
    prop = [p for p in proposals if p["analyzer"] == "downsize"][0]
    assert prop["needs_source_path"] is True
    assert prop["apply_capable"] is True
    # No apply_kind yet: with no registered path there is no deterministic edit,
    # so the row must not reach the endpoint that assumes one.
    assert prop["apply_kind"] == ""
    assert prop["target_path"] == ""
    # The caveat rides its own field so the card can keep it visible beside the
    # button rather than inside the collapsed description.
    assert "NOT measured here" in prop["apply_caveat"]

    body = {"proposal_id": prop["proposal_id"], "source_path": str(repo)}

    # PREVIEW writes nothing: not the source file, not the config.
    before_source = source.read_text(encoding="utf-8")
    before_config = config.config_path.read_text(encoding="utf-8")
    dry = await client.post(
        "/api/v1/relearn/cost-proposals/register-source-path",
        json={**body, "go": False}, headers=hdr,
    )
    assert dry.status_code == 200, dry.text
    assert dry.json()["applied"]["dry_run"] is True
    assert "claude-haiku-4-5" in dry.json()["applied"]["diff"]
    assert dry.json()["target_path"] == str(source)
    assert source.read_text(encoding="utf-8") == before_source
    assert config.config_path.read_text(encoding="utf-8") == before_config

    # APPLY registers the path and writes the swap.
    wrote = await client.post(
        "/api/v1/relearn/cost-proposals/register-source-path",
        json={**body, "go": True}, headers=hdr,
    )
    assert wrote.status_code == 200, wrote.text
    assert 'MODEL = "claude-haiku-4-5"' in source.read_text(encoding="utf-8")
    # Registered PER AGENT, which is what unlocks the same agent's other rows at
    # the next recompute rather than asking again per proposal.
    assert config.agents["svc-a"].source_path == str(repo.resolve())
    assert "svc-a" in config.config_path.read_text(encoding="utf-8")


async def test_register_source_path_refuses_a_path_it_cannot_swap_in(
    client, app, config, tmp_path, monkeypatch,
):
    """A precheck failure after the path is given says WHY on the row, and leaves
    no registration behind. Silently failing would be worse than the copy box it
    replaced, and a registration pointing at a repo the swap is impossible in
    would make every later recompute produce the same dead button.
    """
    import pathlib as pa

    from tokenjam.core.config import write_config
    from tokenjam.core.optimize.analyzers.downsize_agents import build_agent_price_rows
    from tokenjam.core.optimize.cost_proposals import _downsize_agent_proposals
    from tokenjam.core.optimize.types import DowngradeFinding
    from tokenjam.core.optimize import relearn_store

    home = tmp_path / "home"
    plain = home / "not-a-repo"
    plain.mkdir(parents=True)
    monkeypatch.setattr(pa.Path, "home", classmethod(lambda cls: home))
    (plain / "agent.py").write_text('MODEL = "claude-opus-4-8"\n', encoding="utf-8")

    config.config_path = tmp_path / "tj.toml"
    write_config(config, config.config_path)

    rows = build_agent_price_rows([{
        "session_id": "s1", "agent_id": "svc-a", "provider": "anthropic",
        "model": "claude-opus-4-8", "alt_model": "claude-haiku-4-5",
        "input_tokens": 100_000, "output_tokens": 20_000,
        "cache_tokens": 500_000, "cache_write_tokens": 40_000,
        "started_at": utcnow() - timedelta(days=1),
    }], 30.0)
    finding = DowngradeFinding(
        candidate_sessions=1, total_sessions=1, actual_cost_usd=9.0,
        alternative_cost_usd=1.0, monthly_savings_usd=8.0, percent_of_sessions=100.0,
        examples=[], suggestions={"claude-opus-4-8": "claude-haiku-4-5"},
    )
    finding.per_agent = rows
    relearn_store.write_cost_proposals(
        _downsize_agent_proposals(finding, config, persona="sdk"), config=config,
    )

    hdr = {"X-TJ-Local-Token": app.state.relearn_write_token}
    proposals = (await client.get("/api/v1/relearn/cost-proposals")).json()["proposals"]
    prop = [p for p in proposals if p["analyzer"] == "downsize"][0]

    refused = await client.post(
        "/api/v1/relearn/cost-proposals/register-source-path",
        json={"proposal_id": prop["proposal_id"], "source_path": str(plain), "go": True},
        headers=hdr,
    )
    assert refused.status_code == 409
    assert "not inside a git repository" in refused.json()["detail"]
    assert "svc-a" not in config.config_path.read_text(encoding="utf-8")


async def test_register_source_path_refused_by_apply_leaves_no_registration(
    client, app, config, db, tmp_path, monkeypatch,
):
    """A refusal that surfaces only INSIDE ``apply_relearn_fix`` — the
    active-session gate, which ``model_swap_precheck`` doesn't check — must
    leave the config exactly as untouched as a precheck-time refusal does.

    Registering the path before the apply runs would persist a path the swap
    then refused to write through, silently contradicting the endpoint's own
    docstring promise that a refused apply never leaves a path behind.
    """
    import pathlib as pa
    import subprocess

    from tokenjam.core.config import write_config
    from tokenjam.core.optimize.analyzers.downsize_agents import build_agent_price_rows
    from tokenjam.core.optimize.cost_proposals import _downsize_agent_proposals
    from tokenjam.core.optimize.types import DowngradeFinding
    from tokenjam.core.optimize import relearn_store
    from tests.factories import make_session

    home = tmp_path / "home"
    repo = home / "svc-a"
    repo.mkdir(parents=True)
    monkeypatch.setattr(pa.Path, "home", classmethod(lambda cls: home))
    source = repo / "agent.py"
    source.write_text('MODEL = "claude-opus-4-8"\n', encoding="utf-8")
    for args in (["init", "-q"], ["add", "-A"]):
        subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)
    subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "seed"],
        cwd=repo, check=True, capture_output=True,
    )

    config.config_path = tmp_path / "tj.toml"
    write_config(config, config.config_path)

    # A live session in the same repo (label "svc-a"), seen moments ago — the
    # active-session gate that lives INSIDE apply_relearn_fix, not in
    # model_swap_precheck.
    db.upsert_session(make_session(agent_id="svc-a", status="active"))

    rows = build_agent_price_rows([{
        "session_id": "s1", "agent_id": "svc-a", "provider": "anthropic",
        "model": "claude-opus-4-8", "alt_model": "claude-haiku-4-5",
        "input_tokens": 100_000, "output_tokens": 20_000,
        "cache_tokens": 500_000, "cache_write_tokens": 40_000,
        "started_at": utcnow() - timedelta(days=1),
    }], 30.0)
    finding = DowngradeFinding(
        candidate_sessions=1, total_sessions=1, actual_cost_usd=9.0,
        alternative_cost_usd=1.0, monthly_savings_usd=8.0, percent_of_sessions=100.0,
        examples=[], suggestions={"claude-opus-4-8": "claude-haiku-4-5"},
    )
    finding.per_agent = rows
    relearn_store.write_cost_proposals(
        _downsize_agent_proposals(finding, config, persona="sdk"), config=config,
    )

    hdr = {"X-TJ-Local-Token": app.state.relearn_write_token}
    proposals = (await client.get("/api/v1/relearn/cost-proposals")).json()["proposals"]
    prop = [p for p in proposals if p["analyzer"] == "downsize"][0]

    before_source = source.read_text(encoding="utf-8")
    before_config = config.config_path.read_text(encoding="utf-8")

    refused = await client.post(
        "/api/v1/relearn/cost-proposals/register-source-path",
        json={"proposal_id": prop["proposal_id"], "source_path": str(repo), "go": True},
        headers=hdr,
    )
    assert refused.status_code == 409
    assert "active session" in refused.json()["detail"]

    # No rollback needed because nothing was ever written: neither the swap
    # target nor the config changed.
    assert source.read_text(encoding="utf-8") == before_source
    assert config.config_path.read_text(encoding="utf-8") == before_config
    assert "svc-a" not in config.agents
    assert "svc-a" not in config.config_path.read_text(encoding="utf-8")
