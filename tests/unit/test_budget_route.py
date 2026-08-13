"""GET/POST /api/v1/budget — the redesigned Budget page's two-zone response.

Pins the response SHAPE (`coding` zone grouped by tool, `sdk` zone grouped by
literal agent_id) and the write paths for each: a coding-tool group's daily
cap (`scope: "group:<id>"`), the coding-zone default (`scope:
"defaults_coding"`), the SDK-zone default (`scope: "defaults"`), and an SDK
workflow's own agent_id. Does not touch `POST /budget/provider` (a separate,
unrelated forecast concept).
"""
from __future__ import annotations

from datetime import date

import pytest
from fastapi.testclient import TestClient

from tokenjam.api.app import create_app
from tokenjam.core.config import (
    ApiAuthConfig,
    ApiConfig,
    AgentConfig,
    BudgetConfig,
    CodingGroupConfig,
    DefaultsConfig,
    GroupBudgetConfig,
    load_config,
    TjConfig,
    active_config_path,
)
from tokenjam.core.db import InMemoryBackend
from tokenjam.core.ingest import build_default_pipeline
from tests.factories import make_session


def _config(tmp_path, **kwargs) -> TjConfig:
    cfg = TjConfig(version="1", api=ApiConfig(auth=ApiAuthConfig(enabled=False)), **kwargs)
    path = tmp_path / "tokenjam.toml"
    path.write_text('version = "1"\n')
    cfg.config_path = path
    return cfg


def _client(config, db):
    app = create_app(config=config, db=db, ingest_pipeline=build_default_pipeline(db, config))
    return TestClient(app)


@pytest.fixture
def db():
    backend = InMemoryBackend()
    yield backend
    backend.close()


def test_get_budget_shape_has_coding_and_sdk_zones(tmp_path, db):
    config = _config(tmp_path)
    with _client(config, db) as client:
        resp = client.get("/api/v1/budget")
    assert resp.status_code == 200
    body = resp.json()
    assert "coding" in body
    assert "sdk" in body
    assert "groups" in body["coding"]
    assert "defaults" in body["coding"]
    assert "agents" in body["sdk"]
    assert "defaults" in body["sdk"]
    # Unchanged, unrelated concept — still present.
    assert "provider_budgets" in body
    assert "framing" in body


def test_coding_zone_groups_claude_code_projects_into_one_row(tmp_path, db):
    for aid in ["claude-code-proj-a", "claude-code-proj-b", "claude-code"]:
        db.upsert_session(make_session(agent_id=aid, session_id=f"s-{aid}"))
    config = _config(tmp_path)
    with _client(config, db) as client:
        resp = client.get("/api/v1/budget")
    groups = resp.json()["coding"]["groups"]
    assert set(groups.keys()) == {"claude-code"}
    assert set(groups["claude-code"]["members"]) == {
        "claude-code-proj-a", "claude-code-proj-b", "claude-code",
    }


def test_codex_exact_id_forms_its_own_group_separate_from_claude_code(tmp_path, db):
    db.upsert_session(make_session(agent_id="claude-code-proj-a", session_id="s1"))
    db.upsert_session(make_session(agent_id="codex_exec", session_id="s2"))
    config = _config(tmp_path)
    with _client(config, db) as client:
        resp = client.get("/api/v1/budget")
    groups = resp.json()["coding"]["groups"]
    assert set(groups.keys()) == {"claude-code", "codex"}
    assert groups["codex"]["members"] == ["codex_exec"]


def test_sdk_workflow_gets_its_own_row_not_grouped(tmp_path, db):
    db.upsert_session(make_session(agent_id="sdk-workflow-a", session_id="s1"))
    db.upsert_session(make_session(agent_id="sdk-workflow-b", session_id="s2"))
    config = _config(tmp_path)
    with _client(config, db) as client:
        resp = client.get("/api/v1/budget")
    sdk_agents = resp.json()["sdk"]["agents"]
    assert set(sdk_agents.keys()) == {"sdk-workflow-a", "sdk-workflow-b"}


def test_only_present_or_explicitly_configured_groups_are_returned(tmp_path, db):
    """No data at all for codex, but it IS explicitly configured -- still
    shown (so a pre-set cap survives before the first session lands).
    claude-code has data but no explicit config -- also shown. Nothing else
    invented."""
    db.upsert_session(make_session(agent_id="claude-code-proj-a", session_id="s1"))
    config = _config(
        tmp_path,
        coding_agents={"codex": CodingGroupConfig(budget=GroupBudgetConfig(daily_usd=20.0))},
    )
    with _client(config, db) as client:
        resp = client.get("/api/v1/budget")
    groups = resp.json()["coding"]["groups"]
    assert set(groups.keys()) == {"claude-code", "codex"}
    assert groups["codex"]["members"] == []
    assert groups["codex"]["configured"]["daily_usd"] == 20.0


def test_post_group_scope_writes_the_group_cap(tmp_path, db):
    config = _config(tmp_path)
    with _client(config, db) as client:
        resp = client.post("/api/v1/budget", json={"scope": "group:claude-code", "daily_usd": 50.0})
    assert resp.status_code == 200
    body = resp.json()
    assert body["coding"]["groups"]["claude-code"]["configured"]["daily_usd"] == 50.0
    assert body["coding"]["groups"]["claude-code"]["effective"]["daily_usd"] == 50.0


def test_post_group_scope_rejects_session_usd(tmp_path, db):
    config = _config(tmp_path)
    with _client(config, db) as client:
        resp = client.post(
            "/api/v1/budget",
            json={"scope": "group:claude-code", "daily_usd": 50.0, "session_usd": 5.0},
        )
    assert resp.status_code == 400
    assert "session_usd" in resp.json()["error"]


def test_post_defaults_coding_scope_writes_the_zone_default(tmp_path, db):
    config = _config(tmp_path)
    with _client(config, db) as client:
        resp = client.post("/api/v1/budget", json={"scope": "defaults_coding", "daily_usd": 25.0})
    assert resp.status_code == 200
    assert resp.json()["coding"]["defaults"]["daily_usd"] == 25.0


def test_post_sdk_agent_scope_unchanged_daily_and_session(tmp_path, db):
    config = _config(tmp_path)
    with _client(config, db) as client:
        resp = client.post(
            "/api/v1/budget",
            json={"scope": "sdk-workflow-x", "daily_usd": 12.0, "session_usd": 3.0},
        )
    assert resp.status_code == 200
    sdk = resp.json()["sdk"]["agents"]["sdk-workflow-x"]
    assert sdk["configured"]["daily_usd"] == 12.0
    assert sdk["configured"]["session_usd"] == 3.0


def test_get_response_is_a_superset_carrying_the_legacy_top_level_shape(tmp_path, db):
    """The still-committed old BudgetView reads `data.defaults.daily_usd` /
    `data.defaults.session_usd` (unguarded) and `Object.entries(data.agents)`
    directly off the GET response -- not the new `coding`/`sdk` zones. Until
    that view is rewritten (a separate, in-flight change on this same
    branch), the response MUST still carry those two top-level keys with the
    exact old per-agent-flat-row values, or the page throws on render."""
    db.upsert_session(make_session(agent_id="claude-code-proj-a", session_id="s1"))
    db.upsert_session(make_session(agent_id="sdk-workflow-a", session_id="s2"))
    config = _config(
        tmp_path,
        defaults=DefaultsConfig(budget=BudgetConfig(daily_usd=9.0, session_usd=2.0)),
        agents={"sdk-workflow-a": AgentConfig(budget=BudgetConfig(daily_usd=4.0, session_usd=1.0))},
    )
    with _client(config, db) as client:
        resp = client.get("/api/v1/budget")
    body = resp.json()

    # Legacy top-level "defaults" — exactly the old {daily_usd, session_usd} shape.
    assert body["defaults"] == {"daily_usd": 9.0, "session_usd": 2.0}

    # Legacy top-level "agents" — ONE FLAT ROW PER agent_id, coding and SDK
    # alike, un-grouped (the pre-redesign behavior), each with
    # configured/effective sub-objects.
    assert set(body["agents"].keys()) == {"claude-code-proj-a", "sdk-workflow-a"}
    assert body["agents"]["sdk-workflow-a"]["configured"] == {"daily_usd": 4.0, "session_usd": 1.0}
    assert body["agents"]["sdk-workflow-a"]["effective"] == {"daily_usd": 4.0, "session_usd": 1.0}
    # A coding agent with no per-agent override still gets a flat legacy row,
    # falling back to the legacy defaults (not the new coding-group default).
    assert body["agents"]["claude-code-proj-a"]["configured"] == {"daily_usd": None, "session_usd": None}
    assert body["agents"]["claude-code-proj-a"]["effective"] == {"daily_usd": 9.0, "session_usd": 2.0}


def test_legacy_shaped_post_roundtrips_through_the_new_schema(tmp_path, db):
    """Exactly the payload the still-committed old BudgetView.handleSave
    sends: {scope, daily_usd, session_usd}, with scope either "defaults" or
    a literal agent_id -- including a claude-code-<project> id, since the
    old UI has no concept of coding-tool grouping and writes one row per
    agent_id it saw. Must persist through the NEW config schema (BudgetConfig
    on config.agents[<id>].budget, unchanged) and the response must still
    reflect it under the legacy top-level "agents" key."""
    config = _config(tmp_path)
    with _client(config, db) as client:
        # Legacy defaults-scope save.
        resp = client.post(
            "/api/v1/budget", json={"scope": "defaults", "daily_usd": 15.0, "session_usd": 3.0},
        )
        assert resp.status_code == 200
        assert resp.json()["defaults"] == {"daily_usd": 15.0, "session_usd": 3.0}

        # Legacy per-agent save on a literal coding-project agent_id, exactly
        # as the un-rewritten UI still does (one row per project it observed).
        resp2 = client.post(
            "/api/v1/budget",
            json={"scope": "claude-code-my-project", "daily_usd": 6.0, "session_usd": 1.25},
        )
        assert resp2.status_code == 200
        body2 = resp2.json()
        assert body2["agents"]["claude-code-my-project"]["configured"] == {
            "daily_usd": 6.0, "session_usd": 1.25,
        }

    # And it actually persisted to config, not just the in-memory response.
    reloaded = load_config(str(config.config_path))
    assert reloaded.defaults.budget.daily_usd == 15.0
    assert reloaded.defaults.budget.session_usd == 3.0
    assert reloaded.agents["claude-code-my-project"].budget.daily_usd == 6.0
    assert reloaded.agents["claude-code-my-project"].budget.session_usd == 1.25


def test_post_daily_only_preserves_an_existing_session_usd(tmp_path, db):
    """A save that only touches daily_usd must not erase a session_usd the
    user already had configured on that same SDK agent scope."""
    config = _config(
        tmp_path,
        agents={"legacy-agent": AgentConfig(budget=BudgetConfig(daily_usd=5.0, session_usd=1.5))},
    )
    with _client(config, db) as client:
        resp = client.post("/api/v1/budget", json={"scope": "legacy-agent", "daily_usd": 8.0})
    assert resp.status_code == 200
    sdk = resp.json()["sdk"]["agents"]["legacy-agent"]
    assert sdk["configured"]["daily_usd"] == 8.0
    assert sdk["configured"]["session_usd"] == 1.5


# --------------------------------------------------------------------------- #
# Budgets are an SDK-workflow feature (founder decision).
#
# The surface lives in the Sessions screen's SDK-services zone, so a
# coding-agent cap has nowhere honest to render. Scoped in the ROUTE rather
# than the view, so a client that forgets the filter still cannot show one.
# --------------------------------------------------------------------------- #
def test_sdk_scope_drops_every_coding_agent_budget(tmp_path):
    from tokenjam.api.routes.budget import _budget_payload
    from tokenjam.core.config import StorageConfig, TjConfig

    cfg = TjConfig(version="1", storage=StorageConfig(path=str(tmp_path / "t.duckdb")))
    agent_ids = [
        "claude-code-tokenjam", "claude-code-splito", "claude-code",
        "codex-app-server", "billing-service", "sdk-workload-oversized-model",
    ]

    scoped = _budget_payload(cfg, agent_ids, persona="sdk")

    # No coding groups, and none of the flat map's keys is a coding agent.
    assert scoped["coding"]["groups"] == {}
    assert set(scoped["agents"]) == {"billing-service", "sdk-workload-oversized-model"}
    # `codex-app-server` is the case the two classifiers disagree on: SDK under
    # `agent_kind` (codex matches only the exact id `codex_exec` there), coding
    # under the persona predicate. This scope excludes it, because showing a
    # coding agent breaks the decision and hiding an SDK one does not.
    assert "codex-app-server" not in scoped["agents"]
    assert set(scoped["sdk"]["agents"]) == set(scoped["agents"])
    # THE pin, stated as the property rather than as a key list: nothing a
    # reader could render as an agent row may be an interactive coding agent.
    from tokenjam.core.alerts import is_interactive_coding_agent

    for key in (*scoped["agents"], *scoped["sdk"]["agents"], *scoped["coding"]["groups"]):
        assert not is_interactive_coding_agent(key), key

    # Unscoped keeps the full payload — the CLI and any pre-scope caller still
    # read it, so the narrowing must be opt-in rather than a silent change.
    full = _budget_payload(cfg, agent_ids, persona=None)
    assert any(is_interactive_coding_agent(k) for k in full["agents"])
    assert full["coding"]["groups"], "the unscoped payload still carries coding groups"


def test_budget_route_rejects_an_unknown_persona(tmp_path, db):
    config = _config(tmp_path)
    with _client(config, db) as client:
        assert client.get("/api/v1/budget?persona=sdkk").status_code == 400
        assert client.get("/api/v1/budget?persona=sdk").status_code == 200
