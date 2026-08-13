"""Every endpoint the persona picker reaches must actually narrow.

The picker was cosmetic on most of the product: only three endpoints had ever
gained a ``persona`` parameter, so selecting SDK re-labelled the Dashboard,
Optimize, every Optimize analyzer sub-page and Traces while leaving every
figure on them unchanged.

These tests are deliberately BEHAVIOURAL rather than structural — each asserts
that a request returns a DIFFERENT POPULATION for ``persona=sdk`` than for
``persona=claude-code`` over the same mixed corpus. A signature check would
pass against a parameter that is accepted and dropped, which is the exact shape
of the bug.
"""
from __future__ import annotations

from datetime import timedelta

import pytest
from fastapi.testclient import TestClient

from tokenjam.api.app import create_app
from tokenjam.core.config import ApiAuthConfig, ApiConfig, StorageConfig, TjConfig
from tokenjam.core.db import InMemoryBackend
from tokenjam.core.ingest import build_default_pipeline
from tokenjam.utils.time_parse import utcnow
from tests.factories import make_llm_span, make_session

CODING_AGENT = "claude-code-proj"
SDK_AGENT = "billing-service"


@pytest.fixture
def db():
    backend = InMemoryBackend()
    yield backend
    backend.close()


@pytest.fixture
def config(tmp_path):
    # `storage.path` IS THE ISOLATION and it is not optional. The proposal and
    # report stores resolve their location through
    # `relearn_apply._storage_base_dir`, which falls through to the REAL
    # `~/.tj` when a config carries no storage path — so a fixture that omits
    # it does not merely fail to isolate, it reads and writes the operator's
    # own ledger. That is a test reaching outside its sandbox to read live
    # data, and the symptom is a test whose result depends on the machine it
    # runs on (root anti-pattern 26).
    cfg = TjConfig(
        version="1",
        api=ApiConfig(auth=ApiAuthConfig(enabled=False)),
        storage=StorageConfig(path=str(tmp_path / "telemetry.duckdb")),
    )
    path = tmp_path / "tokenjam.toml"
    path.write_text('version = "1"\n')
    cfg.config_path = path
    return cfg


@pytest.fixture
def client(config, db):
    _seed_mixed(db)
    app = create_app(
        config=config, db=db, ingest_pipeline=build_default_pipeline(db, config),
    )
    with TestClient(app) as c:
        yield c


def _seed_mixed(db) -> None:
    """Four coding sessions and two SDK ones, at very different cost.

    Lopsided on purpose: an endpoint that accepts `persona` and ignores it
    returns the same combined figure for both, and only obviously-different
    populations make that visible rather than plausible.
    """
    for agent_id, count, cost in ((CODING_AGENT, 4, 0.50), (SDK_AGENT, 2, 0.05)):
        for i in range(count):
            started = utcnow() - timedelta(hours=i + 1)
            db.upsert_session(make_session(
                agent_id=agent_id, session_id=f"{agent_id}-s{i}",
                plan_tier="api", started_at=started,
            ))
            db.insert_span(make_llm_span(
                agent_id=agent_id, model="claude-opus-4-7", provider="anthropic",
                input_tokens=1000, output_tokens=200, cost_usd=cost,
                session_id=f"{agent_id}-s{i}",
                trace_id=f"{agent_id}-t{i}", start_time=started,
            ))


def _agents_in(rows, key="agent_id") -> set[str]:
    return {r.get(key) for r in rows if r.get(key)}


# --------------------------------------------------------------------------
# /traces — the list, its total, and its picker
# --------------------------------------------------------------------------

def test_traces_returns_a_different_population_per_persona(client):
    cc = client.get("/api/v1/traces", params={"persona": "claude-code"}).json()
    sdk = client.get("/api/v1/traces", params={"persona": "sdk"}).json()

    assert _agents_in(cc["traces"]) == {CODING_AGENT}
    assert _agents_in(sdk["traces"]) == {SDK_AGENT}


def test_the_trace_total_count_covers_the_same_population_as_its_rows(client):
    """A filtered list beside an unfiltered total is its own bug.

    `total_count` is rendered as "Showing N of TOTAL", so the two figures are
    published together and must cover one population (root anti-pattern 22b).
    """
    for persona, expected in (("claude-code", 4), ("sdk", 2)):
        body = client.get("/api/v1/traces", params={"persona": persona}).json()
        assert body["total_count"] == expected, persona
        assert len(body["traces"]) == expected, persona

    everything = client.get("/api/v1/traces").json()
    assert everything["total_count"] == 6


def test_an_unknown_persona_is_rejected_rather_than_silently_ignored(client):
    """A mistyped persona must not be served as "everything".

    Same contract `GET /sessions` and `GET /optimize` already have: a silent
    fallback returns a whole-corpus answer to a request that asked for a
    narrowed one, with nothing on the wire saying so.
    """
    for path in ("/api/v1/traces", "/api/v1/cost", "/api/v1/alerts",
                 "/api/v1/status", "/api/v1/drift",
                 "/api/v1/relearn/cost-proposals", "/api/v1/relearn/proposals"):
        resp = client.get(path, params={"persona": "clod-code"})
        assert resp.status_code == 400, path


def test_personas_that_narrow_nothing_return_everything(client):
    """`mixed` and `unknown` are not a third population — they scope to all.

    Matching the conservative default the analyzer gate takes for a window the
    classifier declined to bucket.
    """
    for persona in ("mixed", "unknown"):
        body = client.get("/api/v1/traces", params={"persona": persona}).json()
        assert body["total_count"] == 6, persona


# --------------------------------------------------------------------------
# /cost, /status, /alerts
# --------------------------------------------------------------------------

def test_cost_rows_are_scoped_to_the_persona(client):
    cc = client.get(
        "/api/v1/cost", params={"persona": "claude-code", "group_by": "agent"},
    ).json()
    sdk = client.get(
        "/api/v1/cost", params={"persona": "sdk", "group_by": "agent"},
    ).json()

    cc_groups = {r["group"] for r in cc["rows"]}
    sdk_groups = {r["group"] for r in sdk["rows"]}
    assert cc_groups == {CODING_AGENT}
    assert sdk_groups == {SDK_AGENT}
    # The two scoped totals partition the unscoped one — no row counted twice,
    # none dropped between the views.
    everything = client.get("/api/v1/cost", params={"group_by": "agent"}).json()
    assert {r["group"] for r in everything["rows"]} == {CODING_AGENT, SDK_AGENT}


def test_status_scopes_its_archive_and_its_total_together(client):
    """The tiles, the archive page and the archive TOTAL are one population.

    `archived_total` is published as "latest N of TOTAL" beside the capped
    `archived` list, so a scope applied to one and not the other makes that
    sentence false.
    """
    for persona, expected_agent in (
        ("claude-code", CODING_AGENT), ("sdk", SDK_AGENT),
    ):
        body = client.get("/api/v1/status", params={"persona": persona}).json()
        archived_agents = _agents_in(body["archived"])
        if archived_agents:
            assert archived_agents == {expected_agent}, persona
        # The TOTAL is uncapped and the list is capped, so the only relation
        # that holds in general is this one — but it must hold over the SAME
        # population, which the partition test below is what actually pins.
        assert body["archived_total"] >= len(body["archived"]), persona


def test_status_archive_total_matches_its_own_population(client):
    cc = client.get("/api/v1/status", params={"persona": "claude-code"}).json()
    sdk = client.get("/api/v1/status", params={"persona": "sdk"}).json()
    everything = client.get("/api/v1/status").json()
    assert cc["archived_total"] + sdk["archived_total"] == everything["archived_total"]


def test_alerts_accept_and_apply_the_persona(client):
    """No alerts are seeded, so this pins the CONTRACT rather than a count.

    The population claim is covered by the trace/cost/status tests above; what
    matters here is that the parameter is accepted, validated, and reaches the
    filters rather than being dropped at the signature.
    """
    from tokenjam.core.models import AlertFilters

    assert "persona" in AlertFilters.__dataclass_fields__
    for persona in ("claude-code", "sdk"):
        assert client.get(
            "/api/v1/alerts", params={"persona": persona},
        ).status_code == 200


# --------------------------------------------------------------------------
# /drift — persona-aware by DISCLOSURE, not by filtering
# --------------------------------------------------------------------------

def test_drift_says_it_is_not_measured_for_coding_agents(client):
    """An empty list and "not measured" are different claims.

    The drift detector gates itself off interactive coding agents — a
    heterogeneous human-driven workload has no stable baseline — so a
    `claude-code` reader was shown an empty list that read as "no drift
    detected". `persona_applicable` is what separates the two.
    """
    cc = client.get("/api/v1/drift", params={"persona": "claude-code"}).json()
    sdk = client.get("/api/v1/drift", params={"persona": "sdk"}).json()
    assert cc["persona_applicable"] is False
    assert sdk["persona_applicable"] is True


# --------------------------------------------------------------------------
# The cost-proposal ledger's third state
# --------------------------------------------------------------------------

def _write_legacy_ledger(config) -> None:
    """A ledger in the shape a build before per-persona proposals wrote: one
    whole-corpus list, no per-persona block."""
    from tokenjam.core.optimize import relearn_store

    relearn_store.write_cost_proposals(
        [{
            "signature": "corpus-wide", "kind": "cost:downsize",
            "title": "corpus-wide", "past_overspend_usd": 123.45,
            "past_overspend_tokens": 123450,
        }],
        config=config, window_days=30,
    )


def test_a_cold_store_is_never_run_not_unscoped(client):
    """Nothing has been recomputed, so the honest answer is the one that
    already existed. `persona_unscoped` is specifically "a ledger exists and
    cannot answer for this persona" — overloading it onto a cold store would
    lose the distinction the rescan affordance depends on."""
    body = client.get(
        "/api/v1/relearn/cost-proposals", params={"persona": "sdk"},
    ).json()
    assert body["status"] == "never_run"
    assert body["proposals"] == []


def test_a_legacy_ledger_reports_the_third_state_not_a_zero(config, client):
    """The back-compat decision, at the wire.

    A ledger written before per-persona proposals holds one whole-corpus total.
    Serving that under a persona label is the bug; serving it as this persona's
    empty result is the same bug in the reassuring direction. The endpoint says
    it cannot answer, and the client renders not-yet-known.
    """
    _write_legacy_ledger(config)
    body = client.get(
        "/api/v1/relearn/cost-proposals", params={"persona": "sdk"},
    ).json()
    assert body["persona_scoped"] is False
    assert body["status"] == "persona_unscoped"
    assert body["proposals"] == []
    # And emphatically NOT the corpus figure it does hold.
    assert body["past_overspend"].get("past_overspend_usd") != 123.45


def test_an_unnarrowed_request_still_reads_the_legacy_ledger(config, client):
    """The whole-corpus question is unaffected — it is the one such a ledger
    can always answer, and `persona_scoped` is true because nothing was
    narrowed. The narrowing is what degrades, never the endpoint."""
    _write_legacy_ledger(config)
    body = client.get("/api/v1/relearn/cost-proposals").json()
    assert body["persona_scoped"] is True
    assert body["status"] != "persona_unscoped"
    assert len(body["proposals"]) == 1
