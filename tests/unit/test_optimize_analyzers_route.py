"""
`GET /api/v1/optimize/analyzers` — the resolved persona gate, per persona.

The dashboard's analyzer guide has to name the checks that run for a setup,
including one the current window did not resolve to. `GET /optimize` can only
ever publish the stored report's OWN persona, so without this endpoint the guide
would have had to re-declare `PERSONA_DISABLED_ANALYZERS` as a JS literal — the
duplication the map exists as a single source of truth to prevent.

So the contract these tests pin is derivation, not content: every value must
come from `ANALYZER_REGISTRY` + `disabled_analyzers_for_persona`, so adding an
analyzer or gating one in Python changes the payload with no edit here and no
edit in the UI.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from tokenjam.api.app import create_app
from tokenjam.core.config import ApiAuthConfig, ApiConfig, TjConfig
from tokenjam.core.db import InMemoryBackend
from tokenjam.core.framing import PERSONAS
from tokenjam.core.ingest import build_default_pipeline
from tokenjam.core.optimize import (
    ANALYZER_REGISTRY,
    disabled_analyzers_for_persona,
)


@pytest.fixture
def client():
    config = TjConfig(version="1", api=ApiConfig(auth=ApiAuthConfig(enabled=False)))
    db = InMemoryBackend()
    app = create_app(
        config=config, db=db, ingest_pipeline=build_default_pipeline(db, config),
    )
    with TestClient(app) as c:
        yield c


@pytest.fixture
def payload(client):
    resp = client.get("/api/v1/optimize/analyzers")
    assert resp.status_code == 200
    return resp.json()


def test_every_known_persona_is_present(payload):
    assert set(payload["personas"]) == set(PERSONAS)


def test_registered_is_the_analyzer_registry_verbatim(payload):
    assert payload["registered"] == sorted(ANALYZER_REGISTRY)


@pytest.mark.parametrize("persona", PERSONAS)
def test_runs_is_the_registry_minus_the_python_gate(payload, persona):
    """The whole point: `runs` is derived, never listed. If this ever has to be
    updated by hand because someone hardcoded a name, the guide has silently
    become a second source of truth."""
    disabled = disabled_analyzers_for_persona(persona)
    expected = [n for n in sorted(ANALYZER_REGISTRY) if n not in disabled]
    assert payload["personas"][persona]["runs"] == expected


@pytest.mark.parametrize("persona", PERSONAS)
def test_disabled_is_the_raw_gate_entry(payload, persona):
    """Reported verbatim, so it can legitimately name a sub-check that is not
    itself a registry entry (`placement`, which `downsize` attaches). Filtering
    it down to registry names would drop an explanation the guide needs."""
    assert payload["personas"][persona]["disabled"] == sorted(
        disabled_analyzers_for_persona(persona)
    )


def test_runs_and_disabled_never_overlap(payload):
    for persona in PERSONAS:
        block = payload["personas"][persona]
        assert not (set(block["runs"]) & set(block["disabled"]))


def test_claude_code_gates_off_more_than_it_runs_is_not_asserted(payload):
    """Deliberately NOT a hardcoded expected list — that would re-create the
    duplication in the tests instead of the UI. Only the structural invariant
    the guide depends on: the persona it documents has something to document,
    and both halves of the contrast it is built around actually run."""
    cc = payload["personas"]["claude-code"]
    assert cc["runs"], "the documented persona must run at least one analyzer"
    assert "downsize" in cc["runs"]
    assert "subagent" in cc["runs"]


def test_answers_on_a_cold_store(payload):
    """Static config, no corpus read. The guide must render before any scan has
    completed — a reader wondering what these checks are is *most* likely to be
    on a fresh install."""
    assert payload["registered"]
