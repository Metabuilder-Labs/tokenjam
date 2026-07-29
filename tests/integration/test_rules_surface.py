"""`tj rules` and `/api/v1/rules/*` — the surface is reachable, and by a USER.

Critical Rule 24 is the whole point of this file. "The route resolves" and "the
component mounts" each prove nothing on their own, so this checks the INVERSE
direction too: something must link TO the Rules view, and the capability must
have a name a user can type. A screen only a hand-typed URL reaches is an
invisible capability, and so is an analyzer with no Click choice.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from tokenjam.cli.main import cli

UI = Path(__file__).resolve().parents[2] / "tokenjam" / "ui" / "index.html"


# --- (d) a capability needs a name a user can type --------------------------#

def test_rules_is_a_typeable_command_with_the_whole_lifecycle():
    result = CliRunner().invoke(cli, ["rules", "--help"])
    assert result.exit_code == 0, result.output
    for verb in ("list", "show", "stage", "check", "apply", "undo", "applied"):
        assert verb in result.output


def test_rules_never_opens_the_database(tmp_path, monkeypatch):
    """It reads the proposal cache the optimize pass already wrote, so it works
    while `tj serve` holds the DuckDB write lock. Answering "what rules are on
    offer" must never trigger an analyzer sweep."""
    import tokenjam.cli.main as main_mod

    monkeypatch.setattr(
        main_mod, "open_db",
        lambda *a, **k: pytest.fail("`tj rules` must not open the DB"),
    )
    monkeypatch.setenv("HOME", str(tmp_path))
    result = CliRunner().invoke(cli, ["rules", "list"])
    assert result.exit_code == 0, result.output


def test_rules_list_on_a_fresh_install_says_so_rather_than_claiming_none(
    tmp_path, monkeypatch,
):
    """Known-and-empty is the only state allowed to say "no rules", and it has
    to point at what would produce some — an empty list that reads as a verdict
    is the recurring defect this product has."""
    monkeypatch.setenv("HOME", str(tmp_path))
    result = CliRunner().invoke(cli, ["rules", "list"])
    assert result.exit_code == 0, result.output
    assert "No permanent rules on offer" in result.output
    assert "tj optimize" in result.output


def test_rules_list_json_is_machine_readable(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    result = CliRunner().invoke(cli, ["rules", "list", "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["rules"] == []
    # The honesty framing travels with the payload, not only with the table.
    assert "not a guaranteed saving" in payload["note"]


# --- (a) the inverse direction: does anything link TO this view --------------#

def test_something_links_to_the_rules_view():
    html = UI.read_text(encoding="utf-8")
    # A sidebar entry, so the screen is reachable from every other screen...
    assert 'href="#/optimize/rules"' in html
    assert 'data-view="optimize" data-param="rules"' in html
    # ...and a contextual link from the Optimize screen body.
    assert html.count('#/optimize/rules') >= 2


def test_the_rules_route_resolves_to_its_own_view():
    html = UI.read_text(encoding="utf-8")
    assert "function RulesView(" in html
    assert "['rules',     RulesView]," in html
    assert "route.param === 'rules'" in html


def test_the_rules_view_never_claims_emptiness_before_its_fetch_resolves():
    """Root anti-pattern 22: `null` is "not yet known" and renders a skeleton;
    `[]` is "known and empty" and is the only state allowed to say "no rules".
    Zero is the worst placeholder — "no waste" reads as reassurance."""
    html = UI.read_text(encoding="utf-8")
    start = html.index("function RulesView(")
    body = html[start:html.index("function SummarizeView(", start)]
    # The fetch gate is unchanged: `null` still means not-yet-known for both reads.
    for state in ("rules", "applied"):
        assert f"{state} === null" in body
    # The EMPTINESS check now runs on the two derived tab populations rather than
    # on `rules` directly — the page splits into Open / Applied (something to do
    # versus already in place), so "no rules" is a claim each tab makes about its
    # own list. Same property, one level down: an empty-state string may only
    # render off a list that is known, never off `null`.
    for derived in ("openRules.length === 0", "inPlaceRules.length === 0"):
        assert derived in body, derived
    # And the derived lists are built from `rules || []`, so before the fetch
    # lands they are empty-but-known — which is why the tab counts have their OWN
    # `rules === null` guard rather than relying on a length of zero.
    assert "rules || []" in body
    assert "useState(null)" in body


# --- the API half -----------------------------------------------------------#

@pytest.mark.asyncio
async def test_the_rules_endpoints_are_registered_and_read_only_by_default(tmp_path):
    import httpx

    from tokenjam.api.app import create_app
    from tokenjam.core.config import StorageConfig, TjConfig
    from tokenjam.core.db import InMemoryBackend
    from tokenjam.core.ingest import IngestPipeline

    config = TjConfig(
        version="1", storage=StorageConfig(path=str(tmp_path / "tj" / "tj.duckdb")),
    )
    db = InMemoryBackend()
    app = create_app(
        db=db, config=config, ingest_pipeline=IngestPipeline(db=db, config=config),
    )
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        listed = await client.get("/api/v1/rules")
        assert listed.status_code == 200
        assert listed.json()["rules"] == []
        assert listed.json()["offered_count"] == 0

        staged = await client.get("/api/v1/rules/staged")
        assert staged.status_code == 200
        assert staged.json()["staged"] == []

        applied = await client.get("/api/v1/rules/applied")
        assert applied.status_code == 200
        assert applied.json()["applied"] == []

        # A stage request for a rule that does not exist 404s rather than
        # silently staging nothing.
        missing = await client.post(
            "/api/v1/rules/stage", json={"signature": "cost:nope"},
        )
        assert missing.status_code == 404

        # Apply defaults to a dry run even with nothing staged, so a caller
        # that forgets `go` can never write.
        dry = await client.post("/api/v1/rules/apply", json={})
        assert dry.status_code == 200
        assert dry.json()["dry_run"] is True
