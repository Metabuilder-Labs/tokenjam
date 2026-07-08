"""Outbound summarize run surface: run / prep / check (Track B).

`run` makes tj serve an outbound LLM caller; here delivery/session are
monkeypatched so nothing real is called — we prove the routes normalize modes,
serialize verdict/amortization, and map refuse/deliver failures to
409/502/400/404. `run` is also gated behind the `[summarize] allow_outbound_run`
opt-in (DEC-031) — refused 403 when off. prep/check are the manual (no-outbound)
path and need no opt-in.
"""
from __future__ import annotations

from dataclasses import replace

import httpx
import pytest

from tokenjam.api.app import create_app
from tokenjam.core.config import (
    ApiAuthConfig,
    ApiConfig,
    SecurityConfig,
    StorageConfig,
    SummarizeConfig,
    TjConfig,
)
from tokenjam.core.db import InMemoryBackend
from tokenjam.core.ingest import IngestPipeline

PROSE = "Always act carefully and never drop a required step when you respond. " * 30


@pytest.fixture
def db():
    backend = InMemoryBackend()
    yield backend
    backend.close()


@pytest.fixture
def config(tmp_path):
    # storage.path's parent drives session.summary_root → point it at tmp so
    # check() stages under a temp dir, never the real ~/.tj/summary.
    return TjConfig(
        version="1",
        security=SecurityConfig(ingest_secret="s"),
        api=ApiConfig(auth=ApiAuthConfig(enabled=False)),
        storage=StorageConfig(path=str(tmp_path / "telemetry.duckdb")),
        # Run tests exercise the outbound route, so opt-in is on here; the default
        # (off) is exercised by test_run_refused_when_outbound_disabled.
        summarize=SummarizeConfig(allow_outbound_run=True),
    )


@pytest.fixture
def client(config, db):
    pipeline = IngestPipeline(db=db, config=config)
    app = create_app(config=config, db=db, ingest_pipeline=pipeline)
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")


# ---- run / prep / check (mocked — no real outbound calls) ----

def _verdict(path="./CLAUDE.md", ok=True):
    from tokenjam.core.summarize.session import CheckVerdict
    return CheckVerdict(
        path=path, structure_ok=ok, reason="", integrity={}, words_before=1000,
        words_after=550, est_tokens_saved=410, must_keep_removed=[], must_keep_added=[],
        diff="@@ ... @@", restored="...", staged=ok, produced_by="api", note="",
    )


def _prep(path="./CLAUDE.md", wrapped="<wrapped>", note=""):
    from tokenjam.core.summarize.session import PrepResult
    return PrepResult(
        path=path, source_sha256="abc", wrapped_prompt=wrapped, system_rules="rules",
        prose_words=200, target_prose_words=100, protected_blocks=1, plan=[], note=note,
    )


async def test_run_normalizes_claude_p_and_serializes_verdict(client, monkeypatch):
    from tokenjam.core.summarize.delivery import RunResult
    seen: dict = {}

    def fake_via(config, path, mode, *, ratio=0.5):
        seen["mode"], seen["path"] = mode, path
        return RunResult(verdict=_verdict(path), amortization=None, skipped_note=None, cost_unknown=False)

    monkeypatch.setattr("tokenjam.core.summarize.delivery.summarize_via", fake_via)
    r = await client.post("/api/v1/summarize/run", json={"path": "./CLAUDE.md", "mode": "claude_p"})
    assert r.status_code == 200
    assert seen["mode"] == "claude-p"                      # underscore normalized to hyphen
    assert r.json()["verdict"]["est_tokens_saved"] == 410


async def test_run_serializes_amortization(client, monkeypatch):
    from tokenjam.core.summarize.delivery import Amortization, RunResult
    am = Amortization(model="claude-x", rewrite_usd=0.01, saving_usd_per_call=0.002,
                      break_even_calls=5, rates_known=True)
    monkeypatch.setattr(
        "tokenjam.core.summarize.delivery.summarize_via",
        lambda config, path, mode, *, ratio=0.5: RunResult(verdict=_verdict(path), amortization=am),
    )
    body = (await client.post("/api/v1/summarize/run", json={"path": "./x.md", "mode": "api"})).json()
    assert body["amortization"]["break_even_calls"] == 5 and body["amortization"]["rates_known"] is True


async def test_run_skipped_below_gate(client, monkeypatch):
    from tokenjam.core.summarize.delivery import RunResult
    monkeypatch.setattr(
        "tokenjam.core.summarize.delivery.summarize_via",
        lambda config, path, mode, *, ratio=0.5: RunResult(verdict=None, skipped_note="too short"),
    )
    body = (await client.post("/api/v1/summarize/run", json={"path": "./x.md", "mode": "api"})).json()
    assert body["verdict"] is None and body["skipped_note"] == "too short"


async def test_run_rejects_manual_and_unknown_mode(client):
    for m in ("manual", "bogus"):
        r = await client.post("/api/v1/summarize/run", json={"path": "./x.md", "mode": m})
        assert r.status_code == 400


async def test_run_maps_refuse_409_and_delivery_502(client, monkeypatch):
    from tokenjam.core.summarize.delivery import DeliveryError
    from tokenjam.core.summarize.session import SummarizeRefused

    def refuse(config, path, mode, *, ratio=0.5):
        raise SummarizeRefused("changed since prep")

    monkeypatch.setattr("tokenjam.core.summarize.delivery.summarize_via", refuse)
    assert (await client.post("/api/v1/summarize/run", json={"path": "./x.md", "mode": "api"})).status_code == 409

    def boom(config, path, mode, *, ratio=0.5):
        raise DeliveryError("model down")

    monkeypatch.setattr("tokenjam.core.summarize.delivery.summarize_via", boom)
    assert (await client.post("/api/v1/summarize/run", json={"path": "./x.md", "mode": "api"})).status_code == 502


async def test_run_missing_file_returns_404(client, monkeypatch):
    # summarize_via preps first (reads the file); a missing path raises
    # FileNotFoundError, which the route maps to 404 like /prep — not a 500.
    def missing(config, path, mode, *, ratio=0.5):
        raise FileNotFoundError(path)

    monkeypatch.setattr("tokenjam.core.summarize.delivery.summarize_via", missing)
    r = await client.post("/api/v1/summarize/run", json={"path": "./nope.md", "mode": "api"})
    assert r.status_code == 404


async def test_prep_returns_wrapped_prompt_and_404_on_missing(client, monkeypatch):
    monkeypatch.setattr("tokenjam.core.summarize.session.prepare",
                        lambda *, path, ratio=0.5: _prep(path))
    ok = (await client.post("/api/v1/summarize/prep", json={"path": "./CLAUDE.md"})).json()
    assert ok["wrapped_prompt"] == "<wrapped>" and ok["source_sha256"] == "abc"

    def missing(*, path, ratio=0.5):
        raise FileNotFoundError(path)

    monkeypatch.setattr("tokenjam.core.summarize.session.prepare", missing)
    assert (await client.post("/api/v1/summarize/prep", json={"path": "./nope.md"})).status_code == 404


async def test_check_stages_and_maps_drift_409(client, monkeypatch):
    monkeypatch.setattr("tokenjam.core.summarize.session.check",
                        lambda config, path, summary, source_hash, **kw: _verdict(path, ok=True))
    ok = (await client.post("/api/v1/summarize/check",
                            json={"path": "./CLAUDE.md", "summary": "s", "source_hash": "abc"})).json()
    assert ok["structure_ok"] is True and ok["staged"] is True

    from tokenjam.core.summarize.session import SummarizeRefused

    def refuse(config, path, summary, source_hash, **kw):
        raise SummarizeRefused("changed")

    monkeypatch.setattr("tokenjam.core.summarize.session.check", refuse)
    assert (await client.post("/api/v1/summarize/check",
                              json={"path": "./x.md", "summary": "s", "source_hash": "abc"})).status_code == 409


async def test_run_refused_when_outbound_disabled(db, config):
    # Default posture (DEC-031): outbound run is off until the user opts in, so a
    # POST is refused 403 — no spend on a default install. Manual path is unaffected.
    off = replace(config, summarize=SummarizeConfig(allow_outbound_run=False))
    app = create_app(config=off, db=db, ingest_pipeline=IngestPipeline(db=db, config=off))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as c:
        r = await c.post("/api/v1/summarize/run", json={"path": "./x.md", "mode": "api"})
    assert r.status_code == 403
    assert "allow_outbound_run" in r.json()["detail"]


async def test_check_missing_file_returns_404(client, monkeypatch):
    # File vanished between prep and check → 404, parity with /prep and /run.
    def gone(config, path, summary, source_hash, **kw):
        raise FileNotFoundError(path)
    monkeypatch.setattr("tokenjam.core.summarize.session.check", gone)
    r = await client.post("/api/v1/summarize/check",
                          json={"path": "./gone.md", "summary": "s", "source_hash": "abc"})
    assert r.status_code == 404


async def test_check_structure_fail_never_stages(client, config, tmp_path):
    # Real core (no mock): a summary that drops a protected block fails the structure
    # gate — the route returns structure_ok=false/staged=false and NOTHING is written
    # to the staging store. Closes the seam where the other check tests monkeypatch core.
    from tokenjam.core.summarize import session
    p = tmp_path / "CLAUDE.md"
    p.write_text(PROSE + "\n```\nkeep = 'me'\n```\n", encoding="utf-8")
    prep = (await client.post("/api/v1/summarize/prep", json={"path": str(p)})).json()
    broken = "Be brief."   # no <tj-keep> markers → protected block dropped → structure fails
    v = (await client.post("/api/v1/summarize/check", json={
        "path": str(p), "summary": broken, "source_hash": prep["source_sha256"]})).json()
    assert v["structure_ok"] is False and v["staged"] is False
    assert session.list_staged(config) == []
