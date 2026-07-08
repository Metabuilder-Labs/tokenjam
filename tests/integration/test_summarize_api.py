"""Summarize API surface (Track B, in-process): capabilities / candidates /
staged / backups + apply / undo.

Core scan + staging + apply are exercised by their own unit tests; here we prove
the routes wire to core, shape the payloads, and (capabilities) reflect the host
honestly. Most tests monkeypatch list_candidates / session / apply for
determinism; the real-core apply tests at the end drive `POST /apply` end-to-end
against a temp file (no mocks) to close that seam — the drift/owner/symlink/backup
guards must hold through the route, not just in the unit tests.
"""
from __future__ import annotations

import re

import httpx
import pytest

from tokenjam.api.app import create_app
from tokenjam.core.config import (
    ApiAuthConfig,
    ApiConfig,
    SecurityConfig,
    StorageConfig,
    TjConfig,
)
from tokenjam.core.db import InMemoryBackend
from tokenjam.core.ingest import IngestPipeline
from tokenjam.core.summarize.candidates import Candidate, ScanResult

PROSE = "Always act carefully and never drop a required step when you respond. " * 30
_MARKER_RE = re.compile(r'<tj-keep id="\d+"[^>]*?(?:/>|>.*?</tj-keep>)', re.DOTALL)


@pytest.fixture
def db():
    backend = InMemoryBackend()
    yield backend
    backend.close()


@pytest.fixture
def config(tmp_path):
    # storage.path's parent drives session.summary_root → point it at tmp so
    # /staged reads an empty temp dir, never the real ~/.tj/summary.
    return TjConfig(
        version="1",
        security=SecurityConfig(ingest_secret="s"),
        api=ApiConfig(auth=ApiAuthConfig(enabled=False)),
        storage=StorageConfig(path=str(tmp_path / "telemetry.duckdb")),
    )


@pytest.fixture
def client(config, db):
    pipeline = IngestPipeline(db=db, config=config)
    app = create_app(config=config, db=db, ingest_pipeline=pipeline)
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")


async def test_capabilities_manual_always_on_dead_paths_flagged(client, monkeypatch):
    monkeypatch.delenv("TJ_ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr("shutil.which", lambda name: None)   # no `claude` on host
    caps = (await client.get("/api/v1/summarize/capabilities")).json()
    assert caps["manual"]["available"] is True
    assert caps["api"]["available"] is False and caps["api"]["reason"]
    assert caps["claude_p"]["available"] is False and caps["claude_p"]["reason"]


async def test_capabilities_api_enabled_when_key_present(client, monkeypatch):
    monkeypatch.setenv("TJ_ANTHROPIC_API_KEY", "sk-test")
    caps = (await client.get("/api/v1/summarize/capabilities")).json()
    assert caps["api"]["available"] is True and caps["api"]["reason"] == ""


async def test_candidates_returns_scan_dict(client, monkeypatch):
    fake = ScanResult(
        candidates=[Candidate(
            path="./CLAUDE.md", prose_words=1000, total_chars=4000, protected_blocks=1,
            est_tokens_saved=410, pricing_mode="api", scope="repo", is_prompt=True,
        )],
        root=".", recursive=False, globals_checked=0, walk_capped=False, note="",
    )
    monkeypatch.setattr("tokenjam.core.summarize.candidates.list_candidates", lambda **kw: fake)
    body = (await client.get("/api/v1/summarize/candidates")).json()
    assert body["count"] == 1
    c = body["candidates"][0]
    assert c["path"] == "./CLAUDE.md" and c["kind"] == "prompt" and c["est_tokens_saved"] == 410


async def test_staged_empty_by_default(client):
    r = await client.get("/api/v1/summarize/staged")
    assert r.status_code == 200
    assert r.json() == {"staged": []}


async def test_staged_lists_and_reads_one(client, monkeypatch):
    monkeypatch.setattr(
        "tokenjam.core.summarize.session.list_staged",
        lambda config: [{"path": "./CLAUDE.md", "est_tokens_saved": 410}],
    )
    monkeypatch.setattr(
        "tokenjam.core.summarize.session.read_staged",
        lambda config, path: {"path": path, "diff": "@@ ... @@"} if path == "./CLAUDE.md" else None,
    )
    listed = (await client.get("/api/v1/summarize/staged")).json()
    assert listed["staged"][0]["path"] == "./CLAUDE.md"
    one = (await client.get("/api/v1/summarize/staged", params={"path": "./CLAUDE.md"})).json()
    assert one["staged"][0]["diff"] == "@@ ... @@"
    miss = (await client.get("/api/v1/summarize/staged", params={"path": "./nope.md"})).json()
    assert miss["staged"] == []


async def test_backups_empty_by_default(client):
    r = await client.get("/api/v1/summarize/backups")
    assert r.status_code == 200
    assert r.json() == {"backups": []}


async def test_backups_lists_undoable_records(client, monkeypatch):
    monkeypatch.setattr(
        "tokenjam.core.summarize.backup.list_backups",
        lambda config: [
            {"source_path": "./CLAUDE.md", "applied_at": "2026-07-04T12:00:00", "undoable": True, "reason": ""},
            {"source_path": "./AGENTS.md", "applied_at": "2026-07-04T12:01:00",
             "undoable": False, "reason": "changed since apply — undo would lose newer edits"},
        ],
    )
    got = (await client.get("/api/v1/summarize/backups")).json()["backups"]
    assert got[0]["source_path"] == "./CLAUDE.md" and got[0]["undoable"] is True
    assert got[1]["undoable"] is False and "changed since apply" in got[1]["reason"]


async def test_apply_defaults_dry_run_and_passes_go_through(client, monkeypatch):
    seen: dict = {}

    def fake_apply(config, path=None, *, go=False):
        seen["path"], seen["go"] = path, go
        return {"applied": [path] if (go and path) else [], "skipped": [], "dry_run": not go}

    monkeypatch.setattr("tokenjam.core.summarize.apply.apply_staged", fake_apply)
    dry = (await client.post("/api/v1/summarize/apply", json={"path": "./CLAUDE.md"})).json()
    assert seen == {"path": "./CLAUDE.md", "go": False} and dry["dry_run"] is True
    wrote = (await client.post("/api/v1/summarize/apply", json={"path": "./CLAUDE.md", "go": True})).json()
    assert seen["go"] is True and wrote["applied"] == ["./CLAUDE.md"]


async def test_apply_all_when_path_omitted(client, monkeypatch):
    captured: dict = {}
    monkeypatch.setattr(
        "tokenjam.core.summarize.apply.apply_staged",
        lambda config, path=None, *, go=False: captured.update(path=path) or {"applied": [], "skipped": [], "dry_run": not go},
    )
    await client.post("/api/v1/summarize/apply", json={})
    assert captured["path"] is None   # omitted → apply all staged


async def test_undo_ok_and_drift_returns_409(client, monkeypatch):
    from tokenjam.core.summarize.session import SummarizeRefused

    def fake_undo(config, path, *, go=False):
        if path == "./drifted.md":
            raise SummarizeRefused("file changed since backup")
        return {"path": path, "restored": go, "dry_run": not go}

    monkeypatch.setattr("tokenjam.core.summarize.apply.undo", fake_undo)
    ok = await client.post("/api/v1/summarize/undo", json={"path": "./CLAUDE.md", "go": True})
    assert ok.status_code == 200 and ok.json()["restored"] is True
    bad = await client.post("/api/v1/summarize/undo", json={"path": "./drifted.md", "go": True})
    assert bad.status_code == 409


# ---- real-core apply through the route (no mocks — closes the monkeypatch seam) ----

def _stage_real(config, tmp_path, name="CLAUDE.md"):
    """Write a real prompt file and stage a structure-preserving rewrite via core
    (prep/check live in the run PR, so we stage directly and drive WRITE via /apply)."""
    from tokenjam.core.summarize import session
    p = tmp_path / name
    p.write_text(PROSE + "\n```\nkeep = 'me'\n```\n", encoding="utf-8")
    prep = session.prepare(path=str(p))
    summary = "Be careful; never skip a step. " + " ".join(_MARKER_RE.findall(prep.wrapped_prompt))
    verdict = session.check(config, str(p), summary, prep.source_sha256)
    assert verdict.staged
    return p


async def test_apply_go_writes_and_backs_up_real_core(client, config, tmp_path):
    from tokenjam.core.summarize import backup, session
    p = _stage_real(config, tmp_path)
    before = p.read_text()
    r = (await client.post("/api/v1/summarize/apply", json={"path": str(p), "go": True})).json()
    assert r["dry_run"] is False
    assert [a["path"] for a in r["applied"]] == [str(p)] and r["skipped"] == []
    assert p.read_text() != before                       # file was rewritten
    assert any(b["source_path"] == str(p) for b in backup.list_backups(config))   # backup written
    assert session.list_staged(config) == []             # staging cleared after apply


async def test_apply_skips_drift_through_route_real_core(client, config, tmp_path):
    p = _stage_real(config, tmp_path)
    p.write_text("hand-edited after staging\n", encoding="utf-8")   # drift
    r = (await client.post("/api/v1/summarize/apply", json={"path": str(p), "go": True})).json()
    assert r["applied"] == []
    assert any("changed since check" in s["reason"] for s in r["skipped"])
    assert p.read_text() == "hand-edited after staging\n"           # never overwritten
