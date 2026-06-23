"""#239 — first-run plan-tier round-trip + backfill count reconciliation.

The synthetic-factory tests set ``plan_tier`` explicitly, so they never exercise
the *full* path a real user hits on first run: a config that declares
``[budget.anthropic] plan = "max_5x"`` → Claude Code backfill stamps every
``SessionRecord.plan_tier`` → the framing the user actually sees in Lens
(``/api/v1/cost``) and ``tj optimize`` (``/api/v1/optimize``) resolves that plan.

The 0.5.0 first-run review burned real time chasing a "Max 20x" framing that
turned out to be stale data, not a code bug. This file codifies the exact check
that cleared that false alarm: declared ``max_5x`` config must round-trip to
``plan_tier=max_5x`` / ``plan_monthly_usd=100`` in both framing blocks.

It also pins the backfill new/existing/total counts against the ``sessions``
table and asserts idempotent re-runs (the count contract behind #238).

Uses a real ``DuckDBBackend`` because the framing mix queries run against
``db.conn`` (``InMemoryBackend`` has none and short-circuits to ``{}``).
"""
from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path

import httpx
import pytest

from tokenjam.api.app import create_app
from tokenjam.core.backfill import ingest_claude_code
from tokenjam.core.config import ProviderBudget, StorageConfig, TjConfig
from tokenjam.core.db import DuckDBBackend
from tokenjam.core.ingest import IngestPipeline
from tokenjam.utils.time_parse import utcnow


@pytest.fixture(autouse=True)
def _isolate_home(monkeypatch, tmp_path):
    """Point Path.home() at an empty dir so config_declared_plan's global
    fallback never reads the dev machine's ~/.config/tj/config.toml — otherwise
    the framing assertions depend on the host's real plan declaration."""
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)


# --- Claude Code on-disk JSONL fixtures (1 file = 1 session) ----------------
# Mirrors tests/unit/test_backfill.py's inline-fixture style. There are no
# checked-in *.jsonl fixtures in the repo; the on-disk format is built here so
# the test exercises the real parser (parse_claude_code_session).


def _assistant_record(uuid: str, model: str, input_tokens: int,
                      output_tokens: int, timestamp: str, session_id: str,
                      cwd: str) -> dict:
    return {
        "type": "assistant",
        "uuid": uuid,
        "timestamp": timestamp,
        "sessionId": session_id,
        "cwd": cwd,
        "message": {
            "model": model,
            "content": [{"type": "text", "text": "ok"}],
            "usage": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cache_read_input_tokens": 0,
                "cache_creation_input_tokens": 0,
            },
        },
    }


def _ts(hours_ago: float) -> str:
    """Recent ISO-8601 'Z' timestamp so the data falls inside the framing
    window (exercises the data-driven mix, not just the config fallback)."""
    return (utcnow() - timedelta(hours=hours_ago)).strftime("%Y-%m-%dT%H:%M:%S.000Z")


def _write_session(tmp_path: Path, session_id: str, cwd: str,
                   records: list[dict]) -> Path:
    project_dir = tmp_path / cwd.replace("/", "-")
    project_dir.mkdir(parents=True, exist_ok=True)
    path = project_dir / f"{session_id}.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in records))
    return path


def _seed_claude_code_dir(root: Path) -> int:
    """Three sessions across two projects, each with two assistant turns.
    Returns the number of sessions written."""
    _write_session(root, "sess-a", "/Users/me/proj-one", [
        {"type": "user", "message": {"role": "user", "content": "hi"}},
        _assistant_record("a1", "claude-opus-4-7", 1200, 300, _ts(5), "sess-a",
                          "/Users/me/proj-one"),
        _assistant_record("a2", "claude-sonnet-4-6", 800, 150, _ts(4), "sess-a",
                          "/Users/me/proj-one"),
    ])
    _write_session(root, "sess-b", "/Users/me/proj-one", [
        _assistant_record("b1", "claude-sonnet-4-6", 2000, 400, _ts(3), "sess-b",
                          "/Users/me/proj-one"),
        _assistant_record("b2", "claude-haiku-4-5", 500, 90, _ts(2), "sess-b",
                          "/Users/me/proj-one"),
    ])
    _write_session(root, "sess-c", "/Users/me/proj-two", [
        _assistant_record("c1", "claude-opus-4-7", 3000, 600, _ts(2), "sess-c",
                          "/Users/me/proj-two"),
        _assistant_record("c2", "claude-opus-4-7", 1000, 200, _ts(1), "sess-c",
                          "/Users/me/proj-two"),
    ])
    return 3


def _db(tmp_path: Path) -> DuckDBBackend:
    return DuckDBBackend(StorageConfig(path=str(tmp_path / "t.duckdb")))


def _max5x_config() -> TjConfig:
    return TjConfig(version="1", budgets={"anthropic": ProviderBudget(plan="max_5x")})


# --- Piece 1: plan-tier round-trip (config → backfill → framing) ------------


def test_backfill_stamps_every_session_with_declared_plan(tmp_path):
    """Declared max_5x config → every backfilled SessionRecord is max_5x."""
    cc_root = tmp_path / "claude"
    n_sessions = _seed_claude_code_dir(cc_root)
    db = _db(tmp_path)

    result = ingest_claude_code(db, root=cc_root, config=_max5x_config())

    assert result.sessions_seen == n_sessions
    assert result.sessions_ingested == n_sessions

    tiers = [r[0] for r in db.conn.execute(
        "SELECT plan_tier FROM sessions").fetchall()]
    assert len(tiers) == n_sessions
    assert set(tiers) == {"max_5x"}, tiers
    db.close()


@pytest.mark.asyncio
async def test_cost_framing_resolves_declared_plan(tmp_path):
    """/api/v1/cost framing reflects the round-tripped plan: subscription
    pricing mode, plan_tier=max_5x, plan_monthly_usd=100 (the exact figures
    that cleared the false 'Max 20x' scare)."""
    cc_root = tmp_path / "claude"
    _seed_claude_code_dir(cc_root)
    db = _db(tmp_path)
    config = _max5x_config()
    ingest_claude_code(db, root=cc_root, config=config)

    pipeline = IngestPipeline(db=db, config=config)
    app = create_app(config=config, db=db, ingest_pipeline=pipeline)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
        framing = (await c.get("/api/v1/cost?since=90d")).json()["framing"]

    assert framing["pricing_mode"] == "subscription", framing
    assert framing["plan_tier"] == "max_5x", framing
    assert framing["plan_monthly_usd"] == 100.0, framing
    db.close()


@pytest.mark.asyncio
async def test_optimize_framing_resolves_declared_plan(tmp_path):
    """/api/v1/optimize framing must agree with /cost on the round-tripped
    plan. fast=true skips the Trim analyzer (no llmlingua dependency)."""
    cc_root = tmp_path / "claude"
    _seed_claude_code_dir(cc_root)
    db = _db(tmp_path)
    config = _max5x_config()
    ingest_claude_code(db, root=cc_root, config=config)

    pipeline = IngestPipeline(db=db, config=config)
    app = create_app(config=config, db=db, ingest_pipeline=pipeline)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
        payload = (await c.get("/api/v1/optimize?since=90d&fast=true")).json()
    framing = payload["framing"]

    assert framing["pricing_mode"] == "subscription", framing
    assert framing["plan_tier"] == "max_5x", framing
    assert framing["plan_monthly_usd"] == 100.0, framing
    db.close()


@pytest.mark.asyncio
async def test_cost_and_optimize_framing_agree(tmp_path):
    """Lens and `tj optimize` must show the same plan to a first-run user —
    the divergence that made the 0.5.0 review distrust the framing. Pin parity
    on the plan-determining fields."""
    cc_root = tmp_path / "claude"
    _seed_claude_code_dir(cc_root)
    db = _db(tmp_path)
    config = _max5x_config()
    ingest_claude_code(db, root=cc_root, config=config)

    pipeline = IngestPipeline(db=db, config=config)
    app = create_app(config=config, db=db, ingest_pipeline=pipeline)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
        cost = (await c.get("/api/v1/cost?since=90d")).json()["framing"]
        opt = (await c.get("/api/v1/optimize?since=90d&fast=true")).json()["framing"]

    for key in ("pricing_mode", "plan_tier", "plan_monthly_usd"):
        assert cost[key] == opt[key], (key, cost[key], opt[key])
    db.close()


# --- Piece 2: backfill count reconciliation + idempotency (#238) ------------


def _scalar(db: DuckDBBackend, sql: str) -> int:
    row = db.conn.execute(sql).fetchone()
    return row[0] if row else 0


def _session_count(db: DuckDBBackend) -> int:
    return _scalar(db, "SELECT COUNT(*) FROM sessions")


def _span_count(db: DuckDBBackend) -> int:
    return _scalar(db, "SELECT COUNT(*) FROM spans")


def test_backfill_counts_reconcile_with_sessions_table(tmp_path):
    """new/existing/total must reconcile with the sessions table on first run.

    Fixtures are 1 file = 1 session, the clean shape #238's reporting should
    converge on. The reported counts decompose as:
        total    = rows in the sessions table
        new      = result.sessions_ingested
        existing = result.sessions_seen - result.sessions_ingested
    """
    cc_root = tmp_path / "claude"
    n_sessions = _seed_claude_code_dir(cc_root)
    db = _db(tmp_path)

    result = ingest_claude_code(db, root=cc_root, config=_max5x_config())

    # First run: everything is new, nothing pre-existing.
    assert result.sessions_seen == n_sessions
    assert result.sessions_ingested == n_sessions          # new
    assert result.sessions_seen - result.sessions_ingested == 0  # existing
    assert _session_count(db) == n_sessions                # total
    assert result.spans_ingested == _span_count(db)
    assert result.spans_skipped_existing == 0
    db.close()


def test_backfill_idempotent_rerun_reports_existing(tmp_path):
    """Re-running over the same data adds nothing: all sessions read as
    existing, all spans skipped, and the table is unchanged. This is the
    '1 new (12 already present) · 13 total' contract from #238 — the summary
    must not read new-only as 'it barely worked'."""
    cc_root = tmp_path / "claude"
    n_sessions = _seed_claude_code_dir(cc_root)
    db = _db(tmp_path)

    first = ingest_claude_code(db, root=cc_root, config=_max5x_config())
    sessions_after_first = _session_count(db)
    spans_after_first = _span_count(db)

    second = ingest_claude_code(db, root=cc_root, config=_max5x_config())

    # Re-run sees the same sessions but ingests none of them.
    assert second.sessions_seen == n_sessions              # total still observed
    assert second.sessions_ingested == 0                   # new
    assert second.sessions_seen - second.sessions_ingested == n_sessions  # existing
    assert second.spans_ingested == 0
    assert second.spans_skipped_existing == first.spans_ingested

    # The table is unchanged — idempotent.
    assert _session_count(db) == sessions_after_first
    assert _span_count(db) == spans_after_first
    db.close()


def test_backfill_incremental_run_counts_only_new(tmp_path):
    """Adding one new session and re-running reports exactly 1 new / N existing,
    and the table grows by one — the mixed new+existing case the summary must
    report clearly."""
    cc_root = tmp_path / "claude"
    n_sessions = _seed_claude_code_dir(cc_root)
    db = _db(tmp_path)

    ingest_claude_code(db, root=cc_root, config=_max5x_config())
    assert _session_count(db) == n_sessions

    # A brand-new session lands after the first backfill.
    _write_session(cc_root, "sess-d", "/Users/me/proj-two", [
        _assistant_record("d1", "claude-sonnet-4-6", 700, 120, _ts(0.5),
                          "sess-d", "/Users/me/proj-two"),
    ])

    result = ingest_claude_code(db, root=cc_root, config=_max5x_config())

    assert result.sessions_seen == n_sessions + 1          # total observed
    assert result.sessions_ingested == 1                   # new
    assert result.sessions_seen - result.sessions_ingested == n_sessions  # existing
    assert _session_count(db) == n_sessions + 1            # total in table
    db.close()
