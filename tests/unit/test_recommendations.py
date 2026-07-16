"""Unit tests for the recommendation-outcome ledger (`tokenjam.core.recommendations`).

Covers the append-only sink round-trip, the measured-vs-estimated aggregation
split, and post-hoc downsize adoption detection against a real DuckDB (measured
delta, adopted/ignored verdict, ripeness gate, and idempotency).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

from tokenjam.core.config import StorageConfig
from tokenjam.core.db import DuckDBBackend
from tokenjam.core.recommendations import (
    KIND_CONFIG_EXPORT,
    KIND_DOWNSIZE_ADOPTION,
    append_outcome,
    detect_downsize_adoption,
    read_outcomes,
    record_config_export,
    record_summarize_apply,
    summarize_outcomes,
)
from tokenjam.utils.time_parse import utcnow

from tests.factories import make_llm_span


@dataclass
class _Storage:
    path: str


@dataclass
class _Cfg:
    storage: _Storage


def _cfg(tmp_path):
    return _Cfg(storage=_Storage(path=str(tmp_path / "tj.duckdb")))


# ---------------------------------------------------------------------------
# Sink round-trip + aggregation
# ---------------------------------------------------------------------------

def test_summarize_apply_records_estimated_only(tmp_path):
    cfg = _cfg(tmp_path)
    record_summarize_apply(cfg, path="/tmp/CLAUDE.md", est_tokens_saved=1200)

    rows = read_outcomes(cfg)
    assert len(rows) == 1
    assert rows[0]["kind"] == "summarize_apply"
    assert rows[0]["measured"] is False

    summ = summarize_outcomes(rows)
    # A summarize apply is an ACTION with an estimated (not measured) recovery.
    assert summ["estimated_recoverable_tokens"] == 1200
    assert summ["measured_recovered_tokens"] == 0
    assert summ["actions_recorded"] == 1


def test_config_export_records_baseline(tmp_path):
    cfg = _cfg(tmp_path)

    @dataclass
    class _Downgrade:
        suggestions: dict
        estimated_recoverable_usd: float = 3.5
        estimated_recoverable_tokens: int = 5000

    since = utcnow() - timedelta(days=7)
    until = utcnow()
    oid = record_config_export(
        cfg, target="claude-code", export_path="/tmp/exp.jsonc",
        downgrade=_Downgrade(suggestions={"claude-opus-4-8": "claude-haiku-4-5"}),
        pricing_mode="api", provider="anthropic",
        since=since, until=until, window_days=7.0,
    )
    assert oid is not None
    rows = read_outcomes(cfg)
    assert rows[0]["kind"] == KIND_CONFIG_EXPORT
    assert rows[0]["detail"]["suggestions"] == {"claude-opus-4-8": "claude-haiku-4-5"}
    assert rows[0]["estimated_usd"] == 3.5


def test_summarize_split_keeps_measured_and_estimated_separate(tmp_path):
    cfg = _cfg(tmp_path)
    append_outcome(cfg, {
        "outcome_id": "a", "kind": "config_export", "measured": False,
        "pricing_mode": "api", "estimated_usd": 2.0, "estimated_tokens": 100,
    })
    append_outcome(cfg, {
        "outcome_id": "b", "kind": "downsize_adoption", "measured": True,
        "status": "adopted", "pricing_mode": "api",
        "recovered_usd": 1.25, "recovered_tokens": 90,
    })
    summ = summarize_outcomes(read_outcomes(cfg))
    assert summ["estimated_recoverable_usd"] == 2.0
    assert summ["measured_recovered_usd"] == 1.25
    assert summ["adopted"] == 1


def test_subscription_dollars_are_suppressed_in_totals(tmp_path):
    cfg = _cfg(tmp_path)
    append_outcome(cfg, {
        "outcome_id": "s", "kind": "downsize_adoption", "measured": True,
        "status": "adopted", "pricing_mode": "subscription",
        "recovered_usd": 99.0, "recovered_tokens": 500,
    })
    summ = summarize_outcomes(read_outcomes(cfg))
    # Dollars mislead for subscription users (Rule 14) — tokens survive, $ is dropped.
    assert summ["measured_recovered_usd"] == 0.0
    assert summ["measured_recovered_tokens"] == 500


# ---------------------------------------------------------------------------
# Post-hoc adoption detection against a real DuckDB
# ---------------------------------------------------------------------------

def _seed_export(cfg, *, export_ts, since, until, window_days=7.0):
    append_outcome(cfg, {
        "outcome_id": "export:claude-code:x",
        "ts": export_ts.isoformat(),
        "kind": KIND_CONFIG_EXPORT,
        "source": "export:claude-code",
        "status": "exported",
        "target": "/tmp/exp.jsonc",
        "provider": "anthropic",
        "pricing_mode": "api",
        "measured": False,
        "estimated_usd": 4.0,
        "estimated_tokens": 8000,
        "detail": {
            "suggestions": {"claude-opus-4-8": "claude-haiku-4-5"},
            "since": since.isoformat(),
            "until": until.isoformat(),
            "window_days": window_days,
        },
    })


def _insert_opus(db, *, when, cost, n=5):
    for _ in range(n):
        db.insert_span(make_llm_span(
            model="claude-opus-4-8", provider="anthropic",
            input_tokens=2000, output_tokens=300, cost_usd=cost,
            start_time=when, session_id=None,
        ))


def test_adoption_detected_when_premium_spend_drops(tmp_path):
    cfg = _cfg(tmp_path)
    db = DuckDBBackend(StorageConfig(path=str(tmp_path / "tj.duckdb")))
    try:
        now = utcnow()
        export_ts = now - timedelta(days=10)
        since = export_ts - timedelta(days=7)
        until = export_ts
        # Heavy opus usage BEFORE the export; almost none AFTER → a real drop.
        _insert_opus(db, when=export_ts - timedelta(days=3), cost=1.0, n=10)
        _insert_opus(db, when=export_ts + timedelta(days=2), cost=1.0, n=1)
        _seed_export(cfg, export_ts=export_ts, since=since, until=until)

        new = detect_downsize_adoption(db.conn, cfg, now=now)
        assert len(new) == 1
        rec = new[0]
        assert rec["kind"] == KIND_DOWNSIZE_ADOPTION
        assert rec["status"] == "adopted"
        assert rec["measured"] is True
        assert rec["recovered_tokens"] > 0
        assert rec["detail"]["relative_drop"] > 0.25
    finally:
        db.close()


def test_ignored_when_premium_spend_holds(tmp_path):
    cfg = _cfg(tmp_path)
    db = DuckDBBackend(StorageConfig(path=str(tmp_path / "tj.duckdb")))
    try:
        now = utcnow()
        export_ts = now - timedelta(days=10)
        since = export_ts - timedelta(days=7)
        until = export_ts
        # Same premium spend rate before and after → not adopted.
        _insert_opus(db, when=export_ts - timedelta(days=3), cost=1.0, n=7)
        _insert_opus(db, when=export_ts + timedelta(days=5), cost=1.0, n=10)
        _seed_export(cfg, export_ts=export_ts, since=since, until=until)

        new = detect_downsize_adoption(db.conn, cfg, now=now)
        assert len(new) == 1
        assert new[0]["status"] == "ignored"
        assert new[0]["recovered_tokens"] == 0
    finally:
        db.close()


def test_detection_is_idempotent(tmp_path):
    cfg = _cfg(tmp_path)
    db = DuckDBBackend(StorageConfig(path=str(tmp_path / "tj.duckdb")))
    try:
        now = utcnow()
        export_ts = now - timedelta(days=10)
        _insert_opus(db, when=export_ts - timedelta(days=3), cost=1.0, n=10)
        _seed_export(cfg, export_ts=export_ts,
                     since=export_ts - timedelta(days=7), until=export_ts)

        first = detect_downsize_adoption(db.conn, cfg, now=now)
        second = detect_downsize_adoption(db.conn, cfg, now=now)
        assert len(first) == 1
        assert len(second) == 0  # already resolved — no duplicate row
        adoptions = [o for o in read_outcomes(cfg) if o["kind"] == KIND_DOWNSIZE_ADOPTION]
        assert len(adoptions) == 1
    finally:
        db.close()


def test_unripe_export_is_left_pending(tmp_path):
    cfg = _cfg(tmp_path)
    db = DuckDBBackend(StorageConfig(path=str(tmp_path / "tj.duckdb")))
    try:
        now = utcnow()
        export_ts = now - timedelta(days=2)  # younger than MIN_OBSERVATION_DAYS
        _insert_opus(db, when=export_ts - timedelta(days=3), cost=1.0, n=10)
        _seed_export(cfg, export_ts=export_ts,
                     since=export_ts - timedelta(days=7), until=export_ts)

        new = detect_downsize_adoption(db.conn, cfg, now=now)
        assert new == []  # not enough post-export telemetry yet
    finally:
        db.close()


def test_dated_model_name_matches_normalised_suggestion(tmp_path):
    cfg = _cfg(tmp_path)
    db = DuckDBBackend(StorageConfig(path=str(tmp_path / "tj.duckdb")))
    try:
        now = utcnow()
        export_ts = now - timedelta(days=10)
        # Span carries a dated model id; the suggestion key is the bare name.
        for _ in range(10):
            db.insert_span(make_llm_span(
                model="claude-opus-4-8-20260115", provider="anthropic",
                input_tokens=2000, output_tokens=300, cost_usd=1.0,
                start_time=export_ts - timedelta(days=3), session_id=None,
            ))
        _seed_export(cfg, export_ts=export_ts,
                     since=export_ts - timedelta(days=7), until=export_ts)

        new = detect_downsize_adoption(db.conn, cfg, now=now)
        assert len(new) == 1
        assert new[0]["status"] == "adopted"  # dated model matched → drop detected
    finally:
        db.close()
