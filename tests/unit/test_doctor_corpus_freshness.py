"""`tj doctor`'s corpus-freshness check: a stale-by-age corpus with real
un-ingested data waiting on disk can never render as healthy, and must FAIL
(not warn) once nothing is positioned to close the gap automatically -- see
the daemon-liveness check this pairs with.

Age alone isn't enough to call it a gap, though: a user who simply hasn't run
Claude Code since the threshold looks identical by wall-clock but has no
newer on-disk data to ingest. That must read as healthy, not an error -- see
test_ok_when_stale_but_idle_with_no_newer_transcripts below.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from tokenjam.cli.cmd_doctor import _check_corpus_freshness
from tokenjam.core.config import IngestConfig, StorageConfig, TjConfig
from tokenjam.core.db import DuckDBBackend
from tokenjam.utils.time_parse import utcnow
from tests.factories import make_session


@pytest.fixture()
def db(tmp_path: Path):
    backend = DuckDBBackend(StorageConfig(path=str(tmp_path / "t.duckdb")))
    yield backend
    backend.close()


@pytest.fixture()
def projects_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "projects"
    root.mkdir()
    monkeypatch.setenv("TJ_CLAUDE_PROJECTS_ROOT", str(root))
    return root


def _config() -> TjConfig:
    config = TjConfig(version="1")
    config.ingest = IngestConfig(interval_minutes=30)
    return config


def _write_session(root: Path, session_id: str, started_at: datetime) -> None:
    project_dir = root / "-tmp-proj"
    project_dir.mkdir(parents=True, exist_ok=True)
    (project_dir / f"{session_id}.jsonl").write_text(json.dumps({
        "type": "assistant",
        "uuid": f"{session_id}-u1",
        "sessionId": session_id,
        "cwd": "/tmp/proj",
        "timestamp": started_at.isoformat(),
        "message": {
            "id": f"msg_{session_id}",
            "model": "claude-sonnet-4-5-20250929",
            "content": [{"type": "text", "text": "ok"}],
            "usage": {
                "input_tokens": 100, "output_tokens": 20,
                "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0,
            },
        },
    }))


def test_info_when_no_sessions_ever(db: DuckDBBackend, projects_root: Path) -> None:
    check = _check_corpus_freshness(_config(), db, daemon_alive=True)
    assert check["level"] == "info"


def test_ok_when_recently_ingested(db: DuckDBBackend, projects_root: Path) -> None:
    db.upsert_session(make_session(started_at=utcnow() - timedelta(minutes=5)))
    check = _check_corpus_freshness(_config(), db, daemon_alive=True)
    assert check["level"] == "ok"


def test_ok_when_stale_but_idle_with_no_newer_transcripts(
    db: DuckDBBackend, projects_root: Path,
) -> None:
    """The legitimate-idle-user case: recorded sessions, nothing new on disk,
    daemon down. Wall-clock age alone would call this an error; it must not."""
    db.upsert_session(make_session(started_at=utcnow() - timedelta(days=2)))
    check = _check_corpus_freshness(_config(), db, daemon_alive=False)
    assert check["level"] == "ok"
    assert "haven't run it" in check["message"]


def test_warning_when_stale_with_newer_transcripts_and_daemon_alive(
    db: DuckDBBackend, projects_root: Path,
) -> None:
    db.upsert_session(make_session(started_at=utcnow() - timedelta(days=2)))
    _write_session(projects_root, "sess-newer", utcnow() - timedelta(hours=1))

    check = _check_corpus_freshness(_config(), db, daemon_alive=True)

    assert check["level"] == "warning"


def test_error_when_stale_with_newer_transcripts_and_daemon_dead(
    db: DuckDBBackend, projects_root: Path,
) -> None:
    """This is the real-gap scenario: newest ingested session 2 days stale,
    newer on-disk data waiting, and nothing running to catch it up. Must
    FAIL, never merely warn."""
    db.upsert_session(make_session(started_at=utcnow() - timedelta(days=2)))
    _write_session(projects_root, "sess-newer", utcnow() - timedelta(hours=1))

    check = _check_corpus_freshness(_config(), db, daemon_alive=False)

    assert check["level"] == "error"
    assert "not running" in check["message"]


def test_a_confirmed_gap_never_renders_as_ok(
    db: DuckDBBackend, projects_root: Path,
) -> None:
    """Never assert more than the data supports: once there's confirmed
    newer on-disk data waiting, this must never come back level 'ok',
    regardless of daemon state."""
    db.upsert_session(make_session(started_at=utcnow() - timedelta(days=5)))
    _write_session(projects_root, "sess-newer", utcnow() - timedelta(hours=1))
    for daemon_alive in (True, False):
        check = _check_corpus_freshness(_config(), db, daemon_alive=daemon_alive)
        assert check["level"] != "ok"


def test_info_when_reconciliation_cannot_be_verified(
    db: DuckDBBackend, projects_root: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A check that can't determine the answer must not render as either
    healthy or failing."""
    db.upsert_session(make_session(started_at=utcnow() - timedelta(days=2)))

    def _boom(*args, **kwargs):
        raise RuntimeError("disk on fire")

    monkeypatch.setattr(
        "tokenjam.core.transcript_sync.reconcile_claude_code", _boom
    )

    check = _check_corpus_freshness(_config(), db, daemon_alive=True)

    assert check["level"] == "info"
    assert "disk on fire" in check["message"]
