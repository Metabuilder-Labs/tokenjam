"""`tj doctor` must surface on-disk sessions that were never ingested.

`_check_span_staleness` catches "nothing is arriving at all". It cannot see the
subtler steady-state failure this check exists for: most sessions arrive, and a
slice silently doesn't, because Claude Code's OTLP exporter has no retry and no
buffer. Those sessions stay recoverable only until Claude Code prunes the
transcript.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from tokenjam.cli.cmd_doctor import _check_transcript_ingest_gap
from tokenjam.core.backfill import ingest_claude_code
from tokenjam.core.config import IngestConfig, StorageConfig, TjConfig
from tokenjam.core.db import DuckDBBackend


def _write_session(root: Path, session_id: str) -> None:
    project_dir = root / "-tmp-proj"
    project_dir.mkdir(parents=True, exist_ok=True)
    (project_dir / f"{session_id}.jsonl").write_text(json.dumps({
        "type": "assistant",
        "uuid": f"{session_id}-u1",
        "sessionId": session_id,
        "cwd": "/tmp/proj",
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
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


@pytest.fixture()
def db(tmp_path: Path) -> DuckDBBackend:
    backend = DuckDBBackend(StorageConfig(path=str(tmp_path / "t.duckdb")))
    yield backend
    backend.close()


@pytest.fixture()
def projects_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "projects"
    root.mkdir()
    monkeypatch.setenv("TJ_CLAUDE_PROJECTS_ROOT", str(root))
    return root


def test_warns_when_an_on_disk_session_was_never_ingested(
    db: DuckDBBackend, projects_root: Path,
) -> None:
    _write_session(projects_root, "sess-dropped")

    check = _check_transcript_ingest_gap(TjConfig(version="1"), db)

    assert check["level"] == "warning"
    assert "1 on-disk session(s) are not ingested" in check["message"]
    assert "day(s) from being pruned" in check["message"]


def test_reports_ok_when_everything_on_disk_is_ingested(
    db: DuckDBBackend, projects_root: Path,
) -> None:
    _write_session(projects_root, "sess-a")
    ingest_claude_code(db, root=projects_root)

    check = _check_transcript_ingest_gap(TjConfig(version="1"), db)

    assert check["level"] == "ok"


def test_remedy_names_the_manual_command_when_auto_catch_up_is_off(
    db: DuckDBBackend, projects_root: Path,
) -> None:
    _write_session(projects_root, "sess-dropped")
    config = TjConfig(version="1")
    config.ingest = IngestConfig(auto_catch_up=False)

    check = _check_transcript_ingest_gap(config, db)

    assert "Automatic catch-up is off" in check["message"]
    assert "tj backfill claude-code" in check["message"]


def test_is_informational_when_no_transcripts_exist(
    db: DuckDBBackend, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TJ_CLAUDE_PROJECTS_ROOT", str(tmp_path / "absent"))

    check = _check_transcript_ingest_gap(TjConfig(version="1"), db)

    assert check["level"] == "info"


def test_escalates_to_error_when_the_daemon_is_not_running(
    db: DuckDBBackend, projects_root: Path,
) -> None:
    """A gap that nothing is positioned to close automatically must FAIL, not
    warn — daemon down means `auto_catch_up` structurally cannot run."""
    _write_session(projects_root, "sess-dropped")

    check = _check_transcript_ingest_gap(TjConfig(version="1"), db, daemon_alive=False)

    assert check["level"] == "error"
    assert "background daemon is not running" in check["message"]


def test_stays_a_warning_when_the_daemon_is_running_and_auto_catch_up_is_on(
    db: DuckDBBackend, projects_root: Path,
) -> None:
    """Explicit True (matching the default) still self-heals — a live daemon
    with auto-catch-up on should close the gap on its own next pass."""
    _write_session(projects_root, "sess-dropped")

    check = _check_transcript_ingest_gap(TjConfig(version="1"), db, daemon_alive=True)

    assert check["level"] == "warning"


def test_never_raises_when_the_comparison_fails(
    db: DuckDBBackend, projects_root: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A health check that crashes takes the whole `tj doctor` run with it."""
    def _boom(*args, **kwargs):
        raise RuntimeError("disk on fire")

    monkeypatch.setattr(
        "tokenjam.core.transcript_sync.reconcile_claude_code", _boom
    )

    check = _check_transcript_ingest_gap(TjConfig(version="1"), db)

    assert check["level"] == "info"
    assert "disk on fire" in check["message"]
