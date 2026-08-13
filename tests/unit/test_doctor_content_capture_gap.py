"""`tj doctor`'s content-capture-backfill-gap check: the OTHER side of
`_check_capture_prompts` (which only flags capture being OFF). When
`[capture]` is ON but a slice of backfilled history predates that, the spans
never got the content on their own — a plain re-run only ever INSERTED new
spans until `bulk_overlay_span_attrs` grew a content-merge half (the
`json_merge_patch` overlay). This pins the check + its `--repair`
end-to-end, against real transcripts, not mocked DB state."""
from __future__ import annotations

import json
from pathlib import Path

from tokenjam.core.backfill import ingest_claude_code
from tokenjam.core.config import CaptureConfig, TjConfig
from tokenjam.core.db import InMemoryBackend
from tokenjam.otel.semconv import GenAIAttributes


def _make_session_file(tmp_path: Path, session_id: str, cwd: str,
                        records: list[dict]) -> Path:
    project_dir = tmp_path / cwd.replace("/", "-")
    project_dir.mkdir(parents=True, exist_ok=True)
    path = project_dir / f"{session_id}.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in records))
    return path


def _content_session_file(tmp_path: Path) -> Path:
    return _make_session_file(
        tmp_path, session_id="sess-gap", cwd="/Users/me/proj",
        records=[
            {"type": "user", "message": {"role": "user",
                                         "content": "please read the config"}},
            {
                "type": "assistant", "uuid": "msg-gap",
                "timestamp": "2026-04-01T10:00:00.000Z",
                "sessionId": "sess-gap", "cwd": "/Users/me/proj",
                "message": {
                    "model": "claude-opus-4-7",
                    "content": [
                        {"type": "text", "text": "Reading the config file now."},
                        {"type": "tool_use", "id": "tu-gap", "name": "Read",
                         "input": {"file_path": "/etc/app/config.toml"}},
                    ],
                    "usage": {"input_tokens": 1000, "output_tokens": 200},
                },
            },
        ],
    )


def test_check_is_info_when_capture_is_off():
    from tokenjam.cli.cmd_doctor import _check_content_capture_backfill_gap

    db = InMemoryBackend()
    try:
        config = TjConfig(version="1")
        config.capture = CaptureConfig(prompts=False, completions=False, tool_inputs=False)
        check = _check_content_capture_backfill_gap(config, db)
        assert check["level"] == "info"
    finally:
        db.close()


def test_check_flags_backfilled_history_missing_content(tmp_path):
    from tokenjam.cli.cmd_doctor import _check_content_capture_backfill_gap

    _content_session_file(tmp_path)
    db = InMemoryBackend()
    try:
        # Ingested with capture OFF, exactly the pre-existing-history state.
        off_cfg = TjConfig(version="1")
        off_cfg.capture = CaptureConfig(prompts=False, tool_inputs=False)
        ingest_claude_code(db, root=tmp_path, config=off_cfg)

        # Now capture is on, but the check queries the ACTUAL span content —
        # so it must fire regardless of what config THIS check call passes.
        on_cfg = TjConfig(version="1")
        on_cfg.capture = CaptureConfig(prompts=True, completions=True, tool_inputs=True)
        check = _check_content_capture_backfill_gap(on_cfg, db)
        assert check["level"] == "warning"
        assert check["repair_action"] == "backfill_missing_content"
        assert "LLM span" in check["message"]
        assert "tool span" in check["message"]
    finally:
        db.close()


def test_doctor_repairs_the_content_gap_end_to_end(tmp_path):
    from tokenjam.cli.cmd_doctor import (
        _attempt_repairs,
        _check_content_capture_backfill_gap,
    )

    _content_session_file(tmp_path)
    db = InMemoryBackend()
    try:
        off_cfg = TjConfig(version="1")
        off_cfg.capture = CaptureConfig(prompts=False, tool_inputs=False)
        ingest_claude_code(db, root=tmp_path, config=off_cfg)

        on_cfg = TjConfig(version="1")
        on_cfg.capture = CaptureConfig(prompts=True, completions=True, tool_inputs=True)
        check = _check_content_capture_backfill_gap(on_cfg, db)
        assert check["level"] == "warning"

        # `_attempt_repairs`'s ingest_claude_code call reads capture off the
        # config it's given directly.
        import tokenjam.core.backfill as backfill_mod
        original_root = backfill_mod.CLAUDE_CODE_PROJECTS_ROOT
        backfill_mod.CLAUDE_CODE_PROJECTS_ROOT = tmp_path
        try:
            _attempt_repairs([check], db, output_json=True, config=on_cfg)
        finally:
            backfill_mod.CLAUDE_CODE_PROJECTS_ROOT = original_root

        raw = db.conn.execute(
            "SELECT attributes FROM spans WHERE name = 'gen_ai.llm.call'"
        ).fetchone()[0]
        attrs = json.loads(raw) if isinstance(raw, str) else raw
        assert attrs[GenAIAttributes.PROMPT_CONTENT] == "please read the config"
        assert attrs[GenAIAttributes.COMPLETION_CONTENT] == \
            "Reading the config file now."

        assert _check_content_capture_backfill_gap(on_cfg, db)["level"] == "ok"
    finally:
        db.close()
