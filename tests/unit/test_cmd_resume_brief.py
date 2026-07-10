"""CLI-level tests for `tj resume-brief --from-hook`.

The SessionStart-hook path reads stdin JSON (session_id / transcript_path per
Claude Code's contract) instead of the global mtime scan, so a concurrent
session in a different project can't cross-leak its brief. Uses Click's
CliRunner (canonical pattern from test_cmd_hook.py).
"""
from __future__ import annotations

import json
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from tokenjam.cli.main import cli
from tokenjam.core.config import TjConfig


@pytest.fixture
def runner():
    return CliRunner()


def _config(tmp_path) -> TjConfig:
    cfg = TjConfig(version="1")
    cfg.storage.path = str(tmp_path / "tj.duckdb")
    return cfg


def _invoke(runner, cfg, args, stdin=""):
    with patch("tokenjam.cli.main.load_config", return_value=cfg):
        return runner.invoke(cli, args, input=stdin)


def test_from_hook_uses_stdin_transcript_path(runner, tmp_path):
    """--from-hook prefers stdin transcript_path over mtime scan."""
    # A hand-built transcript file with a plausible session id
    projects = tmp_path / "projects" / "proj-a"
    projects.mkdir(parents=True)
    tpath = projects / "sess-abc.jsonl"
    tpath.write_text(json.dumps({"type": "user", "message": {"content": "Do the thing"}}) + "\n")

    payload = json.dumps({
        "session_id": "sess-abc",
        "transcript_path": str(tpath),
        "source": "resume",
    })
    result = _invoke(runner, _config(tmp_path), ["resume-brief", "--from-hook"], stdin=payload)
    assert result.exit_code == 0
    # We don't assert on brief content (that's the synthesizer's contract) —
    # only that the command completed cleanly on a real transcript path.


def test_from_hook_with_empty_stdin_exits_silently(runner, tmp_path):
    """No usable signal on stdin → exit 0 with no output (never guess)."""
    result = _invoke(runner, _config(tmp_path), ["resume-brief", "--from-hook"], stdin="")
    assert result.exit_code == 0
    assert result.output == ""


def test_from_hook_with_malformed_json_exits_silently(runner, tmp_path):
    """Malformed stdin JSON must not raise; degrade to no-op."""
    result = _invoke(runner, _config(tmp_path), ["resume-brief", "--from-hook"], stdin="{not json")
    assert result.exit_code == 0
    assert result.output == ""


def test_from_hook_with_stdin_missing_keys_exits_silently(runner, tmp_path):
    """Valid JSON but no session_id / transcript_path → no-op (no cross-leak)."""
    payload = json.dumps({"source": "resume"})
    result = _invoke(runner, _config(tmp_path), ["resume-brief", "--from-hook"], stdin=payload)
    assert result.exit_code == 0
    assert result.output == ""


def test_no_flags_raises_usage_error(runner, tmp_path):
    """`tj resume-brief` with no mode flag is a usage error (unchanged)."""
    result = _invoke(runner, _config(tmp_path), ["resume-brief"], stdin="")
    assert result.exit_code != 0
    assert "Provide --session" in result.output or "Usage" in result.output


# --- top-driver hint (neutral informational wording) ------------------------


def test_top_driver_hint_neutral_wording(tmp_path, monkeypatch):
    """The hint reads as neutral context, not a "TOP RE-READ DRIVER" alert.

    The resume-brief hook can't compute the live re-read share, so a low-share
    user must not see an alarm — only a plain informational line.
    """
    from tokenjam.cli.cmd_resume_brief import _top_driver_hint
    from tokenjam.core.attribution_cache import write_attribution_cache

    cache_path = tmp_path / "attribution_cache.json"
    monkeypatch.setattr(
        "tokenjam.core.attribution_cache._cache_path", lambda: cache_path
    )
    write_attribution_cache("CLAUDE.md", 14, 3, path=cache_path)

    hint = _top_driver_hint()
    assert "Most re-read context (last 30d): CLAUDE.md ×14" in hint
    assert "TOP RE-READ DRIVER" not in hint  # not an alert
    assert "—" not in hint  # house style: no em dashes in user-facing copy


def test_top_driver_hint_empty_when_no_cache(tmp_path, monkeypatch):
    from tokenjam.cli.cmd_resume_brief import _top_driver_hint

    monkeypatch.setattr(
        "tokenjam.core.attribution_cache._cache_path",
        lambda: tmp_path / "missing.json",
    )
    assert _top_driver_hint() == ""
