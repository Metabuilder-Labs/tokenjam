"""`[loop].transcript_path`: pointing the loop at a non-Claude-Code workspace.

A Claude Agent SDK app writes its session transcripts wherever the app puts
them, so without this the loop can't see it at all (the detector globs
~/.claude/projects and finds nothing). Covers the config parse, the resolution
order, and that the default is unchanged.

Nothing here touches the real ~/.claude: every assertion either uses tmp_path or
compares against the resolved default without reading it.
"""
from __future__ import annotations

from pathlib import Path

from tokenjam.core.config import LoopConfig, TjConfig, _parse
from tokenjam.core.transcript import loop_transcript_root, resolve_projects_root


# -- Config parse ------------------------------------------------------------

def test_transcript_path_defaults_to_none():
    assert _parse({"version": "1"}).loop.transcript_path is None


def test_transcript_path_parses_from_the_loop_section():
    cfg = _parse({"version": "1", "loop": {"transcript_path": "/srv/app/.transcripts"}})

    assert cfg.loop.transcript_path == "/srv/app/.transcripts"


def test_empty_transcript_path_reads_as_unset():
    cfg = _parse({"version": "1", "loop": {"transcript_path": ""}})

    assert cfg.loop.transcript_path is None


def test_tjconfig_has_a_loop_section_by_default():
    assert isinstance(TjConfig(version="1").loop, LoopConfig)


# -- Resolution --------------------------------------------------------------

def test_configured_path_wins(tmp_path):
    cfg = TjConfig(version="1", loop=LoopConfig(transcript_path=str(tmp_path)))

    assert loop_transcript_root(cfg) == tmp_path


def test_falls_back_to_the_default_root_when_unset(monkeypatch, tmp_path):
    monkeypatch.setenv("TJ_CLAUDE_PROJECTS_ROOT", str(tmp_path))
    cfg = TjConfig(version="1")

    assert loop_transcript_root(cfg) == tmp_path == resolve_projects_root()


def test_no_config_falls_back_to_the_default_root(monkeypatch, tmp_path):
    monkeypatch.setenv("TJ_CLAUDE_PROJECTS_ROOT", str(tmp_path))

    assert loop_transcript_root(None) == tmp_path


def test_user_home_is_expanded():
    cfg = TjConfig(version="1", loop=LoopConfig(transcript_path="~/app-transcripts"))

    resolved = loop_transcript_root(cfg)

    assert "~" not in str(resolved)
    assert resolved == Path.home() / "app-transcripts"


def test_a_malformed_config_degrades_to_the_default(monkeypatch, tmp_path):
    monkeypatch.setenv("TJ_CLAUDE_PROJECTS_ROOT", str(tmp_path))

    class Broken:
        @property
        def loop(self):
            raise RuntimeError("boom")

    assert loop_transcript_root(Broken()) == tmp_path
