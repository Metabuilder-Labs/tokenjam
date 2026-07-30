"""`setup_project` MCP tool resolves its config path via TJ_CONFIG.

Regression for the shape bug where several `find_config_file()` call sites
called it bare, ignoring `TJ_CONFIG`, even though the surrounding process was
already `TJ_CONFIG`-aware. This tool passes its resolved config_path into
`_tool_setup_project`, which uses it to decide where OTEL_RESOURCE_ATTRIBUTES
gets written — a bare call would silently report the wrong (or no) config
path when the MCP server process has TJ_CONFIG set.
"""
from __future__ import annotations

from unittest.mock import patch

from tokenjam.mcp import server as server_mod


def test_setup_project_honors_tj_config(monkeypatch, tmp_path):
    cfg_file = tmp_path / "custom.toml"
    cfg_file.write_text('version = "1"\n')
    monkeypatch.setenv("TJ_CONFIG", str(cfg_file))

    captured = {}

    def _fake_tool_setup_project(**kwargs):
        captured.update(kwargs)
        return {"ok": True}

    with patch.object(server_mod, "_tool_setup_project", _fake_tool_setup_project):
        result = server_mod.setup_project(agent_id="probe", project_path=str(tmp_path))

    assert result == {"ok": True}
    assert captured["config_path"] == str(cfg_file)


def test_setup_project_no_config_when_nothing_discoverable(monkeypatch, tmp_path):
    monkeypatch.delenv("TJ_CONFIG", raising=False)
    monkeypatch.chdir(tmp_path)
    import tokenjam.core.config as cfg_mod
    monkeypatch.setattr(cfg_mod, "SEARCH_PATHS", [
        tmp_path / "tokenjam.toml",
        tmp_path / ".tj" / "config.toml",
        tmp_path / ".config" / "tj" / "config.toml",
    ])

    captured = {}

    def _fake_tool_setup_project(**kwargs):
        captured.update(kwargs)
        return {"ok": True}

    with patch.object(server_mod, "_tool_setup_project", _fake_tool_setup_project):
        server_mod.setup_project(agent_id="probe", project_path=str(tmp_path))

    assert captured["config_path"] is None
