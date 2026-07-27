"""`tj mcp` config discovery honors TJ_CONFIG.

Regression for the shape bug where several `find_config_file()` call sites
across the codebase called it bare, ignoring `TJ_CONFIG`, even though the
surrounding code already assumes a `TJ_CONFIG`-aware config. `cmd_mcp` boots
the MCP server against whatever config it resolves here — a bare call would
silently fall back to the global/search-path config even when the user (or
an SDK-launched subprocess) set `TJ_CONFIG` to point elsewhere.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from tokenjam.cli.cmd_mcp import cmd_mcp


def test_cmd_mcp_resolves_config_via_tj_config(monkeypatch, tmp_path):
    cfg_file = tmp_path / "custom.toml"
    cfg_file.write_text(
        'version = "1"\n\n[storage]\npath = "%s"\n' % (tmp_path / "db.duckdb")
    )
    monkeypatch.setenv("TJ_CONFIG", str(cfg_file))

    mock_init = MagicMock()
    mock_mcp = MagicMock()
    with patch("tokenjam.mcp.server.init", mock_init), \
         patch("tokenjam.mcp.server.mcp", mock_mcp), \
         patch("tokenjam.cli.cmd_mcp._port_open", return_value=False), \
         patch("tokenjam.cli.cmd_mcp._start_and_wait", return_value=False), \
         patch("tokenjam.cli.cmd_mcp.duckdb.connect", return_value=MagicMock()):
        result = CliRunner().invoke(cmd_mcp, [], obj={})

    assert result.exit_code == 0, result.output
    assert mock_init.called
    _, kwargs = mock_init.call_args
    assert kwargs["config"].storage.path == str(tmp_path / "db.duckdb")


def test_cmd_mcp_falls_back_to_no_config_sentinel_without_tj_config(monkeypatch, tmp_path):
    """No TJ_CONFIG and nothing on the search path → init() never called, so
    the MCP tools return their built-in no-config sentinel."""
    monkeypatch.delenv("TJ_CONFIG", raising=False)
    monkeypatch.chdir(tmp_path)
    import tokenjam.core.config as cfg_mod
    monkeypatch.setattr(cfg_mod, "SEARCH_PATHS", [
        tmp_path / "tokenjam.toml",
        tmp_path / ".tj" / "config.toml",
        tmp_path / ".config" / "tj" / "config.toml",
    ])

    mock_init = MagicMock()
    mock_mcp = MagicMock()
    with patch("tokenjam.mcp.server.init", mock_init), \
         patch("tokenjam.mcp.server.mcp", mock_mcp):
        result = CliRunner().invoke(cmd_mcp, [], obj={})

    assert result.exit_code == 0, result.output
    assert not mock_init.called
