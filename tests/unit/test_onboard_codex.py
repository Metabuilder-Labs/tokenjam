"""Unit tests for Codex config writing helpers in cmd_onboard."""
from __future__ import annotations

from tj.cli.cmd_onboard import _codex_apply_block, _codex_mcp_toml_block


class TestCodexApplyBlock:
    def test_appends_when_absent(self):
        content = "[otel]\nlog_user_prompt = false\n"
        block = _codex_mcp_toml_block()
        result = _codex_apply_block(content, r"\[mcp_servers\.tj\]", False, block, False)
        assert "[mcp_servers.tj]" in result
        assert "[otel]" in result

    def test_skips_when_present_no_force(self):
        block = _codex_mcp_toml_block()
        content = "[otel]\nlog_user_prompt = false\n\n" + block
        result = _codex_apply_block(content, r"\[mcp_servers\.tj\]", True, block, False)
        assert result == content

    def test_replaces_when_present_with_force(self):
        old_block = "[mcp_servers.tj]\ncommand = \"old\"\n"
        new_block = _codex_mcp_toml_block()
        content = "[otel]\nlog_user_prompt = false\n\n" + old_block
        result = _codex_apply_block(content, r"\[mcp_servers\.tj\]", True, new_block, True)
        assert 'command = "tj"' in result
        assert 'command = "old"' not in result

    def test_otel_block_unchanged_when_mcp_appended(self):
        """Appending MCP block must not alter the existing otel section."""
        otel_section = "[otel]\nlog_user_prompt = false\n"
        block = _codex_mcp_toml_block()
        result = _codex_apply_block(otel_section, r"\[mcp_servers\.tj\]", False, block, False)
        assert result.startswith(otel_section.rstrip())
