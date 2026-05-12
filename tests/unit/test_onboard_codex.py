"""Unit tests for Codex config writing helpers in cmd_onboard."""
from __future__ import annotations

from tokenjam.cli.cmd_onboard import (
    _codex_apply_block,
    _codex_mcp_toml_block,
    _codex_purge_legacy_ocw,
)


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


class TestCodexPurgeLegacyOcw:
    def test_empty_content_unchanged(self):
        assert _codex_purge_legacy_ocw("") == ""

    def test_no_legacy_sections_unchanged(self):
        content = (
            '[otel]\n'
            '# Managed by tj — do not edit this block manually\n'
            'log_user_prompt = false\n'
            '\n'
            '[mcp_servers.tj]\n'
            'command = "tj"\n'
            'args = ["mcp"]\n'
        )
        # Trailing newline normalization is fine, but the substantive content
        # should be identical.
        result = _codex_purge_legacy_ocw(content)
        assert "[mcp_servers.ocw]" not in result
        assert "Managed by ocw" not in result
        assert "[mcp_servers.tj]" in result
        assert "[otel]" in result
        assert "Managed by tj" in result

    def test_drops_legacy_mcp_servers_ocw_block(self):
        content = (
            '[mcp_servers.ocw]\n'
            '# Managed by ocw — gives Codex access to OCW observability tools\n'
            'command = "ocw"\n'
            'args = ["mcp"]\n'
            '\n'
            '[mcp_servers.tj]\n'
            'command = "tj"\n'
            'args = ["mcp"]\n'
        )
        result = _codex_purge_legacy_ocw(content)
        assert "[mcp_servers.ocw]" not in result
        assert 'command = "ocw"' not in result
        assert "[mcp_servers.tj]" in result
        assert 'command = "tj"' in result

    def test_drops_legacy_otel_block_marked_managed_by_ocw(self):
        content = (
            '[otel]\n'
            '# Managed by ocw — do not edit this block manually\n'
            'log_user_prompt = false\n'
            '\n'
            '[otel.exporter."otlp-http"]\n'
            'endpoint = "http://127.0.0.1:7391/v1/logs"\n'
            'protocol = "json"\n'
            '\n'
            '[otel.exporter."otlp-http".headers]\n'
            'Authorization = "Bearer abc123"\n'
        )
        result = _codex_purge_legacy_ocw(content)
        assert "[otel]" not in result
        assert "[otel.exporter" not in result
        assert "Managed by ocw" not in result
        assert "Authorization" not in result

    def test_preserves_otel_block_not_marked_ocw(self):
        """A [otel] block without the 'Managed by ocw' comment is left alone —
        only the legacy ocw-marked sections are stripped."""
        content = (
            '[otel]\n'
            'log_user_prompt = false\n'
            '\n'
            '[otel.exporter."otlp-http"]\n'
            'endpoint = "http://127.0.0.1:7391/v1/logs"\n'
        )
        result = _codex_purge_legacy_ocw(content)
        assert "[otel]" in result
        assert "[otel.exporter" in result

    def test_real_world_dual_blocks_from_user_report(self):
        """End-to-end repro of the actual file from issue testing:
        both [mcp_servers.ocw] and [mcp_servers.tj] present, plus an [otel]
        block carrying the legacy 'Managed by ocw' comment."""
        content = (
            '[otel]\n'
            '# Managed by ocw — do not edit this block manually\n'
            'log_user_prompt = false\n'
            '\n'
            '[otel.exporter."otlp-http"]\n'
            'endpoint = "http://127.0.0.1:7391/v1/logs"\n'
            'protocol = "json"\n'
            '\n'
            '[otel.exporter."otlp-http".headers]\n'
            'Authorization = "Bearer d473add7b0a19c18ca8a8014f7df760e29017578fcdc93339424172192c787ea"\n'
            '\n'
            '[mcp_servers.ocw]\n'
            '# Managed by ocw — gives Codex access to OCW observability tools\n'
            'command = "ocw"\n'
            'args = ["mcp"]\n'
            '\n'
            '[mcp_servers.tj]\n'
            '# Managed by tj — gives Codex access to TokenJam observability tools\n'
            'command = "tj"\n'
            'args = ["mcp"]\n'
        )
        result = _codex_purge_legacy_ocw(content)
        # All ocw-managed sections gone.
        assert "[mcp_servers.ocw]" not in result
        assert "Managed by ocw" not in result
        assert 'command = "ocw"' not in result
        # The [otel] block was ocw-managed, so it was stripped — the caller
        # is responsible for writing a fresh tj-managed [otel] block.
        assert "[otel]" not in result
        # The tj MCP block survived.
        assert "[mcp_servers.tj]" in result
        assert 'command = "tj"' in result
