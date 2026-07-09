"""Tests for the tj-managed ~/.zshrc OTEL export block: stable sentinel +
replace-all semantics across onboard and uninstall (#118).

Before this fix, the block was keyed on a single-line comment marker that had
itself already drifted once (the ocw -> tj rebrand renamed
"# ocw harness observability" to "# tj harness observability" without
migrating existing installs). A block written under an older marker was
invisible to both re-onboard's "replace in place" and `tj uninstall`'s
removal: re-onboarding APPENDED a second block with a fresh bearer token
instead of replacing the first (stale secrets accumulate in the user's shell
rc), and uninstall only stripped the current-marker block, leaving the
old-marker one behind.

`_strip_zshrc_otel_blocks` is the single removal routine shared by both
onboard (called before appending exactly one fresh block) and uninstall
(called for removal only) — see cmd_onboard.py and cmd_uninstall.py.
"""
from __future__ import annotations

from tokenjam.cli.cmd_onboard import (
    _ZSHRC_OTEL_END,
    _ZSHRC_OTEL_START,
    _strip_zshrc_otel_blocks,
    _zshrc_otel_block,
)

_LEGACY_OCW_BLOCK = (
    "# ocw harness observability\n"
    "export CLAUDE_CODE_ENABLE_TELEMETRY=1\n"
    "export OTEL_LOGS_EXPORTER=otlp\n"
    "export OTEL_EXPORTER_OTLP_PROTOCOL=http/json\n"
    "export OTEL_EXPORTER_OTLP_ENDPOINT=http://host.docker.internal:7391\n"
    'export OTEL_EXPORTER_OTLP_HEADERS="Authorization=Bearer stale-ocw-token"\n'
)

_LEGACY_TJ_BLOCK = (
    "# tj harness observability\n"
    "export CLAUDE_CODE_ENABLE_TELEMETRY=1\n"
    "export OTEL_LOGS_EXPORTER=otlp\n"
    "export OTEL_EXPORTER_OTLP_PROTOCOL=http/json\n"
    "export OTEL_EXPORTER_OTLP_ENDPOINT=http://host.docker.internal:7391\n"
    'export OTEL_EXPORTER_OTLP_HEADERS="Authorization=Bearer stale-tj-token"\n'
)


def test_zshrc_otel_block_is_sentinel_delimited():
    block = _zshrc_otel_block(7391, "secret123")
    assert block.startswith(_ZSHRC_OTEL_START + "\n")
    assert block.rstrip("\n").endswith(_ZSHRC_OTEL_END)
    assert "Authorization=Bearer secret123" in block
    assert "http://host.docker.internal:7391" in block


def test_strip_removes_current_sentinel_block():
    text = "# unrelated line\n" + _zshrc_otel_block(7391, "secret123") + "\n# after\n"
    cleaned = _strip_zshrc_otel_blocks(text)
    assert _ZSHRC_OTEL_START not in cleaned
    assert "Authorization=Bearer secret123" not in cleaned
    assert "# unrelated line" in cleaned
    assert "# after" in cleaned


def test_strip_removes_legacy_tj_marker_block():
    text = "# unrelated line\n" + _LEGACY_TJ_BLOCK
    cleaned = _strip_zshrc_otel_blocks(text)
    assert "# tj harness observability" not in cleaned
    assert "stale-tj-token" not in cleaned
    assert "# unrelated line" in cleaned


def test_strip_removes_legacy_ocw_marker_block():
    text = "# unrelated line\n" + _LEGACY_OCW_BLOCK
    cleaned = _strip_zshrc_otel_blocks(text)
    assert "# ocw harness observability" not in cleaned
    assert "stale-ocw-token" not in cleaned
    assert "# unrelated line" in cleaned


def test_strip_removes_both_legacy_markers_at_once():
    # The exact real-world shape from #118: a ~/.zshrc that accumulated both
    # an old-marker block (pre-rebrand) AND a current-marker block (post-
    # rebrand, pre-sentinel), each with a different bearer token.
    text = _LEGACY_OCW_BLOCK + "\n" + _LEGACY_TJ_BLOCK
    cleaned = _strip_zshrc_otel_blocks(text)
    assert "harness observability" not in cleaned
    assert "stale-ocw-token" not in cleaned
    assert "stale-tj-token" not in cleaned


def test_strip_removes_sentinel_block_with_no_trailing_newline():
    """A managed block that is the LAST line of the file, with no final
    newline, must still be stripped — the sentinel regex previously required
    a hard `\\n` after `_ZSHRC_OTEL_END`, silently no-opping here and leaving
    a bearer token behind after "removal"."""
    block = _zshrc_otel_block(7391, "secret123")
    text = "# unrelated line\n" + block.rstrip("\n")  # no trailing newline
    assert not text.endswith("\n")
    cleaned = _strip_zshrc_otel_blocks(text)
    assert _ZSHRC_OTEL_START not in cleaned
    assert "Authorization=Bearer secret123" not in cleaned
    assert "# unrelated line" in cleaned


def test_strip_removes_legacy_marker_block_with_no_trailing_newline():
    """Same no-final-newline case for the legacy-marker path."""
    text = "# unrelated line\n" + _LEGACY_TJ_BLOCK.rstrip("\n")
    assert not text.endswith("\n")
    cleaned = _strip_zshrc_otel_blocks(text)
    assert "# tj harness observability" not in cleaned
    assert "stale-tj-token" not in cleaned
    assert "# unrelated line" in cleaned


def test_strip_is_noop_on_text_without_managed_blocks():
    text = "export PATH=/usr/bin:$PATH\nalias ll='ls -la'\n"
    assert _strip_zshrc_otel_blocks(text) == text


def test_onboard_replace_all_leaves_exactly_one_block():
    """Simulates onboard's zshrc write: strip everything managed (both legacy
    markers plus any current sentinel block), then append exactly one fresh
    block. A ~/.zshrc seeded with both legacy blocks ends up with ONE block,
    carrying the new secret only."""
    seeded = "# my own env\nexport FOO=bar\n\n" + _LEGACY_OCW_BLOCK + "\n" + _LEGACY_TJ_BLOCK
    stripped = _strip_zshrc_otel_blocks(seeded)
    fresh_block = _zshrc_otel_block(7391, "fresh-secret")
    result = (stripped.rstrip("\n") + "\n\n" + fresh_block) if stripped.strip() else fresh_block

    assert result.count(_ZSHRC_OTEL_START) == 1
    assert result.count("harness observability") == 0  # both legacy markers gone
    assert "stale-ocw-token" not in result
    assert "stale-tj-token" not in result
    assert "Authorization=Bearer fresh-secret" in result
    assert "export FOO=bar" in result  # user's own content preserved


def test_uninstall_cleanup_removes_all_managed_blocks():
    """Simulates uninstall's zshrc cleanup: strip every managed block (current
    sentinel + every legacy marker). Zero tj OTEL exports remain."""
    seeded = "# my own env\nexport FOO=bar\n\n" + _LEGACY_OCW_BLOCK + "\n" + _LEGACY_TJ_BLOCK + "\n" + _zshrc_otel_block(7391, "current-secret")
    cleaned = _strip_zshrc_otel_blocks(seeded)

    assert "harness observability" not in cleaned
    assert _ZSHRC_OTEL_START not in cleaned
    assert "OTEL_EXPORTER_OTLP_HEADERS" not in cleaned
    assert "export FOO=bar" in cleaned  # user's own content preserved
