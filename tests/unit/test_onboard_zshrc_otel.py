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

import pytest

from tokenjam.cli.cmd_onboard import (
    _DOCKER_GATEWAY_HOST,
    _OTEL_HOST_ENV,
    _ZSHRC_OTEL_END,
    _ZSHRC_OTEL_START,
    _otel_endpoint_host,
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


@pytest.fixture(autouse=True)
def _host_env_off(monkeypatch):
    """The endpoint host is env-overridable; a real `TJ_OTEL_HOST` in the dev
    or CI environment must not decide what these tests assert."""
    monkeypatch.delenv(_OTEL_HOST_ENV, raising=False)


def test_zshrc_otel_block_is_sentinel_delimited(monkeypatch):
    monkeypatch.setattr("tokenjam.cli.cmd_onboard._in_container", lambda: False)
    block = _zshrc_otel_block(7391, "secret123")
    assert block.startswith(_ZSHRC_OTEL_START + "\n")
    assert block.rstrip("\n").endswith(_ZSHRC_OTEL_END)
    assert "Authorization=Bearer secret123" in block
    assert "http://127.0.0.1:7391" in block


# -- endpoint host selection -------------------------------------------------
# `host.docker.internal` is the address a process INSIDE a container uses to
# reach its host. Written unconditionally into a HOST user's ~/.zshrc it does
# not resolve at all, so every span an agent session emitted failed at DNS and
# was dropped with no error surfaced anywhere.

def test_endpoint_host_is_loopback_on_a_host(monkeypatch):
    monkeypatch.setattr("tokenjam.cli.cmd_onboard._in_container", lambda: False)
    assert _otel_endpoint_host() == "127.0.0.1"


def test_container_only_hostname_is_never_written_on_a_host(monkeypatch):
    """Even where the Docker gateway name happens to resolve, a machine that
    is not in a container gets the loopback address."""
    monkeypatch.setattr("tokenjam.cli.cmd_onboard._in_container", lambda: False)
    monkeypatch.setattr("tokenjam.cli.cmd_onboard._host_resolves", lambda h: True)
    assert _DOCKER_GATEWAY_HOST not in _zshrc_otel_block(7391, "secret123")


def test_endpoint_host_is_docker_gateway_inside_a_container(monkeypatch):
    monkeypatch.setattr("tokenjam.cli.cmd_onboard._in_container", lambda: True)
    monkeypatch.setattr("tokenjam.cli.cmd_onboard._host_resolves", lambda h: True)
    assert _otel_endpoint_host() == _DOCKER_GATEWAY_HOST


def test_container_falls_back_to_loopback_when_gateway_does_not_resolve(monkeypatch):
    """In a container whose runtime publishes no host-gateway name, an
    unresolvable hostname is worse than a wrong-but-resolvable one: it fails
    before a connection is ever attempted."""
    monkeypatch.setattr("tokenjam.cli.cmd_onboard._in_container", lambda: True)
    monkeypatch.setattr("tokenjam.cli.cmd_onboard._host_resolves", lambda h: False)
    assert _otel_endpoint_host() == "127.0.0.1"


def test_env_override_wins_over_detection(monkeypatch):
    monkeypatch.setenv(_OTEL_HOST_ENV, "otel.internal.example")
    monkeypatch.setattr("tokenjam.cli.cmd_onboard._in_container", lambda: False)
    assert _otel_endpoint_host() == "otel.internal.example"
    assert "http://otel.internal.example:7391" in _zshrc_otel_block(7391, "s")


def test_block_port_follows_the_port_it_is_given(monkeypatch):
    monkeypatch.setattr("tokenjam.cli.cmd_onboard._in_container", lambda: False)
    assert "http://127.0.0.1:7391" in _zshrc_otel_block(7391, "s")
    assert "http://127.0.0.1:9999" in _zshrc_otel_block(9999, "s")


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


def test_onboard_repairs_an_already_written_bad_endpoint(monkeypatch):
    """Re-onboarding must CORRECT an endpoint already written to the shell
    profile, not merely stop re-offending on fresh installs. The seeded block
    is a well-formed CURRENT-sentinel block carrying the unreachable
    container-only host, which is the state every machine onboarded before
    this fix is sitting in."""
    monkeypatch.setattr("tokenjam.cli.cmd_onboard._in_container", lambda: False)
    bad_block = (
        f"{_ZSHRC_OTEL_START}\n"
        "export CLAUDE_CODE_ENABLE_TELEMETRY=1\n"
        "export OTEL_LOGS_EXPORTER=otlp\n"
        "export OTEL_EXPORTER_OTLP_PROTOCOL=http/json\n"
        "export OTEL_EXPORTER_OTLP_ENDPOINT=http://host.docker.internal:7500\n"
        'export OTEL_EXPORTER_OTLP_HEADERS="Authorization=Bearer stale-secret"\n'
        f"{_ZSHRC_OTEL_END}\n"
    )
    seeded = "# my own env\nexport FOO=bar\n\n" + bad_block
    stripped = _strip_zshrc_otel_blocks(seeded)
    fresh = _zshrc_otel_block(7391, "fresh-secret")
    result = (stripped.rstrip("\n") + "\n\n" + fresh) if stripped.strip() else fresh

    assert result.count(_ZSHRC_OTEL_START) == 1
    assert "host.docker.internal" not in result
    assert ":7500" not in result
    assert "stale-secret" not in result
    assert "export OTEL_EXPORTER_OTLP_ENDPOINT=http://127.0.0.1:7391" in result
    assert "export FOO=bar" in result


def test_uninstall_cleanup_removes_all_managed_blocks():
    """Simulates uninstall's zshrc cleanup: strip every managed block (current
    sentinel + every legacy marker). Zero tj OTEL exports remain."""
    seeded = "# my own env\nexport FOO=bar\n\n" + _LEGACY_OCW_BLOCK + "\n" + _LEGACY_TJ_BLOCK + "\n" + _zshrc_otel_block(7391, "current-secret")
    cleaned = _strip_zshrc_otel_blocks(seeded)

    assert "harness observability" not in cleaned
    assert _ZSHRC_OTEL_START not in cleaned
    assert "OTEL_EXPORTER_OTLP_HEADERS" not in cleaned
    assert "export FOO=bar" in cleaned  # user's own content preserved
