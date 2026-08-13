"""Unit tests for the MCP schema measurement (core/optimize/mcp_probe.py).

The probe is the thing that replaced an unmeasured constant, so the tests that
matter are the ones about HONESTY, not about arithmetic: a server that cannot be
started must produce a measurement that says so, never a number.

One test does start a real subprocess — a tiny stdio MCP server written to
``tmp_path`` and run with this interpreter. That is deliberate. Serializing a
hand-built ``tools/list`` payload proves the arithmetic; only speaking the
protocol to a process proves the probe. It sends exactly ``initialize`` and
``tools/list``, which is the whole read-only contract.
"""
from __future__ import annotations

import sys
from datetime import timedelta
from pathlib import Path

from tokenjam.core import agent_config as ac
from tokenjam.core.optimize import mcp_probe
from tokenjam.utils.time_parse import utcnow

_TOOLS = [
    {
        "name": "search",
        "description": "Search the corpus for a phrase.",
        "inputSchema": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "the phrase"}},
            "required": ["query"],
        },
    },
    {
        "name": "fetch",
        "description": "Fetch one document by id.",
        "inputSchema": {
            "type": "object",
            "properties": {"id": {"type": "string"}},
        },
    },
]

#: A minimal stdio MCP server: line-delimited JSON-RPC, answers `initialize`
#: and `tools/list`, ignores everything else. Written to disk by the test rather
#: than shipped, so it cannot drift into being treated as product code.
_FAKE_SERVER = '''
import json, sys

TOOLS = {tools!r}

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    try:
        msg = json.loads(line)
    except ValueError:
        continue
    if msg.get("method") == "initialize":
        sys.stdout.write(json.dumps({{
            "jsonrpc": "2.0", "id": msg["id"],
            "result": {{"protocolVersion": "2024-11-05", "capabilities": {{}},
                       "serverInfo": {{"name": "fake", "version": "1"}}}},
        }}) + "\\n")
        sys.stdout.flush()
    elif msg.get("method") == "tools/list":
        sys.stdout.write(json.dumps({{
            "jsonrpc": "2.0", "id": msg["id"], "result": {{"tools": TOOLS}},
        }}) + "\\n")
        sys.stdout.flush()
'''


def _write_fake_server(tmp_path: Path, tools=_TOOLS) -> Path:
    path = tmp_path / "fake_mcp_server.py"
    path.write_text(_FAKE_SERVER.format(tools=tools), encoding="utf-8")
    return path


# --- Serialization ----------------------------------------------------------

def test_the_serialized_schema_carries_the_namespaced_tool_name():
    """The model receives ``mcp__<server>__<tool>``, so that is what is sized.

    A measurement of the raw ``tools/list`` payload would miss the prefix, and a
    server with many long tool names really does pay for them.
    """
    blob = mcp_probe.serialize_tool_schemas("apollo", _TOOLS)
    assert "mcp__apollo__search" in blob
    assert "mcp__apollo__fetch" in blob
    # The input schema is part of what is injected and part of what is counted.
    assert "the phrase" in blob


def test_the_deferred_listing_is_smaller_than_the_full_schema():
    """Both lanes are measured off ONE response.

    The deferred lane used to be its own assumed constant. Deriving it from the
    same tools means the two can never drift apart, and the relationship the
    analyzer relies on (a listing is cheaper than a schema) is a measured fact
    rather than an arithmetic coincidence between two hand-picked numbers.
    """
    full = mcp_probe.serialize_tool_schemas("apollo", _TOOLS)
    listing = mcp_probe.serialize_deferred_listing("apollo", _TOOLS)
    assert len(listing) < len(full)
    measurement = mcp_probe.measurement_from_tools("apollo", _TOOLS)
    assert measurement.ok
    assert measurement.tool_count == 2
    assert measurement.deferred_tokens < measurement.tokens


def test_a_server_with_no_tools_measures_to_a_real_zero():
    """Zero tools is a MEASURED zero, and is not the same as unmeasured.

    Both would render as "no cost" to a careless consumer, which is exactly why
    ``status`` exists alongside ``tokens``.
    """
    measurement = mcp_probe.measurement_from_tools("empty", [])
    assert measurement.status == ac.MEASURE_OK
    assert measurement.tokens is not None
    assert measurement.ok


# --- The probe itself -------------------------------------------------------

def test_probing_a_real_stdio_server_measures_its_tools(tmp_path):
    """The end-to-end path: spawn, handshake, list, measure, terminate."""
    server = _write_fake_server(tmp_path)
    result = mcp_probe.measure_server(
        "fake", {"command": sys.executable, "args": [str(server)]}, timeout=30.0,
    )
    assert result.status == ac.MEASURE_OK, result.detail
    assert result.tool_count == 2
    expected = ac.tokens_for_chars(len(mcp_probe.serialize_tool_schemas("fake", _TOOLS)))
    assert result.tokens == expected
    assert result.deferred_tokens is not None
    assert result.deferred_tokens < result.tokens


def test_a_server_that_cannot_start_never_returns_a_number():
    """THE property. No fallback, no floor, no default — ``tokens is None``."""
    result = mcp_probe.measure_server(
        "ghost", {"command": "tj-no-such-binary-anywhere", "args": []}, timeout=5.0,
    )
    assert result.tokens is None
    assert result.status == ac.MEASURE_UNREACHABLE
    assert not result.ok
    assert "could not start" in result.detail


def test_a_process_that_says_nothing_times_out_without_hanging(tmp_path):
    """A server that never answers must not hang the analyzer, and must not be
    guessed at either."""
    silent = tmp_path / "silent.py"
    silent.write_text("import time\ntime.sleep(60)\n", encoding="utf-8")
    result = mcp_probe.measure_server(
        "silent", {"command": sys.executable, "args": [str(silent)]}, timeout=1.0,
    )
    assert result.tokens is None
    assert result.status == ac.MEASURE_UNREACHABLE


def test_a_remote_server_is_reported_unsupported_not_guessed():
    """There is nothing local to start, and probing it would mean issuing an
    authenticated third-party request as a side effect of an analysis run. That
    is a gap, and it is reported as one."""
    result = mcp_probe.measure_server("remote", {"type": "http", "url": "https://x"})
    assert result.tokens is None
    assert result.status == ac.MEASURE_UNSUPPORTED
    assert "third party" in result.detail


# --- Caching ----------------------------------------------------------------

def _store_with(name: str, spec: dict) -> tuple[ac.AgentConfigStore, str]:
    record = ac.ConfigRecord(
        kind=ac.KIND_MCP_SERVER, scope=ac.SCOPE_GLOBAL, root="", name=name,
        path="/tmp/.claude.json", content_hash="spec-hash-1",
        detail=dict(spec, spec_hash="spec-hash-1"),
    )
    store = ac.InMemoryAgentConfigStore()
    store.upsert([record])
    return store, record.config_id


class _Server:
    def __init__(self, config_id: str, spec: dict) -> None:
        self.config_id = config_id
        self.spec = spec


def test_a_successful_measurement_is_reused_until_the_spec_changes():
    """Starting a server is the expensive part, so it happens once."""
    spec = {"command": "x"}
    store, config_id = _store_with("apollo", spec)
    calls: list[str] = []

    def _take(name, _spec):
        calls.append(name)
        return mcp_probe.measurement_from_tools(name, _TOOLS)

    for _ in range(3):
        out = mcp_probe.resolve_schema_measurements(
            {"apollo": _Server(config_id, spec)}, store=store, measurer=_take,
        )
    assert calls == ["apollo"]
    assert out["apollo"].ok
    assert out["apollo"].from_cache
    assert out["apollo"].tool_count == 2
    assert out["apollo"].deferred_tokens is not None


def test_a_changed_launch_spec_invalidates_the_measurement():
    """A server whose command changed is a different server for this purpose.

    Believing the old number would be the same class of error as the constant
    this replaced: a figure carried forward past the evidence for it.
    """
    spec = {"command": "x"}
    store, config_id = _store_with("apollo", spec)
    mcp_probe.resolve_schema_measurements(
        {"apollo": _Server(config_id, spec)}, store=store,
        measurer=lambda n, s: mcp_probe.measurement_from_tools(n, _TOOLS),
    )
    assert store.measurement_for(config_id).status == ac.MEASURE_OK

    changed = ac.ConfigRecord(
        kind=ac.KIND_MCP_SERVER, scope=ac.SCOPE_GLOBAL, root="", name="apollo",
        path="/tmp/.claude.json", content_hash="spec-hash-2",
        detail={"command": "y", "spec_hash": "spec-hash-2"},
    )
    store.upsert([changed])
    assert store.measurement_for(changed.config_id).status == ""


def test_a_failed_measurement_is_retried_after_its_grace_period():
    """"Down" is a statement about a moment, not about the server.

    A success is believed until the spec moves; a failure is believed only for
    :data:`UNREACHABLE_RETRY_AFTER`, so a transiently-broken server is not
    written off forever — while a broken one is not respawned on every pass.
    """
    spec = {"command": "x"}
    store, config_id = _store_with("apollo", spec)
    store.record_measurement(
        config_id, tokens=None, status=ac.MEASURE_UNREACHABLE,
        at=utcnow() - mcp_probe.UNREACHABLE_RETRY_AFTER - timedelta(minutes=1),
    )
    calls: list[str] = []

    def _take(name, _spec):
        calls.append(name)
        return mcp_probe.measurement_from_tools(name, _TOOLS)

    out = mcp_probe.resolve_schema_measurements(
        {"apollo": _Server(config_id, spec)}, store=store, measurer=_take,
    )
    assert calls == ["apollo"]
    assert out["apollo"].ok


def test_measurement_disabled_reports_skipped_and_prices_nothing():
    spec = {"command": "x"}
    store, config_id = _store_with("apollo", spec)
    out = mcp_probe.resolve_schema_measurements(
        {"apollo": _Server(config_id, spec)}, store=store, enabled=False,
        measurer=lambda n, s: mcp_probe.measurement_from_tools(n, _TOOLS),
    )
    assert out["apollo"].tokens is None
    assert out["apollo"].status == ac.MEASURE_SKIPPED


def test_a_measurer_that_raises_does_not_take_the_analyzer_down():
    spec = {"command": "x"}
    store, config_id = _store_with("apollo", spec)

    def _boom(_name, _spec):
        raise RuntimeError("kaboom")

    out = mcp_probe.resolve_schema_measurements(
        {"apollo": _Server(config_id, spec)}, store=store, measurer=_boom,
    )
    assert out["apollo"].tokens is None
    assert out["apollo"].status == ac.MEASURE_UNREACHABLE
    assert "kaboom" in out["apollo"].detail
