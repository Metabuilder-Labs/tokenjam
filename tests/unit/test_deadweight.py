"""Unit tests for the MCP dead-weight + context-tax analyzer
(core/optimize/analyzers/deadweight.py).

Mirrors test_relearn.py's fixture style — hand-written Claude Code on-disk
JSONL records under a tmp_path projects root, no I/O beyond that. The global
``~/.claude.json`` path is resolved lazily inside ``_global_config_path``, so
patching ``HOME`` (via monkeypatch, same as tests/conftest.py's autouse
``_tj_isolated_home`` fixture) is enough to keep every test off the real
developer machine — no test here ever touches the real home.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from tokenjam.core.optimize.analyzers.deadweight import (
    DEFERRED_SCHEMA_TAX_TOKENS,
    FULL_SCHEMA_TAX_TOKENS,
    MIN_SESSIONS_DEADWEIGHT,
    compute_deadweight_finding,
    enumerate_configured_servers,
)

_NOW = datetime.now(timezone.utc)
_SINCE = _NOW - timedelta(days=7)
_UNTIL = _NOW + timedelta(days=1)


# --- Fixture builders (mirrors test_relearn.py) ----------------------------

def _user_prompt(text: str, cwd: str | None = None) -> dict:
    record = {"type": "user", "message": {"role": "user", "content": text}}
    if cwd:
        record["cwd"] = cwd
    return record


def _assistant(text: str | None, tools: list[dict] | None = None, cwd: str | None = None) -> dict:
    content: list[dict] = []
    if text is not None:
        content.append({"type": "text", "text": text})
    for t in tools or []:
        content.append({"type": "tool_use", "id": t["id"], "name": t["name"], "input": t.get("input", {})})
    record = {
        "type": "assistant",
        "message": {"role": "assistant", "model": "claude-opus-4-8", "content": content},
    }
    if cwd:
        record["cwd"] = cwd
    return record


def _write_transcript(root: Path, project: str, session_id: str, records: list[dict]) -> Path:
    project_dir = root / project
    project_dir.mkdir(parents=True, exist_ok=True)
    path = project_dir / f"{session_id}.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")
    return path


def _write_mcp_json(project_dir: Path, servers: dict[str, dict]) -> None:
    project_dir.mkdir(parents=True, exist_ok=True)
    (project_dir / ".mcp.json").write_text(json.dumps({"mcpServers": servers}), encoding="utf-8")


def _plain_session(root: Path, project: str, session_id: str, cwd: str) -> None:
    """A session with no MCP activity at all — a server configured for its
    project is present but never invoked."""
    _write_transcript(root, project, session_id, [
        _user_prompt("say hi", cwd=cwd),
        _assistant("Hello!", cwd=cwd),
    ])


def _invoking_session(root: Path, project: str, session_id: str, cwd: str, tool_name: str) -> None:
    _write_transcript(root, project, session_id, [
        _user_prompt("use the tool", cwd=cwd),
        _assistant("Calling it.", tools=[{"id": "t1", "name": tool_name, "input": {}}], cwd=cwd),
    ])


def _deferred_session(root: Path, project: str, session_id: str, cwd: str, tool_name: str) -> None:
    """A session whose transcript shows the deferred-tools listing naming
    ``tool_name`` — the server's schema was NOT fully loaded this session."""
    reminder = (
        "<system-reminder>\n"
        "The following deferred tools are now available via ToolSearch. "
        "Their schemas are NOT loaded — calling them directly will fail "
        "with InputValidationError. Use ToolSearch to load their schema "
        "before calling them:\n"
        f"{tool_name}\n"
        "</system-reminder>"
    )
    _write_transcript(root, project, session_id, [
        _user_prompt(reminder, cwd=cwd),
        _assistant("Understood.", cwd=cwd),
    ])


# --- enumerate_configured_servers -------------------------------------------

def test_enumerate_project_scoped_server(tmp_path):
    project_dir = tmp_path / "repo-a"
    _write_mcp_json(project_dir, {"apollo": {"command": "apollo-mcp"}})

    servers = enumerate_configured_servers({str(project_dir)})

    assert "apollo" in servers
    assert servers["apollo"].scope == "project"
    assert str(project_dir) in servers["apollo"].cwds


def test_enumerate_global_scoped_server(tmp_path, monkeypatch):
    fake_home = tmp_path / "fake-home"
    fake_home.mkdir()
    (fake_home / ".claude.json").write_text(
        json.dumps({"mcpServers": {"exa": {"command": "exa-mcp"}}}), encoding="utf-8",
    )
    monkeypatch.setenv("HOME", str(fake_home))

    servers = enumerate_configured_servers(set())

    assert "exa" in servers
    assert servers["exa"].scope == "user"


def test_global_scope_wins_over_same_named_project_entry(tmp_path, monkeypatch):
    fake_home = tmp_path / "fake-home"
    fake_home.mkdir()
    (fake_home / ".claude.json").write_text(
        json.dumps({"mcpServers": {"apollo": {}}}), encoding="utf-8",
    )
    monkeypatch.setenv("HOME", str(fake_home))
    project_dir = tmp_path / "repo-a"
    _write_mcp_json(project_dir, {"apollo": {}})

    servers = enumerate_configured_servers({str(project_dir)})

    assert servers["apollo"].scope == "user"


def test_enumerate_no_config_returns_empty(tmp_path):
    assert enumerate_configured_servers({str(tmp_path / "nope")}) == {}


# --- C1: dead-weight detection ----------------------------------------------

def test_no_configured_servers_is_a_no_op(tmp_path):
    project_dir = tmp_path / "root" / "repo-a"
    for i in range(MIN_SESSIONS_DEADWEIGHT):
        _plain_session(tmp_path / "root", "-repo-a", f"s{i}", str(project_dir))

    finding = compute_deadweight_finding(_SINCE, _UNTIL, projects_root=tmp_path / "root")

    assert finding.configured_servers == 0
    assert finding.dead_servers == []
    assert finding.estimated_recoverable_tokens is None


def test_detects_dead_server_at_threshold(tmp_path):
    root = tmp_path / "root"
    project_dir = root / "-repo-a"
    _write_mcp_json(project_dir, {"apollo": {}})
    for i in range(MIN_SESSIONS_DEADWEIGHT):
        _plain_session(root, "-repo-a", f"s{i}", str(project_dir))

    finding = compute_deadweight_finding(_SINCE, _UNTIL, projects_root=root)

    assert finding.configured_servers == 1
    assert len(finding.dead_servers) == 1
    dead = finding.dead_servers[0]
    assert dead.name == "apollo"
    assert dead.sessions_present == MIN_SESSIONS_DEADWEIGHT
    assert dead.invocations == 0
    assert dead.estimated_tax_tokens_per_session == FULL_SCHEMA_TAX_TOKENS
    assert finding.estimated_recoverable_tokens == dead.estimated_tax_tokens_90d
    assert finding.estimated_recoverable_tokens > 0


def test_below_threshold_is_not_flagged_dead(tmp_path):
    root = tmp_path / "root"
    project_dir = root / "-repo-a"
    _write_mcp_json(project_dir, {"apollo": {}})
    for i in range(MIN_SESSIONS_DEADWEIGHT - 1):
        _plain_session(root, "-repo-a", f"s{i}", str(project_dir))

    finding = compute_deadweight_finding(_SINCE, _UNTIL, projects_root=root)

    assert finding.configured_servers == 1
    assert finding.dead_servers == []
    assert finding.estimated_recoverable_tokens is None
    assert finding.notes  # the "no server cleared the bar" note fires


def test_invoked_server_is_never_flagged_dead(tmp_path):
    root = tmp_path / "root"
    project_dir = root / "-repo-a"
    _write_mcp_json(project_dir, {"apollo": {}})
    for i in range(MIN_SESSIONS_DEADWEIGHT - 1):
        _plain_session(root, "-repo-a", f"s{i}", str(project_dir))
    _invoking_session(
        root, "-repo-a", "s-call", str(project_dir), "mcp__apollo__apollo_contacts_search",
    )

    finding = compute_deadweight_finding(_SINCE, _UNTIL, projects_root=root)

    row = next(s for s in finding.servers if s.name == "apollo")
    assert row.invocations == 1
    assert row.dead is False
    assert finding.dead_servers == []


# --- Deferred-tools suppression ---------------------------------------------

def test_deferred_listing_suppresses_full_tax_claim(tmp_path):
    root = tmp_path / "root"
    project_dir = root / "-repo-a"
    _write_mcp_json(project_dir, {"apollo": {}})
    for i in range(MIN_SESSIONS_DEADWEIGHT):
        _deferred_session(
            root, "-repo-a", f"s{i}", str(project_dir), "mcp__apollo__apollo_contacts_search",
        )

    finding = compute_deadweight_finding(_SINCE, _UNTIL, projects_root=root)

    dead = finding.dead_servers[0]
    assert dead.deferred_sessions == MIN_SESSIONS_DEADWEIGHT
    # Every session was deferred -> the blended tax must equal the deferred
    # constant, never the full-schema constant.
    assert dead.estimated_tax_tokens_per_session == DEFERRED_SCHEMA_TAX_TOKENS
    # Never claims the full-schema tax for a fully-deferred server.
    assert str(FULL_SCHEMA_TAX_TOKENS) not in dead.tax_construction


def test_partial_deferral_blends_the_two_constants(tmp_path):
    root = tmp_path / "root"
    project_dir = root / "-repo-a"
    _write_mcp_json(project_dir, {"apollo": {}})
    for i in range(5):
        _plain_session(root, "-repo-a", f"s-full-{i}", str(project_dir))
    for i in range(5):
        _deferred_session(
            root, "-repo-a", f"s-defer-{i}", str(project_dir), "mcp__apollo__apollo_contacts_search",
        )

    finding = compute_deadweight_finding(_SINCE, _UNTIL, projects_root=root)

    dead = finding.dead_servers[0]
    assert dead.sessions_present == 10
    assert dead.deferred_sessions == 5
    expected = round((5 * FULL_SCHEMA_TAX_TOKENS + 5 * DEFERRED_SCHEMA_TAX_TOKENS) / 10)
    assert dead.estimated_tax_tokens_per_session == expected
    assert dead.estimated_tax_tokens_per_session < FULL_SCHEMA_TAX_TOKENS


# --- C2: context tax table --------------------------------------------------

def test_tax_table_includes_claude_md_bucket(tmp_path):
    root = tmp_path / "root"
    project_dir = root / "-repo-a"
    reminder = (
        "<system-reminder>\n"
        "Contents of /Users/dev/CLAUDE.md (project instructions):\n"
        + ("word " * 500) +
        "\n</system-reminder>"
    )
    _write_transcript(root, "-repo-a", "s0", [
        _user_prompt(reminder, cwd=str(project_dir)),
        _assistant("ok", cwd=str(project_dir)),
    ])
    _write_mcp_json(project_dir, {"apollo": {}})

    finding = compute_deadweight_finding(_SINCE, _UNTIL, projects_root=root)

    sources = {row.source for row in finding.tax_table}
    assert "CLAUDE.md" in sources
    claude_row = next(r for r in finding.tax_table if r.source == "CLAUDE.md")
    assert claude_row.avg_tokens_per_session > 0
    assert claude_row.tag == "estimated"


def test_tax_table_includes_mcp_schema_rows_for_every_configured_server(tmp_path):
    root = tmp_path / "root"
    project_dir = root / "-repo-a"
    _write_mcp_json(project_dir, {"apollo": {}})
    _plain_session(root, "-repo-a", "s0", str(project_dir))

    finding = compute_deadweight_finding(_SINCE, _UNTIL, projects_root=root)

    assert any(row.source == "MCP schema: apollo" for row in finding.tax_table)


# --- Dedup rule --------------------------------------------------------------

def test_dead_server_tax_not_double_counted_between_table_and_total(tmp_path):
    root = tmp_path / "root"
    project_dir = root / "-repo-a"
    _write_mcp_json(project_dir, {"apollo": {}})
    for i in range(MIN_SESSIONS_DEADWEIGHT):
        _plain_session(root, "-repo-a", f"s{i}", str(project_dir))

    finding = compute_deadweight_finding(_SINCE, _UNTIL, projects_root=root)

    dead = finding.dead_servers[0]
    mcp_row = next(r for r in finding.tax_table if r.source == "MCP schema: apollo")
    # The tax table's own MCP row and the recoverable total both derive from
    # the SAME per-server figure, but the total must equal exactly the dead
    # servers' sum -- never (tax table total) + (recoverable total).
    assert finding.estimated_recoverable_tokens == dead.estimated_tax_tokens_90d
    assert mcp_row.total_tokens_window == dead.estimated_tax_tokens_per_session * dead.sessions_present


# --- Honesty / string-hygiene guards ----------------------------------------

def test_no_em_dash_or_quota_in_user_facing_strings(tmp_path):
    root = tmp_path / "root"
    project_dir = root / "-repo-a"
    _write_mcp_json(project_dir, {"apollo": {}})
    for i in range(MIN_SESSIONS_DEADWEIGHT):
        _deferred_session(
            root, "-repo-a", f"s{i}", str(project_dir), "mcp__apollo__apollo_contacts_search",
        )

    finding = compute_deadweight_finding(_SINCE, _UNTIL, projects_root=root)

    strings = [finding.caveat, finding.estimate_basis, *finding.notes]
    for server in finding.servers:
        # `fix` embeds the config's on-disk source path, which under pytest is
        # the test's own tmp_path (and can coincidentally contain "quota" as a
        # substring of the test name) -- redact it so the check is over the
        # actual card template wording, not an incidental tmp-dir name.
        strings += [server.fix.replace(server.source, "<source>"), server.tax_construction]
    for row in finding.tax_table:
        strings += [row.construction]
    for s in strings:
        assert "—" not in s, f"em dash found in: {s!r}"
        assert "quota" not in s.lower(), f"'quota' found in: {s!r}"


# --- Registration ------------------------------------------------------------

def test_deadweight_is_registered_in_runner_order():
    from tokenjam.core.optimize.runner import ANALYZER_ORDER
    from tokenjam.core.optimize.registry import ANALYZER_REGISTRY

    assert "deadweight" in ANALYZER_ORDER
    assert "deadweight" in ANALYZER_REGISTRY
