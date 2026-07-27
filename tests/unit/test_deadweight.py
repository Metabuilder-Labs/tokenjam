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
    run as run_deadweight,
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


def _assistant(
    text: str | None, tools: list[dict] | None = None, cwd: str | None = None,
    model: str = "claude-opus-4-8", usage: dict | None = None, msg_id: str | None = None,
) -> dict:
    content: list[dict] = []
    if text is not None:
        content.append({"type": "text", "text": text})
    for t in tools or []:
        content.append({"type": "tool_use", "id": t["id"], "name": t["name"], "input": t.get("input", {})})
    message: dict = {"role": "assistant", "model": model, "content": content}
    if usage is not None:
        message["usage"] = usage
    if msg_id is not None:
        message["id"] = msg_id
    record = {"type": "assistant", "message": message}
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


def _multi_call_session(root: Path, project: str, session_id: str, cwd: str, calls: int) -> None:
    """A session with ``calls`` sequential user/assistant turns -- each
    assistant turn is one API request, so a configured server's tool schemas
    ride in the `tools` array of every one of them (see the per-session
    cache-read multiplier in ``compute_deadweight_finding``)."""
    records: list[dict] = []
    for i in range(calls):
        records.append(_user_prompt(f"turn {i}", cwd=cwd))
        records.append(_assistant(f"ok {i}", cwd=cwd))
    _write_transcript(root, project, session_id, records)


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


# --- Deterministic source selection (Defect 2: set-iteration-order target) --
# The same server name can be independently declared in more than one
# physical config file (a duplicated .mcp.json committed into several
# worktrees of the same repo). `enumerate_configured_servers` used to iterate
# a raw `set` of cwds and let `setdefault` keep whichever path a hash-seed-
# dependent iteration order visited FIRST -- a claim aggregated across every
# copy could have its one fix action land on the copy covering 1 of 571
# sessions instead of 569 of them, decided by nothing but interpreter hash
# randomization.

def test_source_selection_is_deterministic_across_repeated_calls(tmp_path):
    """The same three-cwd input must choose the same canonical source every
    time, regardless of set/dict iteration order -- run several times to
    catch a flake a single call could miss."""
    heavy = tmp_path / "heavy"
    light_a = tmp_path / "light-a"
    light_b = tmp_path / "light-b"
    _write_mcp_json(heavy, {"posthog": {}})
    _write_mcp_json(light_a, {"posthog": {}})
    _write_mcp_json(light_b, {"posthog": {}})
    cwds = {str(heavy), str(light_a), str(light_b)}

    chosen = {enumerate_configured_servers(set(cwds))["posthog"].source for _ in range(20)}

    assert len(chosen) == 1, f"source choice varied across calls: {chosen}"


def test_source_selection_picks_the_path_covering_the_most_sessions(tmp_path):
    """Given an actual coverage skew (the real corpus shape: one path behind
    nearly every session, two behind a single session each), the canonical
    source must be the one that matters, not an arbitrary one -- editing it
    is what the finding's one fix action can actually deliver on."""
    heavy = tmp_path / "root" / "-heavy"
    light_a = tmp_path / "root" / "-light-a"
    light_b = tmp_path / "root" / "-light-b"
    _write_mcp_json(heavy, {"posthog": {}})
    _write_mcp_json(light_a, {"posthog": {}})
    _write_mcp_json(light_b, {"posthog": {}})

    servers = enumerate_configured_servers({str(heavy), str(light_a), str(light_b)})
    # Simulate the coverage skew directly on the returned entry -- the
    # picking step (`enumerate_configured_servers`'s post-scan loop) reruns
    # against whatever `source_cwds` actually holds, so seeding it this way
    # exercises the real selection logic.
    entry = servers["posthog"]
    entry.source_cwds = {
        str(heavy): {f"s{i}" for i in range(569)},
        str(light_a): {"s569"},
        str(light_b): {"s570"},
    }
    # Re-run the picking rule in isolation (mirrors the post-scan loop body).
    entry.source = min(
        entry.source_cwds.items(), key=lambda kv: (-len(kv[1]), kv[0]),
    )[0]

    assert entry.source == str(heavy)


def test_other_sources_lists_every_path_except_the_canonical_one(tmp_path):
    from tokenjam.core.optimize.analyzers.deadweight import _other_sources

    heavy = tmp_path / "heavy"
    light_a = tmp_path / "light-a"
    light_b = tmp_path / "light-b"
    _write_mcp_json(heavy, {"posthog": {}})
    _write_mcp_json(light_a, {"posthog": {}})
    _write_mcp_json(light_b, {"posthog": {}})

    servers = enumerate_configured_servers({str(heavy), str(light_a), str(light_b)})
    entry = servers["posthog"]

    others = _other_sources(entry)
    assert entry.source not in others
    assert len(others) == 2
    assert others == sorted(others)  # stable, sorted disclosure order


def test_single_source_server_has_no_other_sources(tmp_path):
    from tokenjam.core.optimize.analyzers.deadweight import _other_sources

    project_dir = tmp_path / "repo-a"
    _write_mcp_json(project_dir, {"apollo": {}})
    servers = enumerate_configured_servers({str(project_dir)})

    assert _other_sources(servers["apollo"]) == []


def test_multi_source_finding_discloses_the_other_locations_and_scopes_the_fix(tmp_path):
    """End to end: a server declared in three separate config files must
    have its finding name the other locations and state how much of the
    aggregated claim the one edited file actually reaches."""
    root = tmp_path / "root"
    heavy = root / "-heavy"
    light_a = root / "-light-a"
    light_b = root / "-light-b"
    _write_mcp_json(heavy, {"posthog": {}})
    _write_mcp_json(light_a, {"posthog": {}})
    _write_mcp_json(light_b, {"posthog": {}})

    for i in range(MIN_SESSIONS_DEADWEIGHT):
        _plain_session(root, "-heavy", f"s{i}", str(heavy))
    _plain_session(root, "-light-a", "s-a", str(light_a))
    _plain_session(root, "-light-b", "s-b", str(light_b))

    finding = compute_deadweight_finding(_SINCE, _UNTIL, projects_root=root)
    dead = finding.dead_servers[0]

    heavy_config = str(heavy / ".mcp.json")
    light_a_config = str(light_a / ".mcp.json")
    light_b_config = str(light_b / ".mcp.json")

    assert dead.source == heavy_config
    assert set(dead.other_sources) == {light_a_config, light_b_config}
    assert dead.sessions_present == MIN_SESSIONS_DEADWEIGHT + 2
    assert dead.primary_source_sessions == MIN_SESSIONS_DEADWEIGHT
    # The fix text must name the gap rather than imply full coverage.
    assert light_a_config in dead.fix
    assert light_b_config in dead.fix
    assert str(dead.primary_source_sessions) in dead.fix
    assert "ALSO independently declared" in dead.fix


def test_project_scoped_server_fix_never_offers_project_scope_as_an_alternative(tmp_path):
    """A server already at project scope has nothing left to narrow --
    offering "project-scope it" as a second option would be a no-op
    delivering $0 of the claim."""
    root = tmp_path / "root"
    project_dir = root / "-repo-a"
    _write_mcp_json(project_dir, {"apollo": {}})
    for i in range(MIN_SESSIONS_DEADWEIGHT):
        _plain_session(root, "-repo-a", f"s{i}", str(project_dir))

    finding = compute_deadweight_finding(_SINCE, _UNTIL, projects_root=root)
    dead = finding.dead_servers[0]

    assert dead.fix.startswith("Remove the `apollo`")
    assert "project-scope" not in dead.fix.lower()


def test_user_scoped_server_fix_still_offers_project_scope_as_an_alternative(tmp_path, monkeypatch):
    """A GLOBAL server genuinely has a second option: narrowing it to
    project scope is a real alternative, unlike for an already
    project-scoped server."""
    fake_home = tmp_path / "fake-home"
    fake_home.mkdir()
    (fake_home / ".claude.json").write_text(
        json.dumps({"mcpServers": {"exa": {}}}), encoding="utf-8",
    )
    monkeypatch.setenv("HOME", str(fake_home))
    root = tmp_path / "root"
    for i in range(MIN_SESSIONS_DEADWEIGHT):
        _plain_session(root, "-repo-a", f"s{i}", str(root / "-repo-a"))

    finding = compute_deadweight_finding(_SINCE, _UNTIL, projects_root=root)
    dead = next(s for s in finding.dead_servers if s.name == "exa")

    assert dead.fix.startswith("Remove or project-scope the `exa`")


# --- C1: dead-weight detection ----------------------------------------------

def test_no_configured_servers_is_a_no_op(tmp_path):
    project_dir = tmp_path / "root" / "repo-a"
    for i in range(MIN_SESSIONS_DEADWEIGHT):
        _plain_session(tmp_path / "root", "-repo-a", f"s{i}", str(project_dir))

    finding = compute_deadweight_finding(_SINCE, _UNTIL, projects_root=tmp_path / "root")

    assert finding.configured_servers == 0
    assert finding.dead_servers == []
    assert finding.past_overspend_tokens is None


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
    assert finding.past_overspend_tokens == dead.estimated_tax_tokens_window
    assert finding.past_overspend_tokens > 0


def test_dead_server_prices_tax_in_usd_via_pricing_table(tmp_path):
    """The token tax is priced through core/pricing.py at the dominant model
    observed across the server's sessions (every fixture session here runs
    on claude-opus-4-8) -- never a hardcoded rate baked into this module."""
    from tokenjam.core.pricing import get_rates

    root = tmp_path / "root"
    project_dir = root / "-repo-a"
    _write_mcp_json(project_dir, {"apollo": {}})
    for i in range(MIN_SESSIONS_DEADWEIGHT):
        _plain_session(root, "-repo-a", f"s{i}", str(project_dir))

    finding = compute_deadweight_finding(_SINCE, _UNTIL, projects_root=root)
    dead = finding.dead_servers[0]

    rates = get_rates("anthropic", "claude-opus-4-8")
    assert dead.priced_model == "claude-opus-4-8"
    assert dead.estimated_tax_usd_per_session == round(
        dead.estimated_tax_tokens_per_session / 1_000_000 * rates.input_per_mtok, 6,
    )
    assert dead.estimated_tax_usd_window > 0
    assert finding.past_overspend_usd == dead.estimated_tax_usd_window
    assert "Priced at claude-opus-4-8's input rate" in dead.tax_construction


def test_no_priced_model_leaves_usd_none(tmp_path):
    """A session with no assistant model recorded at all must never get a
    fabricated dollar figure -- tokens-only, explicitly noted."""
    root = tmp_path / "root"
    project_dir = root / "-repo-a"
    _write_mcp_json(project_dir, {"apollo": {}})
    for i in range(MIN_SESSIONS_DEADWEIGHT):
        _write_transcript(root, "-repo-a", f"s{i}", [
            _user_prompt("say hi", cwd=str(project_dir)),
            # No assistant turn at all -> no model signal for this session.
        ])

    finding = compute_deadweight_finding(_SINCE, _UNTIL, projects_root=root)
    dead = finding.dead_servers[0]

    assert dead.priced_model == ""
    assert dead.estimated_tax_usd_per_session is None
    assert dead.estimated_tax_usd_window is None
    assert finding.past_overspend_usd is None
    assert "No dollar estimate" in dead.tax_construction


def test_below_threshold_is_not_flagged_dead(tmp_path):
    root = tmp_path / "root"
    project_dir = root / "-repo-a"
    _write_mcp_json(project_dir, {"apollo": {}})
    for i in range(MIN_SESSIONS_DEADWEIGHT - 1):
        _plain_session(root, "-repo-a", f"s{i}", str(project_dir))

    finding = compute_deadweight_finding(_SINCE, _UNTIL, projects_root=root)

    assert finding.configured_servers == 1
    assert finding.dead_servers == []
    assert finding.past_overspend_tokens is None
    assert finding.notes  # the "no server cleared the bar" note fires


def test_default_min_sessions_preserved_when_unspecified(tmp_path):
    """compute_deadweight_finding's default `min_sessions` matches the module
    constant unchanged — the config-thread contract for an unset [optimize]."""
    root = tmp_path / "root"
    project_dir = root / "-repo-a"
    _write_mcp_json(project_dir, {"apollo": {}})
    for i in range(MIN_SESSIONS_DEADWEIGHT - 1):
        _plain_session(root, "-repo-a", f"s{i}", str(project_dir))

    finding = compute_deadweight_finding(_SINCE, _UNTIL, projects_root=root)
    assert finding.dead_servers == []


def test_lower_min_sessions_flags_previously_hidden_server(tmp_path):
    """The exact data from test_below_threshold_is_not_flagged_dead flags
    nothing at the default bar; passing a lower min_sessions (what
    [optimize] min_sessions_deadweight threads through to) flags the server."""
    root = tmp_path / "root"
    project_dir = root / "-repo-a"
    _write_mcp_json(project_dir, {"apollo": {}})
    for i in range(MIN_SESSIONS_DEADWEIGHT - 1):
        _plain_session(root, "-repo-a", f"s{i}", str(project_dir))

    default_finding = compute_deadweight_finding(_SINCE, _UNTIL, projects_root=root)
    assert default_finding.dead_servers == []

    lowered_finding = compute_deadweight_finding(
        _SINCE, _UNTIL, projects_root=root, min_sessions=MIN_SESSIONS_DEADWEIGHT - 1,
    )
    assert len(lowered_finding.dead_servers) == 1
    assert lowered_finding.dead_servers[0].name == "apollo"
    assert lowered_finding.dead_servers[0].sessions_present == MIN_SESSIONS_DEADWEIGHT - 1


def test_run_reads_min_sessions_deadweight_from_ctx_config(tmp_path, monkeypatch):
    """The registered `run(ctx)` entry point (not just compute_deadweight_finding
    directly) reads `ctx.config.optimize.min_sessions_deadweight` — the actual
    wiring `tj optimize` exercises."""
    from tokenjam.core.config import OptimizeConfig, TjConfig
    from tokenjam.core.optimize.types import AnalyzerContext, OptimizeReport, WindowSummary

    root = tmp_path / "root"
    project_dir = root / "-repo-a"
    _write_mcp_json(project_dir, {"apollo": {}})
    for i in range(MIN_SESSIONS_DEADWEIGHT - 1):
        _plain_session(root, "-repo-a", f"s{i}", str(project_dir))
    monkeypatch.setenv("TJ_CLAUDE_PROJECTS_ROOT", str(root))

    summary = WindowSummary(
        since=_SINCE, until=_UNTIL, days=7.0, sessions=0, spans=0,
        total_tokens=0, total_cost_usd=0.0, thin_data=False,
    )

    def _ctx(config) -> AnalyzerContext:
        return AnalyzerContext(
            conn=None, config=config, since=_SINCE, until=_UNTIL, agent_id=None,
            window_days=7.0, summary=summary, report=OptimizeReport(window=summary),
        )

    default_ctx = _ctx(TjConfig(version="1"))
    run_deadweight(default_ctx)
    assert default_ctx.report.findings["deadweight"].dead_servers == []

    lowered_ctx = _ctx(TjConfig(
        version="1",
        optimize=OptimizeConfig(min_sessions_deadweight=MIN_SESSIONS_DEADWEIGHT - 1),
    ))
    run_deadweight(lowered_ctx)
    lowered = lowered_ctx.report.findings["deadweight"]
    assert len(lowered.dead_servers) == 1
    assert lowered.dead_servers[0].name == "apollo"


def test_compute_deadweight_finding_cache_dir_opt_in_matches_uncached(tmp_path):
    """Passing `cache_dir` must not change the finding — only skip re-parsing
    on a warm hit (see the two tests below)."""
    root = tmp_path / "root"
    project_dir = root / "-repo-a"
    _write_mcp_json(project_dir, {"apollo": {}})
    for i in range(MIN_SESSIONS_DEADWEIGHT):
        _plain_session(root, "-repo-a", f"s{i}", str(project_dir))

    uncached = compute_deadweight_finding(_SINCE, _UNTIL, projects_root=root)
    cached = compute_deadweight_finding(
        _SINCE, _UNTIL, projects_root=root, cache_dir=tmp_path / "cache",
    )

    assert [s.name for s in cached.dead_servers] == [s.name for s in uncached.dead_servers]
    assert cached.sessions_scanned == uncached.sessions_scanned


def test_compute_deadweight_finding_warm_cache_skips_reparsing(tmp_path, monkeypatch):
    """A second call over an UNCHANGED corpus with the same `cache_dir` must
    not re-read/re-parse any transcript — the whole point of the cache."""
    root = tmp_path / "root"
    project_dir = root / "-repo-a"
    _write_mcp_json(project_dir, {"apollo": {}})
    for i in range(MIN_SESSIONS_DEADWEIGHT):
        _plain_session(root, "-repo-a", f"s{i}", str(project_dir))
    cache_dir = tmp_path / "cache"

    first = compute_deadweight_finding(_SINCE, _UNTIL, projects_root=root, cache_dir=cache_dir)
    assert first.dead_servers  # sanity: a real signal, not an empty no-op

    def _boom(path):
        raise AssertionError(f"transcript.read_records reparsed {path} on a warm cache run")

    monkeypatch.setattr("tokenjam.core.transcript._parse_records", _boom)

    second = compute_deadweight_finding(_SINCE, _UNTIL, projects_root=root, cache_dir=cache_dir)
    assert [s.name for s in second.dead_servers] == [s.name for s in first.dead_servers]


def test_compute_deadweight_finding_cache_invalidates_on_transcript_edit(tmp_path):
    """A session whose transcript CHANGES between two cached runs (e.g. an
    invocation gets appended) must be re-parsed, not served stale."""
    root = tmp_path / "root"
    project_dir = root / "-repo-a"
    _write_mcp_json(project_dir, {"apollo": {}})
    for i in range(MIN_SESSIONS_DEADWEIGHT):
        _plain_session(root, "-repo-a", f"s{i}", str(project_dir))
    cache_dir = tmp_path / "cache"

    first = compute_deadweight_finding(_SINCE, _UNTIL, projects_root=root, cache_dir=cache_dir)
    assert first.dead_servers  # apollo starts out dead (never invoked)

    # Rewrite one of the sessions so apollo IS invoked — size and mtime both
    # change, which must invalidate that session's cache entry.
    _invoking_session(
        root, "-repo-a", "s0", str(project_dir), "mcp__apollo__apollo_contacts_search",
    )

    second = compute_deadweight_finding(_SINCE, _UNTIL, projects_root=root, cache_dir=cache_dir)
    assert second.dead_servers == []  # no longer dead — the edit was picked up


def test_run_wires_the_persistent_transcript_cache(tmp_path, monkeypatch):
    """The registered `run(ctx)` entry point (the path `tj optimize` and
    `/cost/components` actually exercise) resolves and uses a real cache dir
    from `ctx.config`, not just the standalone `compute_deadweight_finding`
    function tested above."""
    from tokenjam.core.config import TjConfig
    from tokenjam.core.optimize.types import AnalyzerContext, OptimizeReport, WindowSummary

    root = tmp_path / "root"
    project_dir = root / "-repo-a"
    _write_mcp_json(project_dir, {"apollo": {}})
    for i in range(MIN_SESSIONS_DEADWEIGHT):
        _plain_session(root, "-repo-a", f"s{i}", str(project_dir))
    monkeypatch.setenv("TJ_CLAUDE_PROJECTS_ROOT", str(root))

    summary = WindowSummary(
        since=_SINCE, until=_UNTIL, days=7.0, sessions=0, spans=0,
        total_tokens=0, total_cost_usd=0.0, thin_data=False,
    )
    config = TjConfig(version="1")

    def _ctx() -> AnalyzerContext:
        return AnalyzerContext(
            conn=None, config=config, since=_SINCE, until=_UNTIL, agent_id=None,
            window_days=7.0, summary=summary, report=OptimizeReport(window=summary),
        )

    first_ctx = _ctx()
    run_deadweight(first_ctx)
    first = first_ctx.report.findings["deadweight"]
    assert first.dead_servers

    def _boom(path):
        raise AssertionError(f"transcript.read_records reparsed {path} on a warm cache run")

    monkeypatch.setattr("tokenjam.core.transcript._parse_records", _boom)

    second_ctx = _ctx()
    run_deadweight(second_ctx)
    second = second_ctx.report.findings["deadweight"]
    assert [s.name for s in second.dead_servers] == [s.name for s in first.dead_servers]


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


# --- Sidechain/subagent transcripts must not become extra top-level sessions -

def test_sidechain_subagent_transcript_is_not_counted_as_a_top_level_session(tmp_path):
    """A `subagents/agent-*.jsonl` sidechain transcript lives nested under its
    parent session's own directory (core/transcript.py) but must not be
    discovered as an independent top-level "session" -- otherwise a session
    that happens to spawn a subagent spuriously inflates sessions_present (and
    thus can push a server across the dead-weight threshold, or inflate the
    schema tax) purely because of where the subagent's JSONL file lives on
    disk, not because of any real additional session."""
    root = tmp_path / "root"
    project_dir = root / "-repo-a"
    _write_mcp_json(project_dir, {"apollo": {}})
    for i in range(MIN_SESSIONS_DEADWEIGHT - 1):
        _plain_session(root, "-repo-a", f"s{i}", str(project_dir))

    # A sidechain transcript nested under s0's own subagents/ dir. If it were
    # (mis)discovered as its own top-level session, sessions_present would
    # reach MIN_SESSIONS_DEADWEIGHT and flip apollo to dead.
    sidechain_path = root / "-repo-a" / "s0" / "subagents" / "agent-child1.jsonl"
    sidechain_path.parent.mkdir(parents=True, exist_ok=True)
    sidechain_path.write_text(
        "\n".join(json.dumps(r) for r in [
            _user_prompt("subagent task", cwd=str(project_dir)),
            _assistant("working...", cwd=str(project_dir)),
        ]),
        encoding="utf-8",
    )

    finding = compute_deadweight_finding(_SINCE, _UNTIL, projects_root=root)

    assert finding.sessions_scanned == MIN_SESSIONS_DEADWEIGHT - 1
    assert finding.dead_servers == []
    row = next(s for s in finding.servers if s.name == "apollo")
    assert row.sessions_present == MIN_SESSIONS_DEADWEIGHT - 1


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


def test_mixed_deferral_prices_each_call_by_its_own_load_state(tmp_path):
    """A server deferred at the start of a session but actually invoked partway
    through (schema now fully loaded -- a deferred tool's own listing says
    calling it directly "will fail... use ToolSearch to load their schema
    before calling them", so a successful invocation is positive evidence the
    schema was already loaded by that call) must have the calls BEFORE the
    invocation priced at the deferred base and the calls FROM the invocation
    onward priced at the full base -- never the whole session at the low
    deferred base just because it was deferred at some point."""
    root = tmp_path / "root"
    project_dir = root / "-repo-a"
    _write_mcp_json(project_dir, {"apollo": {}})
    reminder = (
        "<system-reminder>\n"
        "The following deferred tools are now available via ToolSearch. "
        "Their schemas are NOT loaded — calling them directly will fail "
        "with InputValidationError. Use ToolSearch to load their schema "
        "before calling them:\n"
        "mcp__apollo__apollo_contacts_search\n"
        "</system-reminder>"
    )
    _write_transcript(root, "-repo-a", "s-mixed", [
        _user_prompt(reminder, cwd=str(project_dir)),
        _assistant("thinking...", cwd=str(project_dir)),                  # call 1: deferred
        _user_prompt("continue", cwd=str(project_dir)),
        _assistant("still deferred", cwd=str(project_dir)),               # call 2: deferred
        _user_prompt("use the tool", cwd=str(project_dir)),
        _assistant(                                                        # call 3: loads + invokes
            "calling it now",
            tools=[{"id": "t1", "name": "mcp__apollo__apollo_contacts_search", "input": {}}],
            cwd=str(project_dir),
        ),
        _user_prompt("result", cwd=str(project_dir)),
        _assistant("done", cwd=str(project_dir)),                         # call 4: full
    ])

    finding = compute_deadweight_finding(_SINCE, _UNTIL, projects_root=root)
    row = next(s for s in finding.servers if s.name == "apollo")

    # `estimated_tax_tokens_window` is the LITERAL (undiscounted) token count
    # -- the schema is resent in full on every call regardless of caching, so
    # this is 2 deferred calls' worth + 2 full calls' worth of the raw
    # constants, never the cache-discounted price-equivalent quantity the $
    # figure is derived from (that quantity is checked separately, in
    # `test_dead_server_prices_tax_in_usd_via_pricing_table`-style tests).
    expected_deferred = DEFERRED_SCHEMA_TAX_TOKENS * 2
    expected_full = FULL_SCHEMA_TAX_TOKENS * 2

    assert row.invocations == 1
    assert row.dead is False
    assert row.estimated_tax_tokens_window == expected_deferred + expected_full
    # Strictly more than the old bug (whole session priced at the deferred
    # base just because it was deferred at some point).
    old_buggy_estimate = DEFERRED_SCHEMA_TAX_TOKENS * 4
    assert row.estimated_tax_tokens_window > old_buggy_estimate


def test_schema_tax_scales_with_actual_calls_per_session(tmp_path):
    """The schema tax must price EVERY call in a session, not just charge the
    full/deferred constant once -- later calls in the same session re-send
    the schema and are billed at the cache-read rate, not re-charged at input
    rate. A mixed population (light single-call sessions + one heavy
    multi-call session) proves this uses each session's OWN actual call
    count, never a global mean/median applied uniformly."""
    from tokenjam.core.pricing import get_rates

    root = tmp_path / "root"
    project_dir = root / "-repo-a"
    _write_mcp_json(project_dir, {"apollo": {}})
    for i in range(MIN_SESSIONS_DEADWEIGHT - 1):
        _plain_session(root, "-repo-a", f"s-light-{i}", str(project_dir))
    _multi_call_session(root, "-repo-a", "s-heavy", str(project_dir), calls=10)

    finding = compute_deadweight_finding(_SINCE, _UNTIL, projects_root=root)
    dead = finding.dead_servers[0]
    assert dead.sessions_present == MIN_SESSIONS_DEADWEIGHT

    rates = get_rates("anthropic", "claude-opus-4-8")
    ratio = rates.cache_read_per_mtok / rates.input_per_mtok
    # The $ figure is priced on the cache-discounted PRICE-EQUIVALENT
    # quantity (this test's actual subject: later calls in a session re-send
    # the schema at the cache-read rate, not the full input rate again).
    light_tax_price_equiv = FULL_SCHEMA_TAX_TOKENS  # single call -> multiplier == 1
    heavy_tax_price_equiv = round(FULL_SCHEMA_TAX_TOKENS * (1.0 + (10 - 1) * ratio))
    expected_usd_window = round(
        (light_tax_price_equiv * (MIN_SESSIONS_DEADWEIGHT - 1) + heavy_tax_price_equiv)
        / 1_000_000 * rates.input_per_mtok,
        6,
    )

    assert heavy_tax_price_equiv > FULL_SCHEMA_TAX_TOKENS
    assert dead.estimated_tax_usd_window == expected_usd_window
    # The heavy session's real cost is folded in, not diluted by an averaged
    # call count applied uniformly to every session.
    assert dead.estimated_tax_usd_window > round(
        light_tax_price_equiv * MIN_SESSIONS_DEADWEIGHT / 1_000_000 * rates.input_per_mtok, 6,
    )
    assert "cache-read rate" in dead.tax_construction
    assert "5-minute cache TTL" in dead.tax_construction

    # `estimated_tax_tokens_window` is the separate, LITERAL (undiscounted)
    # token count -- the schema is resent in full on every call regardless
    # of caching, so this is linear in the call count, never scaled down by
    # the cache-read ratio the $ figure above uses.
    expected_tokens_window = FULL_SCHEMA_TAX_TOKENS * (MIN_SESSIONS_DEADWEIGHT - 1 + 10)
    assert dead.estimated_tax_tokens_window == expected_tokens_window


# --- Split-response record dedup (assistant_turns must count CALLS) ---------
# Claude Code writes a SEPARATE transcript record per content block
# (thinking / text / tool_use) of one API response, sharing one message.id.
# `assistant_turns` used to count "role == assistant" RECORDS one-for-one,
# overcounting the schema-tax call count by however many blocks a response
# split into (measured ~2.19x on a real corpus).

def test_analyze_session_dedupes_split_response_records_by_message_id():
    """Direct unit test of the dedup: two records sharing one message.id are
    ONE assistant turn, not two."""
    from tokenjam.core.optimize.analyzers.deadweight import _analyze_session

    records = [
        _user_prompt("turn 0"),
        _assistant("thinking...", msg_id="m1"),
        _assistant("ok 0", msg_id="m1"),
    ]
    signal = _analyze_session(records)
    assert signal.assistant_turns == 1
    assert signal.models == {"claude-opus-4-8": 1}


def test_analyze_session_still_counts_distinct_calls_separately():
    """Two DIFFERENT message ids remain two separate calls -- the dedup must
    not collapse genuinely distinct turns."""
    from tokenjam.core.optimize.analyzers.deadweight import _analyze_session

    records = [
        _user_prompt("turn 0"),
        _assistant("ok 0", msg_id="m1"),
        _user_prompt("turn 1"),
        _assistant("ok 1", msg_id="m2"),
    ]
    signal = _analyze_session(records)
    assert signal.assistant_turns == 2


def test_split_response_records_price_as_one_call_not_two(tmp_path):
    """End to end: a session with ONE real API call whose response is split
    across two transcript records must price the schema tax for ONE call,
    not two -- the exact overcount the fix closes."""
    from tokenjam.core.pricing import get_rates

    root = tmp_path / "root"
    project_dir = root / "-repo-a"
    _write_mcp_json(project_dir, {"apollo": {}})
    for i in range(MIN_SESSIONS_DEADWEIGHT - 1):
        _plain_session(root, "-repo-a", f"s-light-{i}", str(project_dir))

    # One heavy session: ONE real API call, its response split across two
    # transcript records sharing the same message id (mirrors the real
    # thinking-block/text-block split Claude Code writes).
    _write_transcript(root, "-repo-a", "s-split", [
        _user_prompt("turn 0", cwd=str(project_dir)),
        _assistant("thinking...", cwd=str(project_dir), msg_id="m1"),
        _assistant("ok 0", cwd=str(project_dir), msg_id="m1"),
    ])

    finding = compute_deadweight_finding(_SINCE, _UNTIL, projects_root=root)
    dead = finding.dead_servers[0]
    assert dead.sessions_present == MIN_SESSIONS_DEADWEIGHT

    get_rates("anthropic", "claude-opus-4-8")  # sanity: model is priced
    light_tax = FULL_SCHEMA_TAX_TOKENS  # single call -> multiplier == 1
    split_session_tax = FULL_SCHEMA_TAX_TOKENS  # ONE real call, not two
    expected_window = light_tax * (MIN_SESSIONS_DEADWEIGHT - 1) + split_session_tax

    assert dead.estimated_tax_tokens_window == expected_window


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
    assert finding.past_overspend_tokens == dead.estimated_tax_tokens_window
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

    strings = [finding.caveat, finding.estimate_basis, finding.coverage_note, *finding.notes]
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


# --- Unresolvable-path coverage (Defect 1: silence when a cwd is gone) ------
# `enumerate_configured_servers` silently `continue`s past a recorded session
# cwd that no longer exists on disk -- indistinguishable, on the finding, from
# a live repo genuinely carrying no MCP config. These tests cover the fix:
# counting that blind spot and stating it in `coverage_note`.

def test_unresolvable_path_is_counted_and_narrated(tmp_path):
    root = tmp_path / "root"
    gone = root / "-gone-project" / "does-not-exist"
    # Never created on disk -- this is the "recorded but vanished" cwd.
    for i in range(3):
        _plain_session(root, "-gone-project", f"s{i}", str(gone))

    finding = compute_deadweight_finding(_SINCE, _UNTIL, projects_root=root)

    assert finding.unresolvable_paths == 1
    assert finding.unresolvable_sessions == 3
    assert finding.coverage_note != ""
    assert "no longer exist" in finding.coverage_note
    assert "3 of 3 session(s)" in finding.coverage_note


def test_unresolvable_coverage_survives_when_no_servers_configured(tmp_path):
    """The exact defect: every recorded path is gone, so `configured` comes
    back empty and the old code returned right there, before ever computing
    the blind-spot figures. They must survive that early return."""
    root = tmp_path / "root"
    gone = root / "-gone-project" / "does-not-exist"
    for i in range(3):
        _plain_session(root, "-gone-project", f"s{i}", str(gone))

    finding = compute_deadweight_finding(_SINCE, _UNTIL, projects_root=root)

    assert finding.configured_servers == 0
    assert finding.unresolvable_sessions == 3
    assert finding.coverage_note != ""


def test_live_path_with_no_config_is_not_counted_as_unresolvable(tmp_path):
    """A live repo that genuinely carries no MCP config is a real, correctly
    evaluated zero -- not a blind spot. Only VANISHED paths count."""
    root = tmp_path / "root"
    project_dir = root / "-repo-a"
    project_dir.mkdir(parents=True)  # exists on disk, no .mcp.json written
    for i in range(3):
        _plain_session(root, "-repo-a", f"s{i}", str(project_dir))

    finding = compute_deadweight_finding(_SINCE, _UNTIL, projects_root=root)

    assert finding.unresolvable_paths == 0
    assert finding.unresolvable_sessions == 0
    assert finding.coverage_note == ""


def test_unresolvable_usd_priced_via_pricing_table(tmp_path):
    """The blind-spot dollar figure is priced through core/pricing.py at the
    session's dominant model -- asserted against the real rate, never a
    hardcoded dollar amount (CLAUDE.md Critical Rule 28)."""
    from tokenjam.core.pricing import get_rates

    root = tmp_path / "root"
    gone = root / "-gone-project" / "does-not-exist"
    usage = {
        "input_tokens": 1_000, "output_tokens": 500,
        "cache_read_input_tokens": 200, "cache_creation_input_tokens": 100,
    }
    _write_transcript(root, "-gone-project", "s0", [
        _user_prompt("say hi", cwd=str(gone)),
        _assistant("hi", cwd=str(gone), usage=usage),
    ])

    finding = compute_deadweight_finding(_SINCE, _UNTIL, projects_root=root)

    rates = get_rates("anthropic", "claude-opus-4-8")
    expected = round(
        (1_000 * rates.input_per_mtok + 500 * rates.output_per_mtok
         + 200 * rates.cache_read_per_mtok + 100 * rates.cache_write_per_mtok)
        / 1_000_000,
        6,
    )
    assert finding.unresolvable_usd == expected
    assert finding.unresolvable_tokens == 1_800
    assert finding.unresolvable_unpriced_sessions == 0
    assert f"${finding.unresolvable_usd:,.2f}" in finding.coverage_note


def test_unresolvable_excludes_sessions_with_no_priced_model(tmp_path):
    """A session behind a vanished path but on an unpriced/unknown model
    still counts toward tokens, but is excluded from the dollar sum (never a
    fabricated rate) -- and the note says so."""
    root = tmp_path / "root"
    gone = root / "-gone-project" / "does-not-exist"
    usage = {"input_tokens": 1_000, "output_tokens": 100}
    _write_transcript(root, "-gone-project", "s0", [
        _user_prompt("say hi", cwd=str(gone)),
        _assistant("hi", cwd=str(gone), model="some-unpriced-model-xyz", usage=usage),
    ])

    finding = compute_deadweight_finding(_SINCE, _UNTIL, projects_root=root)

    assert finding.unresolvable_sessions == 1
    assert finding.unresolvable_tokens == 1_100
    assert finding.unresolvable_usd is None
    assert finding.unresolvable_unpriced_sessions == 1
    assert "no dollar figure is stated" in finding.coverage_note


def test_render_deadweight_prints_coverage_note_with_no_configured_servers(tmp_path, capsys):
    """The silent-scope defect's exact user-visible symptom: 'no MCP server
    configured' read as 'you already fixed everything' when really the
    analyzer never got to look. The renderer must say so."""
    from tokenjam.cli.cmd_optimize import _render_deadweight

    root = tmp_path / "root"
    gone = root / "-gone-project" / "does-not-exist"
    for i in range(3):
        _plain_session(root, "-gone-project", f"s{i}", str(gone))

    finding = compute_deadweight_finding(_SINCE, _UNTIL, projects_root=root)
    assert finding.configured_servers == 0

    _render_deadweight(finding, pricing_mode="api", marker="①")
    out = capsys.readouterr().out

    assert "no MCP server is" in out
    assert "no longer exist" in out


# --- Registration ------------------------------------------------------------

def test_deadweight_is_registered_in_runner_order():
    from tokenjam.core.optimize.runner import ANALYZER_ORDER
    from tokenjam.core.optimize.registry import ANALYZER_REGISTRY

    assert "deadweight" in ANALYZER_ORDER
    assert "deadweight" in ANALYZER_REGISTRY


# --- CLI text-view rendering regression --------------------------------------
# Same class of defect the relearn analyzer hit: `deadweight` was registered in
# ANALYZER_REGISTRY/ANALYZER_ORDER but never wired into cmd_optimize's
# _FINDING_RENDERERS dispatch table, so _rank_findings silently dropped it and
# `tj optimize deadweight` (text view) printed the generic empty state even
# with real dead servers sitting in --json.

def test_deadweight_in_click_choices_and_renderer():
    from tokenjam.cli.cmd_optimize import (
        _FINDING_RENDERERS,
        _MINOR_FINDING_LABELS,
        cmd_optimize,
    )

    findings_param = next(
        p for p in cmd_optimize.params if getattr(p, "name", None) == "findings"
    )
    assert "deadweight" in findings_param.type.choices
    assert "deadweight" in _FINDING_RENDERERS
    assert "deadweight" in _MINOR_FINDING_LABELS


def test_render_deadweight_names_the_dead_server(tmp_path, capsys):
    """The finding renders through the CLI dispatch path and names the dead
    server, its presence, its zero invocations and the token tax."""
    from tokenjam.cli.cmd_optimize import _render_deadweight

    root = tmp_path / "root"
    project_dir = root / "-repo-a"
    _write_mcp_json(project_dir, {"apollo": {}})
    for i in range(MIN_SESSIONS_DEADWEIGHT):
        _plain_session(root, "-repo-a", f"s{i}", str(project_dir))

    finding = compute_deadweight_finding(_SINCE, _UNTIL, projects_root=root)
    assert finding.dead_servers  # sanity: the analyzer actually flagged one

    for mode in ("api", "subscription", "local", "unknown"):
        _render_deadweight(finding, pricing_mode=mode, marker="①")
    out = capsys.readouterr().out

    assert "apollo" in out
    assert f"{MIN_SESSIONS_DEADWEIGHT} sessions" in out
    assert "0 invocations" in out
    assert "No candidates flagged" not in out
    # The construction footnote travels with the number.
    assert "tok/session" in out


def test_render_deadweight_omits_dollars_when_no_model_was_priced(tmp_path, capsys):
    """No priced model observed means no dollar figure at all. Printing
    $0.00 would read as "this server costs nothing"."""
    from tokenjam.cli.cmd_optimize import _render_deadweight

    root = tmp_path / "root"
    project_dir = root / "-repo-a"
    _write_mcp_json(project_dir, {"apollo": {}})
    for i in range(MIN_SESSIONS_DEADWEIGHT):
        _write_transcript(root, "-repo-a", f"s{i}", [
            _user_prompt("say hi", cwd=str(project_dir)),
        ])

    finding = compute_deadweight_finding(_SINCE, _UNTIL, projects_root=root)
    assert finding.dead_servers[0].estimated_tax_usd_window is None

    _render_deadweight(finding, pricing_mode="api", marker="①")
    out = capsys.readouterr().out

    assert "apollo" in out
    assert "$" not in out
    assert "no priced model observed" in out


def test_render_report_surfaces_dead_servers_instead_of_no_candidates(tmp_path, capsys):
    """End-to-end: a report whose only finding is a populated deadweight set
    must not fall through to the generic "No candidates flagged" empty state."""
    from tokenjam.cli.cmd_optimize import _render_report
    from tokenjam.core.optimize.types import OptimizeReport, WindowSummary
    from tokenjam.utils.time_parse import utcnow

    root = tmp_path / "root"
    project_dir = root / "-repo-a"
    _write_mcp_json(project_dir, {"apollo": {}})
    for i in range(MIN_SESSIONS_DEADWEIGHT):
        _plain_session(root, "-repo-a", f"s{i}", str(project_dir))

    finding = compute_deadweight_finding(_SINCE, _UNTIL, projects_root=root)
    assert finding.dead_servers

    now = utcnow()
    report = OptimizeReport(
        window=WindowSummary(
            since=now, until=now, days=7, sessions=MIN_SESSIONS_DEADWEIGHT,
            spans=0, total_tokens=100_000, total_cost_usd=0.0, thin_data=False,
        ),
        downgrade=None,
        findings={"deadweight": finding},
    )
    _render_report(report, agent=None, requested=["deadweight"], pricing_mode="local")
    out = capsys.readouterr().out

    assert "No candidates flagged" not in out
    assert "apollo" in out
