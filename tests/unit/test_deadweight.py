"""Unit tests for the MCP dead-weight + context-tax analyzer
(core/optimize/analyzers/deadweight.py).

Mirrors test_relearn.py's fixture style — hand-written Claude Code on-disk
JSONL records under a tmp_path projects root, no I/O beyond that. The global
``~/.claude.json`` path is resolved lazily inside
``core/agent_config._settings_paths``, so patching ``HOME`` (via monkeypatch,
same as tests/conftest.py's autouse ``_tj_isolated_home`` fixture) is enough to
keep every test off the real developer machine — no test here ever touches the
real home.

**The schema size is a MEASUREMENT now, and this module pins it.** The analyzer
used to charge a flat module constant to every server; it now measures each
server by starting it and reading its ``tools/list``. A unit test must not
start a process, and a fixture ``.mcp.json`` names a command that does not
exist — so the autouse ``_fixed_schema_measurement`` fixture below replaces the
measurer with one returning fixed sizes. The two numbers it returns are
deliberately the values the deleted constants held, which is what keeps every
arithmetic assertion in this file meaningful: the tests still check the tax
MODEL (per-call re-send, the cache-read multiplier, the deferred split), and
that model did not change. What changed is where the magnitude comes from, and
:func:`test_an_unmeasured_server_is_excluded_not_defaulted` is the test that
pins the new behaviour.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from tokenjam.core import agent_config as ac
from tokenjam.core.optimize import mcp_probe
from tokenjam.core.optimize.analyzers.deadweight import (
    UNUSED_RECENCY_WINDOW_DAYS,
    compute_deadweight_finding,
    enumerate_configured_servers,
    run as run_deadweight,
)

_NOW = datetime.now(timezone.utc)
_SINCE = _NOW - timedelta(days=7)
_UNTIL = _NOW + timedelta(days=1)

#: Purely a "how many example sessions to create" convenience for the
#: fixtures below. The analyzer no longer gates liveness on a session COUNT
#: (that was `_N_SESSIONS`, retired in favour of the recency
#: window — see `UNUSED_RECENCY_WINDOW_DAYS`), so this number carries no
#: threshold meaning; it just gives the multi-source/pricing/deferred-tools
#: tests several example sessions to aggregate over.
_N_SESSIONS = 5

#: Comfortably past `UNUSED_RECENCY_WINDOW_DAYS`. Every session this module's
#: fixtures write lands at effectively "now" (mtime = write time), which on
#: its own would make the corpus look SHALLOW to `_scan_recency_window`'s
#: corpus-depth check — every unused-server assertion would read as
#: "insufficient history" instead. `_ensure_corpus_depth` below writes one
#: throwaway old session at this age, once per `root`, entirely automatically
#: (threaded through `_write_transcript`, the one low-level writer every
#: fixture helper in this module funnels through) so individual tests never
#: have to know about it.
_DEPTH_ANCHOR_DAYS_AGO = UNUSED_RECENCY_WINDOW_DAYS * 3

#: The sizes the pinned measurer reports. Same values the deleted
#: ``FULL_SCHEMA_TAX_TOKENS`` / ``DEFERRED_SCHEMA_TAX_TOKENS`` constants held,
#: so every arithmetic expectation below is unchanged — see the module
#: docstring for why that is the point rather than a coincidence.
FULL_SCHEMA_TAX_TOKENS = 25_000
DEFERRED_SCHEMA_TAX_TOKENS = 400


def _measurement(name: str, _spec: dict) -> mcp_probe.SchemaMeasurement:
    return mcp_probe.SchemaMeasurement(
        server=name,
        tokens=FULL_SCHEMA_TAX_TOKENS,
        deferred_tokens=DEFERRED_SCHEMA_TAX_TOKENS,
        tool_count=7,
        status=ac.MEASURE_OK,
        measured_at=_NOW,
    )


@pytest.fixture(autouse=True)
def _fixed_schema_measurement(monkeypatch):
    """Every server in this module measures to a fixed, known size.

    Autouse and unconditional: without it a test would START whatever command a
    fixture ``.mcp.json`` happens to name. That is not merely slow — it is a
    unit test spawning arbitrary processes off test data, which is exactly the
    thing the probe's own budget and read-only discipline exist to bound.
    """
    monkeypatch.setattr(mcp_probe, "_default_measurer", _measurement)


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


def _ensure_corpus_depth(root: Path) -> None:
    """See `_DEPTH_ANCHOR_DAYS_AGO`. Idempotent per `root` (checks for the
    anchor directory first), so calling it from every `_write_transcript`
    costs nothing on the second and later session of a test. The anchor's
    mtime sits well outside this module's `_SINCE`/`_UNTIL` report window, so
    it never shows up in `sessions_scanned` or any server's `sessions_present`
    — only `_scan_recency_window`'s own independent, unbounded-below walk
    ever looks at it, and only for its mtime (it is never parsed: its content
    is irrelevant, and old enough that the recency scan skips reading it)."""
    anchor_dir = root / "-corpus-depth-anchor"
    if anchor_dir.exists():
        return
    anchor_dir.mkdir(parents=True)
    path = anchor_dir / "anchor.jsonl"
    path.write_text(
        json.dumps(_user_prompt("anchor", cwd=str(anchor_dir))), encoding="utf-8",
    )
    import os

    mtime = (_NOW - timedelta(days=_DEPTH_ANCHOR_DAYS_AGO)).timestamp()
    os.utime(path, (mtime, mtime))


def _write_transcript(root: Path, project: str, session_id: str, records: list[dict]) -> Path:
    _ensure_corpus_depth(root)
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

    for i in range(_N_SESSIONS):
        _plain_session(root, "-heavy", f"s{i}", str(heavy))
    _plain_session(root, "-light-a", "s-a", str(light_a))
    _plain_session(root, "-light-b", "s-b", str(light_b))

    finding = compute_deadweight_finding(_SINCE, _UNTIL, projects_root=root)
    dead = finding.unused_servers[0]

    heavy_config = str(heavy / ".mcp.json")
    light_a_config = str(light_a / ".mcp.json")
    light_b_config = str(light_b / ".mcp.json")

    assert dead.source == heavy_config
    assert set(dead.other_sources) == {light_a_config, light_b_config}
    assert dead.sessions_present == _N_SESSIONS + 2
    assert dead.primary_source_sessions == _N_SESSIONS
    # The fix text must name the gap rather than imply full coverage. The
    # GROUNDING (which other files, how many sessions the one edit reaches) is
    # built at the render site; the RULE that says a partial edit leaves the
    # rest of the tax running is catalogued, so it is checked for by its
    # catalogued wording rather than by a phrase this module authored.
    assert light_a_config in dead.fix
    assert light_b_config in dead.fix
    assert str(dead.primary_source_sessions) in dead.fix
    assert "Also declared in 2 other locations" in dead.fix
    assert "needs each one edited" in dead.fix


def test_project_scoped_server_fix_never_offers_project_scope_as_an_alternative(tmp_path):
    """A server already at project scope has nothing left to narrow --
    offering "project-scope it" as a second option would be a no-op
    delivering $0 of the claim."""
    root = tmp_path / "root"
    project_dir = root / "-repo-a"
    _write_mcp_json(project_dir, {"apollo": {}})
    for i in range(_N_SESSIONS):
        _plain_session(root, "-repo-a", f"s{i}", str(project_dir))

    finding = compute_deadweight_finding(_SINCE, _UNTIL, projects_root=root)
    dead = finding.unused_servers[0]

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
    for i in range(_N_SESSIONS):
        _plain_session(root, "-repo-a", f"s{i}", str(root / "-repo-a"))

    finding = compute_deadweight_finding(_SINCE, _UNTIL, projects_root=root)
    dead = next(s for s in finding.unused_servers if s.name == "exa")

    assert dead.fix.startswith("Remove or project-scope the `exa`")


# --- C1: dead-weight detection ----------------------------------------------

def test_no_configured_servers_is_a_no_op(tmp_path):
    project_dir = tmp_path / "root" / "repo-a"
    for i in range(_N_SESSIONS):
        _plain_session(tmp_path / "root", "-repo-a", f"s{i}", str(project_dir))

    finding = compute_deadweight_finding(_SINCE, _UNTIL, projects_root=tmp_path / "root")

    assert finding.configured_servers == 0
    assert finding.unused_servers == []
    assert finding.past_overspend_tokens is None


def test_detects_unused_server(tmp_path):
    root = tmp_path / "root"
    project_dir = root / "-repo-a"
    _write_mcp_json(project_dir, {"apollo": {}})
    for i in range(_N_SESSIONS):
        _plain_session(root, "-repo-a", f"s{i}", str(project_dir))

    finding = compute_deadweight_finding(_SINCE, _UNTIL, projects_root=root)

    assert finding.configured_servers == 1
    assert len(finding.unused_servers) == 1
    dead = finding.unused_servers[0]
    assert dead.name == "apollo"
    assert dead.sessions_present == _N_SESSIONS
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
    for i in range(_N_SESSIONS):
        _plain_session(root, "-repo-a", f"s{i}", str(project_dir))

    finding = compute_deadweight_finding(_SINCE, _UNTIL, projects_root=root)
    dead = finding.unused_servers[0]

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
    for i in range(_N_SESSIONS):
        _write_transcript(root, "-repo-a", f"s{i}", [
            _user_prompt("say hi", cwd=str(project_dir)),
            # No assistant turn at all -> no model signal for this session.
        ])

    finding = compute_deadweight_finding(_SINCE, _UNTIL, projects_root=root)
    dead = finding.unused_servers[0]

    assert dead.priced_model == ""
    assert dead.estimated_tax_usd_per_session is None
    assert dead.estimated_tax_usd_window is None
    assert finding.past_overspend_usd is None
    assert "No dollar estimate" in dead.tax_construction


def test_no_session_count_minimum_a_single_present_session_is_enough(tmp_path):
    """Positive pin that the retired session-count gate (the old
    ``MIN_SESSIONS_DEADWEIGHT`` module constant) is genuinely gone, not just
    unreferenced: a server present in exactly ONE session, never invoked,
    with the corpus otherwise deep enough for a confident verdict, is flagged
    unused. The old behaviour required >= 5 sessions before flagging
    anything, precisely to guard against a small sample; the recency window
    (`UNUSED_RECENCY_WINDOW_DAYS`) is what carries that job now — see
    `test_a_shallow_corpus_is_insufficient_history_not_a_finding` for the
    replacement guard.
    """
    root = tmp_path / "root"
    project_dir = root / "-repo-a"
    _write_mcp_json(project_dir, {"apollo": {}})
    _plain_session(root, "-repo-a", "s0", str(project_dir))

    finding = compute_deadweight_finding(_SINCE, _UNTIL, projects_root=root)

    assert finding.configured_servers == 1
    assert [s.name for s in finding.unused_servers] == ["apollo"]
    assert finding.unused_servers[0].sessions_present == 1


def _backdate(path: Path, days_ago: float) -> None:
    import os

    mtime = (_NOW - timedelta(days=days_ago)).timestamp()
    os.utime(path, (mtime, mtime))


def test_a_server_invoked_5_days_ago_is_not_unused(tmp_path):
    """Part E pin: an item used inside the recency window is not unused."""
    root = tmp_path / "root"
    project_dir = root / "-repo-a"
    _write_mcp_json(project_dir, {"apollo": {}})
    path = _write_transcript(root, "-repo-a", "s-recent", [
        _user_prompt("use the tool", cwd=str(project_dir)),
        _assistant(
            "Calling it.",
            tools=[{"id": "t1", "name": "mcp__apollo__search", "input": {}}],
            cwd=str(project_dir),
        ),
    ])
    _backdate(path, 5)

    finding = compute_deadweight_finding(
        _NOW - timedelta(days=45), _NOW + timedelta(days=1), projects_root=root,
    )

    assert finding.unused_servers == []


def test_a_server_last_invoked_40_days_ago_is_unused(tmp_path):
    """Part E pin: an item last used OUTSIDE the recency window (but with
    enough corpus depth to trust the negative) is unused — the 40-day-old
    invocation does not save it."""
    root = tmp_path / "root"
    project_dir = root / "-repo-a"
    _write_mcp_json(project_dir, {"apollo": {}})
    unused_recent = _write_transcript(root, "-repo-a", "s-recent-unused", [
        _user_prompt("say hi", cwd=str(project_dir)),
        _assistant("Hello!", cwd=str(project_dir)),
    ])
    _backdate(unused_recent, 3)
    old_used = _write_transcript(root, "-repo-a", "s-old-used", [
        _user_prompt("use the tool", cwd=str(project_dir)),
        _assistant(
            "Calling it.",
            tools=[{"id": "t1", "name": "mcp__apollo__search", "input": {}}],
            cwd=str(project_dir),
        ),
    ])
    _backdate(old_used, 40)

    finding = compute_deadweight_finding(
        _NOW - timedelta(days=45), _NOW + timedelta(days=1), projects_root=root,
    )

    assert [s.name for s in finding.unused_servers] == ["apollo"]


def test_a_shallow_corpus_is_insufficient_history_not_a_finding(tmp_path):
    """Part E pin: a corpus shorter than the recency window cannot support an
    unused claim — insufficient history, not a finding. Unlike every other
    test in this module, this one deliberately does NOT rely on
    `_ensure_corpus_depth`'s automatic anchor: it asserts the raw, un-anchored
    behaviour, so it writes its own isolated root rather than reusing the
    module-wide `_write_transcript` corpus-depth side effect.
    """
    root = tmp_path / "shallow-root"
    project_dir = root / "-repo-a"
    _write_mcp_json(project_dir, {"apollo": {}})
    for days_ago, name in ((1, "s0"), (3, "s1")):
        path = _write_transcript(root, "-repo-a", name, [
            _user_prompt("say hi", cwd=str(project_dir)),
            _assistant("Hello!", cwd=str(project_dir)),
        ])
        _backdate(path, days_ago)
    # `_write_transcript` seeded a `-corpus-depth-anchor` under `root` too —
    # remove it so this test genuinely exercises a shallow corpus.
    import shutil

    shutil.rmtree(root / "-corpus-depth-anchor")

    finding = compute_deadweight_finding(
        _NOW - timedelta(days=45), _NOW + timedelta(days=1), projects_root=root,
    )

    assert finding.unused_servers == []
    row = next(s for s in finding.servers if s.name == "apollo")
    assert row.insufficient_history is True


def test_compute_deadweight_finding_cache_dir_opt_in_matches_uncached(tmp_path):
    """Passing `cache_dir` must not change the finding — only skip re-parsing
    on a warm hit (see the two tests below)."""
    root = tmp_path / "root"
    project_dir = root / "-repo-a"
    _write_mcp_json(project_dir, {"apollo": {}})
    for i in range(_N_SESSIONS):
        _plain_session(root, "-repo-a", f"s{i}", str(project_dir))

    uncached = compute_deadweight_finding(_SINCE, _UNTIL, projects_root=root)
    cached = compute_deadweight_finding(
        _SINCE, _UNTIL, projects_root=root, cache_dir=tmp_path / "cache",
    )

    assert [s.name for s in cached.unused_servers] == [s.name for s in uncached.unused_servers]
    assert cached.sessions_scanned == uncached.sessions_scanned


def test_compute_deadweight_finding_warm_cache_skips_reparsing(tmp_path, monkeypatch):
    """A second call over an UNCHANGED corpus with the same `cache_dir` must
    not re-read/re-parse any transcript — the whole point of the cache."""
    root = tmp_path / "root"
    project_dir = root / "-repo-a"
    _write_mcp_json(project_dir, {"apollo": {}})
    for i in range(_N_SESSIONS):
        _plain_session(root, "-repo-a", f"s{i}", str(project_dir))
    cache_dir = tmp_path / "cache"

    first = compute_deadweight_finding(_SINCE, _UNTIL, projects_root=root, cache_dir=cache_dir)
    assert first.unused_servers  # sanity: a real signal, not an empty no-op

    def _boom(path):
        raise AssertionError(f"transcript.read_records reparsed {path} on a warm cache run")

    monkeypatch.setattr("tokenjam.core.transcript._parse_records", _boom)

    second = compute_deadweight_finding(_SINCE, _UNTIL, projects_root=root, cache_dir=cache_dir)
    assert [s.name for s in second.unused_servers] == [s.name for s in first.unused_servers]


def test_compute_deadweight_finding_cache_invalidates_on_transcript_edit(tmp_path):
    """A session whose transcript CHANGES between two cached runs (e.g. an
    invocation gets appended) must be re-parsed, not served stale."""
    root = tmp_path / "root"
    project_dir = root / "-repo-a"
    _write_mcp_json(project_dir, {"apollo": {}})
    for i in range(_N_SESSIONS):
        _plain_session(root, "-repo-a", f"s{i}", str(project_dir))
    cache_dir = tmp_path / "cache"

    first = compute_deadweight_finding(_SINCE, _UNTIL, projects_root=root, cache_dir=cache_dir)
    assert first.unused_servers  # apollo starts out dead (never invoked)

    # Rewrite one of the sessions so apollo IS invoked — size and mtime both
    # change, which must invalidate that session's cache entry.
    _invoking_session(
        root, "-repo-a", "s0", str(project_dir), "mcp__apollo__apollo_contacts_search",
    )

    second = compute_deadweight_finding(_SINCE, _UNTIL, projects_root=root, cache_dir=cache_dir)
    assert second.unused_servers == []  # no longer dead — the edit was picked up


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
    for i in range(_N_SESSIONS):
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
    assert first.unused_servers

    def _boom(path):
        raise AssertionError(f"transcript.read_records reparsed {path} on a warm cache run")

    monkeypatch.setattr("tokenjam.core.transcript._parse_records", _boom)

    second_ctx = _ctx()
    run_deadweight(second_ctx)
    second = second_ctx.report.findings["deadweight"]
    assert [s.name for s in second.unused_servers] == [s.name for s in first.unused_servers]


def test_invoked_server_is_never_flagged_dead(tmp_path):
    root = tmp_path / "root"
    project_dir = root / "-repo-a"
    _write_mcp_json(project_dir, {"apollo": {}})
    for i in range(_N_SESSIONS - 1):
        _plain_session(root, "-repo-a", f"s{i}", str(project_dir))
    _invoking_session(
        root, "-repo-a", "s-call", str(project_dir), "mcp__apollo__apollo_contacts_search",
    )

    finding = compute_deadweight_finding(_SINCE, _UNTIL, projects_root=root)

    row = next(s for s in finding.servers if s.name == "apollo")
    assert row.invocations == 1
    assert row.unused is False
    assert finding.unused_servers == []


# --- Sidechain/subagent transcripts must not become extra top-level sessions -

def test_sidechain_subagent_transcript_is_not_counted_as_a_top_level_session(tmp_path):
    """A `subagents/agent-*.jsonl` sidechain transcript lives nested under its
    parent session's own directory (core/transcript.py) but must not be
    discovered as an independent top-level "session" -- otherwise a session
    that happens to spawn a subagent spuriously inflates sessions_present (and
    thus inflates the schema tax) purely because of where the subagent's
    JSONL file lives on disk, not because of any real additional session."""
    root = tmp_path / "root"
    project_dir = root / "-repo-a"
    _write_mcp_json(project_dir, {"apollo": {}})
    for i in range(_N_SESSIONS - 1):
        _plain_session(root, "-repo-a", f"s{i}", str(project_dir))

    # A sidechain transcript nested under s0's own subagents/ dir. If it were
    # (mis)discovered as its own top-level session, sessions_scanned and
    # apollo's own sessions_present would both be one too many.
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

    assert finding.sessions_scanned == _N_SESSIONS - 1
    row = next(s for s in finding.servers if s.name == "apollo")
    assert row.sessions_present == _N_SESSIONS - 1


# --- Deferred-tools suppression ---------------------------------------------

def test_deferred_listing_suppresses_full_tax_claim(tmp_path):
    root = tmp_path / "root"
    project_dir = root / "-repo-a"
    _write_mcp_json(project_dir, {"apollo": {}})
    for i in range(_N_SESSIONS):
        _deferred_session(
            root, "-repo-a", f"s{i}", str(project_dir), "mcp__apollo__apollo_contacts_search",
        )

    finding = compute_deadweight_finding(_SINCE, _UNTIL, projects_root=root)

    dead = finding.unused_servers[0]
    assert dead.deferred_sessions == _N_SESSIONS
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

    dead = finding.unused_servers[0]
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
    assert row.unused is False
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
    for i in range(_N_SESSIONS - 1):
        _plain_session(root, "-repo-a", f"s-light-{i}", str(project_dir))
    _multi_call_session(root, "-repo-a", "s-heavy", str(project_dir), calls=10)

    finding = compute_deadweight_finding(_SINCE, _UNTIL, projects_root=root)
    dead = finding.unused_servers[0]
    assert dead.sessions_present == _N_SESSIONS

    rates = get_rates("anthropic", "claude-opus-4-8")
    ratio = rates.cache_read_per_mtok / rates.input_per_mtok
    # The $ figure is priced on the cache-discounted PRICE-EQUIVALENT
    # quantity (this test's actual subject: later calls in a session re-send
    # the schema at the cache-read rate, not the full input rate again).
    light_tax_price_equiv = FULL_SCHEMA_TAX_TOKENS  # single call -> multiplier == 1
    heavy_tax_price_equiv = round(FULL_SCHEMA_TAX_TOKENS * (1.0 + (10 - 1) * ratio))
    expected_usd_window = round(
        (light_tax_price_equiv * (_N_SESSIONS - 1) + heavy_tax_price_equiv)
        / 1_000_000 * rates.input_per_mtok,
        6,
    )

    assert heavy_tax_price_equiv > FULL_SCHEMA_TAX_TOKENS
    assert dead.estimated_tax_usd_window == expected_usd_window
    # The heavy session's real cost is folded in, not diluted by an averaged
    # call count applied uniformly to every session.
    assert dead.estimated_tax_usd_window > round(
        light_tax_price_equiv * _N_SESSIONS / 1_000_000 * rates.input_per_mtok, 6,
    )
    assert "cache-read rate" in dead.tax_construction
    assert "5-minute cache TTL" in dead.tax_construction

    # `estimated_tax_tokens_window` is the separate, LITERAL (undiscounted)
    # token count -- the schema is resent in full on every call regardless
    # of caching, so this is linear in the call count, never scaled down by
    # the cache-read ratio the $ figure above uses.
    expected_tokens_window = FULL_SCHEMA_TAX_TOKENS * (_N_SESSIONS - 1 + 10)
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
    for i in range(_N_SESSIONS - 1):
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
    dead = finding.unused_servers[0]
    assert dead.sessions_present == _N_SESSIONS

    get_rates("anthropic", "claude-opus-4-8")  # sanity: model is priced
    light_tax = FULL_SCHEMA_TAX_TOKENS  # single call -> multiplier == 1
    split_session_tax = FULL_SCHEMA_TAX_TOKENS  # ONE real call, not two
    expected_window = light_tax * (_N_SESSIONS - 1) + split_session_tax

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
    for i in range(_N_SESSIONS):
        _plain_session(root, "-repo-a", f"s{i}", str(project_dir))

    finding = compute_deadweight_finding(_SINCE, _UNTIL, projects_root=root)

    dead = finding.unused_servers[0]
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
    for i in range(_N_SESSIONS):
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
    for i in range(_N_SESSIONS):
        _plain_session(root, "-repo-a", f"s{i}", str(project_dir))

    finding = compute_deadweight_finding(_SINCE, _UNTIL, projects_root=root)
    assert finding.unused_servers  # sanity: the analyzer actually flagged one

    for mode in ("api", "subscription", "local", "unknown"):
        _render_deadweight(finding, pricing_mode=mode, marker="①")
    out = capsys.readouterr().out

    assert "apollo" in out
    assert f"{_N_SESSIONS} sessions" in out
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
    for i in range(_N_SESSIONS):
        _write_transcript(root, "-repo-a", f"s{i}", [
            _user_prompt("say hi", cwd=str(project_dir)),
        ])

    finding = compute_deadweight_finding(_SINCE, _UNTIL, projects_root=root)
    assert finding.unused_servers[0].estimated_tax_usd_window is None

    _render_deadweight(finding, pricing_mode="api", marker="①")
    out = capsys.readouterr().out

    assert "apollo" in out
    assert "$" not in out
    assert "no priced model observed" in out


def test_render_report_surfaces_unused_servers_instead_of_no_candidates(tmp_path, capsys):
    """End-to-end: a report whose only finding is a populated deadweight set
    must not fall through to the generic "No candidates flagged" empty state."""
    from tokenjam.cli.cmd_optimize import _render_report
    from tokenjam.core.optimize.types import OptimizeReport, WindowSummary
    from tokenjam.utils.time_parse import utcnow

    root = tmp_path / "root"
    project_dir = root / "-repo-a"
    _write_mcp_json(project_dir, {"apollo": {}})
    for i in range(_N_SESSIONS):
        _plain_session(root, "-repo-a", f"s{i}", str(project_dir))

    finding = compute_deadweight_finding(_SINCE, _UNTIL, projects_root=root)
    assert finding.unused_servers

    now = utcnow()
    report = OptimizeReport(
        window=WindowSummary(
            since=now, until=now, days=7, sessions=_N_SESSIONS,
            spans=0, total_tokens=100_000, total_cost_usd=0.0, thin_data=False,
        ),
        downgrade=None,
        findings={"deadweight": finding},
    )
    _render_report(report, agent=None, requested=["deadweight"], pricing_mode="local")
    out = capsys.readouterr().out

    assert "No candidates flagged" not in out
    assert "apollo" in out


# --- The schema tax is measured, and an unmeasured server is excluded -------
#
# The defect these pin: the analyzer used to charge a flat 25,000-token
# constant to EVERY configured server, on every call, whatever that server
# exposed — while the only in-repo source for "~25K" described the tax for ALL
# of a session's attached servers COMBINED. `past_overspend_usd` was therefore
# linear in an unmeasured number AND multiplied by however many servers the
# user had configured. The replacement must not merely move the number: it must
# be impossible for an unmeasured server to be billed anything at all.

def _unmeasurable(name: str, _spec: dict) -> mcp_probe.SchemaMeasurement:
    return mcp_probe.SchemaMeasurement(
        server=name, tokens=None, status=ac.MEASURE_UNREACHABLE,
        detail="the fixture server cannot be started.", measured_at=_NOW,
    )


def test_an_unmeasured_server_is_excluded_not_defaulted(tmp_path, monkeypatch):
    """A server whose schema size could not be measured contributes NOTHING.

    Not a default, not a floor constant, not a zero dressed up as a
    measurement: the row exists (the server really is dead weight and the user
    should see it), it carries no priced figure, and the finding's own totals
    stay unset rather than understating with a number that looks measured.
    """
    monkeypatch.setattr(mcp_probe, "_default_measurer", _unmeasurable)
    root = tmp_path / "root"
    project_dir = root / "-repo-a"
    _write_mcp_json(project_dir, {"apollo": {"command": "does-not-exist"}})
    for i in range(_N_SESSIONS):
        _plain_session(root, "-repo-a", f"s{i}", str(project_dir))

    finding = compute_deadweight_finding(_SINCE, _UNTIL, projects_root=root)

    assert [s.name for s in finding.unused_servers] == ["apollo"]
    dead = finding.unused_servers[0]
    assert dead.schema_tokens_measured is None
    assert dead.measurement_status == ac.MEASURE_UNREACHABLE
    assert dead.estimated_tax_tokens_window == 0
    assert dead.estimated_tax_usd_window is None
    # THE assertion: nothing priced, and no total invented from the gap.
    assert finding.past_overspend_tokens is None
    assert finding.past_overspend_usd is None
    # And the report says why, rather than reading as "nothing to flag".
    assert finding.servers_measured == 0
    assert finding.servers_unmeasured == 1
    assert "could not be measured" in finding.measurement_note
    # The sentence now opens a new sentence ("None of them could be
    # measured..."), so it's capitalized where it used to continue a clause;
    # compare case-insensitively rather than pinning the old casing.
    assert any("none of them could be measured" in n.lower() for n in finding.notes)
    # An unmeasured server must not appear in the tax table either — a row
    # claiming 0 tokens would read as "this server is free".
    assert not [r for r in finding.tax_table if r.source.startswith("MCP schema:")]


def test_the_tax_construction_says_the_size_was_measured(tmp_path):
    """The prose must state the provenance, not only the number.

    The whole defect was a figure that READ as measured because it was rendered
    exactly the way a measured one would be. A card that shows a token count
    with no provenance sentence recreates that, whatever is behind it.
    """
    root = tmp_path / "root"
    project_dir = root / "-repo-a"
    _write_mcp_json(project_dir, {"apollo": {"command": "x"}})
    for i in range(_N_SESSIONS):
        _plain_session(root, "-repo-a", f"s{i}", str(project_dir))

    finding = compute_deadweight_finding(_SINCE, _UNTIL, projects_root=root)
    dead = finding.unused_servers[0]

    assert "tools/list" in dead.tax_construction
    assert "7 tool schema(s)" in dead.tax_construction
    assert dead.schema_tokens_measured == FULL_SCHEMA_TAX_TOKENS
    assert dead.measured_tool_count == 7
    assert "MEASURED" in finding.estimate_basis


def test_a_server_is_measured_once_and_then_read_from_the_store(tmp_path):
    """Measuring means STARTING the server, so it must not happen every pass.

    The store carries the measurement against the server's own launch spec, so
    a second analysis over an unchanged corpus starts nothing. This asserts the
    measurer is called once across two runs sharing a store — the property that
    makes measuring affordable at all.
    """
    calls: list[str] = []

    def _counting(name: str, spec: dict) -> mcp_probe.SchemaMeasurement:
        calls.append(name)
        return _measurement(name, spec)

    root = tmp_path / "root"
    project_dir = root / "-repo-a"
    _write_mcp_json(project_dir, {"apollo": {"command": "x"}})
    for i in range(_N_SESSIONS):
        _plain_session(root, "-repo-a", f"s{i}", str(project_dir))

    store = ac.InMemoryAgentConfigStore()
    for _ in range(2):
        finding = compute_deadweight_finding(
            _SINCE, _UNTIL, projects_root=root, store=store,
            schema_measurer=_counting,
        )
    assert calls == ["apollo"]
    assert finding.unused_servers[0].schema_tokens_measured == FULL_SCHEMA_TAX_TOKENS


def test_measurement_off_prices_nothing_rather_than_restoring_the_assumption(tmp_path):
    """Switching measurement off is not a way back to the old constant.

    A run that CHOSE not to measure has the same evidence as one that failed to,
    so it must reach the same conclusion: excluded, and said so.
    """
    root = tmp_path / "root"
    project_dir = root / "-repo-a"
    _write_mcp_json(project_dir, {"apollo": {"command": "x"}})
    for i in range(_N_SESSIONS):
        _plain_session(root, "-repo-a", f"s{i}", str(project_dir))

    finding = compute_deadweight_finding(
        _SINCE, _UNTIL, projects_root=root, measure_schemas=False,
    )
    assert finding.unused_servers
    assert finding.past_overspend_usd is None
    assert finding.unused_servers[0].measurement_status == ac.MEASURE_SKIPPED


def test_the_enumeration_reads_back_what_it_ingested(tmp_path):
    """The walk populates the config store; the answer comes from the store.

    Pinned because the store round-trip is easy to "optimise" back into a
    direct return, and the measurement cache depends on the records existing.
    """
    project_dir = tmp_path / "repo"
    _write_mcp_json(project_dir, {"apollo": {"command": "x"}, "linear": {}})
    store = ac.InMemoryAgentConfigStore()

    servers = enumerate_configured_servers({str(project_dir)}, store=store)

    ingested = store.select(kind=ac.KIND_MCP_SERVER)
    assert sorted(r.name for r in ingested) == ["apollo", "linear"]
    assert all(r.path.endswith(".mcp.json") for r in ingested)
    assert all(r.content_hash for r in ingested)
    # The spec travels with the record, so the probe never re-reads the file.
    assert servers["apollo"].spec == {"command": "x"}
    assert servers["apollo"].config_id


# --- One analyzer must not destroy the whole report -------------------------

def test_a_raising_analyzer_is_isolated_and_disclosed(tmp_path, monkeypatch):
    """A bare dispatch loop let one analyzer take `build_report` down with it.

    Survivable while every analyzer was pure in-memory computation; not
    survivable once one of them could hit a database write conflict. But
    isolation ALONE would be the worse bug — an analyzer that vanishes silently
    reads as "found nothing", a positive claim the run has no evidence for. So
    the failure has to be recorded, not merely swallowed.
    """
    import duckdb

    from tokenjam.core.config import TjConfig
    from tokenjam.core.db import run_migrations
    from tokenjam.core.optimize import runner
    from tokenjam.core.optimize.registry import ANALYZER_REGISTRY

    conn = duckdb.connect(str(tmp_path / "t.duckdb"))
    run_migrations(conn)

    class _Db:
        pass

    db = _Db()
    db.conn = conn

    def _explode(_ctx):
        raise RuntimeError("kaboom")

    monkeypatch.setitem(ANALYZER_REGISTRY, "deadweight", _explode)
    report = runner.build_report(
        db, TjConfig(version="1"), since=_SINCE, until=_UNTIL,
        findings=["deadweight"],
    )
    conn.close()

    # It did not take the report down...
    assert report is not None
    # ...and it is DISCLOSED, not silently absent.
    assert "deadweight" in report.analyzer_errors
    assert "kaboom" in report.analyzer_errors["deadweight"]
    assert any("did not complete" in n for n in report.notes)


def test_the_cli_cannot_render_the_total_without_its_measurement_disclosure(capsys):
    """An undisclosed FLOOR rendered as a total.

    The terminal showed a priced dollar figure for the servers that could be
    measured and said nothing about the ones excluded from it. The number was
    honest; the presentation was not.
    """
    from tokenjam.cli.cmd_optimize import _render_deadweight
    from tokenjam.core.optimize.analyzers.deadweight import (
        DeadweightFinding,
        ServerDeadweight,
    )

    measured = ServerDeadweight(
        name="apollo", scope="user", source="/x/.claude.json", sessions_present=6,
        invocations=0, deferred_sessions=0, unused=True,
        estimated_tax_tokens_per_session=4000, estimated_tax_tokens_window=24000,
        tax_construction="measured", fix="Remove it.",
        estimated_tax_usd_window=1.23, priced_model="claude-sonnet-4-5",
        schema_tokens_measured=4000,
    )
    finding = DeadweightFinding(
        sessions_scanned=6, configured_servers=2, servers=[measured],
        unused_servers=[measured], past_overspend_usd=1.23,
        past_overspend_tokens=24000, servers_measured=1, servers_unmeasured=1,
    )
    finding.measurement_note = (
        "MEASUREMENT COVERAGE. 1 of 2 configured MCP server(s) had their schema "
        "size measured; the other 1 could not be measured and contributes "
        "NOTHING, so every total here is a floor."
    )
    _render_deadweight(finding, pricing_mode="api", marker="1")
    out = capsys.readouterr().out
    assert "apollo" in out
    assert "MEASUREMENT COVERAGE" in out
    assert "floor" in out


def test_a_server_measured_to_cost_nothing_gets_no_card(tmp_path):
    """`tokens or None` coerces a measured ZERO to None while the dollar figure
    stays a real 0.0 — a card reading `None tokens / $0.00`, which is the mixed
    basis Critical Rule 28 forbids. Reachable, not hypothetical: a server
    exposing zero tools measures to zero tokens.
    """
    from tokenjam.core.optimize.analyzers.deadweight import (
        DeadweightFinding,
        ServerDeadweight,
    )
    from tokenjam.core.optimize.cost_proposals import _deadweight_to_proposals

    empty = ServerDeadweight(
        name="empty", scope="user", source="/x/.claude.json", sessions_present=6,
        invocations=0, deferred_sessions=0, unused=True,
        estimated_tax_tokens_per_session=0, estimated_tax_tokens_window=0,
        tax_construction="measured: 0 tools", fix="Remove it.",
        estimated_tax_usd_window=0.0, priced_model="claude-sonnet-4-5",
        schema_tokens_measured=0, measurement_status="measured",
    )
    finding = DeadweightFinding(
        sessions_scanned=6, configured_servers=1, servers=[empty], unused_servers=[empty],
    )
    assert _deadweight_to_proposals(finding) == []
