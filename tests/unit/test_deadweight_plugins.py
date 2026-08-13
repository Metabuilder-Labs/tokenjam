"""The plugin lane: what an ENABLED, IN-SCOPE plugin costs every session.

Two gates decide whether a plugin costs anything and NEITHER is visible on the
filesystem. Measured on a real machine while building this: 1,299 SKILL.md files
were installed under ``~/.claude/plugins`` and 15 of them were resident — the
other 1,284 belonged to plugins that were switched off or scoped to one project.
Pricing installed-ness would have overstated the population by 87x.

The second gate is what gets counted rather than which plugin. A skill's BODY
arrives when the skill is invoked; its ``name: description`` line is listed
before anything is invoked. On the same machine those differ by more than three
orders of magnitude, so counting bodies is not a conservative overestimate — it
is a different number about a different thing.

Both are pinned below, plus the rule that plugin files never enter the summarize
catalog: they are third-party, under a versioned cache path, and the next plugin
update reverts any edit, so a "shorten this file" fix would appear to succeed
and then silently regress.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from tokenjam.core import agent_config as ac
from tokenjam.core.optimize.analyzers import deadweight as dw

_SKILL = """---
name: {name}
description: {desc}
---

# {name}

{body}
"""


def _plugin(root: Path, marketplace: str, plugin: str, version: str, skills: int) -> Path:
    """An installed plugin with ``skills`` skills, each with a huge body.

    The bodies are deliberately enormous relative to the frontmatter: a test
    where they were similar in size could not tell "counted the listing" from
    "counted the file".
    """
    install = root / "plugins" / "cache" / marketplace / plugin / version
    for i in range(skills):
        skill_dir = install / "skills" / f"s{i}"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            _SKILL.format(
                name=f"{plugin}-skill-{i}",
                desc="Does one specific thing.",
                body="body prose that is never resident " * 400,
            ),
            encoding="utf-8",
        )
    return install


@pytest.fixture
def claude_dir(tmp_path):
    """Enabled+user, disabled+user, and enabled+project — the three cases."""
    root = tmp_path / ".claude"
    root.mkdir()
    installs = {
        "on@mkt": _plugin(root, "mkt", "on", "1.0.0", skills=3),
        "off@mkt": _plugin(root, "mkt", "off", "1.0.0", skills=40),
        "scoped@mkt": _plugin(root, "mkt", "scoped", "1.0.0", skills=25),
    }
    (root / "settings.json").write_text(json.dumps({
        "enabledPlugins": {"on@mkt": True, "off@mkt": False, "scoped@mkt": True},
    }), encoding="utf-8")
    (root / "plugins" / "installed_plugins.json").write_text(json.dumps({
        "version": 2,
        "plugins": {
            "on@mkt": [{"scope": "user", "installPath": str(installs["on@mkt"]),
                        "version": "1.0.0"}],
            "off@mkt": [{"scope": "user", "installPath": str(installs["off@mkt"]),
                         "version": "1.0.0"}],
            "scoped@mkt": [{"scope": "project", "projectPath": "/somewhere/else",
                            "installPath": str(installs["scoped@mkt"]),
                            "version": "1.0.0"}],
        },
    }), encoding="utf-8")
    return root


# --- The two gates ----------------------------------------------------------

def test_only_enabled_and_in_scope_plugins_are_priced(claude_dir):
    """THE gate test. Installed on disk is not evidence of being loaded."""
    by_name = {r.name: r for r in ac.scan_plugins(claude_dir=claude_dir)}
    assert set(by_name) == {"on@mkt", "off@mkt", "scoped@mkt"}

    assert by_name["on@mkt"].detail["resident"] is True
    assert by_name["on@mkt"].tokens > 0

    # Gate 1: switched off. 40 skills on disk, zero of them resident.
    assert by_name["off@mkt"].detail["resident"] is False
    assert by_name["off@mkt"].detail["skills"] == 40
    assert by_name["off@mkt"].tokens == 0
    assert "disabled" in by_name["off@mkt"].detail["not_resident_because"]

    # Gate 2: enabled, but installed for one project — it does not load here.
    assert by_name["scoped@mkt"].detail["resident"] is False
    assert by_name["scoped@mkt"].detail["skills"] == 25
    assert by_name["scoped@mkt"].tokens == 0
    assert "project-scoped" in by_name["scoped@mkt"].detail["not_resident_because"]


def test_a_gated_off_plugin_still_gets_a_row(claude_dir):
    """"Disabled, so free" and "we never looked" must not be the same absence.

    And the disabled rows are exactly what a user deciding whether to ENABLE
    something large needs to see.
    """
    rows = ac.scan_plugins(claude_dir=claude_dir)
    assert len(rows) == 3
    assert all(r.detail["not_resident_because"] or r.detail["resident"] for r in rows)


def test_skill_bodies_are_never_counted_as_resident(claude_dir):
    """The other half, and the one that would inflate the figure most.

    The fixture's bodies are ~13,000 characters each against a ~35-character
    listing line. If a body ever leaks into the resident figure this fails by
    two orders of magnitude, not by a rounding error.
    """
    row = {r.name: r for r in ac.scan_plugins(claude_dir=claude_dir)}["on@mkt"]
    skills = sorted((claude_dir / "plugins" / "cache" / "mkt" / "on").rglob("SKILL.md"))
    assert len(skills) == 3
    whole_files = sum(len(p.read_text(encoding="utf-8")) for p in skills)

    assert row.detail["resident_chars"] < whole_files / 50
    assert row.tokens < ac.tokens_for_chars(whole_files) / 50
    # And positively: it IS the name+description line, for every skill.
    expected = sum(len(ac.skill_listing_line(p)) for p in skills)
    assert row.detail["resident_chars"] == expected


def test_the_listing_line_is_frontmatter_only():
    """A helper that could return a body is one refactor from pricing one."""
    tmp = Path(__file__).parent
    path = tmp / "_plugin_probe_SKILL.md"
    path.write_text(_SKILL.format(
        name="probe", desc="A short description.", body="SECRET BODY TEXT " * 100,
    ), encoding="utf-8")
    try:
        line = ac.skill_listing_line(path)
        assert line == "probe: A short description."
        assert "SECRET BODY TEXT" not in line
    finally:
        path.unlink()


def test_plugin_usage_reads_recorded_counts_and_omits_the_unrecorded(tmp_path):
    home = tmp_path
    (home / ".claude.json").write_text(json.dumps({"pluginUsage": {
        "used@mkt": {"usageCount": 42, "lastUsedAt": 1},
        "never@mkt": {"usageCount": 0, "lastUsedAt": 1},
        "malformed@mkt": {"lastUsedAt": 1},
    }}), encoding="utf-8")
    usage = ac.plugin_usage(home)
    assert usage == {"used@mkt": 42, "never@mkt": 0}
    # A key with no recorded count is ABSENT, not zero: never-recorded is
    # absence of evidence and only a recorded zero is evidence of absence.
    assert "malformed@mkt" not in usage


# --- Through the analyzer ---------------------------------------------------
#
# Liveness is decided by a RECENCY WINDOW now (nothing attributable fired in
# the trailing `UNUSED_RECENCY_WINDOW_DAYS` days), never by `pluginUsage`'s
# cumulative `usageCount` — that counter carries no timestamps and structurally
# cannot answer a recency question (see `deadweight.UNUSED_RECENCY_WINDOW_DAYS`'s
# own docstring). `_finding` below therefore controls WHEN sessions ran and
# WHETHER a skill was invoked, not a `pluginUsage` dict.

def _write_session(project: Path, name: str, days_ago: int, *, skill: str | None = None) -> None:
    """One session's transcript, timestamped `days_ago` before now. When
    `skill` is given, the session invokes it via the `Skill` tool — the
    timestamped evidence the recency scan reads (never `pluginUsage`)."""
    content = (
        [{"type": "tool_use", "name": "Skill", "input": {"skill": skill}}]
        if skill else [{"type": "text", "text": "hello"}]
    )
    records = [
        {"type": "user", "message": {"role": "user", "content": "hi"}, "cwd": str(project)},
        {"type": "assistant", "message": {
            "id": "m1", "role": "assistant", "model": "claude-sonnet-4-5-20250929",
            "content": content,
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }},
    ]
    path = project / f"{name}.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")
    mtime = (datetime.now(timezone.utc) - timedelta(days=days_ago)).timestamp()
    os.utime(path, (mtime, mtime))


def _finding(
    claude_dir: Path, tmp_path: Path, *,
    invoked_skill: str | None = None, session_days_ago: tuple[int, ...] = (3, 40),
):
    """A finding over a corpus whose sessions sit at `session_days_ago` before
    now. Default: one recent session plus one 40 days back — deep enough
    corpus history to make a confident unused verdict. `invoked_skill`, when
    given, is invoked in the MOST RECENT session (so it lands inside the
    trailing recency window whenever that session is itself inside it).
    """
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)

    root = tmp_path / "projects"
    project = root / "-repo"
    project.mkdir(parents=True)
    most_recent = min(session_days_ago) if session_days_ago else None
    for i, days_ago in enumerate(session_days_ago):
        skill = invoked_skill if (invoked_skill and days_ago == most_recent) else None
        _write_session(project, f"s{i}", days_ago, skill=skill)

    now = datetime.now(timezone.utc)
    return dw.compute_deadweight_finding(
        now - timedelta(days=45), now + timedelta(days=1),
        projects_root=root, claude_home=home, claude_dir=claude_dir,
        store=ac.InMemoryAgentConfigStore(), measure_schemas=False,
    )


def test_an_enabled_never_used_plugin_is_flagged_and_priced(claude_dir, tmp_path):
    finding = _finding(claude_dir, tmp_path)

    assert [p.name for p in finding.unused_plugins] == ["on@mkt"]
    unused = finding.unused_plugins[0]
    assert unused.resident is True
    assert unused.resident_tokens > 0
    assert unused.estimated_tax_tokens_window > 0
    assert "name: description" in unused.tax_construction
    assert "BODIES are NOT counted" in unused.tax_construction
    assert "claude plugin disable" in unused.fix
    # The gated-off ones are visible but cost nothing, even though nothing
    # attributable to them fired either.
    assert finding.plugins_resident == 1
    assert len(finding.plugins) == 3
    assert finding.past_overspend_tokens == unused.estimated_tax_tokens_window


def test_a_plugin_used_5_days_ago_is_not_flagged(claude_dir, tmp_path):
    """Part E pin: an item used inside the recency window is not unused.

    `on@mkt` ships THREE skills (`s0`..`s2`); all three have to fire for the
    plugin-level verdict to be "every component used" rather than
    `partial_use_no_fix` — see `test_a_plugin_has_no_fix_when_only_some_
    components_fire` for the partial case, which invoking only one of them
    would otherwise collide with here.
    """
    root = tmp_path / "projects"
    project = root / "-repo"
    project.mkdir(parents=True, exist_ok=True)

    content = [
        {"type": "tool_use", "name": "Skill", "input": {"skill": f"s{i}"}}
        for i in range(3)
    ]
    records = [
        {"type": "user", "message": {"role": "user", "content": "hi"}, "cwd": str(project)},
        {"type": "assistant", "message": {
            "id": "m1", "role": "assistant", "model": "claude-sonnet-4-5-20250929",
            "content": content, "usage": {"input_tokens": 10, "output_tokens": 5},
        }},
    ]
    path = project / "s_recent.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")
    mtime = (datetime.now(timezone.utc) - timedelta(days=5)).timestamp()
    os.utime(path, (mtime, mtime))
    _write_session(project, "s_old", 40)  # corpus depth, nothing invoked

    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    now = datetime.now(timezone.utc)
    finding = dw.compute_deadweight_finding(
        now - timedelta(days=45), now + timedelta(days=1),
        projects_root=root, claude_home=home, claude_dir=claude_dir,
        store=ac.InMemoryAgentConfigStore(), measure_schemas=False,
    )
    assert finding.unused_plugins == []
    assert finding.past_overspend_tokens is None
    assert any("fire in the last" in n for n in finding.notes)


def test_a_plugin_has_no_fix_when_only_some_components_fire(claude_dir, tmp_path):
    """Part C: enable/disable is whole-plugin only, so a plugin with some
    components used and some not gets NO fix — never a tempting number with
    no action behind it."""
    finding = _finding(claude_dir, tmp_path, invoked_skill="s0", session_days_ago=(5, 40))
    assert finding.unused_plugins == []
    row = next(p for p in finding.plugins if p.name == "on@mkt")
    assert row.partial_use_no_fix is True
    assert row.unused is False
    assert row.fix  # names the unused ones, says plainly no fix is on offer
    assert "s1" in row.fix or "s2" in row.fix
    assert any("no fix is offered" in n for n in finding.notes)


def test_a_plugin_last_used_40_days_ago_is_flagged(claude_dir, tmp_path):
    """Part E pin: an item last used OUTSIDE the recency window (but with
    enough corpus depth to trust the negative) is unused — the 40-day-old
    invocation does not save it.

    `_finding`'s `invoked_skill` only lands in the MOST RECENT session, so
    placing the invocation 40 days back (with a newer, silent session on top
    of it) needs the sessions written directly rather than through that
    helper.
    """
    root = tmp_path / "projects"
    project = root / "-repo"
    project.mkdir(parents=True, exist_ok=True)
    _write_session(project, "s_recent_unused", 3)
    _write_session(project, "s_old_used", 40, skill="s0")
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    now = datetime.now(timezone.utc)
    finding = dw.compute_deadweight_finding(
        now - timedelta(days=45), now + timedelta(days=1),
        projects_root=root, claude_home=home, claude_dir=claude_dir,
        store=ac.InMemoryAgentConfigStore(), measure_schemas=False,
    )
    assert [p.name for p in finding.unused_plugins] == ["on@mkt"]


def test_a_plugin_with_insufficient_history_is_never_flagged(claude_dir, tmp_path):
    """Part E pin: a corpus shorter than the recency window cannot support an
    unused claim — insufficient history, not a finding.

    Renamed from the old usage_count-based version of this test: liveness is
    decided by the recency scan now, not by `pluginUsage`, so the "absence of
    evidence is not evidence of absence" property has to be re-pinned against
    the NEW absence (a shallow corpus) rather than the retired one (an
    unrecorded `pluginUsage` key).
    """
    finding = _finding(claude_dir, tmp_path, session_days_ago=(3, 5))
    assert finding.unused_plugins == []
    assert finding.past_overspend_tokens is None
    row = next(p for p in finding.plugins if p.name == "on@mkt")
    assert row.insufficient_history is True
    assert any("history" in n for n in finding.notes)


def test_a_plugin_with_zero_component_files_never_gets_a_priced_disable_row(tmp_path):
    """Part E: no existing fixture had zero SKILL.md files. A plugin whose
    install path carries no skill, agent or MCP server has NOTHING priced,
    so it must never render a `$0 tokens, disable it` row.

    This is the vacuous-truth trap (Critical Rule 42 in
    `.claude/rules/optimize-analyzers.md`): `all(c.used is False for c in
    components)` over an EMPTY `components` list is vacuously True in
    Python, and would otherwise read as "every component agrees it's
    unused" for a plugin nothing was ever measured for.
    """
    root = tmp_path / ".claude"
    root.mkdir()
    install = root / "plugins" / "cache" / "mkt" / "empty" / "1.0.0"
    install.mkdir(parents=True)  # no SKILL.md, no agents/, no .mcp.json
    (root / "settings.json").write_text(json.dumps({
        "enabledPlugins": {"empty@mkt": True},
    }), encoding="utf-8")
    (root / "plugins" / "installed_plugins.json").write_text(json.dumps({
        "version": 2,
        "plugins": {"empty@mkt": [{"scope": "user", "installPath": str(install),
                                    "version": "1.0.0"}]},
    }), encoding="utf-8")

    rows = ac.scan_plugins(claude_dir=root)
    plugin_row = next(r for r in rows if r.name == "empty@mkt")
    assert plugin_row.detail["resident"] is True
    assert plugin_row.detail["components"] == []
    assert plugin_row.tokens == 0

    finding = _finding(root, tmp_path)
    plugin = next(p for p in finding.plugins if p.name == "empty@mkt")
    assert plugin.components == []
    assert plugin.unused is False
    assert plugin.estimated_tax_tokens_window == 0
    assert plugin not in finding.unused_plugins

    from tokenjam.cli.cmd_optimize import _render_deadweight

    _render_deadweight(finding, pricing_mode="api", marker="1")  # must not crash


# --- The catalog exclusion --------------------------------------------------

def test_plugin_paths_never_enter_the_summarize_catalog():
    """A PIN, so a future change cannot quietly pull them in.

    Plugin files live under a versioned third-party cache path the user did not
    author. Offering to shorten one produces a fix that appears to succeed and
    is reverted by the next plugin update — a saving the product would keep
    claiming and never actually collect.
    """
    from tokenjam.core.summarize.catalog import load_catalog

    catalog = load_catalog()
    everything = (
        list(catalog.project_files) + list(catalog.project_globs)
        + list(catalog.global_paths)
    )
    offenders = [entry for entry in everything if "plugin" in entry.lower()]
    assert not offenders, (
        "plugin paths must never be summarize candidates — they are third-party "
        "files under a versioned cache path, and the next plugin update reverts "
        f"any edit: {offenders}"
    )


def test_the_summarize_scan_does_not_pick_up_a_plugin_skill(tmp_path, monkeypatch):
    """The behavioural half of the pin: not just absent from the catalog, but
    genuinely not scanned even when a plugin tree sits under the scan root."""
    from tokenjam.core.summarize import candidates

    project = tmp_path / "repo"
    (project / ".claude" / "plugins" / "cache" / "mkt" / "p" / "1.0" / "skills" / "s").mkdir(
        parents=True,
    )
    (project / ".claude" / "plugins" / "cache" / "mkt" / "p" / "1.0" / "skills" / "s"
     / "SKILL.md").write_text(_SKILL.format(
         name="p", desc="d", body="plugin body prose " * 300,
     ), encoding="utf-8")
    (project / "CLAUDE.md").write_text("real instructions " * 200, encoding="utf-8")

    scan = candidates.list_candidates(
        project_roots=[str(project)], config=None, include_global=False,
    )
    paths = [c.path for c in scan.candidates]
    assert any(p.endswith("CLAUDE.md") for p in paths)
    assert not [p for p in paths if "plugins" in p], paths


def test_the_plugin_lane_renders_even_with_no_mcp_servers(capsys):
    """A capability nobody has a path to does not exist.

    The deadweight renderer returns early when no MCP server is configured, and
    a user in that state can still be paying for an enabled plugin in every
    session. Pinned because the early return is exactly the kind of thing a
    later edit reinstates.
    """
    from tokenjam.cli.cmd_optimize import _render_deadweight
    from tokenjam.core.optimize.analyzers.deadweight import (
        DeadweightFinding,
        PluginComponent,
        PluginDeadweight,
    )

    plugin = PluginDeadweight(
        name="on@mkt", enabled=True, install_scope="user", resident=True,
        not_resident_because="",
        components=[PluginComponent(kind="skill", name="s0", resident_tokens=90, used=False)],
        skills=3, resident_tokens=90, usage_count=0,
        sessions_present=4, unused=True, estimated_tax_tokens_window=400,
        estimated_tax_usd_window=0.0012, priced_model="claude-sonnet-4-5",
        tax_construction="90 tok resident per call.",
        fix="claude plugin disable on@mkt.",
    )
    finding = DeadweightFinding(
        sessions_scanned=4, configured_servers=0,
        plugins=[plugin], unused_plugins=[plugin], plugins_resident=1,
    )
    _render_deadweight(finding, pricing_mode="api", marker="1")
    out = capsys.readouterr().out
    assert "no MCP server is" in out
    assert "on@mkt" in out
    assert "1 of 1 installed are resident" in out
    assert "claude plugin disable" in out
