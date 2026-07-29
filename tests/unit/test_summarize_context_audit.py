"""Context-load audit classification (`core/summarize/context_audit.py`).

The whole point of this scanner is getting the evidence-based classification
right, so these tests focus on the three correctness requirements from the
feature brief:

  1. A present-but-unreferenced file lands in the not-loaded bucket, at zero
     cost — never priced as if it were resident.
  2. A disabled plugin's skills/hooks are excluded; `~/.claude/rules/` files
     are counted on their own merit regardless of any plugin's enabled state.
  3. No total is meaningful before its scan has actually run (covered at the
     route/UI layer; the core-level analogue here is that `ScopeAudit`'s
     totals are computed only over rows the scan actually produced).
"""
from __future__ import annotations

import json

import pytest

from tokenjam.core.summarize import context_audit as ca


def _write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


# --------------------------------------------------------------------------- #
# Requirement 1: unreferenced files are reported, at zero cost, never priced.
# --------------------------------------------------------------------------- #

def test_present_but_unreferenced_file_lands_in_unloaded_bucket(tmp_path):
    root = tmp_path / "repo"
    _write(root / "CLAUDE.md", "# Rules\nAlways do X.\n")
    # A whole directory of real prose that nothing in the catalog auto-loads —
    # the concrete case from the brief (.claude/product-state/*.md).
    _write(root / ".claude" / "product-state" / "BENCHMARK.md", "x" * 5000)

    audit = ca.scan_project(root)

    loaded_sources = {r.source for r in (*audit.class1, *audit.class2, *audit.class3)}
    unloaded_sources = {u.source for u in audit.unloaded}
    stray = str(root / ".claude" / "product-state" / "BENCHMARK.md")

    assert stray in unloaded_sources
    assert stray not in loaded_sources
    # It must never contribute to the class1 total (Critical Rule 22: on-disk
    # presence is not evidence of being loaded).
    assert audit.class1_total_chars < 5000


def test_unloaded_row_is_never_summed_into_class1_total(tmp_path):
    root = tmp_path / "repo"
    _write(root / "CLAUDE.md", "short")
    _write(root / ".claude" / "scratch" / "notes.md", "y" * 9999)

    audit = ca.scan_project(root)

    assert audit.class1_total_chars == len("short")
    assert any(u.chars == 9999 for u in audit.unloaded)


def test_catalog_known_file_is_not_double_reported_as_unloaded(tmp_path):
    root = tmp_path / "repo"
    _write(root / "CLAUDE.md", "# hi\n")
    _write(root / ".claude" / "rules" / "style.md", "Be terse.\n")

    audit = ca.scan_project(root)

    unloaded_sources = {u.source for u in audit.unloaded}
    assert str(root / "CLAUDE.md") not in unloaded_sources
    assert str(root / ".claude" / "rules" / "style.md") not in unloaded_sources
    assert len(audit.class1) == 2


# --------------------------------------------------------------------------- #
# CLAUDE.md / unscoped rule -> class1; paths:-scoped rule -> class2.
# --------------------------------------------------------------------------- #

def test_unscoped_rule_is_class1_and_path_scoped_rule_is_class2(tmp_path):
    root = tmp_path / "repo"
    _write(root / "CLAUDE.md", "always resident\n")
    _write(root / ".claude" / "rules" / "unscoped.md", "always resident too\n")
    _write(
        root / ".claude" / "rules" / "scoped.md",
        "---\npaths:\n  - \"src/**/*.py\"\n---\nOnly for python files.\n",
    )

    audit = ca.scan_project(root)

    class1_sources = {r.source for r in audit.class1}
    class2_sources = {r.source for r in audit.class2}
    assert str(root / "CLAUDE.md") in class1_sources
    assert str(root / ".claude" / "rules" / "unscoped.md") in class1_sources
    assert str(root / ".claude" / "rules" / "scoped.md") in class2_sources
    assert str(root / ".claude" / "rules" / "scoped.md") not in class1_sources


# --------------------------------------------------------------------------- #
# Skill/command/agent frontmatter is class1; the body is class2 only when the
# description reads as self-invoking, class3 otherwise. A command body is
# always class3.
# --------------------------------------------------------------------------- #

def test_skill_frontmatter_is_class1_body_class3_by_default(tmp_path):
    root = tmp_path / "repo"
    _write(
        root / ".claude" / "skills" / "ship" / "SKILL.md",
        "---\nname: ship\ndescription: Ship the current branch.\n---\n# Ship\nDo the steps.\n",
    )

    audit = ca.scan_project(root)

    assert len(audit.class1) == 1
    assert audit.class1[0].frequency == "every turn"
    assert len(audit.class3) == 1
    assert len(audit.class2) == 0


def test_skill_description_implying_auto_invoke_puts_body_in_class2(tmp_path):
    root = tmp_path / "repo"
    _write(
        root / ".claude" / "skills" / "debug" / "SKILL.md",
        "---\nname: debug\ndescription: Proactively use this whenever a bug is reported.\n---\n"
        "# Debug\nDo the steps.\n",
    )

    audit = ca.scan_project(root)

    assert len(audit.class2) == 1
    assert "auto-invoke" in audit.class2[0].trigger
    assert len(audit.class3) == 0


def test_command_body_is_always_class3_even_with_auto_invoke_language(tmp_path):
    root = tmp_path / "repo"
    _write(
        root / ".claude" / "commands" / "ship.md",
        "---\nname: ship\ndescription: Proactively ships the branch.\n---\nDo it.\n",
    )

    audit = ca.scan_project(root)

    assert len(audit.class1) == 1
    assert len(audit.class3) == 1
    assert len(audit.class2) == 0


# --------------------------------------------------------------------------- #
# Hooks — settings.json events become class2 rows with the right frequency.
# --------------------------------------------------------------------------- #

def test_project_settings_hooks_become_class2_rows(tmp_path):
    root = tmp_path / "repo"
    settings = {
        "hooks": {
            "PreToolUse": [
                {"matcher": "Bash", "hooks": [{"type": "command", "command": "echo hi"}]},
            ],
            "SessionStart": [
                {"hooks": [{"type": "command", "command": "echo start"}]},
            ],
        }
    }
    _write(root / ".claude" / "settings.json", json.dumps(settings))

    audit = ca.scan_project(root)

    assert len(audit.class2) == 2
    frequencies = {r.frequency for r in audit.class2}
    assert "every tool call" in frequencies
    assert "per session" in frequencies


# --------------------------------------------------------------------------- #
# Requirement 2: disabled plugins are excluded (skills/hooks); rules files
# load regardless of plugin state.
# --------------------------------------------------------------------------- #

def test_disabled_plugin_skills_excluded_but_rules_still_counted(tmp_path, monkeypatch):
    home = tmp_path / "home"
    claude_dir = home / ".claude"
    settings_path = claude_dir / "settings.json"
    installed_path = claude_dir / "plugins" / "installed_plugins.json"

    plugin_install = claude_dir / "plugins" / "cache" / "acme" / "widget" / "1.0.0"
    _write(plugin_install / "skills" / "auto" / "SKILL.md",
           "---\nname: auto\ndescription: Proactively does things.\n---\nBody.\n")

    # A rules file written by/for that plugin — this is Class 1 regardless of
    # the plugin's enabled state; the harness reads ~/.claude/rules/** directly.
    _write(claude_dir / "rules" / "from-plugin.md", "Some rule text.\n")

    _write(settings_path, json.dumps({"enabledPlugins": {"widget@acme": False}}))
    _write(installed_path, json.dumps({
        "plugins": {"widget@acme": [{"installPath": str(plugin_install)}]}
    }))

    monkeypatch.setattr(ca, "SETTINGS_PATH", settings_path)
    monkeypatch.setattr(ca, "INSTALLED_PLUGINS_FILE", installed_path)
    monkeypatch.setattr(ca, "_global_paths", lambda: [claude_dir / "rules" / "from-plugin.md"])

    audit = ca.scan_global(claude_dir=claude_dir)
    plugin_c1, plugin_c2, plugin_c3, plugin_unloaded, enabled, disabled = ca._scan_plugins()

    # Disabled plugin contributes nothing.
    assert enabled == 0
    assert disabled == 1
    assert plugin_c1 == []
    assert plugin_c2 == []
    assert plugin_c3 == []

    # The rules file is counted on its own merit, independent of the plugin gate.
    assert any(r.source == str(claude_dir / "rules" / "from-plugin.md") for r in audit.class1)


def test_enabled_plugin_skill_is_counted(tmp_path, monkeypatch):
    home = tmp_path / "home"
    claude_dir = home / ".claude"
    settings_path = claude_dir / "settings.json"
    installed_path = claude_dir / "plugins" / "installed_plugins.json"

    plugin_install = claude_dir / "plugins" / "cache" / "acme" / "widget" / "1.0.0"
    _write(plugin_install / "skills" / "auto" / "SKILL.md",
           "---\nname: auto\ndescription: Proactively does things.\n---\nBody.\n")
    hooks = {"hooks": {"SessionStart": [{"hooks": [{"type": "command", "command": "run.sh"}]}]}}
    _write(plugin_install / "hooks" / "hooks.json", json.dumps(hooks))

    _write(settings_path, json.dumps({"enabledPlugins": {"widget@acme": True}}))
    _write(installed_path, json.dumps({
        "plugins": {"widget@acme": [{"installPath": str(plugin_install)}]}
    }))

    monkeypatch.setattr(ca, "SETTINGS_PATH", settings_path)
    monkeypatch.setattr(ca, "INSTALLED_PLUGINS_FILE", installed_path)

    plugin_c1, plugin_c2, plugin_c3, plugin_unloaded, enabled, disabled = ca._scan_plugins()

    assert enabled == 1
    assert disabled == 0
    assert len(plugin_c1) == 1          # SKILL.md frontmatter
    # Body is auto-invoke -> class2, plus the plugin's own hook -> class2.
    assert len(plugin_c2) == 2
    assert plugin_c3 == []


# --------------------------------------------------------------------------- #
# Requirement 3 (core-level analogue): totals only ever reflect rows the
# scan actually produced — an empty scope has a zero total, not a guess.
# --------------------------------------------------------------------------- #

def test_empty_scope_has_zero_total_not_a_guess(tmp_path):
    audit = ca.scan_project(tmp_path / "empty-repo")
    assert audit.class1 == ()
    assert audit.class1_total_chars == 0
    assert audit.class1_total_tokens == 0


def test_result_to_dict_keeps_global_and_project_totals_separate():
    result = ca.ContextAuditResult(
        global_scope=ca.ScopeAudit(scope="global", class1=(ca.Row("g", "t", 100, "every turn", ca.CLASS_1, "global"),)),
        projects=(ca.ScopeAudit(scope="/p", class1=(ca.Row("p", "t", 50, "every turn", ca.CLASS_1, "/p"),)),),
    )
    d = result.to_dict()
    assert d["global"]["class1_total_chars"] == 100
    assert d["projects"][0]["class1_total_chars"] == 50
    # No cross-scope sum field exists at all -- the payload has no key that
    # would invite a caller to add the two together.
    assert "class1_total_chars" not in d


# --------------------------------------------------------------------------- #
# Presentation redesign: rows_for_display groups by family, sorts by tokens
# descending, and never groups a family with exactly one member.
# --------------------------------------------------------------------------- #

def _row(source, chars, family_kind="", family_qualifier="", description=""):
    return ca.Row(source, "trigger", chars, "every turn", ca.CLASS_1, "global",
                  description, family_kind, family_qualifier)


def test_rows_for_display_groups_multi_member_families_and_sums_their_cost():
    rows = [
        _row("/r/a.md", 400, "rule_dir", "~/.claude/rules/ecc/common"),
        _row("/r/b.md", 600, "rule_dir", "~/.claude/rules/ecc/common"),
        _row("/r/c.md", 200, "rule_dir", "~/.claude/rules/ecc/common"),
    ]
    display = ca.rows_for_display(rows)
    assert len(display) == 1
    group = display[0]
    assert group["kind"] == "group"
    assert group["label"] == "~/.claude/rules/ecc/common/* x3"
    assert group["chars"] == 1200
    assert group["tokens"] == sum(r.tokens for r in rows)
    assert len(group["members"]) == 3


def test_rows_for_display_renders_a_single_member_family_as_a_plain_row():
    """The instruction is explicit: a group with one member is a plain row,
    no expander — never a 'group of one'."""
    rows = [_row("/plugins/x/skills/only/SKILL.md", 300, "skill_desc", "acme")]
    display = ca.rows_for_display(rows)
    assert len(display) == 1
    assert display[0]["kind"] == "row"
    assert display[0]["members"] == []


def test_rows_for_display_sorts_by_tokens_descending():
    rows = [
        _row("/a", 40),               # smallest, ungrouped
        _row("/b", 2000, "rule_dir", "~/.claude/rules/x"),
        _row("/c", 2000, "rule_dir", "~/.claude/rules/x"),
        _row("/big.md", 20000),       # biggest single file, well clear of the group's summed total
    ]
    display = ca.rows_for_display(rows)
    tokens = [d["tokens"] for d in display]
    assert tokens == sorted(tokens, reverse=True)
    assert display[0]["label"] == "/big.md"  # the single biggest row leads


def test_rows_for_display_derives_a_names_list_when_members_disagree_on_description():
    """If a group's members don't share one description, the group's
    description falls back to a comma-joined list of the members' own short
    names (never a guessed sentence) — evidence already on hand, not an
    invention. Supersedes the earlier version of this test, which pinned the
    old blank-on-disagreement behavior; leaving a bare '—' on most of the
    Class 1 table was itself the defect (product feedback)."""
    rows = [
        _row("/plugins/x/skills/alpha/SKILL.md", 200, "skill_desc", "acme", "Does A."),
        _row("/plugins/x/skills/beta/SKILL.md", 100, "skill_desc", "acme", "Does B."),
    ]
    display = ca.rows_for_display(rows)
    assert display[0]["description"] == "alpha, beta"


def test_skill_frontmatter_description_flows_through_as_the_plain_english_column(tmp_path):
    root = tmp_path / "repo"
    _write(
        root / ".claude" / "skills" / "ship" / "SKILL.md",
        "---\nname: ship\ndescription: Ship the current branch.\n---\n# Ship\nDo the steps.\n",
    )
    audit = ca.scan_project(root)
    assert audit.class1[0].description == "Ship the current branch."


def test_claude_md_description_is_its_own_first_heading(tmp_path):
    root = tmp_path / "repo"
    _write(root / "CLAUDE.md", "# tokenjam (meta-repo)\n\nSome body text.\n")
    audit = ca.scan_project(root)
    assert audit.class1[0].description == "tokenjam (meta-repo)"


def test_claude_md_with_no_heading_gets_a_blank_description_not_a_guess(tmp_path):
    root = tmp_path / "repo"
    _write(root / "CLAUDE.md", "No heading here, just prose.\n")
    audit = ca.scan_project(root)
    assert audit.class1[0].description == ""


def test_plugin_skills_group_under_the_plugin_short_name(tmp_path, monkeypatch):
    home = tmp_path / "home"
    claude_dir = home / ".claude"
    settings_path = claude_dir / "settings.json"
    installed_path = claude_dir / "plugins" / "installed_plugins.json"
    plugin_install = claude_dir / "plugins" / "cache" / "acme" / "widget" / "1.0.0"
    for name in ("one", "two"):
        _write(plugin_install / "skills" / name / "SKILL.md",
               f"---\nname: {name}\ndescription: does {name}.\n---\nBody.\n")
    _write(settings_path, json.dumps({"enabledPlugins": {"widget@acme": True}}))
    _write(installed_path, json.dumps({
        "plugins": {"widget@acme": [{"installPath": str(plugin_install)}]}
    }))
    monkeypatch.setattr(ca, "SETTINGS_PATH", settings_path)
    monkeypatch.setattr(ca, "INSTALLED_PLUGINS_FILE", installed_path)

    plugin_c1, _, _, _, enabled, _ = ca._scan_plugins()
    display = ca.rows_for_display(plugin_c1)

    assert enabled == 1
    assert len(display) == 1
    assert display[0]["label"] == "widget 2 skill descriptions"
