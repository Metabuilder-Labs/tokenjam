"""Per-file load semantics — which part of an agent file is always in context.

The defect these pin: `summarize` priced every catalog file's WHOLE body as
always-resident, so a 160 KB skill library that had not been invoked once in
the window read as the most expensive prompt file a user owns. The opposite
error is just as wrong — pricing an invoked skill body at zero — so the split
has to be a real measurement of both halves, not a rule that discards one.
"""
from __future__ import annotations

import pytest

from tokenjam.core.summarize import load_semantics as ls

FRONTMATTER = "---\nname: ship\ndescription: Ship the thing.\n---\n"
BODY = "# Ship\n\nDo the steps.\n"


@pytest.mark.parametrize(("path", "expected"), [
    ("/home/u/CLAUDE.md", ls.ALWAYS),
    ("/repo/AGENTS.md", ls.ALWAYS),
    ("/repo/.claude/rules/style.md", ls.ALWAYS),
    ("/home/u/.claude/skills/ship/SKILL.md", ls.SKILL),
    ("/repo/.claude/commands/govern.md", ls.COMMAND),
    ("/repo/.claude/agents/reviewer.md", ls.AGENT),
    # Windows separators classify identically -- the fragments are matched
    # against a POSIX-normalised path.
    (r"C:\u\.claude\skills\ship\SKILL.md", ls.SKILL),
])
def test_classify(path, expected):
    assert ls.classify(path) == expected


def test_classify_defaults_to_always_for_an_unknown_path():
    assert ls.classify("/repo/docs/notes.md") == ls.ALWAYS
    assert ls.classify("") == ls.ALWAYS


@pytest.mark.parametrize(("path", "expected"), [
    ("/home/u/.claude/skills/ship/SKILL.md", "ship"),
    ("/repo/.claude/commands/govern.md", "govern"),
    ("/repo/.claude/agents/reviewer.md", "reviewer"),
    # An always-resident file is never "invoked", so it has no key at all --
    # NOT a key that would accidentally match some skill of the same name.
    ("/repo/CLAUDE.md", ""),
])
def test_invocation_key(path, expected):
    assert ls.invocation_key(path) == expected


def test_always_class_file_is_entirely_resident():
    text = "# Rules\n\nAlways do X.\n"
    assert ls.split_always_resident(text, ls.ALWAYS) == (text, "")


def test_on_demand_file_is_resident_only_up_to_its_frontmatter():
    resident, on_demand = ls.split_always_resident(FRONTMATTER + BODY, ls.SKILL)
    assert resident == FRONTMATTER
    assert on_demand == BODY


def test_on_demand_file_without_frontmatter_is_resident_in_no_part():
    """Nothing surfaces it in the harness's listing, so nothing is always in
    context — an empty resident half, never a guessed prefix."""
    resident, on_demand = ls.split_always_resident(BODY, ls.COMMAND)
    assert resident == ""
    assert on_demand == BODY


def test_a_horizontal_rule_further_down_is_not_frontmatter():
    """`---` only opens frontmatter at position 0. A rule mid-body must not
    swallow the whole file into the always-resident half."""
    text = "# Title\n\nSome prose.\n\n---\n\nmore prose\n"
    resident, on_demand = ls.split_always_resident(text, ls.SKILL)
    assert resident == ""
    assert on_demand == text


def test_crlf_frontmatter_still_splits():
    text = "---\r\nname: x\r\n---\r\nbody\r\n"
    resident, on_demand = ls.split_always_resident(text, ls.SKILL)
    assert resident.startswith("---") and "name: x" in resident
    assert on_demand == "body\r\n"


def test_split_halves_reconstruct_the_original():
    """No text is lost or duplicated between the two load classes — the token
    figures they drive must add up to the whole file's, not overlap."""
    text = FRONTMATTER + BODY
    for load_class in (ls.ALWAYS, ls.SKILL, ls.COMMAND, ls.AGENT):
        resident, on_demand = ls.split_always_resident(text, load_class)
        assert resident + on_demand == text


def test_scan_candidate_carries_the_split(tmp_path):
    """End-to-end through the scan: a skill file's reduction lands mostly in
    the on-demand half, and an always-on file's entirely in the resident one."""
    from tokenjam.core.summarize.candidates import list_candidates

    prose = ("This sentence exists purely to supply prose words to the "
             "detector so the file clears the worth-it gate. ") * 40
    skill = tmp_path / ".claude" / "skills" / "ship" / "SKILL.md"
    skill.parent.mkdir(parents=True)
    skill.write_text(FRONTMATTER + prose, encoding="utf-8")
    claude_md = tmp_path / "CLAUDE.md"
    claude_md.write_text(prose, encoding="utf-8")

    scan = list_candidates(tmp_path, include_global=False)
    by_path = {c.path: c for c in scan.candidates}

    always = by_path[str(claude_md)]
    assert always.load_class == "always"
    assert always.on_demand_tokens_saved == 0
    assert always.always_resident_tokens_saved == always.est_tokens_saved
    assert always.always_resident_chars == always.total_chars

    on_demand = by_path[str(skill)]
    assert on_demand.load_class == "skill"
    assert on_demand.invocation_key == "ship"
    # The body is the overwhelming majority of a skill file's compressible
    # prose; the frontmatter here has none worth counting.
    assert on_demand.on_demand_tokens_saved > on_demand.always_resident_tokens_saved
    assert on_demand.always_resident_chars == len(FRONTMATTER)
