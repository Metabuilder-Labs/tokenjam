"""The never-renumber gate.

These numbers are cited from source comments across the codebase ("Critical Rule
27", "root anti-pattern 21"). Renumbering breaks every citation silently: the
files still parse, every rule is still present, and the only symptom is that a
reference now resolves to the WRONG rule, which reads as correct.
"""
from __future__ import annotations

from tokenjam.core.summarize.numbering import (
    describe_drift,
    numbered_items,
    numbering_drift,
)

RULES = "## Rules\n\n1. First rule.\n2. Second rule.\n3. Third rule.\n"


def _drift(src_after: str, tgt_after: str, src_before: str = RULES, tgt_before: str = ""):
    return numbering_drift(
        source_before=src_before, target_before=tgt_before,
        source_after=src_after, target_after=tgt_after,
    )


def test_a_pure_move_across_the_two_files_is_clean():
    """The whole reason the multiset spans BOTH files: a move takes the numbers
    out of one and puts them into the other, so per-file checks would reject
    every legitimate relocation."""
    assert _drift("## Rules\n\nMoved.\n", RULES) == {}


def test_renumbering_the_survivors_is_caught():
    """The canonical failure: rule 2 moves out and rules 1 and 3 close the gap.

    Every rule is still present and both files parse, so nothing else in the
    tree notices — but "Critical Rule 3" now resolves to what used to be rule 2.
    The drift reports BOTH halves (a surplus `2.` and a missing `3.`) because
    the multiset is taken over the two files together.
    """
    after = "## Rules\n\n1. First rule.\n2. Third rule.\n"
    drift = _drift(after, "## Rules\n\n2. Second rule.\n")
    assert drift == {"2": 1, "3": -1}
    assert "3. x1 lost" in describe_drift(drift)
    assert "2. x1 added" in describe_drift(drift)


def test_dropping_an_item_outright_is_caught():
    assert _drift("## Rules\n\n1. First rule.\n2. Second rule.\n", "") == {"3": -1}


def test_an_invented_number_is_caught():
    assert _drift(RULES + "4. Invented.\n", "") == {"4": 1}


def test_duplicates_are_counted_as_a_multiset_not_a_set():
    """Two sections may each open a list at `1.`. Collapsing the duplicates
    would let one of them be dropped without the gate noticing."""
    before = "1. a\n2. b\n\n## Other\n\n1. c\n"
    assert numbered_items(before)["1"] == 2
    assert numbering_drift(
        source_before=before, target_before="",
        source_after="1. a\n2. b\n", target_after="",
    ) == {"1": -1}


def test_a_numbered_line_inside_a_code_fence_is_not_a_list_item():
    """A `1.` in a shell transcript is not a number anyone cites, and counting
    it would make the gate fire on a move that changed nothing."""
    fenced = "```\n1. not a rule\n2. not a rule\n```\n\n1. real rule\n"
    assert numbered_items(fenced) == {"1": 1}


def test_nested_items_count_because_their_numbers_are_cited_too():
    assert numbered_items("1. top\n   1. nested\n")["1"] == 2


def test_describe_drift_is_empty_when_clean():
    assert describe_drift({}) == ""
