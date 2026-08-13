"""Section model: verbatim slices, no overlap, and fences are not parsed."""
from __future__ import annotations

import pytest

from tokenjam.core.summarize.sections import Section, parse_sections, remove_sections

DOC = """# Title

Preamble prose.

## Alpha

Alpha body.

### Alpha sub

Sub body.

## Beta

Beta body.
"""


def test_sections_are_verbatim_slices_of_the_source():
    for s in parse_sections(DOC):
        assert s.text == DOC[s.start:s.end]
        assert s.heading + s.body == s.text


def test_a_section_owns_its_subsections():
    alpha = parse_sections(DOC)[0]
    assert alpha.title == "Alpha"
    assert "Alpha sub" in alpha.text
    assert "Beta" not in alpha.text


def test_sections_do_not_overlap_and_are_in_document_order():
    sections = parse_sections(DOC)
    for earlier, later in zip(sections, sections[1:]):
        assert earlier.end <= later.start


def test_an_h1_never_returns_a_section_spanning_the_whole_document():
    """Returning "level or shallower" would make the H1 overlap every H2 under
    it, and a caller removing both would delete the file."""
    assert [s.title for s in parse_sections(DOC)] == ["Alpha", "Beta"]


def test_a_heading_inside_a_code_fence_is_not_a_section():
    """This is not a corner case: real instruction files carry PR-body
    templates inside ```markdown fences, and a line-oriented regex reads their
    `## Summary` / `## Tests` lines as real sections — then happily relocates
    the middle of a code block."""
    text = (
        "## Real\n\nbody\n\n```markdown\n## Summary\n- bullet\n## Tests\n```\n\n"
        "## AlsoReal\n\nbody\n"
    )
    assert [s.title for s in parse_sections(text)] == ["Real", "AlsoReal"]
    assert "## Summary" in parse_sections(text)[0].text


def test_an_inner_fence_does_not_close_an_outer_one():
    text = "## Real\n\n````\n```\n## NotAHeading\n```\n````\n\n## After\n\nx\n"
    assert [s.title for s in parse_sections(text)] == ["Real", "After"]


def test_text_with_no_headings_yields_no_sections():
    assert parse_sections("just prose\n\nmore prose\n") == []


def test_remove_sections_replaces_right_to_left_without_shifting_offsets():
    sections = parse_sections(DOC)
    out = remove_sections(DOC, sections, ["## Alpha\n\nSTUB-A\n", "## Beta\n\nSTUB-B\n"])
    assert "STUB-A" in out and "STUB-B" in out
    assert "Alpha body" not in out and "Beta body" not in out
    assert out.startswith("# Title\n\nPreamble prose.\n\n")


def test_remove_sections_requires_one_replacement_per_section():
    with pytest.raises(ValueError):
        remove_sections(DOC, parse_sections(DOC), ["only one"])


def test_a_heading_on_the_final_line_terminates_at_end_of_file():
    text = "## A\n\nbody\n\n## B"
    last = parse_sections(text)[-1]
    assert isinstance(last, Section)
    assert last.end == len(text)
