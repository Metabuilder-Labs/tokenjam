"""Heading-delimited section model for a markdown instruction file.

Relocation moves whole SECTIONS, so it needs a section boundary it can trust:
one wrong offset and the operation stops being lossless. This parses ATX
headings into byte ranges over the exact source text, with three properties the
rest of the relocate path depends on:

* **Every section is a verbatim slice.** ``section.text is text[start:end]``
  byte for byte, so a move can be asserted lossless by substring identity
  rather than by re-rendering.
* **Sections do not overlap and are in document order**, so removing several in
  one pass needs no offset bookkeeping.
* **Fenced content is not parsed.** A ``#`` inside a code fence is not a
  heading, and this is not a corner case: the PR-body template in this repo's
  own ``CLAUDE.md`` contains ``## Summary`` / ``## Tests / Verification`` inside
  a ```` ```markdown ```` fence, which a line-oriented regex reads as four
  real sections and would happily relocate the middle of a code block.

A section OWNS its subsections: an H2's range runs to the next heading at the
same or a shallower level, so relocating it takes its H3s with it. That is what
makes the unit a thing a reader can follow a pointer to.

Stdlib only, like the rest of this package.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

#: An ATX heading line. Up to three leading spaces (CommonMark), one to six
#: hashes, mandatory space, non-empty title, optional trailing hashes.
_ATX_RE = re.compile(r"^[ \t]{0,3}(#{1,6})[ \t]+(\S.*?)[ \t]*#*[ \t]*$")
#: A fence opener/closer: three or more backticks or tildes. A fence closes on a
#: line of the same character that is at least as long, which is what stops an
#: inner ```` ``` ```` inside a ```` ```` ```` block from closing it early.
_FENCE_RE = re.compile(r"^[ \t]{0,3}(`{3,}|~{3,})")


@dataclass(frozen=True)
class Section:
    """One heading and everything under it, as a verbatim slice of the source.

    ``start`` is the offset of the heading line; ``end`` is exclusive and lands
    on the start of the next heading at the same or a shallower level (or the
    end of the file). ``body_start`` is the offset just past the heading line,
    so ``text[:body_start - start]`` is the heading and the rest is the body.
    """

    level: int
    title: str
    start: int
    body_start: int
    end: int
    text: str

    @property
    def body(self) -> str:
        """Everything under the heading, heading line excluded."""
        return self.text[self.body_start - self.start:]

    @property
    def heading(self) -> str:
        """The heading line itself, including its trailing newline if it had one."""
        return self.text[: self.body_start - self.start]


def _heading_lines(text: str) -> list[tuple[int, int, int, str]]:
    """``(start, body_start, level, title)`` for every ATX heading outside a fence."""
    out: list[tuple[int, int, int, str]] = []
    offset = 0
    fence: str | None = None
    for line in text.splitlines(keepends=True):
        stripped = line.rstrip("\n").rstrip("\r")
        fence_match = _FENCE_RE.match(stripped)
        if fence is not None:
            # Inside a fence: only a closer of the same char and >= length ends it.
            if (
                fence_match
                and fence_match.group(1)[0] == fence[0]
                and len(fence_match.group(1)) >= len(fence)
            ):
                fence = None
            offset += len(line)
            continue
        if fence_match:
            fence = fence_match.group(1)
            offset += len(line)
            continue
        m = _ATX_RE.match(stripped)
        if m:
            out.append((offset, offset + len(line), len(m.group(1)), m.group(2)))
        offset += len(line)
    return out


def parse_sections(text: str, *, level: int = 2) -> list[Section]:
    """Sections headed at exactly ``level``, each spanning its own subsections.

    Deliberately EXACTLY that level rather than "that level or shallower": an
    H1 title line at the top of a file would otherwise return a section
    spanning the entire document, overlapping every H2 inside it, and a caller
    removing both would delete the file. Deeper headings are part of the
    enclosing section's body, which is what makes a section a self-contained
    unit that can be moved. Anything outside a returned range — a preamble, an
    H1 line, content under a heading at another level — is simply not offered
    for relocation.

    Returns ``[]`` for text with no qualifying heading. Never raises.
    """
    heads = _heading_lines(text)
    if not heads:
        return []
    out: list[Section] = []
    for i, (start, body_start, head_level, title) in enumerate(heads):
        if head_level != level:
            continue
        end = len(text)
        for later_start, _, later_level, _ in heads[i + 1:]:
            if later_level <= head_level:
                end = later_start
                break
        out.append(Section(
            level=head_level, title=title, start=start, body_start=body_start,
            end=end, text=text[start:end],
        ))
    return out


def remove_sections(text: str, sections: list[Section], replacements: list[str]) -> str:
    """``text`` with each of ``sections`` replaced by its stub, in one pass.

    Sections must come from :func:`parse_sections` over this exact ``text``.
    Applied right-to-left so no replacement shifts a later section's offsets.
    """
    if len(sections) != len(replacements):
        raise ValueError("one replacement per section is required")
    out = text
    for section, stub in sorted(
        zip(sections, replacements), key=lambda pair: pair[0].start, reverse=True,
    ):
        out = out[: section.start] + stub + out[section.end:]
    return out
