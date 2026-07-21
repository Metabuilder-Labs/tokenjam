"""No tracker reference may reach a string the product prints.

An id in a printed string does not stay in this repo. It escapes into users'
terminals, and from there into screenshots, pasted bug reports and issues,
where a bare number-sign reference reads as a link to a public artifact that
has nothing to do with it. Internal and public number spaces genuinely
collide, so the reader is confidently pointed at the wrong thing.

This guard exists because a test used to assert the OPPOSITE: it required an
id to be PRESENT in the doctor's MCP warning, so the suite actively defended
the leak, and anyone who removed it was told by CI that they had broken
something. See the leak-pinning entry in CLAUDE.md.

Scope is deliberately the printed surface only. Comments and docstrings are
developer-facing prose where a tracker reference is legitimate context, so
docstrings are skipped and comments are never parsed as string constants at
all.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

PACKAGE_ROOT = Path(__file__).resolve().parents[2] / "tokenjam"

#: The shape of a tracker reference: a number sign and at least TWO digits.
#: Two digits is the discriminator against an ordinal, as in "Your #1 fix",
#: which is a number sign and a single digit used as a word.
_TRACKER_REF = re.compile(r"#\d{2,}")

#: A tag opening: a number sign inside a markup or stylesheet blob is a colour
#: literal (``color: #555``), never a tracker reference. Detected by the shape
#: of the surrounding literal rather than by listing the files that contain
#: one, so a new report template is covered the day it lands.
_MARKUP = re.compile(r"<[a-zA-Z/]")


def _is_markup(value: str) -> bool:
    return bool(_MARKUP.search(value)) or "style=" in value


def _docstring_nodes(tree: ast.AST) -> set[int]:
    """Identity of every docstring constant, which this guard does not cover."""
    found: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(
            node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            continue
        body = getattr(node, "body", None)
        if not body:
            continue
        first = body[0]
        if (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
        ):
            found.add(id(first.value))
    return found


def _offending_literals(path: Path) -> list[tuple[int, str]]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    docstrings = _docstring_nodes(tree)
    hits: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
            continue
        if id(node) in docstrings or _is_markup(node.value):
            continue
        match = _TRACKER_REF.search(node.value)
        if match:
            excerpt = node.value[max(0, match.start() - 60):match.end() + 20]
            hits.append((node.lineno, " ".join(excerpt.split())))
    return hits


def _source_files() -> list[Path]:
    return sorted(PACKAGE_ROOT.rglob("*.py"))


def test_the_scan_actually_reaches_the_source_tree():
    """A guard that silently walks an empty tree passes forever."""
    files = _source_files()
    assert len(files) > 100
    assert any(f.name == "cmd_doctor.py" for f in files)


@pytest.mark.parametrize("path", _source_files(), ids=lambda p: p.name)
def test_no_tracker_reference_in_a_printed_string(path):
    hits = _offending_literals(path)
    assert not hits, (
        f"{path.relative_to(PACKAGE_ROOT.parent)} has a tracker reference in a "
        f"string the product can print: {hits}. Describe what it referred to "
        f"instead; the reference means nothing to the user reading it and "
        f"points at an unrelated artifact on GitHub."
    )


def test_the_pattern_separates_references_from_colours_and_ordinals():
    """The discriminators, pinned. If someone loosens the pattern to silence a
    failure, this says which cases it was built to tell apart."""
    # Tracker references: caught.
    assert _TRACKER_REF.search("landed (see #55); ingest fails")
    assert _TRACKER_REF.search("(+36% measured, ticket #59)")
    # An ordinal is a number sign and ONE digit used as a word.
    assert not _TRACKER_REF.search("[bold]Your #1 fix:[/bold]")
    # Colour literals live in markup, which is excluded by literal shape.
    assert _is_markup("<span style='color:#555'>x</span>")
    assert _is_markup("h1 { color: #222; }\n</style>")
    # A plain sentence with a comparison is not markup.
    assert not _is_markup("keep this under <  10 sessions")
