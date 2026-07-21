"""No UNQUALIFIED tracker reference may reach a string the product prints.

The problem is ambiguity, not references. A bare ``#99999`` resolves against
whatever repository the reader happens to be looking at, and internal and
public number spaces genuinely collide, so the same token can be correct here
and confidently wrong once pasted into an issue, a screenshot or another
repo's thread. A printed string travels further than any other kind, straight
out of the terminal and into all three.

So this guard bans the BARE form and permits a qualified one, either
``owner/repo#99999`` or a full URL. A pointer to real background is useful
information the user should get; deleting it is a loss. An earlier version of
this guard banned every reference, and it promptly pushed a scrub into
removing a correct public issue link from the spans-column-statistics warning,
which is precisely the failure mode a blunt rule produces. Prefer the URL form
in anything printed: a terminal linkifies nothing, and a URL survives being
pasted anywhere.

This guard also exists because a test used to assert the OPPOSITE: it required
an internal id to be PRESENT in the doctor's MCP warning, so the suite
actively defended the leak, and anyone who removed it was told by CI that they
had broken something. See the leak-pinning entry in CLAUDE.md.

Scope is deliberately the printed surface only. Comments and docstrings are
developer-facing prose where a bare reference is legitimate shorthand, so
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

#: An ``owner/repo`` immediately before the number sign qualifies the
#: reference, so it resolves to the same artifact from anywhere. A full URL
#: contains no number sign at all and never reaches this check.
_QUALIFIER = re.compile(r"[A-Za-z0-9._-]+/[A-Za-z0-9._-]+\Z")


def _is_qualified(value: str, ref_start: int) -> bool:
    """True when the reference at ``ref_start`` names its repository."""
    return bool(_QUALIFIER.search(value[:ref_start]))

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
        for match in _TRACKER_REF.finditer(node.value):
            if _is_qualified(node.value, match.start()):
                continue
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
        f"{path.relative_to(PACKAGE_ROOT.parent)} has an UNQUALIFIED tracker "
        f"reference in a string the product can print: {hits}. A bare number "
        f"resolves against whatever repo the reader is looking at. If it names "
        f"a real public artifact, qualify it (a full "
        f"https://github.com/<owner>/<repo>/issues/<n> URL reads best in a "
        f"terminal). If it names our own queue, describe what it referred to "
        f"instead."
    )


def test_the_pattern_separates_references_from_colours_and_ordinals():
    """The discriminators, pinned. If someone loosens the pattern to silence a
    failure, this says which cases it was built to tell apart."""
    # Tracker references: caught.
    assert _TRACKER_REF.search("landed (see #99999); ingest fails")
    assert _TRACKER_REF.search("(+36% measured, ticket #99998)")
    # An ordinal is a number sign and ONE digit used as a word.
    assert not _TRACKER_REF.search("[bold]Your #1 fix:[/bold]")
    # Colour literals live in markup, which is excluded by literal shape.
    assert _is_markup("<span style='color:#555'>x</span>")
    assert _is_markup("h1 { color: #222; }\n</style>")
    # A plain sentence with a comparison is not markup.
    assert not _is_markup("keep this under <  10 sessions")


def test_a_qualified_reference_passes_where_its_bare_twin_fails(tmp_path):
    """THE WHOLE RULE, as one pair. Same number, same sentence: unqualified is
    ambiguous and rejected, qualified is unambiguous and kept. The guard must
    never make deleting a true reference look like the correct move."""
    sentence = 'MSG = "column statistics are corrupt. Background: {ref}."\n'
    bare = tmp_path / "bare.py"
    bare.write_text(sentence.format(ref="see issue #99999"), encoding="utf-8")
    qualified = tmp_path / "qualified.py"
    qualified.write_text(
        sentence.format(ref="see Metabuilder-Labs/tokenjam#99999"), encoding="utf-8",
    )
    url = tmp_path / "url.py"
    url.write_text(
        sentence.format(
            ref="see https://github.com/Metabuilder-Labs/tokenjam/issues/99999",
        ),
        encoding="utf-8",
    )

    assert _offending_literals(bare), "a bare reference must be rejected"
    assert not _offending_literals(qualified), "owner/repo#n must be permitted"
    assert not _offending_literals(url), "a full URL must be permitted"
