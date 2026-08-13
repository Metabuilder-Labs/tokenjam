"""`print_capped_table` — the shared bounded-listing renderer.

Several `tj` screens print one block or row per entity with no cap, so their
length is set by how long the user has been running tj rather than by the
terminal. That degrades worst for the heaviest user, who is exactly the one
with something to find in the output.

The contract these pin:

* the summary/total line renders ABOVE the rows, never after them;
* at most `limit` rows render;
* whatever was cut is named, together with the LITERAL command that reaches
  it — a cap with no way past it is just missing data;
* nothing is cut silently, and a listing with nothing in it says so rather
  than printing a bare header over nothing.
"""
from __future__ import annotations

from tokenjam.utils.formatting import CAPPED_ROW_LIMIT, print_capped_table


def _flat(out: str) -> str:
    """Collapse Rich's terminal-width wrapping so a phrase can be asserted
    whole regardless of the runner's column count."""
    return " ".join(out.split())


def _rows(n: int):
    return [(f"row-{i:02d}", str(i)) for i in range(n)]


def test_caps_rows_and_names_the_command_that_reaches_the_rest(capsys):
    hidden = print_capped_table(
        ("NAME", "N"), _rows(25), limit=10, more_command="tj thing --name <id>",
        noun="thing",
    )
    out = _flat(capsys.readouterr().out)

    assert hidden == 15
    assert "row-09" in out
    assert "row-10" not in out
    assert "+15 more things" in out
    assert "tj thing --name <id>" in out


def test_summary_renders_above_the_rows(capsys):
    print_capped_table(
        ("NAME", "N"), _rows(3), more_command="tj thing", summary="3 things.",
    )
    lines = [ln for ln in capsys.readouterr().out.splitlines() if ln.strip()]

    assert "3 things." in lines[0]
    assert "NAME" in lines[1]


def test_no_trailer_when_nothing_was_cut(capsys):
    hidden = print_capped_table(
        ("NAME", "N"), _rows(4), limit=10, more_command="tj thing",
    )
    out = _flat(capsys.readouterr().out)

    assert hidden == 0
    assert "more" not in out
    assert "tj thing" not in out


def test_singular_noun_when_exactly_one_row_is_cut(capsys):
    print_capped_table(
        ("NAME", "N"), _rows(11), limit=10, more_command="tj thing", noun="agent",
    )
    out = _flat(capsys.readouterr().out)

    assert "+1 more agent." in out
    assert "agents" not in out


def test_empty_listing_says_so_rather_than_printing_a_bare_header(capsys):
    hidden = print_capped_table(
        ("NAME", "N"), [], more_command="tj thing", empty_note="Nothing tracked yet.",
    )
    out = _flat(capsys.readouterr().out)

    assert hidden == 0
    assert "Nothing tracked yet." in out
    assert "NAME" not in out


def test_default_limit_is_a_screenful(capsys):
    print_capped_table(("NAME", "N"), _rows(50), more_command="tj thing")
    out = capsys.readouterr().out

    assert f"row-{CAPPED_ROW_LIMIT - 1:02d}" in out
    assert f"row-{CAPPED_ROW_LIMIT:02d}" not in out
