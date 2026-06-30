"""Unit tests for summarize structure detection + the worth-it gate."""
from __future__ import annotations

from tokenjam.core.summarize import detect


def test_code_fence_excluded_from_prose():
    text = "word " * 50 + "\n```python\n" + "code_token " * 100 + "\n```\n" + "word " * 50
    b = detect.analyze(text)
    assert b.protected_blocks == 1           # the fenced block
    assert b.prose_words == 100              # only the 100 prose words, not the 100 code tokens


def test_unclosed_fence_counts_as_prose():
    # Documents the stdlib-regex behavior: a lone ``` with no closer is NOT a
    # protected span (so it can't swallow the rest of the file). Edge: real
    # markdown would treat it as code-to-EOF; we don't, by design (no md parser).
    text = "```\n" + "word " * 120
    b = detect.analyze(text)
    assert b.protected_blocks == 0
    assert b.prose_words >= 100


def test_longest_span_wins_on_overlap():
    text = ("intro " * 8 + "<example>" + "x " * 20 + "`inline` " + "y " * 20
            + "</example>" + " outro " * 8)
    b = detect.analyze(text)
    assert b.protected_blocks == 1           # tag block absorbs the inline span (no double count)
    assert b.prose_words == 16               # only the 8 intro + 8 outro words


def test_gate_boundary():
    assert detect.is_candidate("w " * 99) is False
    assert detect.is_candidate("w " * 100) is True


def test_structure_only_and_empty():
    assert detect.analyze("").prose_words == 0
    assert detect.is_candidate("") is False
    assert detect.is_candidate("```\n" + "x " * 500 + "\n```") is False   # all code → ~0 prose


def test_markdown_table_protected():
    table = "| Sym | Meaning |\n|-----|---------|\n| a | b |\n| c | d |\n"
    text = "word " * 60 + "\n\n" + table + "\nword " * 60
    b = detect.analyze(text)
    assert b.protected_blocks == 1                              # the table is one protected block
    assert tuple(k for _, _, k in detect.protected_spans(text)) == ("table",)
    assert b.prose_words == 120                                 # table cells are NOT counted as prose
    assert "Sym" not in detect.prose_text(text)                 # …and never reach the summarizer


def test_tj_keep_marker_literal_protected():
    text = 'Docs mention <tj-keep id="99"/> as literal marker syntax. ' + "word " * 120
    spans = detect.protected_spans(text)
    assert tuple(k for _, _, k in spans) == ("tj_keep_marker",)
    assert '<tj-keep id="99"/>' not in detect.prose_text(text)
