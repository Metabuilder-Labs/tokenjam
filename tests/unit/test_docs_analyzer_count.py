"""The analyzer count claimed in user-facing docs must match the registry.

This number has drifted every time an analyzer landed: the docs said six while
the registry held twelve, and the four places that stated it did not even agree
with each other. A reader cannot check a count, so a stale one is simply a
false claim. Derive it from ``ANALYZER_REGISTRY`` instead of trusting prose.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

import tokenjam.core.optimize.analyzers  # noqa: F401  (populates the registry)
from tokenjam.core.optimize.registry import ANALYZER_REGISTRY

REPO_ROOT = Path(__file__).resolve().parents[2]

#: Every doc that states how many analyzers ship. Historical release-test
#: records under tests/results/ are excluded on purpose: they record what a
#: past run observed and must not be rewritten.
DOCS = (
    "README.md",
    "CLAUDE.md",
    "npm-wrapper/README.md",
    "docs/first-hour.md",
    "docs/agent-capability-matrix.md",
    "docs/cli-reference.md",
)

_NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
    "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20,
}

_ANALYZERS_RE = re.compile(r"\banalyzers?\b", re.IGNORECASE)
_WORD_RE = re.compile(r"[\w-]+")

#: How many words before "analyzers" a count may sit: "twelve analyzers",
#: "twelve cost-optimization analyzers", "all twelve registry-driven analyzers".
_LOOKBACK_WORDS = 3

#: The reversed table-header form: "Analyzers (twelve: downsize / ...)".
_TRAILING_COUNT_RE = re.compile(r"\banalyzers? \(([a-z]+)[:,]", re.IGNORECASE)


def _as_count(token: str) -> int | None:
    """The number a token spells, or None. Spelled-out words only: a bare
    integer near the word "analyzer" is nearly always a line number, a version
    or a percentage, and every claim site here spells the count out."""
    return _NUMBER_WORDS.get(token.lower())


def _claimed_counts(text: str) -> list[int]:
    """Every "<n> analyzers" claim in `text`, as integers.

    Scans backwards from each occurrence of the word rather than matching a
    single regex forwards: "all six analyzers" and "the twelve
    cost-optimization analyzers" put the number a different distance away, and
    a forward pattern with an optional middle greedily swallows the number.
    """
    counts = []
    for match in _ANALYZERS_RE.finditer(text):
        preceding = _WORD_RE.findall(text[:match.start()])[-_LOOKBACK_WORDS:]
        for token in reversed(preceding):
            count = _as_count(token)
            if count is not None:
                counts.append(count)
                break
    for raw in _TRAILING_COUNT_RE.findall(text):
        count = _as_count(raw)
        if count is not None:
            counts.append(count)
    return counts


@pytest.mark.parametrize("relpath", DOCS)
def test_doc_analyzer_count_matches_the_registry(relpath):
    expected = len(ANALYZER_REGISTRY)
    text = (REPO_ROOT / relpath).read_text(encoding="utf-8")
    wrong = [n for n in _claimed_counts(text) if n != expected]
    assert not wrong, (
        f"{relpath} claims {wrong} analyzer(s); the registry holds {expected}: "
        f"{sorted(ANALYZER_REGISTRY)}"
    )


def test_the_registry_is_the_number_the_docs_were_updated_to():
    """A tripwire on the number itself. Adding or removing an analyzer should
    fail here first, as the prompt to go update every doc listed above."""
    assert len(ANALYZER_REGISTRY) == 12
