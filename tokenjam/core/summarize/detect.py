"""Structured-input detection for the summarize surface (stdlib only).

Segments a prompt into PROSE vs PROTECTED structured regions (fenced code, tag
blocks, inline code, templates, markdown tables, and any literal tj-keep markers
the source itself contains), counts prose words, and exposes the worth-it gate. Deliberately stdlib-only — no markdown-it / numpy / yaml — so it adds no
dependency to the base install. The protected-span detectors mirror the
research verifier's regexes so the eventual wrap/restore mechanism stays
consistent with what the detector counts as structure.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# Minimum prose words (after structure is set aside) for a prompt to be worth
# summarizing. Below this the savings don't justify a rewrite. The default;
# overridable via a future [summarize] config section.
MIN_PROSE_WORDS = 100

# Rough English chars-per-token, matching the trim analyzer's basis so token
# estimates are comparable across surfaces.
CHARS_PER_TOKEN = 4

_CODE_FENCE_RE = re.compile(r"```.*?```", re.DOTALL)
_INLINE_RE = re.compile(r"``[^`]+``|(?<!`)`[^`\n]+`(?!`)")
_TAGBLOCK_RE = re.compile(r"<([a-zA-Z][\w:-]*)(?:\s[^>]*?)?>.*?</\1\s*>", re.DOTALL)
_TEMPLATE_RE = re.compile(r"\$\{[^}]*\}|\{\{[^}]*\}\}|\{%[^%]*%\}|<%[^%]*%>")
_TJ_KEEP_MARKER_RE = re.compile(r'<tj-keep id="\d+"[^>]*?(?:/>|>.*?</tj-keep>)', re.DOTALL)
# Markdown table (GitHub-flavored canonical form: leading + trailing pipes, a dash delimiter row).
# Conservative on purpose — prose rarely has a `|...|` line followed by a `|---|---|` row.
_TABLE_RE = re.compile(
    r"^[ \t]*\|.+\|[ \t]*\n"                        # header row
    r"[ \t]*\|(?:[ \t]*:?-+:?[ \t]*\|)+[ \t]*\n"    # delimiter row ( ---|--- )
    r"(?:[ \t]*\|.+\|[ \t]*\n?)*",                  # zero+ body rows
    re.MULTILINE)
_WORD_RE = re.compile(r"\S+")

# Order matters only for the kind label on overlap; merge is longest-wins below.
_DETECTORS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("code_fence", _CODE_FENCE_RE),
    ("table", _TABLE_RE),
    ("tj_keep_marker", _TJ_KEEP_MARKER_RE),
    ("tag_block", _TAGBLOCK_RE),
    ("inline_code", _INLINE_RE),
    ("template", _TEMPLATE_RE),
)


def protected_spans(text: str) -> list[tuple[int, int, str]]:
    """Non-overlapping protected spans ``[(start, end, kind)]``, longest-wins.

    Candidates from all detectors are sorted by (earliest start, longest span)
    and greedily kept if they don't overlap an already-kept span — so an outer
    fenced block wins over an inline span nested inside it.
    """
    spans: list[tuple[int, int, str]] = []
    for kind, rx in _DETECTORS:
        for m in rx.finditer(text):
            spans.append((m.start(), m.end(), kind))
    spans.sort(key=lambda s: (s[0], -(s[1] - s[0])))
    out: list[tuple[int, int, str]] = []
    last = -1
    for start, end, kind in spans:
        if start >= last:
            out.append((start, end, kind))
            last = end
    return out


def prose_text(text: str) -> str:
    """Return ``text`` with every protected span removed (what a summarizer rewrites)."""
    spans = protected_spans(text)
    if not spans:
        return text
    parts: list[str] = []
    cur = 0
    for start, end, _ in spans:
        parts.append(text[cur:start])
        cur = end
    parts.append(text[cur:])
    return "".join(parts)


@dataclass(frozen=True)
class StructureBreakdown:
    """Prose-vs-structure measurement of a single prompt."""

    total_chars: int
    prose_chars: int
    protected_chars: int
    prose_words: int
    protected_blocks: int


def analyze(text: str) -> StructureBreakdown:
    """Measure prose vs protected structure in ``text``."""
    spans = protected_spans(text)
    protected_chars = sum(end - start for start, end, _ in spans)
    prose = prose_text(text)
    return StructureBreakdown(
        total_chars=len(text),
        prose_chars=len(prose),
        protected_chars=protected_chars,
        prose_words=len(_WORD_RE.findall(prose)),
        protected_blocks=len(spans),
    )


def is_candidate(text: str, min_prose_words: int = MIN_PROSE_WORDS) -> bool:
    """True if ``text`` has enough prose to be worth summarizing."""
    return analyze(text).prose_words >= min_prose_words
