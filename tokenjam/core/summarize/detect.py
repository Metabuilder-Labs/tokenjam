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
class LineBreakdown:
    """How a prompt's LINES divide between protected structure and prose.

    The chars-based :class:`StructureBreakdown` is what pricing runs on; this is
    the line-based view a *line* target needs (Anthropic publishes one for
    always-resident instruction files, see
    ``core/summarize/estimate.PUBLISHED_LINE_TARGET``).

    **Only a span that occupies whole lines counts as a protected LINE.** A
    fenced block, a table, or a multi-line tag block is restored verbatim and
    therefore pins every line it covers. An INLINE span (`` `like this` ``) does
    not: it must survive, but it can be packed into a much shorter sentence, so
    the line it currently sits on is compressible prose. Counting it as a
    protected line was measurably wrong — a prose-heavy `CLAUDE.md` with
    backticks throughout read as majority-structure and the line target was
    withheld from exactly the files it exists for.

    ``protected_lines + prose_lines == total_lines`` always holds.
    """

    total_lines: int
    protected_lines: int
    prose_lines: int


def line_breakdown(text: str) -> LineBreakdown:
    """Split ``text``'s lines into protected-structure lines and prose lines."""
    if not text:
        return LineBreakdown(total_lines=0, protected_lines=0, prose_lines=0)
    # Line index of every character offset, derived once from the newline
    # positions so a large file costs one pass rather than one per span.
    newlines = [i for i, ch in enumerate(text) if ch == "\n"]
    total = len(newlines) + (0 if text.endswith("\n") else 1)

    def _line_of(offset: int) -> int:
        lo, hi = 0, len(newlines)
        while lo < hi:                       # bisect_left over the newline offsets
            mid = (lo + hi) // 2
            if newlines[mid] < offset:
                lo = mid + 1
            else:
                hi = mid
        return lo

    touched: set[int] = set()
    for start, end, _ in protected_spans(text):
        if "\n" not in text[start:end]:
            continue                         # inline span: its line is still prose
        first = _line_of(start)
        last = _line_of(max(start, end - 1))
        touched.update(range(first, last + 1))
    protected = len({ln for ln in touched if ln < total})
    return LineBreakdown(
        total_lines=total, protected_lines=protected, prose_lines=total - protected,
    )


_WS_RUN_RE = re.compile(r"\s+")


def content_chars(text: str) -> int:
    """Characters of ACTUAL CONTENT: whitespace runs collapsed to one space.

    Every token estimate in this package must be built on this rather than on
    ``len(text)``, because a raw character count books **reflow as compression**.
    Un-hard-wrapping a deliberately hard-wrapped file deletes one newline per
    line and changes nothing a tokenizer bills, yet a `len()` delta reports it
    as a saving. Measured on real instruction files, that artifact was the
    majority of the claimed reduction — 71% and 68% on the two worst — and it
    is worst precisely on the files whose authors wrapped them ON PURPOSE, so
    the product was selling the destruction of a formatting convention as a
    saving. Collapsing whitespace first makes that reduction structurally
    unclaimable: reflow moves this number by zero.
    """
    return len(_WS_RUN_RE.sub(" ", text or "").strip())


@dataclass(frozen=True)
class StructureBreakdown:
    """Prose-vs-structure measurement of a single prompt.

    ``*_chars`` are RAW character counts, kept because they describe the file
    as it sits on disk. Every token estimate uses the ``*_content_chars``
    counterparts instead — see :func:`content_chars` for why the difference is
    load-bearing rather than cosmetic.
    """

    total_chars: int
    prose_chars: int
    protected_chars: int
    prose_words: int
    protected_blocks: int
    #: Whitespace-normalized counterparts. Defaulted so any existing
    #: construction of this dataclass keeps working; `analyze` always sets them.
    total_content_chars: int = 0
    prose_content_chars: int = 0


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
        total_content_chars=content_chars(text),
        prose_content_chars=content_chars(prose),
    )


#: A prose line that OPENS a discrete instruction: a markdown bullet, a numbered
#: item, or a heading. Deliberately narrow — these are the three shapes a rule
#: is actually written in; anything else is treated as running prose.
_DIRECTIVE_OPENER_RE = re.compile(r"^[ \t]*(?:[-*+]|\d+[.)]|#{1,6})[ \t]+\S")

#: A unit that OPENS with a date. An append-only log (a `learnings.md`, a
#: dated changelog) is long because entries accumulated over time, and its own
#: remedy is expiry — promote what proved durable, delete what went stale —
#: not compression. Deliberately narrow: an ISO date or a `YYYY-MM` prefix near
#: the start of the unit, which is how a dated entry is actually written.
_DATED_UNIT_RE = re.compile(r"^[^\n]{0,40}?\b(20\d{2})-(0[1-9]|1[0-2])(-\d{2})?\b")


@dataclass(frozen=True)
class ProseShape:
    """How a prompt's prose is SHAPED: discrete directives vs running prose.

    A long instruction file is long for one of two very different reasons, and
    the remedy differs. Prose-heavy means padded explanation, which summarizing
    genuinely compresses. Rule-heavy means directives ACCUMULATED, and squeezing
    words there makes each surviving rule shorter and vaguer rather than fewer.

    **This measures FORM, not NECESSITY.** It can say a file is written as 120
    bullets averaging 14 words; it cannot say which of them earn their place.
    That is the pruning question, and only the file's owner can answer it. Any
    copy derived from this must keep the distinction — a diagnosis of shape
    presented as a judgement about need would be exactly the wrong guess.

    A "unit" starts at a directive opener or after a blank line, so a wrapped
    bullet stays one unit and a blank-line-separated paragraph stays one unit.
    """

    units: int
    directive_units: int
    paragraph_units: int
    prose_words: int
    directive_words: int
    paragraph_words: int
    #: Units whose text opens with a date — the signature of an append-only log,
    #: whose remedy is expiry rather than compression.
    dated_units: int = 0

    @property
    def dated_share(self) -> float:
        return self.dated_units / self.units if self.units else 0.0

    @property
    def directive_share(self) -> float:
        """Share of prose words living inside directives (0.0 with no prose)."""
        return self.directive_words / self.prose_words if self.prose_words else 0.0

    @property
    def mean_words_per_directive(self) -> float:
        return self.directive_words / self.directive_units if self.directive_units else 0.0


def prose_shape(text: str) -> ProseShape:
    """Measure how ``text``'s prose divides into directives vs running prose."""
    prose = prose_text(text)
    units: list[tuple[bool, int, bool]] = []    # (is_directive, words, is_dated)
    start_new = True
    for line in prose.splitlines():
        words = len(_WORD_RE.findall(line))
        if not line.strip():
            start_new = True                    # blank line closes the current unit
            continue
        opens = bool(_DIRECTIVE_OPENER_RE.match(line))
        if opens or start_new or not units:
            units.append((opens, words, bool(_DATED_UNIT_RE.match(line.strip()))))
        else:
            is_dir, had, dated = units[-1]
            units[-1] = (is_dir, had + words, dated)   # continuation of a wrapped unit
        start_new = False
    d_units = [w for is_dir, w, _ in units if is_dir]
    p_units = [w for is_dir, w, _ in units if not is_dir]
    return ProseShape(
        units=len(units), directive_units=len(d_units), paragraph_units=len(p_units),
        prose_words=sum(w for _, w, _ in units),
        directive_words=sum(d_units), paragraph_words=sum(p_units),
        dated_units=sum(1 for _, _, dated in units if dated),
    )


def is_candidate(text: str, min_prose_words: int = MIN_PROSE_WORDS) -> bool:
    """True if ``text`` has enough prose to be worth summarizing."""
    return analyze(text).prose_words >= min_prose_words
