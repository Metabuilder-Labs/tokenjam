"""Is this section REFERENCE (what exists) or INSTRUCTION (what to do)?

Relocation is only safe on reference material. Move an instruction out of the
file that is supposed to carry it and the agent stops following it — that is a
correctness bug, not a missed saving, and it is silent. Compressing the wrong
section merely wastes a rewrite. **The two error directions therefore cost
wildly different amounts, and this module is built to be wrong in the cheap
direction**: every ambiguous case resolves to INSTRUCTION (leave it in place),
and REFERENCE requires several independent signals to agree.

The reference side takes its definition from Anthropic's own published exclude
list for an instruction file (:data:`EXCLUDE_QUOTE`, quoted verbatim in
``route.PRUNE_EXCLUDE_QUOTE``): things Claude can work out by reading the code,
standard language conventions, detailed API documentation, information that
changes frequently, long explanations and tutorials, file-by-file descriptions
of the codebase. Those are all statements of what EXISTS. An instruction is a
statement of what to DO, and English marks it: deontic modals (must, never,
always, should), negative imperatives, and the second person.

**What is measured.** Features over the section's prose (fenced code, tables and
tag blocks removed first, because a section full of SQL is not thereby
descriptive; inline code is KEPT — see :func:`_prose_keeping_inline_code`):

* ``deontic_share`` — prose units carrying a deontic or second-person marker.
* ``imperative_share`` — prose units opening with a bare imperative verb.
* ``raw_deontic_per_100w`` — the same markers over the RAW text, so a rule
  encoded in a TABLE ROW is still seen. A table-only section has no prose units
  at all, and without this backstop it clears every prose test vacuously.
* ``descriptive_share`` — prose units whose subject is a thing and whose verb
  says what it IS (``X is / holds / lives in / returns / contains``).
* ``artifact_share`` — prose units naming a concrete code artifact (a module
  path, a dotted symbol, a filename with a known source extension). This is
  what separates a file-by-file description from an essay.
* ``link_share`` — prose units that are links out to another document, which is
  the finished form of "link to docs instead".
* ``structure_share`` — how much of the section is table / fenced-code
  structure. An API dump is reference precisely because it has no prose.
* the TITLE, which votes on the reference side and VETOES on the instruction
  side, including when it merely opens with an imperative verb ("Avoid Layout
  Thrashing" is a rule with worked examples).

**The decision.** Instruction vetoes run first and are never outvoted. Only then
does the reference side have to earn a verdict, by getting
:data:`_REQUIRED_REFERENCE_SIGNALS` independent signals to agree — because every
single signal has a plausible instruction-shaped counterexample, and requiring
agreement is what keeps the expensive error direction rare.

**What is NOT measured, on purpose.** Whether the section is CORRECT, whether
anyone reads it, or whether it earns its place. Those are the pruning question
(``route.PRUNE_TEST_QUOTE``) and only the file's owner can answer them.
Relocation never needs the answer, because it deletes nothing.

The thresholds below were tuned against a hand-labelled set of real sections from
real instruction files on a real machine, and both error directions were measured
rather than assumed. The accuracy claim itself lives in
``tests/unit/test_summarize_classify.py``, which pins the property that matters
(no labelled instruction is ever called reference) plus a live regression against
this repository's own committed ``CLAUDE.md`` — the hardest real case there is,
because its module inventory has binding rules interleaved into it at a
granularity finer than any heading.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

#: Anthropic's exclude list, the source of the reference side's definition.
#: Re-exported under a name local to this module so a reader can see what the
#: reference side is derived from without chasing the import.
from tokenjam.core.summarize.route import PRUNE_EXCLUDE_QUOTE as EXCLUDE_QUOTE

__all__ = [
    "EXCLUDE_QUOTE",
    "INSTRUCTION",
    "REFERENCE",
    "UNCLASSIFIED",
    "SectionClassification",
    "classify_section",
]

#: The verdicts. ``INSTRUCTION`` is also the answer for "not confidently
#: reference", because leaving a section in place is always safe.
REFERENCE = "reference"
INSTRUCTION = "instruction"
#: Too little prose to read anything from. Distinct from INSTRUCTION so a caller
#: can tell "judged to be instruction" from "not judged" — but it is treated
#: identically by the relocate path, which only ever acts on REFERENCE.
UNCLASSIFIED = "unclassified"

_WORD_RE = re.compile(r"\S+")

#: Deontic modals, negative imperatives and the second person: the marks of a
#: sentence telling the reader what to do. Any ONE of these in a prose unit
#: makes that unit count as instruction-shaped.
#:
#: **Several obvious candidates are deliberately absent, because in this genre
#: they are overwhelmingly DESCRIPTIVE rather than deontic** and including them
#: made the feature fire on plainly descriptive prose: ``always`` ("**Always
#: synchronous** on the ingest thread"), ``required`` ("auth required"),
#: ``cannot`` / ``can't`` / ``doesn't`` (all three state a limitation of a
#: system, not an obligation on a reader), and the first person plural. Each
#: was checked against real matches before being dropped. ``never`` is kept
#: despite having the same ambiguity ("never propagated") because its directive
#: use ("never modify existing ones") is common enough that dropping it loses
#: more than it gains — and because on the instruction side a false positive is
#: the cheap error.
_DEONTIC_RE = re.compile(
    r"\b(?:must|never|should|shall|do not|don't|"
    r"avoid|ensure|make sure|be sure|forbidden|"
    r"you|your|yours)\b",
    re.IGNORECASE,
)
#: A unit that OPENS with a bare imperative verb. Deliberately a closed list of
#: verbs that only appear sentence-initially as commands in this genre — a list
#: broad enough to include, say, "Note" or "See" would fire on reference prose.
_IMPERATIVE_OPENER_RE = re.compile(
    r"^(?:[-*+]|\d+[.)])?\s*(?:\*\*)?(?:"
    r"use|run|add|write|keep|put|call|check|verify|prefer|avoid|never|always|"
    r"do|don't|read|follow|treat|grep|delete|remove|update|set|pass|return|"
    r"start|stop|commit|push|open|close|file|ask|confirm|scope|split|extract|"
    r"assert|pin|guard|disclose|report|reuse|route|apply|install"
    r")\b",
    re.IGNORECASE,
)
#: "X is / holds / lives in / returns ..." — a statement about what a thing IS.
#: The subject is allowed to be a backticked identifier, which is how this genre
#: writes a module inventory.
_DESCRIPTIVE_RE = re.compile(
    r"\b(?:is|are|was|were|holds?|contains?|lives?|provides?|returns?|exposes?|"
    r"defines?|implements?|includes?|consists?|comprises?|maps?|wraps?|accepts?|"
    r"emits?|stores?|reads?|writes?|owns?|handles?|supports?|has|have)\b",
    re.IGNORECASE,
)
#: A concrete code artifact: a source filename, a module path, a dotted symbol,
#: or an endpoint route. Reference sections name these constantly; instructions
#: name them occasionally.
_ARTIFACT_RE = re.compile(
    r"(?:[\w./-]+\.(?:py|ts|tsx|js|jsx|md|toml|json|sql|sh|ya?ml|rs|go|java|rb)\b)"
    r"|(?:\b\w+(?:\.\w+){2,}\b)"
    r"|(?:\b(?:GET|POST|PUT|PATCH|DELETE)\s+/[\w/{}-]+)"
    r"|(?:(?<![\w/])/(?:api|v\d)/[\w/{}-]+)",
)

#: A markdown link entry — a unit that is mostly a link to somewhere else.
#: "detailed API documentation (link to docs instead)" is on the exclude list by
#: name, and a section built of link entries is the finished form of that advice.
_LINK_UNIT_RE = re.compile(r"^\s*(?:[-*+]|\d+[.)])?\s*\**\[[^\]]+\]\([^)]+\)")

#: Titles that name a body of reference material. Matched on the whole title,
#: case-insensitively, as a whole word so "API" does not fire inside "APIs we
#: must never call".
_REFERENCE_TITLE_RE = re.compile(
    r"\b(?:architecture|architectural overview|data model|data flow|schema|"
    r"module|modules|package layout|directory layout|directory structure|"
    r"file structure|layout|inventory|reference|api|apis|api surface|endpoints|"
    r"rest api|cli commands|commands|glossary|terminology|components|"
    r"key modules|key files|further reading|background|history|appendix|"
    r"examples|packaging|dependencies|tech stack|stack|overview|structure|"
    r"contents|table of contents|subdirectory|subdirectories|registry|catalog|"
    r"events|integrations|options|parameters|fields|types)\b",
    re.IGNORECASE,
)
#: Titles that name a body of INSTRUCTION. This is a VETO, not a vote: whatever
#: the prose looks like, a section called "Critical Rules" or "Anti-patterns"
#: carries rules, and a classifier that can be talked out of that by sentence
#: statistics is the classifier that relocates the rules.
_INSTRUCTION_TITLE_RE = re.compile(
    r"\b(?:rule|rules|anti-?pattern|anti-?patterns|convention|conventions|"
    r"guideline|guidelines|policy|policies|checklist|workflow|workflows|"
    r"process|procedure|protocol|discipline|standards|requirements|"
    r"do|don'?ts?|dos|caveats?|gotchas?|warnings?|constraints?|"
    r"how to|when to|before|after|never|always|must|style|practices|"
    r"best practices|testing|tests|security|review|instructions?|"
    r"agreements?|principles?|norms|expectations?|contract|etiquette)\b",
    re.IGNORECASE,
)
#: A title that OPENS with an imperative verb is itself an instruction — "Avoid
#: Layout Thrashing", "Use defer or async on Script Tags", "Prevent Hydration
#: Mismatch". Leading section numbering (``7.1``) is stripped first. This vetoes
#: for the same reason :data:`_INSTRUCTION_TITLE_RE` does: such a section is a
#: rule with worked examples, and its example code can make it look structural.
_TITLE_NUMBER_RE = re.compile(r"^\s*\d+(?:\.\d+)*[.)]?\s*")
_INSTRUCTION_TITLE_OPENER_RE = re.compile(
    r"^(?:avoid|prevent|use|eliminate|eliminating|avoiding|preventing|using|"
    r"optimize|optimizing|minimize|minimizing|reduce|reducing|ensure|ensuring|"
    r"keep|keeping|prefer|preferring|handle|handling|never|always|do|don'?t|"
    r"write|writing|add|adding|remove|removing|hoist|hoisting|batch|batching|"
    r"defer|deferring|cache|caching|dedupe|deduplicate)\b",
    re.IGNORECASE,
)

#: Below this many prose units there is no shape to read from PROSE. A section
#: can still be classified below it when it is made of tables or code (an API
#: dump is reference precisely because it has no prose), which is what
#: :data:`_STRUCTURE_FLOOR` is for.
_MIN_UNITS = 6
#: Any real density of deontic language vetoes REFERENCE outright.
_DEONTIC_VETO = 0.22
#: ...and REFERENCE additionally requires the density to be genuinely low, not
#: merely under the veto. The gap between the two is deliberate dead space: a
#: section landing in it is left in place with no verdict flipping on a
#: hair's-breadth difference.
_DEONTIC_CEILING = 0.12
#: Deontic markers per 100 words of the RAW section text, including everything
#: the prose-unit features cannot see. A rule encoded in a table row
#: (``| never | ... |``) or in a code comment reaches no prose unit at all, so
#: without this a table-only section could clear every prose test vacuously.
_RAW_DEONTIC_PER_100W_VETO = 1.2
#: Floors for the individual reference signals. None of them is sufficient
#: alone — see :func:`classify_section`, which requires two to agree.
_DESCRIPTIVE_FLOOR = 0.50
_ARTIFACT_FLOOR = 0.50
_LINK_FLOOR = 0.50
#: Share of a section's content that is table / fenced-code / tag-block
#: structure. A section that is mostly a table or an API dump is describing what
#: exists; there is barely a sentence in it to be an instruction.
_STRUCTURE_FLOOR = 0.50
#: A section this close to PURE structure — a table, an inventory, an API dump —
#: counts as two signals on its own, because there is almost no prose left in it
#: that could be carrying an instruction. The deontic vetoes still run first and
#: still apply, over the raw text as well as the prose, so a rules TABLE cannot
#: reach this branch. Measured on the labelled set, every section at or above
#: this share that survived the vetoes was reference.
_STRUCTURE_STRONG = 0.90
#: How many independent reference signals must agree. Two, not one: any single
#: signal has a plausible instruction-shaped counterexample (a rules table, a
#: "read these before starting" link list, a reference-sounding title over a
#: section of hard rules), and requiring agreement is what keeps the expensive
#: error direction rare.
_REQUIRED_REFERENCE_SIGNALS = 2


@dataclass(frozen=True)
class SectionClassification:
    """The verdict plus every feature behind it, so a card can show evidence.

    ``verdict`` is the machine-readable answer; ``reason`` is one sentence
    naming which test decided it, which is what a reviewer of a proposed
    relocation actually needs.
    """

    verdict: str
    reason: str
    units: int
    deontic_share: float = 0.0
    descriptive_share: float = 0.0
    artifact_share: float = 0.0
    imperative_share: float = 0.0
    link_share: float = 0.0
    structure_share: float = 0.0
    raw_deontic_per_100w: float = 0.0
    title_reference: bool = False
    title_instruction: bool = False
    #: Which reference signals agreed — the evidence a reviewer reads.
    signals: tuple[str, ...] = ()

    @property
    def is_reference(self) -> bool:
        return self.verdict == REFERENCE

    def to_dict(self) -> dict:
        return {
            "verdict": self.verdict, "reason": self.reason, "units": self.units,
            "deontic_share": round(self.deontic_share, 4),
            "descriptive_share": round(self.descriptive_share, 4),
            "artifact_share": round(self.artifact_share, 4),
            "imperative_share": round(self.imperative_share, 4),
            "link_share": round(self.link_share, 4),
            "structure_share": round(self.structure_share, 4),
            "raw_deontic_per_100w": round(self.raw_deontic_per_100w, 3),
            "title_reference": self.title_reference,
            "title_instruction": self.title_instruction,
            "signals": list(self.signals),
        }


def _prose_keeping_inline_code(text: str) -> str:
    """``text`` with BLOCK structure removed but inline code left in place.

    ``detect.prose_text`` strips inline spans too, which is right for its job
    (an inline span survives a rewrite verbatim, so it is not compressible
    prose) and wrong for this one: in this genre a module inventory names its
    modules as `` `tokenjam/core/db.py` ``, so stripping inline code deletes
    exactly the evidence :data:`_ARTIFACT_RE` looks for. Measured on this
    machine's own instruction files, the artifact share of a real
    file-by-file architecture section read 0.07 with inline code stripped and
    an order of magnitude higher with it kept — i.e. the feature was blind.
    """
    from tokenjam.core.summarize.detect import protected_spans

    spans = [
        (start, end) for start, end, _ in protected_spans(text)
        if "\n" in text[start:end]
    ]
    if not spans:
        return text
    parts: list[str] = []
    cur = 0
    for start, end in spans:
        parts.append(text[cur:start])
        cur = end
    parts.append(text[cur:])
    return "".join(parts)


def _units(text: str) -> list[str]:
    """Prose units: a bullet / numbered item, or a blank-line-separated paragraph.

    Mirrors ``detect.prose_shape``'s unit rule so the two modules agree on what
    a unit is, but returns the TEXT of each unit because every feature here is
    a per-unit regex rather than a word count.
    """
    from tokenjam.core.summarize.detect import _DIRECTIVE_OPENER_RE

    units: list[str] = []
    start_new = True
    for line in _prose_keeping_inline_code(text).splitlines():
        if not line.strip():
            start_new = True
            continue
        if _DIRECTIVE_OPENER_RE.match(line) or start_new or not units:
            units.append(line.strip())
        else:
            units[-1] = units[-1] + " " + line.strip()
        start_new = False
    # A heading-only unit carries no claim either way; dropping it stops a
    # section of many H3s from reading as neither descriptive nor deontic.
    return [u for u in units if len(_WORD_RE.findall(u)) >= 4]


def _structure_share(text: str) -> float:
    """Share of a section's content that is table / fenced-code / tag-block."""
    from tokenjam.core.summarize.detect import content_chars, protected_spans

    total = content_chars(text)
    if total <= 0:
        return 0.0
    block = sum(
        content_chars(text[start:end]) for start, end, _ in protected_spans(text)
        if "\n" in text[start:end]
    )
    return min(1.0, block / total)


def _raw_deontic_per_100w(text: str) -> float:
    """Deontic markers per 100 words of the RAW section text.

    The prose-unit features cannot see a rule written inside a table row or a
    code comment, and a table-only section has no prose units at all — so
    without this backstop such a section clears every prose test vacuously.
    """
    words = len(_WORD_RE.findall(text))
    if words <= 0:
        return 0.0
    return len(_DEONTIC_RE.findall(text)) * 100.0 / words


def classify_section(title: str, body: str) -> SectionClassification:
    """Decide whether ``body`` under ``title`` describes what EXISTS.

    Returns :data:`INSTRUCTION` for anything not confidently reference, which
    includes every error case: the caller acts only on :data:`REFERENCE`, so a
    failure to classify can never move an instruction.

    Shape of the decision, in order: **instruction vetoes first** (title, then
    every density measure, including one over the raw text that the prose
    features cannot see), and only then does the reference side have to earn a
    verdict by getting :data:`_REQUIRED_REFERENCE_SIGNALS` independent signals
    to agree. A veto is never outvoted; the signals only ever decide between
    REFERENCE and "leave it alone".
    """
    units = _units(body)
    n = len(units)
    title_stripped = _TITLE_NUMBER_RE.sub("", (title or "").strip())
    title_ref = bool(_REFERENCE_TITLE_RE.search(title_stripped))
    title_ins = bool(_INSTRUCTION_TITLE_RE.search(title_stripped)) or bool(
        _INSTRUCTION_TITLE_OPENER_RE.match(title_stripped)
    )

    deontic = sum(1 for u in units if _DEONTIC_RE.search(u)) / n if n else 0.0
    descriptive = sum(1 for u in units if _DESCRIPTIVE_RE.search(u)) / n if n else 0.0
    artifact = sum(1 for u in units if _ARTIFACT_RE.search(u)) / n if n else 0.0
    imperative = sum(1 for u in units if _IMPERATIVE_OPENER_RE.match(u)) / n if n else 0.0
    link = sum(1 for u in units if _LINK_UNIT_RE.match(u)) / n if n else 0.0
    structure = _structure_share(body)
    raw_deontic = _raw_deontic_per_100w(body)

    def _verdict(
        verdict: str, reason: str, signals: tuple[str, ...] = (),
    ) -> SectionClassification:
        return SectionClassification(
            verdict=verdict, reason=reason, units=n, deontic_share=deontic,
            descriptive_share=descriptive, artifact_share=artifact,
            imperative_share=imperative, link_share=link,
            structure_share=structure, raw_deontic_per_100w=raw_deontic,
            title_reference=title_ref, title_instruction=title_ins,
            signals=signals,
        )

    # --- instruction vetoes, in order of how hard they are ------------------#
    # Title first and unconditionally. A section headed "Critical Rules" carries
    # rules however its sentences are shaped, and no measurement may overrule it.
    if title_ins:
        return _verdict(INSTRUCTION, (
            f'titled "{title}", which names or opens as an instruction — never '
            "relocated regardless of how its prose reads"
        ))
    if raw_deontic >= _RAW_DEONTIC_PER_100W_VETO:
        return _verdict(INSTRUCTION, (
            f"{raw_deontic:.1f} deontic markers per 100 words across its whole "
            "text (tables and code included), which is instruction density"
        ))
    if deontic >= _DEONTIC_VETO:
        return _verdict(INSTRUCTION, (
            f"{deontic:.0%} of its prose units carry a deontic or second-person "
            "marker (must / never / should / you), which is instruction"
        ))
    if imperative >= _DEONTIC_VETO:
        return _verdict(INSTRUCTION, (
            f"{imperative:.0%} of its prose units open with a bare imperative "
            "verb, which is instruction"
        ))
    if deontic > _DEONTIC_CEILING:
        return _verdict(INSTRUCTION, (
            f"{deontic:.0%} of its prose units carry a deontic or second-person "
            f"marker — under the {_DEONTIC_VETO:.0%} veto but over the "
            f"{_DEONTIC_CEILING:.0%} a relocation requires, so it stays put"
        ))

    # --- reference side: signals must AGREE ---------------------------------#
    # A share computed over fewer than _MIN_UNITS units is not a measurement —
    # "100% of its prose units" over ONE unit is a single sentence, and reading
    # it as evidence is how a short section with a reference-sounding title gets
    # relocated on the strength of nothing. The prose-derived signals are
    # therefore withheld below the floor; a section can still qualify on
    # structure, which is measured over content rather than over units.
    readable = n >= _MIN_UNITS
    signals: list[str] = []
    if title_ref:
        signals.append(f'titled "{title}"')
    if readable and descriptive >= _DESCRIPTIVE_FLOOR:
        signals.append(
            f"{descriptive:.0%} of its prose units state what something IS")
    if readable and artifact >= _ARTIFACT_FLOOR:
        signals.append(
            f"{artifact:.0%} of its prose units name a concrete code artifact")
    if readable and link >= _LINK_FLOOR:
        signals.append(
            f"{link:.0%} of its prose units are links out to another document")
    if structure >= _STRUCTURE_FLOOR:
        signals.append(
            f"{structure:.0%} of its content is table or code structure")
    if structure >= _STRUCTURE_STRONG:
        # Counts twice: at this share the section is an inventory or an API
        # dump with almost no prose, and the deontic vetoes above have already
        # cleared it of instruction density over its raw text.
        signals.append(
            f"— and at {structure:.0%} structure it is an inventory or API dump "
            "rather than prose")

    # A title alone is not evidence of anything READABLE, so it does not stop a
    # section from being reported as unjudged.
    if len(signals) < _REQUIRED_REFERENCE_SIGNALS:
        if not readable and structure < _STRUCTURE_FLOOR:
            return _verdict(UNCLASSIFIED, (
                f"only {n} prose unit(s) and no structural evidence — too "
                "little to tell a description of what exists from an "
                "instruction, so no verdict is offered"
            ))
        return _verdict(INSTRUCTION, (
            f"{len(signals)} of the {_REQUIRED_REFERENCE_SIGNALS} independent "
            "reference signals needed — not confidently a description of what "
            "exists, so it stays where it is"
        ))
    return _verdict(REFERENCE, (
        f"{', '.join(signals)} — and only {deontic:.0%} of its prose units "
        f"carry a deontic marker ({raw_deontic:.1f} per 100 words overall). "
        "That is a description of what exists, needed when touching the code "
        "rather than on every turn of every session."
    ), tuple(signals))
