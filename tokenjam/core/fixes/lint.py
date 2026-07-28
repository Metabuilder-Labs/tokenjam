"""Properties every fix text must hold, checked mechanically.

The reason this is a lint and not a review checklist: the defects it catches
already shipped, twice each, past readers who were looking. The identical
sizing contradiction lived in two constants at once; a rubric shipped an escape
hatch the agent grades itself against; a card shipped a fix whose own text said
to do nothing. None of those are subtle once stated — they are just invisible
when the policy has no single home to be checked in.

Four of the five properties come from what Anthropic publishes about
instruction text (keep it short, keep it concrete, do not contradict yourself,
escalate to a hook when it must run at a fixed point). The fifth is ours and is
the one that matters most here:

    **A fix may not re-license the behaviour its own analyzer bills for.**

`past_overspend_usd` is the maximum its analyzer knows: applying the fix should
ERASE the number it found. A fix that gives a pass to the exact shape the
analyzer flagged leaves the number where it was, which is a correctness bug
dressed as copy. That is the whole D-class, and it is mechanical: the record
names the shapes it must not excuse, and this checks the text against them.
"""
from __future__ import annotations

import re
from collections.abc import Sequence

from tokenjam.core.fixes.catalog import FIX_CATALOG, FixRecord

#: Anthropic's published guidance for instruction files. A rule longer than
#: this competes with everything after it for attention.
MAX_FIX_LINES = 200

#: Below this a "fix" is a label, not an instruction an agent can follow. Same
#: floor the write budget's quality gate applies.
MIN_FIX_CHARS = 40

#: An escape hatch the agent grades ITSELF against. "Unless the subtask
#: genuinely needs deep reasoning" asks the dispatching agent to rate its own
#: task's difficulty, and an agent asked that question answers yes — so the
#: exception swallows the rule. An exception has to be stated in terms an
#: outside reader could check (a condition present in the dispatch, a stated
#: fact), never in terms of the agent's own judgement of difficulty or need.
_SELF_GRADED = re.compile(
    r"\b(?:unless|except\s+when|only\s+when|if)\b[^.]{0,80}\b"
    r"(?:genuinely|truly|really|actually)\s+(?:needs?|requires?|warrants?)",
    re.IGNORECASE,
)
#: The same trap without an intensifier: "when the task needs deep reasoning".
_SELF_GRADED_JUDGEMENT = re.compile(
    r"\b(?:unless|except\s+when|only\s+when)\b[^.]{0,80}\b"
    r"(?:needs?|requires?)\s+(?:deep|serious|real|careful)\s+"
    r"(?:reasoning|thought|thinking|judgement|judgment)",
    re.IGNORECASE,
)

#: A fix whose own text says no action is required. Legitimate as an
#: OBSERVATION, never as an offered write — a card whose fix says to do nothing
#: must not occupy an apply slot. The record has to admit it via
#: ``advisory_only`` so every surface can gate on a field instead of re-reading
#: the prose.
#:
#: The alternation tolerates a LIST of nouns ("no rule or hook is needed"),
#: which is how the sentence is naturally written. The first draft matched only
#: a single noun and reported a correctly-marked advisory record as
#: mis-marked — a false positive that would have taught the next author to
#: reach for an exception rather than to fix the text.
_NO_ACTION_NOUN = r"(?:hook|rule|action|fix|change|edit)"
_SAYS_NO_ACTION = re.compile(
    rf"no\s+{_NO_ACTION_NOUN}(?:\s*(?:,|/|or|and)\s*{_NO_ACTION_NOUN})*\s+"
    r"(?:is\s+|are\s+)?(?:needed|required)"
    r"|advisory\s+awareness\s+only"
    r"|there\s+is\s+no\s+change\s+to\s+make"
    r"|nothing\s+to\s+do\s+here",
    re.IGNORECASE,
)

#: Text that must run at a fixed point in the loop — a checkpoint, every time,
#: before a specific tool — is a hook's job. Delivered as a rule it is a
#: request the agent may or may not honour at that instant.
_FIXED_POINT = re.compile(
    r"\b(?:before every|after every|on every (?:tool|command|bash)"
    r"|block the (?:tool|command)|refuse to run)\b",
    re.IGNORECASE,
)


def lint_fix(record: FixRecord) -> list[str]:
    """Every property ``record`` violates, as human-readable strings.

    An empty list is a pass. Returns all violations rather than the first, so a
    fix is corrected in one pass instead of being re-run per complaint.
    """
    text = record.text or ""
    problems: list[str] = []

    if len(text.strip()) < MIN_FIX_CHARS:
        problems.append(
            f"too short to be an instruction ({len(text.strip())} chars, "
            f"minimum {MIN_FIX_CHARS}) — this is a label, not a fix",
        )
    lines = text.splitlines()
    if len(lines) > MAX_FIX_LINES:
        problems.append(
            f"{len(lines)} lines exceeds the {MAX_FIX_LINES}-line ceiling for "
            "instruction text — a rule this long competes with everything "
            "after it",
        )
    if _SELF_GRADED.search(text) or _SELF_GRADED_JUDGEMENT.search(text):
        problems.append(
            "contains an escape hatch the agent grades itself against "
            "(\"unless it genuinely needs ...\"): an agent asked to rate its "
            "own task's difficulty answers yes, so the exception swallows the "
            "rule. State the exception as a condition an outside reader can "
            "check.",
        )
    # THE load-bearing check. See the module docstring.
    for shape in sorted(record.must_not_relicense):
        if shape and shape.lower() in text.lower():
            problems.append(
                f"re-licenses the behaviour its analyzer bills for: the text "
                f"contains {shape!r}, which gives a pass to the shape "
                f"{record.key!r} is written to route away from. Applying this "
                "fix would leave the number that produced it where it was.",
            )
    says_no_action = bool(_SAYS_NO_ACTION.search(text))
    if says_no_action and not record.advisory_only:
        problems.append(
            "the text says no action is needed, but the record is not marked "
            "advisory_only — a fix that says to do nothing must never be "
            "OFFERED as a write, though its observation still stands.",
        )
    if record.advisory_only and not says_no_action:
        problems.append(
            "marked advisory_only but the text reads as an actionable "
            "instruction — say plainly that no action is required, or offer "
            "the action.",
        )
    if _FIXED_POINT.search(text) and record.delivery == "claude_md_rule":
        problems.append(
            "asks for behaviour at a FIXED POINT in the loop but is delivered "
            "as an instruction-file rule, which is read once at session start "
            "and honoured at the agent's discretion. Escalate to a hook.",
        )
    return problems


#: Two records this similar are two wordings of one instruction. Measured on
#: content words (the shared vocabulary is what makes two rules read as
#: duplicates), so a shared boilerplate caveat does not trip it while a
#: genuinely restated instruction does.
NEAR_DUPLICATE_OVERLAP = 0.6

#: Words that carry no instruction and would inflate any two texts' overlap.
_STOPWORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "для", "for",
    "from", "in", "is", "it", "its", "not", "of", "on", "or", "own", "so",
    "than", "that", "the", "their", "them", "then", "there", "this", "to",
    "up", "was", "were", "what", "when", "which", "with", "you", "your",
})


def _content_words(text: str) -> set[str]:
    return {
        w for w in re.findall(r"[a-z][a-z0-9_-]{2,}", (text or "").lower())
        if w not in _STOPWORDS
    }


def _overlap(left: str, right: str) -> float:
    """CONTAINMENT over content words, not Jaccard.

    Jaccard divides by the union, so it collapses when the two texts differ in
    LENGTH — and that is precisely the shape of the case this check exists for.
    Measured against the real defect: the driver-role wording scored 22% Jaccard
    against the canonical rule it duplicated, far under any usable threshold, so
    a Jaccard check would have reported the three-analyzers-one-rule case clean.
    A check that cannot catch the defect it was written for is worse than no
    check, because it certifies the problem as absent.

    Containment (``|a ∩ b| / min(|a|, |b|)``) asks the right question instead:
    is the shorter text's instruction largely contained in the longer one? Two
    rules saying the same thing at different lengths are still two rules.
    """
    a, b = _content_words(left), _content_words(right)
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


def lint_duplicates() -> dict[str, list[str]]:
    """Pairs of records carrying substantially the same instruction.

    THE check that catches the three-analyzers-one-rule case. Three fixes told
    the agent to delegate context-heavy work to a subagent in three
    separately-authored wordings, so three near-identical blocks could land in
    one CLAUDE.md — and the write budget's one-block-per-family rule could not
    see it, because they were three families from three analyzers.

    The harm is not untidiness. Length and redundancy REDUCE adherence, so
    writing a rule three times makes it less likely to be followed than writing
    it once: each analyzer's duplicate actively defeats the others.
    """
    out: dict[str, list[str]] = {}
    records = sorted(FIX_CATALOG.values(), key=lambda r: r.key)
    for i, left in enumerate(records):
        for right in records[i + 1:]:
            score = _overlap(left.text, right.text)
            if score < NEAR_DUPLICATE_OVERLAP:
                continue
            note = (
                f"carries substantially the same instruction as {right.key!r} "
                f"({score:.0%} content-word overlap). Two wordings of one rule "
                "can both land in the same file, and length plus redundancy "
                "REDUCE adherence — so writing it twice is worse than writing "
                "it once. Collapse them onto one record and let each analyzer "
                "reference it."
            )
            out.setdefault(left.key, []).append(note)
    return out


#: Two SENTENCES this similar, in one artifact, are one instruction written
#: twice. Deliberately the same threshold as the record-level check: the
#: question is identical, only the unit of comparison changes.
COMPOSED_SENTENCE_OVERLAP = 0.6

#: Below this a sentence carries too little to compare. Containment divides by
#: the SHORTER text, which is what makes it the right metric for two rules of
#: different lengths — and also what makes it unstable when one side is tiny: a
#: six-word clause needs only four words in common with a long paragraph to
#: score 67%. Measured on the real catalog, the floor is what separates signal
#: from noise: "this file is too large to read whole" scored 60% against the
#: offload rule on the words `file`, `read` and `context` alone, while every
#: genuine duplicate found here runs to eighteen content words or more.
#:
#: Erring high is the right direction. A false positive teaches the next author
#: to reach for an exception rather than to fix the text, and an exception in
#: this lint is permanent; a missed short clause is caught by the pairwise
#: whole-record check, which has no such floor.
MIN_COMPARABLE_WORDS = 8

#: Verbs that make a sentence an INSTRUCTION rather than an explanation. Only
#: instruction sentences are compared: two records may freely restate the same
#: REASON (the cost mechanism, the caveat) — that is context, and it is what
#: makes each record readable alone. What must never appear twice is the thing
#: the user is being asked to do.
_DIRECTIVE = re.compile(
    r"\b(?:pin|set|add|use|prefer|route|offload|delegate|dispatch|create|"
    r"define|write|run|replace|trim|adopt|read|right-size|default|check|"
    r"issue|pass|include|reserve|template|consider|block|point)\b",
    re.IGNORECASE,
)

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n{2,}")

#: The delivery label many fix texts open with — "CLAUDE.md/skill note:",
#: "PostToolUseFailure hook (Edit/MultiEdit):". It names the MECHANISM, which is
#: already a structured field on the record, so leaving it in the comparison
#: makes every record of one delivery kind look partly like every other.
#:
#: That is not a hypothetical tidy-up: without this, two unrelated Read rules
#: ("Read takes a file path" / "this file is too large to read whole") scored
#: 71% purely on their shared prefix. A false positive here is worse than a gap,
#: because it teaches the next author to reach for an exception rather than to
#: fix the text — so the label comes off before anything is measured, and what
#: remains is compared on its own merits.
_DELIVERY_LABEL = re.compile(
    r"^\s*(?:CLAUDE\.md(?:/skill)?\s+note|Skill/scoped\s+note|"
    r"\w*Hook|PostToolUseFailure\s+hook|PreToolUse\s+hook)"
    r"(?:\s*\([^)]*\))?\s*:\s*",
    re.IGNORECASE,
)


def _strip_delivery_label(sentence: str) -> str:
    return _DELIVERY_LABEL.sub("", sentence, count=1)


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_SPLIT.split(text or "") if s.strip()]


def _is_instruction(sentence: str) -> bool:
    return bool(_DIRECTIVE.search(sentence))


def lint_composed_text(artifact: str, parts: Sequence[tuple[str, str]]) -> list[str]:
    """Instructions that appear twice once ``parts`` are composed into one block.

    THE check :func:`lint_duplicates` structurally cannot make. That one is
    PAIRWISE OVER WHOLE RECORDS, so it can only see redundancy that exists
    between two records taken entire — and this defect class does not have that
    shape. It already shipped in the smaller form: consolidating three duplicate
    wordings folded a model-pinning sentence into the offload rule, and the
    compound card renders offload and right-sizing back to back, so the pinning
    instruction appeared twice inside one written block. Whole-record
    containment scored the pair at 42%, under threshold, because each record was
    only PARTLY redundant. It was found by rendering the artifact and reading
    it, which is not a check.

    Comparing at sentence level asks the question the reader actually asks: is
    any single thing I am being told to do stated here more than once? Only
    sentences carrying a directive are compared — two records restating the same
    REASON is fine and often necessary, since each has to read sensibly alone.

    ``parts`` is ``(label, text)`` per composed piece; labels appear in the
    violation so the author knows which two to collapse. Comparison is ACROSS
    parts only: repetition inside one record is that record's own problem and
    :func:`lint_fix` owns it.
    """
    problems: list[str] = []
    indexed = [
        (label, sentence, bare)
        for label, text in parts
        for sentence in _sentences(text)
        for bare in (_strip_delivery_label(sentence),)
        if _is_instruction(bare)
        and len(_content_words(bare)) >= MIN_COMPARABLE_WORDS
    ]
    for i, (left_label, left, left_bare) in enumerate(indexed):
        for right_label, right, right_bare in indexed[i + 1:]:
            if left_label == right_label:
                continue
            score = _overlap(left_bare, right_bare)
            if score < COMPOSED_SENTENCE_OVERLAP:
                continue
            problems.append(
                f"{artifact!r} states one instruction twice ({score:.0%} "
                f"containment): {left_label} says {left!r} and {right_label} "
                f"says {right!r}. A user reading the composed block is told the "
                "same thing in two places, and length plus redundancy REDUCE "
                "adherence — so give the instruction one owner and let the "
                "other record stop at its own job.",
            )
    return problems


def composable_groups() -> dict[str, tuple[FixRecord, ...]]:
    """Record sets the product can compose into ONE artifact.

    Derived rather than hand-listed, so a record added tomorrow is covered
    without anyone remembering to extend a list — the hand-listed version is
    how the pairwise check came to have a blind spot in the first place.

    Two records can land in the same written block when they share a delivery
    kind (they go in the same KIND of file), a persona (the same user is shown
    both) and an analyzer (the same finding hands both out). That is exactly the
    "per delivery kind, per analyzer, per destination" surface.
    """
    groups: dict[str, list[FixRecord]] = {}
    for record in sorted(FIX_CATALOG.values(), key=lambda r: r.key):
        for analyzer in sorted(record.analyzers):
            for persona in sorted(record.personas):
                groups.setdefault(
                    f"{analyzer}/{persona}/{record.delivery}", [],
                ).append(record)
    return {k: tuple(v) for k, v in groups.items() if len(v) > 1}


def lint_composed() -> dict[str, list[str]]:
    """Every composable artifact's duplicated instructions, keyed by artifact."""
    out: dict[str, list[str]] = {}
    for name, records in composable_groups().items():
        problems = lint_composed_text(
            name, [(r.key, r.text) for r in records],
        )
        if problems:
            out[name] = problems
    return out


def lint_catalog() -> dict[str, list[str]]:
    """Every catalogued fix's violations, keyed by fix key. Empty dict = clean.

    Includes the cross-record duplicate check, so a caller cannot get a clean
    bill of health while two records restate one instruction.
    """
    out = {}
    for key, record in FIX_CATALOG.items():
        problems = lint_fix(record)
        if problems:
            out[key] = problems
    for key, problems in lint_duplicates().items():
        out.setdefault(key, []).extend(problems)
    # Kept alongside the pairwise check rather than replacing it: the pairwise
    # one is cheap and catches the simpler shape (two records that are wholly
    # one instruction), while this catches the shape it cannot see.
    for name, problems in lint_composed().items():
        out.setdefault(f"composed:{name}", []).extend(problems)
    return out


__all__ = [
    "COMPOSED_SENTENCE_OVERLAP",
    "MAX_FIX_LINES",
    "MIN_COMPARABLE_WORDS",
    "MIN_FIX_CHARS",
    "NEAR_DUPLICATE_OVERLAP",
    "composable_groups",
    "lint_catalog",
    "lint_composed",
    "lint_composed_text",
    "lint_duplicates",
    "lint_fix",
]
