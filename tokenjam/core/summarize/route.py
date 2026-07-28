"""Which route to a smaller instruction file this candidate actually wants.

Compression is one of four routes to Anthropic's published size target, and for
an instruction file it is often the worst one. The other three remove tokens
without making any surviving instruction vaguer:

* **prune** — delete rules that do not earn their place. Anthropic's test:
  :data:`PRUNE_TEST_QUOTE`, plus an explicit exclude list.
* **path-scope** — `.claude/rules/*.md` with `paths:` frontmatter, so the file
  loads only when the model touches matching files (:data:`PATH_SCOPE_QUOTE`).
* **hook** — for an instruction that must run at a fixed point, a hook is
  deterministic where an instruction file is advisory.
* **compress** — what `summarize` does. It is the ONLY one of the four that
  trades adherence for tokens.

**Why that trade is real and why it is a hard constraint.** A long `CLAUDE.md`
is usually long because rules ACCUMULATED, not because the prose is padded.
Compression squeezes words, so each surviving rule gets shorter and vaguer —
and Anthropic's own guidance is that vaguer instructions are followed less
reliably (:data:`SPECIFICITY_QUOTE`). Winning tokens by degrading the thing the
file exists to do is a quality tax, and Critical Rule 26's third gate forbids a
saving that comes from making the agent terser or dumber. So the offer must
never present compression as the only or default route for an instruction file.

**What is diagnosed, and what is not.** `detect.prose_shape` measures the FORM
of a file's prose: how much of it lives in discrete directives versus running
paragraphs. That is a real, verifiable measurement and it separates the
prose-heavy file (a genuine compression candidate) from the rule-heavy one (a
pruning/scoping candidate). It is emphatically NOT a measurement of whether any
individual rule is needed — that is the pruning question, and only the file's
owner can answer it. Where the file is too small to have a shape worth reading,
the diagnosis is WITHHELD rather than guessed: a wrong diagnosis is worse than
none.

Nothing here writes anything. `summarize` mentions the other routes; it does not
build path-scoped rules or hooks, and it never prunes.
"""
from __future__ import annotations

from dataclasses import dataclass

from tokenjam.core.summarize import load_semantics
from tokenjam.core.summarize.detect import prose_shape
from tokenjam.core.summarize.estimate import (
    PUBLISHED_LINE_TARGET,
    PUBLISHED_LINE_TARGET_QUOTE,
    PUBLISHED_LINE_TARGET_SOURCE,
)

#: Anthropic's own wording, verified on the pages cited. Kept as constants so
#: the advice cannot drift from what they actually say — every one of these is
#: quoted TO the user, and a paraphrase attributed to them would be a fabricated
#: citation.
BEST_PRACTICES_SOURCE = "https://code.claude.com/docs/en/best-practices"
MEMORY_SOURCE = PUBLISHED_LINE_TARGET_SOURCE
#: Inner quotes rendered single: this is quoted INSIDE a double-quoted span in
#: every surface that prints it, and nesting doubles reads as broken markup.
PRUNE_TEST_QUOTE = (
    "For each line, ask: 'Would removing this cause Claude to make mistakes?' "
    "If not, cut it."
)
PRUNE_EXCLUDE_QUOTE = (
    "anything Claude can figure out by reading code; standard language "
    "conventions Claude already knows; detailed API documentation (link to docs "
    "instead); information that changes frequently; long explanations or "
    "tutorials; file-by-file descriptions of the codebase; self-evident "
    "practices"
)
PATH_SCOPE_QUOTE = (
    "If your instructions are growing large, use path-scoped rules so "
    "instructions load only when Claude works with matching files."
)
HOOK_QUOTE = (
    "Unlike CLAUDE.md instructions which are advisory, hooks are deterministic "
    "and guarantee the action happens."
)
SPECIFICITY_QUOTE = (
    "The more specific and concise your instructions, the more consistently "
    "Claude follows them."
)

#: Routes. The value is the field a consumer reads; the copy lives below.
ROUTE_COMPRESS = "compress"
ROUTE_PRUNE = "prune"
ROUTE_MIXED = "mixed"
#: An append-only dated log (a `learnings.md`, a dated changelog). Long because
#: entries ACCUMULATED over time, and the remedy is expiry: promote what proved
#: durable, delete what went stale. Compressing a log rewrites history and still
#: keeps every stale entry, so it is close to useless here.
ROUTE_EXPIRE = "expire"
#: Diagnosis withheld — too little prose to read a shape from. Never guessed.
ROUTE_UNDIAGNOSED = "undiagnosed"
#: Not an always-resident instruction file, so the instruction-file argument
#: does not apply and compression is the appropriate lever.
ROUTE_NOT_INSTRUCTION = "not-instruction-file"

#: Share of prose words living in discrete directives, above/below which the
#: file reads as rule-heavy / prose-heavy. Between them it is genuinely mixed
#: and both routes are named rather than one being picked arbitrarily.
_RULE_HEAVY_SHARE = 0.60
_PROSE_HEAVY_SHARE = 0.30
#: Below this many prose units there is no shape to read, so no diagnosis is
#: offered. Guessing here would put a confident wrong recommendation on a card.
_MIN_UNITS_FOR_DIAGNOSIS = 8
#: Share of units opening with a date, above which the file reads as an
#: append-only log rather than an instruction set. Entries are dated headings
#: and their bodies are not, so this is deliberately well below half.
_DATED_LOG_SHARE = 0.15

_QUALITY_TAX = (
    "Compression is the only one of these routes that trades adherence for "
    "tokens: it shortens the surviving rules rather than removing any, and "
    "Anthropic's guidance is that \"{specificity}\" ({memory}) Only the token "
    "reduction is ever claimed as a saving here; whether a rewrite preserved "
    "your meaning is not checked by anything automatic. The structure gate "
    "verifies that code blocks, tables and tags came back verbatim. It does "
    "not read the prose. You do, in the diff, which is why applying is a "
    "dry-run until you pass --go."
)
_PRUNE_ADVICE = (
    "Pruning is the route that costs no specificity: \"{prune_test}\" "
    "({best_practices}) Anthropic's exclude list is {exclude}. "
)
_SCOPE_ADVICE = (
    "Path-scoping is the other: \"{path_scope}\" ({memory}) A rule moved into "
    "`.claude/rules/` with a `paths:` glob stops loading in every session "
    "without a word of it being cut. For an instruction that must run at a "
    "fixed point, a hook is stronger still: \"{hook}\" ({best_practices}) "
    "tokenjam only names these; it does not write them for you. "
)

_SHAPE_RULE_HEAVY = (
    "This file's prose is written as {directive_units:,} discrete directives "
    "carrying {share:.0f}% of its words, averaging {mean:.0f} words each. That "
    "is a file that is long because rules ACCUMULATED, not because the prose is "
    "padded, so compression would make each surviving rule shorter and vaguer "
    "rather than make the file hold fewer rules. Prune or scope it first. "
)
_SHAPE_PROSE_HEAVY = (
    "This file's prose is written as {paragraph_units:,} running paragraphs "
    "carrying {para_share:.0f}% of its words, not as discrete directives. That "
    "is padding rather than accumulated rules, so it is a genuine compression "
    "candidate. "
)
_SHAPE_MIXED = (
    "This file's prose is mixed: {directive_units:,} discrete directives hold "
    "{share:.0f}% of its words and {paragraph_units:,} running paragraphs hold "
    "the rest. Compress the explanation, prune or scope the rules. "
)
_SHAPE_EXPIRE = (
    "{dated:,} of this file's {units:,} prose units open with a date, so it "
    "reads as an append-only LOG rather than an instruction set. A log is long "
    "because entries accumulated over time, and its remedy is expiry: promote "
    "what proved durable into the instructions and delete what went stale. "
    "Compressing it rewrites the history and still keeps every stale entry, so "
    "it is close to useless here. "
)
_SHAPE_UNDIAGNOSED = (
    "This file has too little prose ({units:,} unit(s)) to tell accumulated "
    "rules from padded explanation, so no route is recommended for it rather "
    "than one being guessed. "
)
#: Every shape statement is followed by this, because the measurement is of
#: FORM and a reader will otherwise take it as a verdict on necessity.
_SHAPE_DISCLAIMER = (
    "That is a measurement of how the prose is SHAPED, not of which rules earn "
    "their place; only you can answer the removal test. "
)
_ALREADY_SCOPED = (
    "This rule already declares `paths:` frontmatter, so it loads only when "
    "Claude works with matching files and path-scoping is not a route left to "
    "take here. "
)
#: Deliberately does NOT assert this file is over the target — the diagnosis
#: below is worth reading either way, since a rule-heavy file that is already
#: short is still the wrong thing to compress.
_TARGET_STATEMENT = (
    "Anthropic publishes a size target for an always-resident instruction file: "
    "under {lines} lines (\"{quote}\" {memory}). That target is theirs, not "
    "tokenjam's, and there are four routes to it — prune, path-scope, escalate "
    "to a hook, and compress. summarize only performs the last. "
)
_NOT_INSTRUCTION = (
    "This is not an always-resident instruction file, so the size guidance "
    "published for a CLAUDE.md does not apply to it and compression is the "
    "appropriate lever. The reduction target here is tokenjam's own, and "
    "nothing enforces it. "
)


@dataclass(frozen=True)
class RouteAdvice:
    """Which route this file wants, and the user-facing reasoning for it.

    ``route`` is the machine-readable verdict; ``advice`` is the paragraph a
    surface prints. ``directive_share`` is carried so a consumer can show the
    evidence rather than only the conclusion.
    """

    route: str
    advice: str
    directive_share: float = 0.0
    units: int = 0
    directive_units: int = 0
    paragraph_units: int = 0
    already_path_scoped: bool = False

    def to_dict(self) -> dict:
        return {
            "route": self.route, "advice": self.advice,
            "directive_share": round(self.directive_share, 4),
            "units": self.units, "directive_units": self.directive_units,
            "paragraph_units": self.paragraph_units,
            "already_path_scoped": self.already_path_scoped,
        }


def _quality_tax() -> str:
    return _QUALITY_TAX.format(specificity=SPECIFICITY_QUOTE, memory=MEMORY_SOURCE)


def _alternatives(already_scoped: bool) -> str:
    out = _PRUNE_ADVICE.format(
        prune_test=PRUNE_TEST_QUOTE, best_practices=BEST_PRACTICES_SOURCE,
        exclude=PRUNE_EXCLUDE_QUOTE,
    )
    if already_scoped:
        return out + _ALREADY_SCOPED
    return out + _SCOPE_ADVICE.format(
        path_scope=PATH_SCOPE_QUOTE, memory=MEMORY_SOURCE,
        hook=HOOK_QUOTE, best_practices=BEST_PRACTICES_SOURCE,
    )


def recommend_route(*, text: str, load_class: str) -> RouteAdvice:
    """Diagnose why this file is long, and name the route that follows from it.

    Never raises and never guesses: a file too small to have a readable prose
    shape gets :data:`ROUTE_UNDIAGNOSED` and says so.
    """
    if load_class != load_semantics.ALWAYS:
        return RouteAdvice(route=ROUTE_NOT_INSTRUCTION, advice=_NOT_INSTRUCTION)

    shape = prose_shape(text)
    scoped = load_semantics.has_paths_scope(text)
    head = _TARGET_STATEMENT.format(
        lines=PUBLISHED_LINE_TARGET, quote=PUBLISHED_LINE_TARGET_QUOTE,
        memory=MEMORY_SOURCE,
    )
    def _advice(route: str, body: str) -> RouteAdvice:
        return RouteAdvice(
            route=route, advice=head + body + _alternatives(scoped) + _quality_tax(),
            directive_share=shape.directive_share, units=shape.units,
            directive_units=shape.directive_units,
            paragraph_units=shape.paragraph_units, already_path_scoped=scoped,
        )

    if shape.units < _MIN_UNITS_FOR_DIAGNOSIS or shape.prose_words <= 0:
        return _advice(
            ROUTE_UNDIAGNOSED, _SHAPE_UNDIAGNOSED.format(units=shape.units))

    # Expiry is checked FIRST: a dated log is a log whether its entries happen
    # to be written as bullets or as paragraphs, so the directive share cannot
    # answer this one and would mislabel it either way.
    if shape.dated_share >= _DATED_LOG_SHARE:
        return _advice(ROUTE_EXPIRE, _SHAPE_EXPIRE.format(
            dated=shape.dated_units, units=shape.units) + _SHAPE_DISCLAIMER)

    share = shape.directive_share
    if share >= _RULE_HEAVY_SHARE:
        route, shape_text = ROUTE_PRUNE, _SHAPE_RULE_HEAVY.format(
            directive_units=shape.directive_units, share=share * 100,
            mean=shape.mean_words_per_directive,
        )
    elif share <= _PROSE_HEAVY_SHARE:
        route, shape_text = ROUTE_COMPRESS, _SHAPE_PROSE_HEAVY.format(
            paragraph_units=shape.paragraph_units, para_share=(1 - share) * 100,
        )
    else:
        route, shape_text = ROUTE_MIXED, _SHAPE_MIXED.format(
            directive_units=shape.directive_units, share=share * 100,
            paragraph_units=shape.paragraph_units,
        )
    return _advice(route, shape_text + _SHAPE_DISCLAIMER)
