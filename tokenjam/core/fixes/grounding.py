"""Say the rule in the user's own terms, using what the analyzer already saw.

The behavioural rules are written generically — "offload context-heavy
sub-tasks (broad file reads, multi-file search, long tool-output loops,
exploratory investigation)" — while the analyzer that produced them knows which
directories the user worked in, which tools drove the tail, and which models
ran it. That is the shape of instruction that gets ignored: the published test
for instruction text is that it be *concrete enough to verify*, contrasting
"API handlers live in `src/api/handlers/`" against "keep files organized". We
were writing the second kind while holding everything needed to write the
first.

**Substitution, not generation.** A template carrying "you did X in Y" beats
both a generic sentence and a model call: it adds no cost, no latency and no
new failure mode, and it cannot hallucinate a directory the user does not have.
Nothing here calls an LLM, and if a pattern genuinely cannot be expressed by
substitution the honest move is to leave the generic wording and say so.

**Grounding lives in RENDERING, never in the record set.** One catalog record
per fix, always — minting a record per user or per finding would make the
catalog and its duplicate lint meaningless overnight. The record owns the
canonical instruction plus the spans that may be replaced; this module fills
them from the observation.

**SPECIFIC MUST NOT MEAN LONGER.** The guidance is "specific *and* concise",
and a longer instruction file reduces adherence — which is the very thing
grounding is for. So a substitution REPLACES a vague span with a concrete one
and never appends an evidence paragraph to a generic sentence. "Offload
context-heavy sub-tasks (broad file reads, ...)" becoming "offload the
multi-file sweeps you run in `optimize/`" is the goal; the generic sentence
plus three lines of observed detail is the failure mode, and it would make the
fix worse while looking like an improvement. :func:`ground` enforces this
directly: a rendering that comes out materially longer is discarded in favour
of the generic text.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from tokenjam.core.fixes.catalog import FixRecord, Substitution

#: How much longer than the generic text a grounded rendering may be before it
#: is rejected. A little slack is needed — naming two directories costs more
#: characters than the word "your projects" — but the whole point is that
#: grounding trades vague words for concrete ones, so anything past this is a
#: substitution that appended rather than replaced.
MAX_GROUNDED_GROWTH = 1.10

#: Above this many named items a list stops reading as concrete and starts
#: reading as noise, which costs length and buys nothing. Past it the rendering
#: names the top few and stops, rather than enumerating everything observed.
MAX_NAMED_ITEMS = 3


@dataclass(frozen=True)
class Evidence:
    """What one analyzer observed, in the terms a rule can be written in.

    Every field is something the analyzer already computed — no field here is
    a new measurement, and nothing is inferred. An empty field means "not
    observed", and a substitution that needs it is skipped rather than filled
    with a plausible-looking guess.
    """

    #: Directory or repo names the flagged sessions ran in, most-affected
    #: first. From ``core/optimize/rule_placement``'s resolved destinations.
    repos: tuple[str, ...] = field(default_factory=tuple)
    #: Tool names that drove the observed pattern (``Read``, ``Grep``, ...).
    tools: tuple[str, ...] = field(default_factory=tuple)
    #: Models involved, for a rule about which model runs what.
    models: tuple[str, ...] = field(default_factory=tuple)
    #: Subagent / dispatch types the observation involved.
    agents: tuple[str, ...] = field(default_factory=tuple)

    def slot(self, name: str) -> str:
        """The rendered value for a substitution slot, or ``""``.

        ``""`` is the honest answer for an unobserved slot and is what makes
        the degrade graceful: the caller skips that substitution and keeps the
        generic span rather than emitting an empty phrase.
        """
        values = {
            "repos": self.repos, "tools": self.tools,
            "models": self.models, "agents": self.agents,
        }.get(name, ())
        return _join(values)


def _join(values: tuple[str, ...]) -> str:
    """Up to :data:`MAX_NAMED_ITEMS` names, in backticks, comma-joined.

    Backticked because these are things the user can look at — a directory, a
    tool, a model id — and the one accent this product spends on "something you
    can type or click" is the same idea in prose.
    """
    cleaned = [str(v).strip() for v in values if str(v or "").strip()]
    if not cleaned:
        return ""
    named = cleaned[:MAX_NAMED_ITEMS]
    quoted = [f"`{v}`" for v in named]
    if len(quoted) == 1:
        return quoted[0]
    return ", ".join(quoted[:-1]) + " and " + quoted[-1]


def _apply(text: str, sub: Substitution, evidence: Evidence) -> str:
    values = {}
    for slot in sub.slots:
        rendered = evidence.slot(slot)
        if not rendered:
            return text          # unobserved -> keep the generic span
        values[slot] = rendered
    replacement = sub.template.format(**values)
    pattern = re.compile(re.escape(sub.find), re.IGNORECASE)
    if not pattern.search(text):
        return text
    return pattern.sub(replacement.replace("\\", "\\\\"), text, count=1)


def ground(record: FixRecord, evidence: Evidence | None = None) -> str:
    """``record``'s text with its vague spans replaced by observed ones.

    Returns the generic text unchanged when there is no evidence, when the
    record declares no substitutions, or when the result would be materially
    longer than the generic form — see :data:`MAX_GROUNDED_GROWTH`. That last
    guard is not a nicety: a grounded rendering that grew is one where a
    substitution appended instead of replacing, and shipping it would reduce
    adherence while looking like an improvement.

    Never strengthens a claim. Substitution names what was OBSERVED; it does
    not add an assertion about what the fix will achieve (Critical Rule 14).
    """
    text = record.text
    if evidence is None or not record.grounding:
        return text
    grounded = text
    for sub in record.grounding:
        grounded = _apply(grounded, sub, evidence)
    if grounded == text:
        return text
    if len(grounded) > len(text) * MAX_GROUNDED_GROWTH:
        # Specific must not mean longer. Discard rather than ship the
        # regression: the generic sentence is worse than a grounded one but
        # better than a grounded one that is also longer.
        return text
    return grounded


__all__ = [
    "MAX_GROUNDED_GROWTH",
    "MAX_NAMED_ITEMS",
    "Evidence",
    "Substitution",
    "ground",
]
