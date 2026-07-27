"""Net-of-standing-cost accounting and the write budget for permanent fixes.

Several analyzers propose writing a PERMANENT artifact into the user's agent
files: a rung-1 ``CLAUDE.md`` rule, a rung-2 ``.claude/skills/<slug>/SKILL.md``
note. Those artifacts are not free. A CLAUDE.md rule is re-sent on every future
session, forever, and every write-bearing analyzer used to report its saving
GROSS of that permanent cost with nothing anywhere netting the two. A rule
costing ~200 tokens per session that prevents a 1,500-token failure twice a
month is NET NEGATIVE, and the product would still have shown it as a saving.

This module is the one place that does the arithmetic. It answers four
questions for any set of proposed writes:

1. **What does the artifact cost to keep?**  ``standing_tokens`` prices the
   block against :mod:`tokenjam.core.optimize.projection`'s shared 30-day
   session pace, so the cost and the saving are on the SAME basis.
2. **Is the fix worth its own keep?**  ``net_tokens = gross - standing``.
   A non-positive net suppresses the write outright.
3. **How many permanent rules may be offered at all?**  A ranked budget, in
   per-session standing tokens, sized as a SHARE of the agent-file footprint
   the ``summarize`` analyzer already measures. The product can no longer grow
   a file by 18% in one window while the same report recommends compressing
   it, and past a pathological footprint it offers no new rule at all.
4. **Is this even a real rule?**  A generic "Review examples" placeholder is
   never offered as a permanent artifact.

**Why a ratio and not a break-even call count.** ``core/summarize/delivery.py``
amortizes a ONE-TIME rewrite cost over future calls, so its
``break_even_calls`` is a count. Here the cost is RECURRING: both sides scale
with the same session count, so the count cancels and what survives is a
ratio (``payback_ratio = gross / standing``). Same amortization shape, one
term different; this deliberately reuses that shape rather than inventing a
second scheme.

**The net-negative verdict is a MODEL, not a measurement — and it has one
known blind spot.** Everything above is arithmetic over two modelled
quantities: what a rule costs to carry (session pace x block size) and what
the cluster it addresses is worth gross. The cost side is counted; the
PREVENTION side is not. A rule that stops an agent flailing may save more
than the failure it is priced against — the avoided retry, the abandoned
branch, the human interrupt — and none of that appears in ``gross``. So a
sub-1.0 ``payback_ratio`` means "this rule does not pay for itself *on the
quantities we can count*", never "codifying this was shown to lose money".

The measurement that would settle it was run on 2026-07-26 and returned
neither a validation nor a refutation: the before/after signal is not
obtainable from the current corpus at any usable confidence, and the one
project expected to show a saving measured null. See
``docs/internal/repeat-task-codification-measurement.md``. The suppression
stands — there is no evidence it is wrong, only no evidence it is right — but
no string in this module or its consumers may describe the verdict as an
observed result. Re-open the empirical question only after the
task-statement / model instrumentation lands.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, replace
from typing import Iterable, Sequence

from tokenjam.core.optimize.projection import ProjectionBasis
from tokenjam.core.summarize.detect import CHARS_PER_TOKEN

# --- Standing cost by intervention rung --------------------------------------

#: How much of a written artifact is re-sent on EVERY future session, by rung
#: of the intervention ladder (relearn SPEC 6). This distinction is the whole
#: reason the module is not a flat "count the block" helper:
#:
#:   rung 1  CLAUDE.md note  -> the entire block, re-sent in full, forever.
#:   rung 2  SKILL.md        -> only the frontmatter name + description. The
#:                              harness lists skills by description and loads
#:                              a body only when that skill is invoked.
#:   rung 3+ hook / wrapper / config -> ZERO. A settings file is executed, not
#:                              sent as prompt text, so it has no standing
#:                              per-session token cost at all.
#:
#: So relearn's rung-3 hook families keep their full gross saving and only the
#: rung-1/rung-2 prompt-text writes get netted down. That is the correct
#: answer, not a loophole.
RUNG_CLAUDE_MD = 1
RUNG_SKILL = 2

#: ``relearn_apply.render_skill_content`` truncates the description at 200
#: chars; the name line and frontmatter fences add a little more. The
#: always-loaded surface of a skill is bounded by that, however long its body.
SKILL_ALWAYS_LOADED_CHARS = 260

# --- Write budget -------------------------------------------------------------

#: New permanent rules may grow the user's ALWAYS-LOADED agent-file footprint
#: by at most this share per window. Relative, not absolute, because that is
#: where the real defect lives: appending one 107-token rule to a 25,000-token
#: CLAUDE.md is 0.4% growth and obviously fine, while appending forty-one of
#: them is +18% on every future session and obviously not. An absolute ceiling
#: gets this backwards, killing the lane outright for any real project (a
#: mature CLAUDE.md is routinely tens of thousands of tokens) while waving
#: through unbounded growth on a small one.
AGENT_FILE_GROWTH_SHARE_PER_WINDOW = 0.10

#: Floor under the relative share, so a user whose agent files are nearly empty
#: can still be offered a rule or two (10% of almost nothing is nothing).
MIN_WRITE_BUDGET_TOKENS = 300

#: The absolute stop. Past this the always-loaded footprint is pathological on
#: its own terms — a quarter of a 200k context window spent before the user
#: types anything — and the only honest recommendation is to compress, which
#: is exactly what the summarize findings in the same report say. No new
#: permanent rule is offered at all above this line.
AGENT_FILE_STANDING_CEILING_TOKENS = 50_000

#: Per-lane hard caps. The two write lanes (relearn clusters, cost proposals)
#: are built in separate passes that never see each other's output, so each
#: carries its own explicit ceiling rather than sharing a pool one lane could
#: silently drain. relearn gets the larger share: it is the write-analyzer by
#: design and routinely produces tens of candidate clusters, while the cost
#: lane has at most a handful of write-bearing families.
RELEARN_WRITE_BUDGET_TOKENS = 1_000
RELEARN_MAX_OFFERED_WRITES = 5
COST_WRITE_BUDGET_TOKENS = 500
COST_MAX_OFFERED_WRITES = 3

# --- Value floor ---------------------------------------------------------------

#: A permanent write has to be WORTH ASKING FOR, not merely net-positive.
#:
#: Netting already stops a rule that costs more to keep than it recovers, but
#: "net positive" is a very low bar: a rule clearing its own standing cost by
#: $0.40 passes it, and the product then spends one of five scarce write slots
#: — and one block of the user's permanently-re-sent agent file — on a figure
#: no user would act on. The write budget's slots and the agent file's byte
#: budget are the genuinely scarce resources here, so the floor is applied to
#: what a rule RETURNS, in dollars, not to whether it merely breaks even.
#:
#: Deliberately a floor on the WRITE OFFER, never on the observation. A family
#: below the floor keeps its `past_overspend_*` (that money was spent whether
#: or not we choose to write a rule about it) and keeps its forward net claim
#: and its copyable snippet — it is treated exactly like a deferred write, not
#: like a suppressed one. Zeroing an observed cost because our remedy is too
#: small to bother writing is the "no action for this" / "this was free"
#: conflation the repo CLAUDE.md's rule 32 exists to stop.
#:
#: A candidate with no dollar figure at all is NOT held to this floor — the
#: comparison cannot be made, and inventing a rate to make it would be worse
#: than letting the tokens-only netting stand alone.
MIN_NET_WRITE_USD = 5.0

# --- Quality floor -------------------------------------------------------------

#: A permanent rule needs to actually say something. Shorter than this is a
#: label, not an instruction an agent can follow.
MIN_FIX_CHARS = 40

#: Anchored at the start of the fix text, so these only fire when the fix IS
#: the placeholder rather than when a real fix happens to mention reviewing
#: something (every honesty caveat in the tree says "review ... before
#: applying"; none of those may be mistaken for a placeholder).
#: The first two are anchored on purpose: "review examples" / "TODO" are common
#: enough words that matching them mid-text would false-positive on a genuine
#: rule that merely mentions them. The third is deliberately UNANCHORED, and
#: must stay that way — the text handed to :func:`is_placeholder_fix` on the
#: live relearn path is not the raw fix but the fully RENDERED artifact (a
#: marker comment, a heading, the fix, an evidence line; see
#: ``relearn_apply.artifact_for_rung``), so the placeholder sentence sits in the
#: middle of the block and an anchored pattern never sees it. Anchoring this one
#: for symmetry silently switches the whole quality floor off for that path.
_PLACEHOLDER_PATTERNS = (
    re.compile(r"^\s*review\s+examples?\b", re.IGNORECASE),
    re.compile(r"^\s*(tbd|todo|n/?a)\b", re.IGNORECASE),
    re.compile(r"no known fix template matched", re.IGNORECASE),
)

# --- Suppression reasons (stable strings; surfaced on the card) ----------------

REASON_PLACEHOLDER = (
    "No specific fix was derived for this pattern, so there is no rule worth "
    "writing permanently. The example sessions are still listed for review."
)
#: Deliberately hedged ("on the quantities we can count"): this is a modelled
#: comparison, not an observed one, and it does not count what a rule saves by
#: PREVENTING a flail. See the module docstring's blind-spot note.
REASON_NET_NEGATIVE = (
    "On the quantities we can count, a permanent rule for this would cost more "
    "to keep than it recovers: it is re-sent on every future session, and that "
    "modelled standing cost exceeds the estimated saving. Not counted: what the "
    "rule might save by preventing the failure in the first place."
)
REASON_BUDGET_FULL = (
    "Deferred: this window's permanent-rule budget is already allocated to "
    "higher-value fixes. The recommendation is still shown as a copyable "
    "snippet, and this becomes offerable once an earlier rule is applied or "
    "the agent files are compressed."
)
REASON_FAMILY_MERGED = (
    "Covered by the single rule offered for this family: one block is written "
    "for the whole family rather than one per cluster."
)
#: Not a suppression either — the same shape as REASON_BUDGET_FULL. The saving
#: is real and the snippet stays copyable; we simply will not spend a permanent
#: block of an always-re-sent file on a return this small.
REASON_BELOW_VALUE_FLOOR = (
    f"Below the ${MIN_NET_WRITE_USD:.0f} floor for a permanent rule: this one "
    "clears its own standing cost but not by enough to be worth a block in a "
    "file re-sent on every future session. The recommendation is still shown "
    "as a copyable snippet, and what the recurrence already cost is reported "
    "in full."
)
#: Not a suppression: the write IS offered, but the disclosure says the netting
#: could not run. Reaching a suppression verdict from a MISSING measurement
#: would kill real fixes for the sin of being unquantified, which is strictly
#: worse than the gross-claim bug this module exists to fix.
BASIS_NOT_PRICEABLE = (
    "standing cost not netted: this finding carries no token estimate to net "
    "the permanent rule's own per-session cost against"
)
REASON_CEILING_REACHED = (
    "Your agent files already carry more standing per-session context than the "
    "budget allows to grow. Compress them (see the summarize findings) before "
    "adding new permanent rules."
)

#: The same five verdicts as a SHORT label, for a dense list where the full
#: sentence would be a paragraph per row (a real corpus gates ~50 of 55
#: clusters, so the long form turns the list into a wall of text). Keyed by the
#: exact long string, so a reason added above without a short form here
#: degrades to a generic label instead of silently rendering blank.
#:
#: Same rule as the long forms: this module owns the wording, so the CLI list,
#: the Review inbox row and the API payload cannot drift into three different
#: names for one flag. The long sentence still renders wherever there is room
#: for it (the expanded card, `--json`).
#: Populated at the bottom of this section, once every long reason is defined.
REASON_SHORT_BY_REASON: dict[str, str] = {}


def short_reason(reason: str) -> str:
    """The compact label for a suppression reason, for list rendering.

    Returns "" for an empty reason (a write that WAS offered) so callers can
    treat falsy as "nothing to say", and a generic label for an unrecognised
    reason rather than dropping the fact that something gated the write.
    """
    reason = str(reason or "").strip()
    if not reason:
        return ""
    return REASON_SHORT_BY_REASON.get(reason, "no permanent fix offered")


REASON_SHORT_BY_REASON.update({
    REASON_PLACEHOLDER: "no fix template derived",
    REASON_NET_NEGATIVE: "a rule would cost more to keep than it returns",
    REASON_BUDGET_FULL: "deferred — this window's rule budget is spent",
    REASON_FAMILY_MERGED: "covered by another cluster's rule",
    REASON_CEILING_REACHED: "blocked — agent files already too large to grow",
    REASON_BELOW_VALUE_FLOOR: (
        f"returns under ${MIN_NET_WRITE_USD:.0f} — not worth a permanent rule"
    ),
})

#: The net-negative sentence as it read BEFORE the modelled/observed hedging.
#: A stored proposal cache written by the previous build still carries this
#: exact string, and the Review inbox reads that cache directly — without the
#: alias every net-negative row would degrade to the generic label for one
#: upgrade cycle, which is precisely the invisibility the short label exists to
#: fix. Delete once no supported release can still have written it.
LEGACY_REASON_NET_NEGATIVE = (
    "A permanent rule for this would cost more to keep than it recovers: it is "
    "re-sent on every future session, and that standing cost exceeds the "
    "estimated saving."
)
REASON_SHORT_BY_REASON[LEGACY_REASON_NET_NEGATIVE] = (
    REASON_SHORT_BY_REASON[REASON_NET_NEGATIVE]
)

#: Named once so every card, basis string and test refers to the same sentence.
STANDING_COST_BASIS = (
    "net of the fix's own standing cost: a permanent rule is re-sent on every "
    "future session, so its block is priced against the same 30-day session "
    "pace the saving is projected on, and subtracted"
)


def tokens_from_chars(chars: int) -> int:
    """Chars to tokens on the shared summarize-pipeline constant, never a
    local ``/4``. The one conversion every figure in this module goes through
    so a size in chars and a size in tokens can never drift apart."""
    return math.ceil(max(0, int(chars)) / CHARS_PER_TOKEN)


def artifact_tokens(text: str) -> int:
    """Token estimate for a written artifact."""
    return tokens_from_chars(len(text)) if text else 0


def standing_tokens_per_session(rung: int, text: str) -> int:
    """Tokens this artifact adds to EVERY future session, by rung.

    See ``RUNG_CLAUDE_MD`` above for why rung 3 and up are zero: a hook or a
    config edit is never sent to the model as prompt text.
    """
    tokens = artifact_tokens(text)
    if rung >= 3:
        return 0
    if rung == RUNG_SKILL:
        return min(tokens, tokens_from_chars(SKILL_ALWAYS_LOADED_CHARS))
    return tokens


def is_placeholder_fix(text: str) -> bool:
    """True when the fix text is a generic placeholder, not a real rule.

    The quality floor: a permanent artifact is never offered for one of these.
    """
    stripped = (text or "").strip()
    if len(stripped) < MIN_FIX_CHARS:
        return True
    return any(p.search(stripped) for p in _PLACEHOLDER_PATTERNS)


def _always_loaded_chars(candidate: object) -> int:
    """How much of one catalog file is re-sent on EVERY session.

    The same always-loaded model :func:`standing_tokens_per_session` charges a
    NEW write at, applied to the files that already exist. Measuring the
    existing footprint one way and pricing the new write another would let the
    budget compare two different quantities.

    Prefers the MEASURED frontmatter size the ``summarize`` scan now carries
    (``always_resident_chars``, from ``core/summarize/load_semantics`` — the
    single source of truth for the read side too). Falls back to classifying
    the path and applying the ``SKILL_ALWAYS_LOADED_CHARS`` bound for a
    candidate that predates that field, so an older serialized report still
    sizes the budget the way it always did.
    """
    from tokenjam.core.summarize import load_semantics

    total_chars = int(getattr(candidate, "total_chars", 0) or 0)
    path = str(getattr(candidate, "path", "") or "")
    measured = getattr(candidate, "always_resident_chars", None)
    load_class = load_semantics.classify(path)
    if load_class not in load_semantics.ON_DEMAND_CLASSES:
        return total_chars
    if measured:
        return min(int(measured), total_chars)
    return min(total_chars, SKILL_ALWAYS_LOADED_CHARS)


def measured_agent_file_tokens(summarize_finding: object | None) -> int | None:
    """Standing per-session token footprint of the user's agent files, read
    straight off the ``summarize`` analyzer's own filesystem scan.

    This is the cross-reference the two halves of the loop were missing.
    ``summarize``'s catalog covers exactly the files the write-analyzers append
    to (``CLAUDE.md``, ``.claude/rules/*.md``, ``.claude/skills/*/SKILL.md``,
    ``.claude/commands/*.md``, ``.claude/agents/*.md`` and the ``~/.claude``
    equivalents), so its measured size IS the budget the writers spend against.
    Without it the product could recommend adding dozens of rules and
    compressing the file they land in, in the same report.

    ``None`` when the analyzer did not run or found nothing, which leaves the
    lane cap intact rather than inventing a footprint. The figure is a floor,
    not a total: the scan only lists files with prose left to compress, so an
    already-tight file contributes nothing. Erring low means erring toward MORE
    headroom, which is the conservative direction for a suppression gate.
    """
    if summarize_finding is None:
        return None
    candidates = getattr(summarize_finding, "candidates", None) or []
    if not candidates:
        return None
    total_chars = sum(_always_loaded_chars(c) for c in candidates)
    if total_chars <= 0:
        return None
    return tokens_from_chars(total_chars)


@dataclass(frozen=True)
class WriteBudget:
    """How much new permanent standing cost this window may take on.

    ``existing_tokens`` is the always-loaded agent-file footprint the
    ``summarize`` analyzer measured; ``budget_tokens`` is how many per-session
    standing tokens of NEW rules that permits, and ``max_writes`` how many
    rules. ``ceiling_reached`` is True only at the absolute stop, where the
    footprint is pathological on its own terms and the honest recommendation is
    to compress rather than add. Both are zero there.
    """

    budget_tokens: int
    max_writes: int
    existing_tokens: int
    ceiling_reached: bool


def build_write_budget(
    *,
    lane_budget_tokens: int,
    lane_max_writes: int,
    existing_agent_file_tokens: int | None = None,
) -> WriteBudget:
    """Size this lane's write budget against the measured agent-file footprint.

    ``existing_agent_file_tokens`` is the standing per-session cost the user's
    agent files ALREADY carry, as measured by the ``summarize`` analyzer.
    ``None`` (analyzer not run, scan failed) leaves the lane cap intact rather
    than inventing a footprint, which is the only honest default: an unmeasured
    file is not evidence of a full one.

    Three terms, in order of who wins: the absolute ceiling zeroes the budget
    outright; otherwise the growth share caps it relative to what is already
    there; the floor keeps a nearly-empty file from being budgeted to nothing;
    and the lane cap bounds the whole thing regardless.
    """
    lane_cap = max(0, lane_budget_tokens)
    if existing_agent_file_tokens is None:
        return WriteBudget(
            budget_tokens=lane_cap, max_writes=max(0, lane_max_writes),
            existing_tokens=0, ceiling_reached=False,
        )
    existing = max(0, int(existing_agent_file_tokens))
    if existing >= AGENT_FILE_STANDING_CEILING_TOKENS:
        return WriteBudget(
            budget_tokens=0, max_writes=0, existing_tokens=existing,
            ceiling_reached=True,
        )
    allowed_growth = max(
        MIN_WRITE_BUDGET_TOKENS, int(existing * AGENT_FILE_GROWTH_SHARE_PER_WINDOW),
    )
    return WriteBudget(
        budget_tokens=min(lane_cap, allowed_growth),
        max_writes=max(0, lane_max_writes),
        existing_tokens=existing,
        ceiling_reached=False,
    )


@dataclass(frozen=True)
class WriteCandidate:
    """One proposed permanent write, as the budget pass needs to see it.

    ``family`` groups candidates that would write the SAME rule: relearn's
    known-failure ``family_key``, or an analyzer name in the cost lane, where
    every cluster writes one shared intro note. One block is offered per
    family, never one per cluster.

    ``gross_tokens``/``gross_usd`` are the proposal's WINDOW-basis figures, the
    same basis :class:`~tokenjam.core.optimize.projection.ProjectionBasis`
    counted its sessions over. Mixing a window-scoped saving against a 30-day
    standing cost is the exact time-basis error this whole module exists to
    stop, so the basis is stated here and honoured throughout.
    """

    key: str
    family: str
    rung: int
    artifact_text: str
    gross_tokens: int
    gross_usd: float | None = None
    #: How many of the window's sessions actually re-send this artifact. A
    #: user-global rule is paid on every session; a project-scoped one only on
    #: that project's. Charging a project rule against the whole corpus would
    #: net a cluster-scoped saving against a corpus-scoped cost, which is the
    #: same basis mismatch in a different costume. ``None`` falls back to the
    #: basis's own session count (the user-global case).
    exposure_sessions: int | None = None


@dataclass(frozen=True)
class WriteDecision:
    """The budget pass's verdict for one candidate.

    Every token figure is on the WINDOW basis (see ``WriteCandidate``), except
    ``standing_tokens_per_session``, which is basis-free by construction: a
    caller netting a 30-day field multiplies it by
    ``ProjectionBasis.projected_sessions`` instead of ``.sessions``, and both
    nets stay consistent because those two counts differ by exactly the
    projection ratio.

    ``claimed_*`` is what the proposal is allowed to report as recoverable. It
    is the NET figure whenever a fix is genuinely available (offered now, or
    deferred to a later window but still copyable today), and ZERO when there
    is no fix worth doing at all: a placeholder, or a family whose rule costs
    more to keep than it recovers. A proposal can never claim more than this.
    """

    key: str
    offered: bool
    reason: str
    standing_tokens_per_session: int
    standing_tokens: int
    standing_usd: float | None
    net_tokens: int
    net_usd: float | None
    payback_ratio: float | None
    net_negative: bool
    #: The session count the standing cost was charged against, so a caller
    #: netting a 30-day field rescales the SAME exposure by the projection
    #: ratio instead of reaching for the window-wide count again.
    exposure_sessions: int
    claimed_tokens: int
    claimed_usd: float | None
    #: True when there is no fix worth doing at all, so the proposal may claim
    #: NOTHING on any basis (a placeholder, or a family whose rule costs more
    #: to keep than it recovers). Distinct from a merely deferred write, whose
    #: saving is real and whose snippet is still copyable today.
    claim_suppressed: bool
    basis: str


def _rate_per_token(gross_usd: float | None, gross_tokens: int) -> float | None:
    """The candidate's OWN implied $/token, so the standing cost is priced at
    exactly the rate its saving was priced at. Never a default rate: without
    both sides the netting stays tokens-only (the dollar figure is then left
    untouched rather than adjusted with an invented rate)."""
    if gross_usd is None or gross_tokens <= 0:
        return None
    return gross_usd / gross_tokens


def _unpriceable_decision(candidate: WriteCandidate) -> WriteDecision:
    """Verdict for a candidate whose saving carries no token figure.

    The rule's standing cost is knowable; the saving it is meant to beat is
    not, so no comparison exists and none is invented. The write stays on
    offer (ranked last, since a zero net cannot outrank a measured one) and the
    figures pass through untouched, with the basis saying plainly that the
    netting did not run.
    """
    return WriteDecision(
        key=candidate.key, offered=True, reason="",
        standing_tokens_per_session=standing_tokens_per_session(
            candidate.rung, candidate.artifact_text,
        ),
        standing_tokens=0, standing_usd=None,
        net_tokens=0, net_usd=candidate.gross_usd,
        payback_ratio=None, net_negative=False, exposure_sessions=0,
        claimed_tokens=candidate.gross_tokens, claimed_usd=candidate.gross_usd,
        claim_suppressed=False, basis=BASIS_NOT_PRICEABLE,
    )


def _decide_family(
    members: Sequence[WriteCandidate], basis: ProjectionBasis,
) -> tuple[WriteCandidate, WriteDecision, list[WriteDecision]]:
    """Price a family's single block and split the verdict across its members.

    The representative is the family's largest-gross member, and its artifact
    is the one block that would actually be written. Siblings carry no standing
    cost of their own precisely because nothing extra is written for them, so
    the family's total net is ``sum(gross) - one block``, which is the true
    arithmetic. Choosing the largest member as representative is deliberately
    conservative: if even it cannot pay for the block, no smaller sibling can.
    """
    ordered = sorted(members, key=lambda c: (-c.gross_tokens, c.key))
    rep = ordered[0]
    if rep.gross_tokens <= 0:
        # The representative's saving is unpriceable, so nothing can be netted.
        # Its siblings still lose the write for the ordinary reason — the block
        # is written ONCE for the family — and must NOT each be handed an
        # `offered=True` unpriceable verdict of their own, which would put the
        # same artifact text on offer N times and break the one-block-per-family
        # invariant. Same shape as the priceable branch's sibling handling.
        return rep, _unpriceable_decision(rep), [
            replace(
                _unpriceable_decision(c),
                offered=False, reason=REASON_FAMILY_MERGED,
            )
            for c in ordered[1:]
        ]
    # The family's gross is the SUM across every member, not just the
    # representative's own — siblings carry no standing cost of their own
    # (the block is written once for the whole family), but their savings
    # still count toward whether that ONE shared write is worth keeping.
    # Netting only `rep.gross_*` understates a multi-member family and can
    # withhold the write even when the family's COMBINED return clears the
    # value floor below.
    family_gross_tokens = sum(c.gross_tokens for c in ordered)
    per_session = standing_tokens_per_session(rep.rung, rep.artifact_text)
    exposure = max(
        rep.exposure_sessions if rep.exposure_sessions is not None else basis.sessions, 0,
    )
    standing = per_session * exposure
    # Priced at the REPRESENTATIVE's own implied rate (its `gross_usd` is the
    # only figure guaranteed present, since the unpriceable branch above
    # already returned early otherwise), then applied to the family total so
    # every member's dollars are converted at one consistent rate rather than
    # summing separately-derived per-candidate USD figures.
    rate = _rate_per_token(rep.gross_usd, rep.gross_tokens)
    family_gross_usd = family_gross_tokens * rate if rate is not None else None
    standing_usd = standing * rate if rate is not None else None
    net_tokens = family_gross_tokens - standing
    net_usd = (family_gross_usd - standing_usd) if (
        family_gross_usd is not None and standing_usd is not None
    ) else family_gross_usd
    payback = (family_gross_tokens / standing) if standing > 0 else None
    net_negative = net_tokens <= 0
    clamped_net_usd = None if net_usd is None else max(net_usd, 0.0)

    rep_decision = WriteDecision(
        key=rep.key, offered=not net_negative,
        reason=REASON_NET_NEGATIVE if net_negative else "",
        standing_tokens_per_session=per_session,
        standing_tokens=standing, standing_usd=standing_usd,
        net_tokens=max(net_tokens, 0), net_usd=clamped_net_usd,
        payback_ratio=payback, net_negative=net_negative, exposure_sessions=exposure,
        claimed_tokens=0 if net_negative else max(net_tokens, 0),
        claimed_usd=(
            None if family_gross_usd is None
            else (0.0 if net_negative else (clamped_net_usd or 0.0))
        ),
        claim_suppressed=net_negative, basis=STANDING_COST_BASIS,
    )
    siblings = [
        WriteDecision(
            key=c.key, offered=False,
            reason=REASON_NET_NEGATIVE if net_negative else REASON_FAMILY_MERGED,
            standing_tokens_per_session=0,
            standing_tokens=0, standing_usd=0.0 if c.gross_usd is not None else None,
            net_tokens=c.gross_tokens, net_usd=c.gross_usd,
            payback_ratio=None, net_negative=net_negative, exposure_sessions=0,
            claimed_tokens=0 if net_negative else c.gross_tokens,
            claimed_usd=(
                None if c.gross_usd is None else (0.0 if net_negative else c.gross_usd)
            ),
            claim_suppressed=net_negative, basis=STANDING_COST_BASIS,
        )
        for c in ordered[1:]
    ]
    return rep, rep_decision, siblings


def allocate_writes(
    candidates: Iterable[WriteCandidate], budget: WriteBudget, basis: ProjectionBasis,
) -> dict[str, WriteDecision]:
    """Rank every proposed permanent write by net value and offer only what
    the budget affords. Returns one decision per candidate, keyed by ``key``.

    Order of operations, each stage feeding the next:

    1. **Quality floor** — a placeholder fix is dropped before anything else;
       it never reaches the ranking and never claims a saving.
    2. **Family merge** — same-family candidates collapse onto ONE block.
    3. **Net value** — the block's standing cost over the projected sessions is
       subtracted; a non-positive net suppresses the family's write.
    4. **Budget** — surviving families are ranked by net tokens, highest first,
       and offered until either the standing-token budget or the write count
       is exhausted. Anything past that is deferred, not dropped: the saving is
       real and the fix stays copyable, so it keeps its net claim.
    """
    decisions: dict[str, WriteDecision] = {}
    families: dict[str, list[WriteCandidate]] = {}

    for candidate in candidates:
        if is_placeholder_fix(candidate.artifact_text):
            decisions[candidate.key] = WriteDecision(
                key=candidate.key, offered=False, reason=REASON_PLACEHOLDER,
                standing_tokens_per_session=0, standing_tokens=0, standing_usd=None,
                net_tokens=0, net_usd=0.0 if candidate.gross_usd is not None else None,
                payback_ratio=None, net_negative=False, exposure_sessions=0,
                claimed_tokens=0, claimed_usd=0.0 if candidate.gross_usd is not None else None,
                claim_suppressed=True, basis=STANDING_COST_BASIS,
            )
            continue
        families.setdefault(candidate.family, []).append(candidate)

    ranked: list[tuple[WriteCandidate, WriteDecision]] = []
    for members in families.values():
        rep, rep_decision, siblings = _decide_family(members, basis)
        for sibling in siblings:
            decisions[sibling.key] = sibling
        if rep_decision.net_negative:
            decisions[rep.key] = rep_decision
            continue
        # The VALUE floor, applied after netting and before the budget so a
        # too-small family never consumes one of the scarce write slots a
        # bigger one could use. Shaped exactly like the budget-full verdict
        # below — not offered, but the claim and the snippet both survive —
        # because a small return is still a real one; see MIN_NET_WRITE_USD.
        # Skipped entirely when the family carries no dollar figure: there is
        # no comparison to make and none is invented.
        if (
            rep_decision.net_usd is not None
            and rep_decision.net_usd < MIN_NET_WRITE_USD
        ):
            decisions[rep.key] = replace(
                rep_decision, offered=False, reason=REASON_BELOW_VALUE_FLOOR,
            )
            continue
        ranked.append((rep, rep_decision))

    ranked.sort(key=lambda pair: (-pair[1].net_tokens, pair[0].key))
    spent_tokens = 0
    offered_count = 0
    for rep, decision in ranked:
        # Spent in PER-SESSION tokens, the same unit the ceiling and the
        # summarize-measured footprint are in. Charging the window TOTAL here
        # would make the budget shrink as the user's session count grows, which
        # is backwards: a busier user is exactly who a good permanent rule pays
        # off for. The window total is what the SAVING is netted against; the
        # per-session figure is what the FILE has to carry.
        fits = (
            offered_count < budget.max_writes
            and spent_tokens + decision.standing_tokens_per_session <= budget.budget_tokens
        )
        if fits:
            spent_tokens += decision.standing_tokens_per_session
            offered_count += 1
            decisions[rep.key] = decision
            continue
        decisions[rep.key] = WriteDecision(
            key=decision.key, offered=False,
            reason=REASON_CEILING_REACHED if budget.ceiling_reached else REASON_BUDGET_FULL,
            standing_tokens_per_session=decision.standing_tokens_per_session,
            standing_tokens=decision.standing_tokens, standing_usd=decision.standing_usd,
            net_tokens=decision.net_tokens, net_usd=decision.net_usd,
            payback_ratio=decision.payback_ratio, net_negative=False,
            exposure_sessions=decision.exposure_sessions,
            claimed_tokens=decision.claimed_tokens, claimed_usd=decision.claimed_usd,
            claim_suppressed=False, basis=decision.basis,
        )
    return decisions
