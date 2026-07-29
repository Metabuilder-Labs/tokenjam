"""``list`` — every permanent rule the last analyzer run is offering, and where.

Four analyzers write rules (``downsize``, ``resend``, ``subagent``, ``relearn``).
Before this module each hand-rolled its own text and its own single-destination
write, and the only surface that showed them was the Review inbox, one card per
analyzer. A card can express one write. It cannot express "this rule, into these
4 of your 11 projects, here is each diff, apply selectively" — which is what a
placed rule actually is, and why this lifecycle exists beside the inbox rather
than inside it.

Reads only CONFIG and the already-computed proposal cache (``relearn_store``),
never the DB: the analyzers do the expensive work on their own schedule
(``core/optimize/runner``), and re-running a sweep to answer "what rules are on
offer" would put an analyzer pass on a user-facing command.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Any

from tokenjam.core.config import TjConfig
from tokenjam.core.rulewrite.kinds import DEFAULT_DELIVERY
from tokenjam.core.rulewrite.legacy import delivery_from_legacy_record
from tokenjam.core.rulewrite.types import RuleDestination, RuleWrite

#: The analyzers whose fix is a permanent rule. Not a copy of a gating map —
#: membership here is decided by fix SHAPE, and a name absent from the stored
#: proposals simply contributes nothing, so a persona gate upstream
#: (``runner.PERSONA_DISABLED_ANALYZERS``) removes an analyzer from this
#: surface without any edit here.
RULE_WRITING_ANALYZERS = ("downsize", "resend", "subagent", "relearn")

#: Stated once, so the CLI, the UI and the payload cannot word it three ways.
ALREADY_APPLIED_REASON = (
    "You already applied this one. What it cost before you did is still "
    "reported in full; only the offer to write it again is withdrawn."
)
DISMISSED_REASON = (
    "You dismissed this one. What the behaviour already cost is still reported "
    "in full; only the offer is withdrawn, and you can bring it back."
)
ALREADY_PRESENT_REASON = (
    "This guidance is already in your instruction files — you wrote it, not "
    "tokenjam, so no apply ledger knew about it. What the recurrence cost before "
    "is still reported in full."
)


def _delivery_of(raw: dict[str, Any]) -> str:
    """The mechanism this stored proposal names, mapped on read when it comes
    from a build that named a ladder number instead.

    Read from the payload rather than hardcoded, so an analyzer that starts
    proposing a different delivery needs no edit here.
    """
    return str(raw.get("delivery") or delivery_from_legacy_record(raw) or "")


def _destinations_from_proposal(raw: dict[str, Any]) -> tuple[RuleDestination, ...]:
    """The files this proposal's rule lands in, with their session counts.

    A proposal computed before placement existed carries no ``placement_paths``;
    it degrades to no destinations rather than to a wrong one, so an older
    cache still lists and simply cannot stage.

    ``placement_sessions`` is read positionally alongside the paths. It was
    absent at first, which left every destination reporting zero sessions —
    and per-destination exposure is the entire justification for placing a
    rule, so a zero there does not merely look wrong, it hides whether the
    mechanism did anything. A short or missing list degrades to 0 for the
    unmatched entries rather than raising: an older payload has no counts, and
    "not recorded" is the honest reading of that.
    """
    paths = [str(p) for p in (raw.get("placement_paths") or []) if p]
    sessions = [int(n or 0) for n in (raw.get("placement_sessions") or [])]
    scope = str(raw.get("placement_scope", "") or "user-global")
    if not paths:
        return ()
    return tuple(
        RuleDestination(
            path=path, scope=scope,
            sessions=sessions[i] if i < len(sessions) else 0,
        )
        for i, path in enumerate(paths)
    )


def _rule_from_cost_proposal(raw: dict[str, Any]) -> RuleWrite | None:
    """One stored cost proposal as a rule write, or ``None`` if it writes none."""
    analyzer = str(raw.get("analyzer", "") or "")
    if analyzer not in RULE_WRITING_ANALYZERS:
        return None
    # A suppressed write has its `proposed_fix` cleared and its text moved to
    # `suggestion` by the budget pass, so the artifact has to be read from
    # whichever slot survived. Losing it would make a deferred rule
    # indistinguishable from one that was never derived.
    artifact = str(raw.get("proposed_fix", "") or raw.get("suggestion", "") or "")
    offered = bool(raw.get("write_offered", False))
    if not artifact:
        return None
    # An empty `delivery` means this proposal offers no write at all — the
    # persona gate cleared it. That is a statement about the OFFER, so a
    # proposal that also carries no blocked-reason has nothing to list.
    delivery = _delivery_of(raw)
    if not delivery and not raw.get("write_blocked_reason"):
        return None
    return RuleWrite(
        signature=str(raw.get("signature", "") or ""),
        analyzer=analyzer,
        title=str(raw.get("title", "") or ""),
        artifact_text=artifact,
        delivery=delivery or DEFAULT_DELIVERY,
        destinations=_destinations_from_proposal(raw),
        offered=offered,
        blocked_reason=str(raw.get("write_blocked_reason", "") or ""),
        placement_basis=str(raw.get("placement_basis", "") or ""),
        placement_coverage_note=str(raw.get("placement_coverage_note", "") or ""),
        past_overspend_tokens=raw.get("past_overspend_tokens"),
        past_overspend_usd=raw.get("past_overspend_usd"),
    )


def _rule_from_relearn_cluster(raw: dict[str, Any]) -> RuleWrite | None:
    """One stored relearn cluster as a rule write.

    relearn's clusters already carry a resolved ``suggested_target`` and their
    own per-repo exposure accounting (``relearn._write_exposure_sessions``), so
    the destination is read off the cluster rather than re-derived. The two
    lanes stay separate upstream — different budgets, different populations,
    Critical Rule 27 — and meet only here, at the fix surface.
    """
    target = str(raw.get("suggested_target", "") or "")
    artifact = str(raw.get("proposed_fix", "") or "")
    if not artifact:
        return None
    repos = [str(r) for r in (raw.get("repos") or [])]
    scope = str(raw.get("scope", "") or "user-global")
    destinations = (
        (RuleDestination(
            path=target, scope=scope, sessions=int(raw.get("sessions", 0) or 0),
        ),) if target else ()
    )
    return RuleWrite(
        signature=str(raw.get("signature", "") or ""),
        analyzer="relearn",
        title=str(raw.get("title", "") or ""),
        artifact_text=artifact,
        delivery=_delivery_of(raw) or DEFAULT_DELIVERY,
        destinations=destinations,
        offered=bool(raw.get("write_offered", False)) and bool(target),
        blocked_reason=str(raw.get("write_blocked_reason", "") or ""),
        placement_basis=(
            f"scoped to {len(repos)} repo(s) by the recurrence's own sessions"
            if scope == "project" and repos else ""
        ),
        past_overspend_tokens=raw.get("past_overspend_tokens"),
        past_overspend_usd=raw.get("past_overspend_usd"),
    )


def dismissals_covers(signature: str, dismissed: set[str]) -> bool:
    """Whether ``signature`` reads as dismissed.

    Delegates the MATCHING to ``cost_apply.signature_is_applied``, which
    already resolves a legacy agent-only mark as covering the later
    model-qualified signatures for that agent. A dismissal has to honour the
    same equivalence or a card the user dismissed under one signature would
    reappear under its refinement — the same defect a second matcher would
    reintroduce for applies.
    """
    from tokenjam.core.optimize import cost_apply

    return cost_apply.signature_is_applied(signature, dismissed)


def _mark_applied(rules: list[RuleWrite], config: TjConfig) -> list[RuleWrite]:
    """Flag every rule the user has already dealt with, and withdraw its offer.

    **The offer only.** ``past_overspend_usd``/``_tokens`` are not touched here
    and must never be: the waste happened inside the analyzed window, and the
    user fixing it afterwards does not un-spend the money (Critical Rule 32).
    An action-availability gate that edits a past figure is the exact
    "we have no remedy" / "this was free" conflation that rule exists to stop.

    Matching goes through ``cost_apply.signature_is_applied``, which already
    resolves a legacy agent-only mark as covering the later model-qualified
    signatures for that agent. Writing a second matcher here would silently
    reopen every card that helper settles.

    Never raises: an unreadable ledger reads as "nothing applied", which leaves
    every rule on offer. That is the safe direction — it can waste a user's
    attention, where the opposite hides a fix they never made.
    """
    from tokenjam.core.optimize import cost_apply, dismissals, relearn_apply

    try:
        cost_sigs = cost_apply.applied_signatures(config)
    except Exception:
        cost_sigs = set()
    try:
        relearn_sigs = relearn_apply.applied_signatures(config)
    except Exception:
        relearn_sigs = set()
    try:
        dismissed_sigs = dismissals.dismissed_signatures(config)
    except Exception:
        dismissed_sigs = set()

    out: list[RuleWrite] = []
    for rule in rules:
        known = relearn_sigs if rule.analyzer == "relearn" else cost_sigs
        # ONE resolution pass over all three ledgers, matched by the ONE
        # matcher. A dismissal is a different statement from an apply — "not
        # this one" versus "I did this" — so it carries its own flag and its
        # own reason, but it resolves through the same helper rather than a
        # fourth copy of the signature filter.
        if dismissals_covers(rule.signature, dismissed_sigs):
            out.append(replace(
                rule, dismissed=True, offered=False,
                blocked_reason=rule.blocked_reason or DISMISSED_REASON,
            ))
            continue
        if not cost_apply.signature_is_applied(rule.signature, known):
            out.append(rule)
            continue
        out.append(replace(
            rule,
            already_applied=True,
            # Withdrawing the OFFER. Every figure above is carried through
            # untouched by construction — `replace` names only these two.
            offered=False,
            blocked_reason=rule.blocked_reason or ALREADY_APPLIED_REASON,
        ))
    return out


def _mark_present(rules: list[RuleWrite], config: TjConfig) -> list[RuleWrite]:
    """Flag every rule whose guidance is already in the user's own files.

    Reads the STORED verdicts (``presence.load_presence``) — the model call runs
    on the analyzer cycle, never here, because this function is on the path of a
    user-facing route and an analyzer may not run on a request.

    Ordered AFTER ``_mark_applied`` on purpose: a rule tokenjam itself applied is
    already accounted for by the ledger, which is exact, and re-labelling it from
    a model's verdict would replace a fact with an inference.

    Withdraws the OFFER and nothing else (Critical Rule 32) — ``past_overspend_*``
    is untouched. The recurrence happened and cost what it cost; the guidance
    having been there the whole time does not un-spend it. If anything, a rule
    that was present AND still recurring is evidence about the rule's wording,
    which is a reason to keep the figure visible rather than to erase it.

    Never raises: an unreadable store reads as "nothing known", which leaves
    every rule on offer.
    """
    from tokenjam.core.rulewrite.presence import load_presence

    try:
        known = load_presence(config)
    except Exception:  # noqa: BLE001 - see docstring
        return rules
    if not known:
        return rules
    out: list[RuleWrite] = []
    for rule in rules:
        record = known.get(rule.signature)
        if record is None or rule.already_applied or rule.dismissed:
            out.append(rule)
            continue
        out.append(replace(
            rule,
            already_present=True,
            offered=False,
            blocked_reason=rule.blocked_reason or ALREADY_PRESENT_REASON,
            presence_evidence=str(record.get("evidence") or ""),
            presence_path=str(record.get("source_path") or ""),
        ))
    return out


def _is_listable(rule: RuleWrite) -> bool:
    """Whether this rule belongs on the rule surface at all.

    THE GATE MOVED DOWN THE STACK. Every rule used to be listed, including the
    ones the write budget refused, each carrying its refusal as prose: "below the
    $5 floor", "a permanent rule would cost more to keep than it recovers",
    "your instruction files already carry more standing context than the budget
    allows to grow". Individually those are honest. Collectively they turned the
    rule surface into a list of things the product declined to do — a page of
    refusals where a user came looking for something to apply.

    The economics belong lower in the stack than the surface (founder call): a
    proposal that is not worth writing permanently should not reach this page at
    all, rather than reach it and explain itself. There is then no refusal prose
    to render, because there is nothing to refuse.

    **This is a filter on the OFFER, never on the MEASUREMENT.** The money for
    every rule dropped here is still summed by the Review inbox's headline and
    still carried on its own analyzer card — those surfaces read the proposal
    caches directly, not this module. An action-availability gate that reached
    back and edited what a behaviour already cost is the exact defect Critical
    Rule 32 exists to stop, and it is the easy mistake to make here, because
    "we're not offering it" feels like it should zero the number. It does not:
    ``$3,225`` of measured overspend stays ``$3,225`` whether or not any of it
    has a rule worth writing.

    Kept: anything actionable (``offered``), and anything whose STATE the user
    needs to see — already applied, already present in their files, or dismissed
    (which must stay listed, or there is nothing left to un-dismiss).
    """
    return bool(
        rule.offered
        or rule.already_applied
        or rule.already_present
        or rule.dismissed
    )


def all_rule_writes(config: TjConfig) -> list[RuleWrite]:
    """Every rule the last pass produced, INCLUDING the ones nothing will offer.

    The unfiltered set, for callers whose question is about the proposals rather
    than about the surface. There is one: ``presence`` detection has to ask about
    rules the write budget declined, because a rule already in the user's files is
    very often declined *because* those files are what saturate the budget — so
    asking only about the listable subset would never find the population the
    whole mechanism exists for.

    Everything a reader-facing caller wants is :func:`list_rule_writes`.
    """
    return _collect(config)


def list_rule_writes(config: TjConfig) -> list[RuleWrite]:
    """Every permanent rule worth writing, plus every one already in place.

    Ranked by what it addresses. Never raises — an unreadable cache reads as an
    empty list, which is what a fresh install genuinely is.

    What is NOT here: rules the write budget declined. They used to be listed
    with their refusal stated; the gate now sits below this surface, so they are
    absent rather than explained. See :func:`_is_listable` — including why that
    filter cannot touch the measured figures.
    """
    return [rule for rule in _collect(config) if _is_listable(rule)]


def _collect(config: TjConfig) -> list[RuleWrite]:
    """Build, resolve state, and rank. No listability filter — callers apply it."""
    from tokenjam.core.optimize import relearn_proposals, relearn_store

    out: list[RuleWrite] = []
    try:
        block = relearn_store.read_cost_proposals(config=config) or {}
    except Exception:
        block = {}
    for raw in (block.get("cost_proposals") or []):
        if not isinstance(raw, dict):
            continue
        rule = _rule_from_cost_proposal(raw)
        if rule is not None and rule.signature:
            out.append(rule)
    try:
        clusters = relearn_proposals.list_proposals(config)
    except Exception:
        clusters = []
    for cluster in clusters:
        raw = cluster if isinstance(cluster, dict) else getattr(cluster, "__dict__", {})
        rule = _rule_from_relearn_cluster(dict(raw))
        if rule is not None and rule.signature:
            out.append(rule)
    # Applied rules keep their figures and lose their offer — see
    # `_mark_applied`. They stay in the list on purpose: a user who applied
    # something and then sees nothing cannot tell "done" from "broken".
    out = _mark_applied(out, config)
    # THEN the model's verdict, for the rules no ledger of ours knows about. The
    # ledger is exact and goes first; this only ever labels what it left over.
    out = _mark_present(out, config)
    # Ranked by what the rule addresses, unpriced last. `None` sorts last
    # rather than as zero: "not measured" is not "worth nothing".
    out.sort(key=lambda r: (r.past_overspend_usd is None, -(r.past_overspend_usd or 0.0)))
    return out


def find_rule(config: TjConfig, signature: str) -> RuleWrite | None:
    """One rule by signature, or ``None``."""
    for rule in list_rule_writes(config):
        if rule.signature == signature:
            return rule
    return None
