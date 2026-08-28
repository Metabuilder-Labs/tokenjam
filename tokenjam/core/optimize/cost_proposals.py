"""Adapt cost-analyzer findings into Review-inbox proposals ("advisories").

The self-improve loop's relearn detector already produces
``RelearnCluster`` proposals that the Lens Improve inbox renders and that a
user can mark and apply. The *cost* analyzers (``downsize`` model
over-sizing, ``cache`` efficacy, ``cache-recommend`` breakpoint placement,
``trim`` prompt bloat, ``subagent`` right-sizing, ``deadweight`` dead MCP
servers, ``script`` deterministic workflows, ``reuse`` repeated planning
skeletons, ``verbosity`` high-output outliers) produce findings of a
different shape. This module adapts each finding into a ``CostProposal`` so
the inbox can list them BESIDE the relearn proposals, typed by a distinct
``kind`` field.

Two structural facts carry over from the relearn ``advise_only`` lane and are
NOT optional here:

  * **Advise-only by default, apply-capable where a real workspace surface
    exists.** Most cost fixes live in the user's own application code (a
    model-routing decision, a cache-prefix change, a prompt edit) that
    tokenjam cannot write into — those cards have NO apply path, exactly like
    an ``advise_only`` ``RelearnCluster`` (empty ``suggested_target``). A
    minority (``subagent``, the per-agent slice of ``downsize``, ``script``,
    ``reuse``) DO have a workspace surface an orchestrating agent reads
    before acting (a CLAUDE.md rubric, a model-id key, a new skill note) and
    route through the same ``relearn_apply.apply_relearn_fix`` machinery the
    relearn lane uses (``apply_capable=True``, ``delivery``,
    ``scope``, ``proposed_fix``). ``verbosity`` shares that same class of
    surface in principle but is deliberately kept advise-only for every
    persona (see ``_verbosity_to_proposals``) — the finding is cohort-scoped
    but the artifact would be global, which fails the no-quality-tax gate.
    Every other card carries a recommendation and, where sensible, a
    copyable config/code suggestion; the user applies it themselves.
  * **Estimated, never causal.** Every saving figure a cost finding carries is
    a heuristic ESTIMATE (house style, CLAUDE.md Rule 14). The adapter
    preserves the finding's own ``estimate_basis`` and labels the figure
    ``estimated`` — never proof tokenjam's advice caused a savings change.

The adapter is pure: it reads an already-built ``OptimizeReport`` and returns
proposals. It never touches the DB, the store, or the network.
"""
from __future__ import annotations

import hashlib
import json
import re
import threading
from dataclasses import asdict, dataclass, field, is_dataclass, replace
from pathlib import Path
from typing import Any, Callable

from tokenjam.core import fixes
from tokenjam.core.rulewrite.kinds import DELIVERY_CLAUDE_MD_RULE, DELIVERY_SKILL
from tokenjam.core.optimize.report_window import (
    FALLBACK_WINDOW_DAYS as _REPORT_FALLBACK_WINDOW_DAYS,
)

# House-style label strings. Kept verbatim on every cost proposal so no channel
# can surface a savings figure without the honesty framing (Rule 14).
COST_ESTIMATE_CONFIDENCE = "estimated"
COST_CORRELATIONAL_CAVEAT = (
    "Estimated, correlational figure; not a causal savings claim. The "
    "recommendation lives in your own application code. Review the evidence "
    "before changing anything."
)

#: The analyzers this wiring covers, by registration name. ``cache-recommend``
#: requires ``[capture] prompts = true``; when that's off the analyzer itself
#: returns an ``enabled=False`` finding with no candidates (same shape as
#: ``trim``'s own disabled-state guard), so including it here unconditionally
#: costs nothing when capture is off.
#:
#: ``resend`` (context re-send, the product's headline waste category — see
#: ``analyzers/context_resend.py``) is included: it has real recoverable
#: savings and previously had no adapter here, silently excluding it from the
#: Review inbox's Cost-advisories tab.
#:
#: ``summarize`` (prompt summarization) IS here (re-filing the
#: purged #326). It used to be deliberately excluded — own review surface,
#: "don't add cards" — and #326 tried to soften the consequence with a
#: link-only "$X more, not summed here" footnote instead of a card. The
#: founder decision that revisited this rejects the old reasoning outright:
#: **the Review inbox is the complete index of everything actionable, not the
#: list of things whose apply flow happens to live here.** Where the fix is
#: APPLIED is a routing detail, not a reason to keep a real, measured figure
#: off the one surface a user reads for "what's outstanding". So
#: ``summarize`` gets a normal card like every other analyzer here — see
#: ``_summarize_to_proposals`` — except its card routes to the ``tj summarize``
#: curate/diff surface instead of offering an inline apply (``advise_only``,
#: no ``apply_kind``): unlike a model-id swap or an MCP-server removal, the fix
#: is a reviewed rewrite (structure kept, prose compressed), not a value this
#: adapter can safely one-click.
#: ``relearn`` (recurring agent failures) is a member but produces NO cost
#: proposal any more — there is no adapter for it. Membership is what keeps the
#: analyzer running inside the inbox recompute (its per-cluster rows depend on
#: the finding); the one aggregate card it used to also emit is deleted,
#: because the only figure that card ever carried was the retired
#: total-observed-cost field. See the note above ``_summarize_to_proposals``'s
#: neighbours where that card used to live.
COST_ANALYZERS = (
    "downsize", "cache", "cache-recommend", "trim", "subagent", "deadweight",
    "script", "reuse", "verbosity", "resend", "summarize", "relearn",
)


def _disabled_analyzers(persona: str) -> frozenset[str]:
    """The persona skip-gate set, imported lazily.

    Deferred import: ``runner`` reaches this module through the analyzer
    registry, so a module-level import would cycle.
    """
    from tokenjam.core.optimize.runner import disabled_analyzers_for_persona

    return disabled_analyzers_for_persona(persona)


def cost_analyzers_for_persona(persona: str) -> tuple[str, ...]:
    """``COST_ANALYZERS`` minus the ones with no fix this persona can apply.

    ``COST_ANALYZERS`` is an INDEPENDENT second analyzer-selection surface
    (the Review inbox's recompute selects from it, not from
    ``ANALYZER_ORDER``), so the persona skip gate has to be mirrored here or a
    disabled analyzer's findings still reach the inbox as apply-able cards.
    Same map, same reasons — see ``runner.PERSONA_DISABLED_ANALYZERS``.
    """
    disabled = _disabled_analyzers(persona)
    return tuple(name for name in COST_ANALYZERS if name not in disabled)

# The apply notes below all route through the SAME workspace-note machinery
# `subagent` already uses (`relearn_apply.apply_relearn_fix`: a CLAUDE.md rule,
# or a skill at a new .claude/skills/<slug>/SKILL.md). None of
# these three analyzers has a workspace file it can edit outright the way a
# model-routing swap does — the fix is behavioral (an orchestrator or the model
# itself reading guidance), same class of surface as the subagent rubric.

# The skill note a `script` proposal writes: the observed tool-call
# pattern is deterministic enough that a script could run it directly instead
# of dispatching a full agent turn.
# THE text lives in `core/fixes/registry.py`, so the lint sees it.
_SCRIPT_SKILL_INTRO = fixes.fix_text("script.replace_agent_turn_with_script")

# The CLAUDE.md rule a `reuse` proposal writes: the planning skeleton recurs.
# THE text lives in `core/fixes/registry.py`, so the lint sees it.
_REUSE_NOTE_INTRO = fixes.fix_text("reuse.template_the_plan_skeleton")

# The sizing-rubric rule a CC-origin subagent proposal writes into the
# workspace CLAUDE.md when applied. A shape-based default, not a per-subagent
# edit — it names the observed oversized dispatches and states the routing rule.
#
# TWO THINGS THIS TEXT MUST NOT DO AGAIN, both of which it once did.
#
# (a) It must not pass a long, tool-heavy dispatch. The sentence that used to
# close this rubric ("a subagent that does little tool work and returns a short
# result rarely needs the premium tier") encoded the very gate that was DELETED
# from `analyzers/subagent_rightsizing.py` — the `output_tokens < 2000` /
# `tool_calls <= 5` clauses that made the most expensive dispatches the LEAST
# eligible to be flagged (Critical Rule 29's gate inversion). Every dispatch in
# the claim this rubric is written against is large, tool-heavy and
# long-output, so that sentence told the agent to keep doing precisely what the
# card is billing it for: applying the fix would have left the number where it
# found it, which is a correctness bug, not a wording preference.
#
# (b) The premium escape hatch must be CHECKABLE BY AN OUTSIDE READER, never
# self-assessed. "Unless the subtask genuinely needs deep reasoning" asks the
# dispatching agent to rate its own task's difficulty, and an agent asked that
# question answers yes — the exception then swallows the rule. The conditions
# below are stated as things that are either written into the dispatch or not,
# so a reader of the transcript can decide whether the exception applied
# without re-running anyone's judgement.
# THE text lives in `core/fixes/registry.py`, not here. It used to be a
# constant in this module and an all-but-identical one in `context_resend`,
# which is exactly how the same sizing contradiction shipped twice in different
# words and was fixed in only one of them. One definition, linted.
SUBAGENT_RUBRIC_INTRO = fixes.fix_text("subagent.sizing_rubric")

# The downsize card's claude-code CTA. Mirrors `cmd_optimize._render_downgrade_
# cta`'s claude-code branch: an interactive CC session can't pass `--original`/
# `--candidate` to swap its own model mid-turn, so "route to a cheaper model"
# is not a fix this persona can act on. Used wherever the card would otherwise
# hand a CC window the same generic model-swap text an SDK caller gets (see
# ticket-level "no CC user gets a raw-model-swap CTA" requirement).
#: `/compact` IS NOT ON THIS LIST, and that is deliberate. It sits below the
#: actionable ceiling recorded in the persona matrix under product-state: it is
#: transient, persists nothing past the session, and users who feel a session
#: filling already do it unprompted. Listing it beside three durable levers
#: implies it is the same kind of thing, which is the mistake that entry exists
#: to name — and it pads a list whose length is itself a cost to adherence.
#: It survives as explicitly-labelled immediate relief where a card has room
#: for one (see the resend card's "Immediate relief" position); it is never a
#: fix.
# THE text lives in `core/fixes/registry.py`, so the lint sees it.
_DOWNSIZE_CC_LEVER = fixes.fix_text("downsize.claude_code_levers")

# Appended (not substituted) for a "mixed" window: the swap text above already
# applies to the sdk share, this just adds the claude-code share's own lever —
# same "both, labeled" precedent as `_render_downgrade_cta`'s mixed branch.
# ONE record, not a second shorter copy of the list above. This was a
# separately-authored restatement of `_DOWNSIZE_CC_LEVER`, which is how the
# same instruction comes to drift in two places; only the framing differs, so
# only the framing is written here (as a catalogued lead-in).
_DOWNSIZE_MIXED_CC_NOTE = " " + fixes.fix_text_for(
    "downsize.claude_code_levers", "downsize.mixed_window",
)


@dataclass
class CostProposal:
    """One cost analyzer's finding, shaped for the Review inbox.

    Mirrors the fields the inbox already reads off a ``RelearnCluster`` (title,
    evidence, an estimate with its basis, ``advise_only``) plus a cost-specific
    ``target_key``.
    """
    kind:      str                     # always "cost" — the inbox discriminator
    analyzer:  str                     # "downsize" | "cache" | "trim" | "subagent"
    signature: str                     # stable identity for dedup
    title:     str
    # WHICH thing is flagged, machine-readable (downsize: the oversized
    # model(s); cache: a provider/model; trim: an agent/step).
    target_key: dict[str, Any]
    # Human-readable evidence line: which model/step/cache + the measured
    # baseline number.
    evidence:   str
    # The measured baseline numbers, machine-readable (for rendering + as the
    # verify pass's pre-window reference where useful).
    baseline:   dict[str, Any]
    # Recommendation the user applies themselves + an optional copyable snippet.
    advise_text: str
    suggestion:  str = ""
    # PAST OVERSPEND — **THE canonical per-analyzer dollar/token figure.** One
    # field, one meaning, one time basis; there is no second name for it and no
    # paced variant of it anywhere in the tree (see the field contract in the
    # repo `CLAUDE.md`). Every surface — Dashboard hero, Review inbox headline,
    # per-card headline, CLI, MCP — renders THIS field or the rollup of it.
    #
    # **It is the AVOIDABLE amount, observed over the analyzed window, past
    # tense.** Waste is what could have been avoided; unavoidable spend is
    # cost, not waste (the same rule `reuse` already applies by pricing
    # `reps - 1` rather than `reps` — the necessary first instance is not
    # waste). A figure a reader will call "overspend" may therefore only ever
    # be an avoidable figure.
    #
    # Set by the ADAPTER, straight off its analyzer's own
    # `past_overspend_usd`/`_tokens`; `_with_past_overspend` adds only the basis
    # string, so no adapter invents its own tense or wording. ``None`` when the
    # finding produced no priced figure for this item — never coerced to 0.0,
    # which would state "worth nothing" for "not measured".
    #
    # There is no second per-analyzer dollar field beside it. A total-observed-cost
    # pair (`cost_of_waste_*` on the analyzer, `observed_cost_*` here) used to ride
    # alongside, spanning EVERY session while the avoidable figure spanned a
    # filtered subset. It is DELETED, by founder decision: two analyzers of twelve
    # emitted it, only one ever rendered it, and the aggregate it fed made a claim
    # that was provably false (see `past_overspend_rollup`). What the avoidable
    # figure does and does not cover is stated in `coverage_note` instead, in
    # words, which is the part that was actually load-bearing.
    #
    # Per-analyzer derivation, verified against source: downsize
    # (`actual_cost - alt_cost` over the window), deadweight (window tax, "NO
    # projection folded in"), subagent (deltas priced off already-incurred
    # tokens), summarize (per-call reduction over observed calls), reuse (avg
    # cost x (reps - 1)), resend (the measured offload + right-size terms).
    #
    # **NEVER paced.** There is no ratio to apply: pacing left the tree with
    # `estimated_monthly_*` and `compute_projection_ratio`. Multiplying an
    # observation by a forward ratio would make it a forecast — and a paired
    # display whose two sides sat on different time bases would attribute a
    # pacing artifact (measured 1.0714 on the reference corpus) to
    # avoidability.
    past_overspend_usd:           float | None = None
    past_overspend_tokens:        int | None   = None
    past_overspend_basis:         str          = ""
    # Plain-language statement of what the figure above does and does NOT cover,
    # supplied by the analyzer. It exists because a filtered avoidable figure
    # invites the reader to assume everything outside it was unavoidable, and
    # that inference is wrong: what is outside was analysed on another card,
    # filtered out before the calculation, or outside the definition the analyzer
    # measures. The note has to end by saying so.
    #
    # It survived the deletion of the total-cost figure it originally sat beside,
    # on purpose. The pairing was what made the false claim, and the note was what
    # corrected it; the correction is worth keeping on its own, as a collapsed
    # disclosure, because the filtering it describes is still happening.
    coverage_note:                str          = ""
    estimate_basis:       str = ""
    estimate_confidence:  str = COST_ESTIMATE_CONFIDENCE
    correlational:        bool = True
    # Structural: a cost proposal never has an apply path (see module docstring).
    advise_only:          bool = True
    caveat:               str = COST_CORRELATIONAL_CAVEAT
    # Best-effort service scope for the marker/expectation the user creates on
    # "mark applied" (Expectation.agent_id). "" when the finding spans agents.
    agent_id:             str = ""
    # Workspace-apply plumbing (subagent right-sizing only). Unlike the three
    # advise-only analyzers, a CC-origin subagent finding HAS a writable surface
    # — a sizing rubric rule in the workspace's CLAUDE.md — so its card
    # can route an actual, reversible, human-gated write through the existing
    # relearn apply path (``relearn_apply.apply_relearn_fix``). The adapter (not
    # the analyzer) supplies these; ``apply_capable`` gates the apply action, and
    # a proposal with no clean workspace surface degrades to advise-only like the
    # other three (``apply_capable=False``, ``advise_only=True``).
    apply_capable:        bool = False
    #: HOW this proposal's fix reaches the agent, and therefore what gets
    #: written (``core/rulewrite/kinds``). Empty means no write is offered —
    #: the persona gate cleared it, or this proposal was never write-bearing.
    delivery:             str  = ""
    scope:                str  = ""
    proposed_fix:         str  = ""
    # Model-routing apply kinds (``core.optimize.model_apply``). Set only where
    # the edit is a deterministic rewrite of a value already written down: an
    # agent file's ``model:`` key, or one exact model-id string in a repo the
    # user registered. Empty everywhere else, which leaves the card advise-only
    # with its one-paste artifact.
    apply_kind:           str  = ""
    agent_name:           str  = ""
    current_model:        str  = ""
    proposed_model:       str  = ""
    source_path:          str  = ""
    target_path:          str  = ""
    #: ``apply_capable`` but not yet applyable: the fix is a deterministic edit
    #: and the one missing input is WHERE, which only the user can answer (see
    #: ``_model_swap_plumbing``). A row carrying this asks for the path and then
    #: applies, rather than falling back to "Mark applied" — which records that
    #: the user did something by hand and is the weakest thing the inbox can
    #: offer. ``apply_kind`` stays unset while this is true.
    needs_source_path:    bool = False
    #: The caveat that must stay VISIBLE beside an Apply control, never folded
    #: into the collapsed description with the rest of the prose. Set only where
    #: an apply is on offer and the fix carries a risk the measurement does not
    #: cover — today the model swap, whose cost delta is measured and whose
    #: quality equivalence is never claimed (Critical Rule 14). An Apply button
    #: must not imply the change is free of judgement.
    apply_caveat:         str  = ""
    # Why the direct apply is not on offer, when it is not. Rendered on the card
    # next to the one-paste fix so a fallback is never silent.
    apply_blocked_reason: str  = ""
    # The exact fix, with this agent's own measured values already substituted
    # in. Every advise-only card carries one.
    one_paste_fix:        str  = ""
    # WHERE the permanent rule would be written (`core/optimize/rule_placement`).
    # Every field here describes placement, never a saving: `placement_paths` is
    # the set of CLAUDE.md files the rule lands in, `placement_scope` is
    # "project" or "user-global", and the two standing figures are the chosen
    # and the rejected alternative, kept side by side so the decision is
    # inspectable rather than a bare verdict. `placement_footprint_tokens` is
    # what the FILES have to carry (per-session x files written) as distinct
    # from what the rule costs to KEEP.
    #
    # These carry no dollars on purpose: placement changes what a rule costs to
    # KEEP, never whether it is offered — a rule is offered whenever the card
    # is `apply_capable`, with no budget/ceiling gate on top.
    placement_scope:          str          = ""
    placement_paths:          list[str]    = field(default_factory=list)
    #: Per-destination session counts, parallel to `placement_paths`. Carried
    #: because per-destination exposure IS what placement computes — a payload
    #: that serialized only the paths left every consumer reporting `sessions:
    #: 0` for destinations the resolver had counted correctly, so the one
    #: figure that justifies the whole mechanism was invisible downstream.
    placement_sessions:       list[int]    = field(default_factory=list)
    placement_standing_tokens: int         = 0
    placement_alternative_standing_tokens: int = 0
    placement_footprint_tokens: int        = 0
    placement_basis:          str          = ""
    #: What the placement split covers and what it could not place, ending by
    #: saying the difference is what was not analysed (Critical Rule 30). Kept
    #: separate from `coverage_note` above, which is about the FIGURE's
    #: population rather than the RULE's.
    placement_coverage_note:  str          = ""
    #: EVERY session `past_overspend_*` was priced over, when the adapter
    #: knows the exact population (today: `reuse`, `script` — both cluster on
    #: a repeated-tool-sequence shape and can genuinely claim the same
    #: sessions). Empty for every other analyzer, which keeps them exactly as
    #: they behaved before this field existed: no adapter is required to
    #: populate it, and `_net_cross_analyzer_session_overlap` only acts on the
    #: pairs that do (CLAUDE.md Critical Rule 27's "prove no shipped analyzer
    #: already claims those rows" made mechanical rather than hand-derived per
    #: pair, for the analyzers that can state their own population exactly).
    claimed_session_ids:      tuple[str, ...] = ()


# --------------------------------------------------------------------------- #
# Per-analyzer adapters. Each reads ONE finding dataclass and returns 0..N
# proposals. All tolerate a None/empty finding (returns []).
# --------------------------------------------------------------------------- #

def _downsize_to_proposal(
    finding: Any, config: Any = None, persona: str = "unknown",
    resend_finding: Any = None,
) -> list[CostProposal]:
    """The model-over-sizing card(s).

    When the finding carries per-agent price rows, each agent gets its own card
    with its own arithmetic (and, where the preconditions hold, the gated model-
    id swap). Those rows partition the same candidate sessions, so the window-
    wide card is NOT emitted alongside them: one source of over-sized spend,
    one card.

    Without those rows (no pricing data for a model on either side) the finding
    falls back to the single window-wide card. ``DowngradeFinding.suggestions``
    maps each oversized model to its cheaper same-family alternative, and the
    delta-verify pass measures the model-mix cost delta across ALL flagged
    models, so one proposal listing them keeps that aggregate estimate coherent.

    ``persona`` gates which of the two cards fire, not just their wording.
    DECISION (persona/actionability matrix, downsize row): for
    ``"claude-code"``, this analyzer emits ONLY the driver-role card. Both
    the window-wide and per-agent tiny-session cards end in "switch your own
    interactive model" for this persona — above the CC actionable ceiling —
    so a Claude Code reader used to hit a card that stated a number with no
    fix. Retired for that persona only: ``_downsize_agent_proposals`` and the
    window-wide card below are unchanged and still fire for ``sdk``, where
    routing a request to a cheaper model is a real lever.
    """
    if finding is None:
        return []
    # The driver-role card is a SEPARATE card from the tiny-session one: it
    # describes a different session population (disjoint by construction — see
    # `analyzers/resend_tail.premium_driver_role`) and a different lever, so
    # merging the two would put one number on top of two unrelated derivations.
    # Exactly one window-wide card, never one per agent, so this adds at most a
    # single row to the inbox.
    proposals = _driver_role_proposals(finding, persona, resend_finding=resend_finding)
    if persona == "claude-code":
        # See the docstring above: the tiny-session/per-agent cards never had
        # a fix on this persona's action surface, only a pointer to other
        # commands — retiring them here means a claude-code window either
        # gets the driver-role card (a real one-paste fix) or nothing, never
        # a number with no fix attached.
        return proposals
    if getattr(finding, "candidate_sessions", 0) <= 0:
        return proposals
    per_agent = _downsize_agent_proposals(finding, config, persona)
    if per_agent:
        return proposals + per_agent
    suggestions: dict[str, str] = dict(getattr(finding, "suggestions", {}) or {})
    if not suggestions:
        return proposals
    models = sorted(suggestions.keys())
    model_list = ", ".join(models)
    evidence = (
        f"{finding.candidate_sessions} of {finding.total_sessions} sessions "
        f"({finding.percent_of_sessions:.0f}%) ran on a larger-than-needed model "
        f"({model_list}); candidate sessions are {finding.percent_of_tokens:.0f}% "
        f"of the window's tokens."
    )
    caveat = str(getattr(finding, "caveat", "") or "")
    suggestion = "\n".join(f"{m} -> {alt}" for m, alt in sorted(suggestions.items()))
    if persona == "claude-code":
        advise = (_DOWNSIZE_CC_LEVER + " " + caveat).strip()
        suggestion = ""
    else:
        advise = (
            "Route the flagged structural-shaped work to the cheaper same-family "
            "model before it runs. Suggested swaps: "
            + "; ".join(f"{m} → {alt}" for m, alt in sorted(suggestions.items()))
            + ". " + caveat
        ).strip()
        if persona == "mixed":
            advise += _DOWNSIZE_MIXED_CC_NOTE
    return [CostProposal(
        kind="cost",
        analyzer="downsize",
        signature="cost:downsize",
        title="Model over-sizing (route to a cheaper same-family model)",
        target_key={"models": models, "suggestions": suggestions},
        evidence=evidence,
        baseline={
            "candidate_sessions": int(finding.candidate_sessions),
            "total_sessions": int(finding.total_sessions),
            "actual_cost_usd": float(finding.actual_cost_usd),
            "alternative_cost_usd": float(finding.alternative_cost_usd),
            "percent_of_tokens": float(finding.percent_of_tokens),
        },
        advise_text=advise,
        suggestion=suggestion,
        one_paste_fix=suggestion,
        # The TINY-SESSION share only: the finding's figures are the sum of
        # both cases, and the driver-role half already has its own card above,
        # so carrying the sum here would claim it twice under two signatures —
        # exactly the failure Critical Rule 27 describes. Subtractive rather
        # than recomputed, so a window with no driver case passes the finding's
        # own numbers through byte for byte (including a `None`).
        past_overspend_usd=_minus_driver_usd(finding),
        past_overspend_tokens=_minus_driver_tokens(finding),
        estimate_basis=_tiny_session_basis(finding),
        # `model_downgrade.py`'s own `monthly_savings_usd` /
        # `monthly_tokens_in_candidates` is a `30/window_days` projection
        # computed inside the analyzer. It is deliberately NOT carried onto
        # this card: a card states one past-tense observation, never a paced
        # one. That field survives only to back the CLI's `tj optimize` line.
    )]


def _minus_driver_usd(finding: Any) -> float | None:
    """The finding's recoverable dollars with the driver-role card's share
    removed, so the two downsize cards partition one figure instead of each
    carrying the total. ``None`` passes straight through."""
    total = getattr(finding, "past_overspend_usd", None)
    if total is None:
        return None
    driver = float(getattr(finding, "driver_recoverable_usd", 0.0) or 0.0)
    return round(max(float(total) - driver, 0.0), 6)


def _minus_driver_tokens(finding: Any) -> int | None:
    """Token counterpart of :func:`_minus_driver_usd`."""
    total = getattr(finding, "past_overspend_tokens", None)
    if total is None:
        return None
    driver = int(getattr(finding, "driver_tokens", 0) or 0)
    return max(int(total) - driver, 0)


def _tiny_session_basis(finding: Any) -> str:
    """The basis for the tiny-session card.

    The finding's own `estimate_basis` describes BOTH cases, which would be
    misleading on a card that only claims one of them — but only once the
    driver-role case actually fired. With no driver session in the window the
    finding's basis already describes exactly this card, so it passes through.
    """
    if int(getattr(finding, "driver_sessions", 0) or 0) <= 0:
        return str(getattr(finding, "estimate_basis", "") or "")
    return (
        "candidate sessions routed to a cheaper model over the window — "
        "structural fit only, no quality validation. The premium-driver-role "
        "share of this window is claimed on its own card, not here."
    )


# The driver-role card's advice: one lead-in sentence naming why THIS card is
# showing the rule, then the canonical rule itself. The lead-in is per-analyzer
# framing; the rule is not, and a second copy of it here is exactly what let
# three analyzers write three wordings of one instruction into one file.
def _driver_role_advice() -> str:
    """The driver-role card's advice: a lead-in naming why THIS card shows the
    rule, then BOTH halves composed from the records that own them.

    A function rather than a module constant because it composes helpers
    defined further down; the alternative was reordering the module around a
    string. Composed rather than restated on purpose — this card's original
    wording carried its own copy of each half, which is how one instruction
    came to be authored three times across three analyzers.
    """
    return (
        "Route this shape of work to workers instead of doing it inline. "
        + compound_offload_fix(
            {},
            fixes.fix_text("resend.offload_to_subagent"),
            fixes.fix_text("resend.rightsize_worker"),
        )
    )


def _driver_role_proposals(
    finding: Any, persona: str = "unknown", resend_finding: Any = None,
) -> list[CostProposal]:
    """The model-ROLE card: a premium model drove undelegated work inline.

    One window-wide card, deliberately never one per agent — the standing
    don't-fill-the-inbox constraint means this case adds exactly one row no
    matter how many sessions or agents it covers.

    Persona-agnostic advice, unlike the tiny-session card's: the fix here is a
    CLAUDE.md rule plus an agent-file `model:` pin, both of which are on the
    Claude Code action surface, so there is no "you can't switch your own
    interactive model" caveat to apply.

    ``resend_finding``, when present, states the reciprocal of resend's own
    `coverage_note` (#613): the mirror sentence that these sessions also
    carry a cost on resend's card, read off resend's own fields so the two
    cards cannot disagree.
    """
    sessions = int(getattr(finding, "driver_sessions", 0) or 0)
    usd = float(getattr(finding, "driver_recoverable_usd", 0.0) or 0.0)
    if sessions <= 0 or usd <= 0:
        return []
    substitutes: dict[str, str] = dict(getattr(finding, "driver_substitutes", {}) or {})
    models = sorted(substitutes.keys())
    total_sessions = int(getattr(finding, "total_sessions", 0) or 0)
    offload = float(getattr(finding, "driver_offload_usd", 0.0) or 0.0)
    tier = float(getattr(finding, "driver_tier_usd", 0.0) or 0.0)
    tail = int(getattr(finding, "driver_tail_tokens", 0) or 0)
    swap_text = ", ".join(f"{m} → {substitutes[m]}" for m in models)
    evidence = (
        f"{sessions} of {total_sessions} sessions ran "
        f"{', '.join(models) or 'a premium model'} as the driver and never "
        f"dispatched a subagent: {tail:,} tokens were re-read purely because "
        f"that work stayed in the main thread. Routing it to a worker "
        f"({swap_text or 'a cheaper same-family model'}) would have saved "
        f"${offload:,.2f} of re-reads plus ${tier:,.2f} of tier difference."
    )
    # Reciprocal of resend's own coverage_note (#613). Read resend's
    # already-computed session count and dollar figure rather than
    # recomputing, so the two cards cannot disagree.
    #
    # resend classifies a session as driver-role from `premium_driver_role`
    # alone; this card's own `sessions` additionally requires a priced,
    # contiguous tool-driven stretch (`_driver_session_arithmetic`'s mask and
    # rate lookups), so `sessions` is always a SUBSET of resend's
    # `resend_sessions`, never the same population under a different name.
    # Quoting resend_sessions/resend_usd as if they described exactly THESE
    # `sessions` overstates the overlap whenever the populations diverge
    # (resend counts sessions this card's own gates reject); the wording below
    # only claims what is actually true of the relationship in either case.
    resend_sessions = int(getattr(resend_finding, "driver_role_sessions", 0) or 0)
    resend_usd = float(getattr(resend_finding, "cost_driver_role_usd", 0.0) or 0.0)
    coverage_note = ""
    if resend_sessions and resend_usd:
        if resend_sessions > sessions:
            coverage_note = (
                f"COVERAGE. These {sessions:,} session(s) sit inside resend's "
                f"broader {resend_sessions:,}-session driver-role class, which "
                f"also carries ${resend_usd:,.2f} of resend's observed cost, "
                f"analysed there as re-sent context. The two figures price "
                f"overlapping populations and must not be added together."
            )
        else:
            coverage_note = (
                f"COVERAGE. These {resend_sessions:,} session(s) also carry "
                f"${resend_usd:,.2f} of resend's observed cost, analysed there as "
                f"re-sent context. The two figures price the same sessions and "
                f"must not be added together."
            )
    # A FOURTH copy of the offload rule used to live here, abbreviated. The
    # consolidation that gave the three analyzers one record reached the card's
    # advise text and missed its one-paste block, so the user still received a
    # separately-authored wording — in the artifact they actually paste into a
    # CLAUDE.md, which is the worst place for it to drift.
    one_paste = (
        "# CLAUDE.md\n"
        + fixes.fix_text("resend.offload_to_subagent") + "\n"
        + "\n".join(
            f"\n# .claude/agents/<name>.md\n---\nmodel: {substitutes[m]}\n---"
            for m in models[:1]
        )
    )
    return [CostProposal(
        kind="cost",
        analyzer="downsize",
        signature="cost:downsize:driver-role",
        title="Premium model in the driver role (route the work, not the thread)",
        target_key={"models": models, "suggestions": substitutes},
        evidence=evidence,
        coverage_note=coverage_note,
        baseline={
            "driver_sessions": sessions,
            "total_sessions": total_sessions,
            "driver_offload_usd": offload,
            "driver_tier_usd": tier,
            "driver_tail_tokens": tail,
            "driver_tokens": int(getattr(finding, "driver_tokens", 0) or 0),
            "substitutes": substitutes,
        },
        advise_text=_driver_role_advice(),
        suggestion=one_paste,
        one_paste_fix=one_paste,
        past_overspend_usd=round(usd, 6),
        past_overspend_tokens=int(getattr(finding, "driver_tokens", 0) or 0),
        estimate_basis=str(getattr(finding, "driver_estimate_basis", "") or ""),
    )]


def _per_agent_cache_recoverable_by_model(finding: Any) -> dict[tuple[str, str], tuple[float, int]]:
    """Sum of ``past_overspend_usd``/``past_overspend_tokens``
    already claimed by the root-caused per-agent cards (A1 uncached / A2
    thrash / A3 lookback), keyed by (provider, model).

    The generic per-(provider, model) efficacy row and these per-agent checks
    both read from the SAME underlying spans — a flagged agent's own calls
    are part of the aggregate the generic row's efficacy is computed over. So
    the dollars a per-agent card claims must be subtracted from the generic
    row's figure before it's surfaced, or the Review-inbox rollup (which sums
    every open card's ``past_overspend_usd`` with no analyzer
    allowlist) double-counts the same waste under two signatures. See
    ``_cache_to_proposals``.
    """
    totals: dict[tuple[str, str], tuple[float, int]] = {}
    groups = (
        getattr(finding, "uncached_agents", []) or [],
        getattr(finding, "thrash_agents", []) or [],
        getattr(finding, "lookback_miss_agents", []) or [],
    )
    for group in groups:
        for c in group:
            usd = getattr(c, "past_overspend_usd", None) or 0.0
            tokens = getattr(c, "past_overspend_tokens", None) or 0
            if usd <= 0 and tokens <= 0:
                continue
            key = (c.provider, c.model)
            prev_usd, prev_tokens = totals.get(key, (0.0, 0))
            totals[key] = (prev_usd + usd, prev_tokens + tokens)
    return totals


def _money(value: float) -> str:
    """A dollar figure with enough precision to stay honest at small values.

    Never rendered bare: every call site pairs it with an estimated/measured tag
    and the construction footnote.
    """
    if abs(value) >= 1.0:
        return f"${value:,.2f}"
    return f"${value:.4f}"


def _agent_arithmetic_line(row: Any) -> str:
    """The card's arithmetic, spelled out with both sides of the comparison."""
    window = (
        f"{row.sessions} session(s) over {row.window_days:.0f} day(s): "
        f"{row.input_tokens:,} input, {row.output_tokens:,} output, "
        f"{row.cache_tokens:,} cache read and {row.cache_write_tokens:,} cache "
        f"write tokens. At {row.model} rates that is "
        f"{_money(row.current_cost_usd)}; the same tokens at {row.alt_model} "
        f"rates are {_money(row.alternative_cost_usd)}. Difference: "
        f"{_money(row.delta_usd)} over the window, up to "
        f"{_money(row.projected_30d_delta_usd)} per 30 days at this rate "
        f"(estimated)."
    )
    if row.thinking_share_of_output is not None:
        window += (
            f" Thinking tokens were {row.thinking_share_of_output * 100:.0f}% of "
            f"this agent's output tokens over the same sessions "
            f"({row.thinking_tokens:,} of {row.output_tokens:,}, measured); "
            f"they bill as output on both models."
        )
    return window


#: The one sentence an Apply button on a model swap must never be allowed to
#: drown out (Critical Rule 14). The token-cost delta IS measured; quality
#: equivalence is never claimed, and a one-click write makes that distinction
#: easier to lose, not harder. Spliced into ``advise_text`` AND carried on its own
#: ``apply_caveat`` field from this ONE constant, so the sentence beside the button
#: and the sentence in the prose cannot drift into two different strengths of
#: claim — and so the card can render it OUTSIDE the collapsed description, where
#: a caveat behind a "Read more" would not have counted as visible.
MODEL_SWAP_QUALITY_CAVEAT = (
    "The price difference is arithmetic on this agent's own measured tokens. "
    "Whether the cheaper model answers as well is NOT measured here, so review "
    "the example sessions before applying."
)


def _model_swap_plumbing(row: Any, config: Any) -> dict[str, Any]:
    """Whether this agent's swap can be written directly, and where.

    Three outcomes, not two. The middle one is the point of this docstring.

    **Applyable now.** A registered source path and every precondition in
    ``model_apply.model_swap_precheck`` holding: the card carries ``apply_kind``
    and the resolved ``target_path``, and Apply writes.

    **Applyable once answered** (``needs_source_path``). The ONLY thing missing
    is that nobody ever told tokenjam where this agent's source lives — and
    tokenjam will not go looking, by design (``config.AgentConfig.source_path``:
    opt-in, never inferred, because scanning a filesystem for an agent's source
    is not a thing this product does). That is a QUESTION, so the row asks it
    instead of degrading to "Mark applied". ``apply_capable`` is true and
    ``apply_kind`` is deliberately UNSET: with no registered path there is no
    deterministic edit yet, so the row must not route to the apply endpoint that
    assumes one. It routes to the register-then-apply endpoint, which persists
    the answer to the user's config and re-runs every gate below against it.

    **Not applyable.** Any later gate fails — not a git repo, the model id in
    several files, the file dirty. None of those is answerable from a card, so
    the row stays advise-only with its one-paste artifact and says why.

    Branching on ``needs_source_path`` rather than on the reason string is
    load-bearing: two gates that report through one prose channel eventually get
    treated as one condition.
    """
    from tokenjam.core.optimize.model_apply import (
        APPLY_KIND_MODEL_SWAP,
        MODEL_SWAP_NEEDS_SOURCE_PATH,
        model_swap_precheck,
    )

    agents = getattr(config, "agents", None) or {}
    agent_cfg = agents.get(row.agent_id) if hasattr(agents, "get") else None
    source_path = str(getattr(agent_cfg, "source_path", "") or "")
    check = model_swap_precheck(source_path, row.model)
    if not check["ok"] and check.get(MODEL_SWAP_NEEDS_SOURCE_PATH):
        return {
            "apply_capable": True,
            # No apply_kind and no target_path until the user supplies the path.
            "needs_source_path": True,
            "current_model": row.model,
            "proposed_model": row.alt_model,
            "apply_blocked_reason": "",
        }
    if not check["ok"]:
        return {"apply_capable": False, "apply_blocked_reason": check["reason"]}
    return {
        "apply_capable": True,
        "apply_kind": APPLY_KIND_MODEL_SWAP,
        "source_path": source_path,
        "target_path": check["target_path"],
        "current_model": row.model,
        "proposed_model": row.alt_model,
        "apply_blocked_reason": "",
    }


def _downsize_agent_proposals(
    finding: Any, config: Any, persona: str = "unknown",
) -> list[CostProposal]:
    """One card per agent, carrying that agent's own price arithmetic.

    Replaces the window-wide card when per-agent rows exist, so one source of
    over-sized spend produces exactly one card rather than an aggregate plus its
    own parts.

    When the direct model-id swap is ``apply_capable`` (a registered, git-clean
    source path — see ``_model_swap_plumbing``), the fix is a real write to
    that agent's own config, not "switch your interactive model": it stays
    persona-agnostic. Only the NON-apply-capable fallback text needs the
    claude-code CTA, same reasoning as the window-wide card above.
    """
    proposals: list[CostProposal] = []
    for row in getattr(finding, "per_agent", []) or []:
        if row.delta_usd <= 0:
            # The proposed model is not actually cheaper for this agent's token
            # mix. There is nothing to recover, so there is no card: a rollup
            # that summed this would be summing a loss.
            continue
        plumbing = _model_swap_plumbing(row, config) if config is not None else {
            "apply_capable": False,
            "apply_blocked_reason": (
                "no tj config was available to look up a registered source path."
            ),
        }
        # THERE IS NOTHING TO REDEPLOY ON CLAUDE CODE, so this lane must not
        # word its offer that way. On Claude Code `agent_id` is
        # `claude-code-<cwd-basename>` (`core/backfill._agent_id_from_cwd`): it
        # names a PROJECT DIRECTORY whose sessions are ephemeral, not a service
        # with a process behind it. "Set the model id where it is configured,
        # then redeploy or restart the agent" names three things that do not
        # exist for that user, which reads as the product not understanding
        # their setup.
        #
        # The gate is a PERSONA check at the point of production, not a dollar
        # threshold (Critical Rule 26(c)): the observation is just as real for
        # a Claude Code window, so it keeps its figure and its card and is
        # handed the levers that persona actually has. Only the redeploy-shaped
        # OFFER is withheld — the same observation-stands / offer-withdrawn
        # split the applied-detection layer uses.
        redeployable = persona not in {"claude-code"}
        if redeployable:
            one_paste = (
                f"{row.model} -> {row.alt_model}\n"
                f"# Set this agent's model id to {row.alt_model} where it is "
                f"configured, then redeploy or restart the agent."
            )
        else:
            one_paste = _DOWNSIZE_CC_LEVER
        advise = (
            f"Route {row.agent_id}'s flagged structural-shaped work from "
            f"{row.model} to {row.alt_model}. " + MODEL_SWAP_QUALITY_CAVEAT
        )
        if not redeployable:
            advise += (
                f" {row.agent_id} is a Claude Code project directory, not a "
                f"deployed service: there is no process to restart and no "
                f"model id written down anywhere to change. What this window "
                f"already cost on {row.model} is reported above; the levers "
                f"that exist for this setup are below."
            )
        if not redeployable:
            # Levers this persona actually has. Stated once, from the shared
            # constant, so the CC action surface is described in one place.
            advise += " " + _DOWNSIZE_CC_LEVER
            plumbing = {
                "apply_capable": False,
                "apply_blocked_reason": (
                    "this agent id names a project directory, not a deployed "
                    "service, so there is no model id for tokenjam to rewrite."
                ),
            }
        elif plumbing.get("needs_source_path"):
            # Asks, rather than announcing a target it does not have. The
            # honesty caveat above is untouched: nothing here claims the cheaper
            # model answers as well, only that the substitution can be made.
            advise += (
                f" tokenjam can make this exact substitution for you, committed "
                f"and revertable in one call, once you point it at "
                f"{row.agent_id}'s local checkout below. After it is applied "
                f"you must redeploy or restart the agent: measurement starts at "
                f"the first call that runs on {row.alt_model}, not at the "
                f"moment of the write."
            )
        elif plumbing.get("apply_capable"):
            advise += (
                f" tokenjam can make this exact substitution in "
                f"{plumbing['target_path']}, with the change committed and "
                f"revertable in one call. After it is applied you must redeploy "
                f"or restart the agent: measurement starts at the first call "
                f"that runs on {row.alt_model}, not at the moment of the write."
            )
        elif plumbing.get("apply_blocked_reason"):
            advise += f" Applying it here is not on offer: {plumbing['apply_blocked_reason']}"
            if persona == "claude-code":
                advise += " " + _DOWNSIZE_CC_LEVER
            elif persona == "mixed":
                advise += _DOWNSIZE_MIXED_CC_NOTE
        proposals.append(CostProposal(
            kind="cost",
            analyzer="downsize",
            # The signature has to be as fine-grained as the ROW, or the rollup's
            # dedup-by-signature silently drops money that still renders.
            # `build_agent_price_rows` groups by (agent, provider, model,
            # alt_model), so one agent that ran two over-sized models yields two
            # rows — two distinct inbox cards, two distinct titles, two distinct
            # figures. Keyed on the agent alone they collided, and
            # `past_overspend_rollup` kept whichever sorted first and discarded
            # the rest, so the headline understated a total whose parts the user
            # could see listed underneath it. Mirror the grouping key exactly.
            signature=(
                f"cost:downsize:{row.agent_id}:{row.provider}:"
                f"{row.model}:{row.alt_model}"
            ),
            title=f"Model over-sizing in {row.agent_id} ({row.model} to {row.alt_model})",
            target_key={
                "agent_id": row.agent_id,
                "models": [row.model],
                "suggestions": {row.model: row.alt_model},
            },
            evidence=_agent_arithmetic_line(row),
            baseline={
                "agent_id": row.agent_id,
                "provider": row.provider,
                "model": row.model,
                "alt_model": row.alt_model,
                "sessions": row.sessions,
                "input_tokens": row.input_tokens,
                "output_tokens": row.output_tokens,
                "cache_tokens": row.cache_tokens,
                "cache_write_tokens": row.cache_write_tokens,
                "current_cost_usd": row.current_cost_usd,
                "alternative_cost_usd": row.alternative_cost_usd,
                "delta_usd": row.delta_usd,
                "projected_30d_delta_usd": row.projected_30d_delta_usd,
                "thinking_tokens": row.thinking_tokens,
                "thinking_share_of_output": row.thinking_share_of_output,
            },
            advise_text=advise,
            suggestion=one_paste,
            one_paste_fix=one_paste,
            past_overspend_usd=row.delta_usd,
            past_overspend_tokens=row.total_tokens,
            estimate_basis=row.estimate_basis,
            # `row.projected_30d_delta_usd` is this row's own `window_days`-based
            # self-projection and never reaches the card, for the same reason as
            # the window-wide downsize card above: one past-tense figure, never
            # a paced one.
            agent_id=row.agent_id if row.agent_id != "unknown" else "",
            advise_only=not plumbing.get("apply_capable", False),
            apply_capable=bool(plumbing.get("apply_capable")),
            apply_kind=str(plumbing.get("apply_kind", "")),
            source_path=str(plumbing.get("source_path", "")),
            target_path=str(plumbing.get("target_path", "")),
            current_model=str(plumbing.get("current_model", "")),
            proposed_model=str(plumbing.get("proposed_model", "")),
            apply_blocked_reason=str(plumbing.get("apply_blocked_reason", "")),
            needs_source_path=bool(plumbing.get("needs_source_path")),
            # Only where an Apply is actually on offer. On an advise-only row the
            # sentence is already in the prose and there is no button to qualify.
            apply_caveat=(
                MODEL_SWAP_QUALITY_CAVEAT if plumbing.get("apply_capable") else ""
            ),
        ))
    return proposals


def _placement_to_proposals(
    finding: Any, *, pricing_mode: str = "api", persona: str = "unknown",
) -> list[CostProposal]:
    """One card for the batch-placement candidates (advise-only).

    Advise-only is not a formality here: moving a workload to the batch lane is
    an architectural change in the user's own application, and the card says so
    beside the number.

    The Batch API's flat discount is an api-billed price lever — a
    subscription or local plan can't pull it, so ``pricing_mode`` gates the
    dollar figure exactly like the CLI's ``_render_placement`` already does
    (CLAUDE.md anti-pattern #22: never show a figure the reader can't act
    on). Without this the web Review inbox showed a batch-placement dollar
    figure the CLI deliberately suppresses for the same finding.

    ``persona`` gates the WHOLE card, not just the dollar figure: the Batch
    API is a synchronous-endpoint replacement in the user's own APPLICATION
    code, and an interactive Claude Code session has no application code of
    its own to move to a batch lane — the lever is structurally unreachable
    for that persona regardless of ``pricing_mode``. A ``"claude-code"``
    window gets nothing here (not even the advise-only/no-dollar branch);
    ``"mixed"``/``"sdk"``/``"unknown"`` are unaffected — the sdk share of a
    mixed window can still act on it.
    """
    if finding is None or persona == "claude-code":
        return []
    candidates = list(getattr(finding, "candidates", []) or [])
    if not candidates:
        return []
    agents = ", ".join(c.agent_id for c in candidates[:5])
    total = float(getattr(finding, "candidate_cost_usd", 0.0) or 0.0)
    saving = float(getattr(finding, "past_overspend_usd", 0.0) or 0.0)
    percent = float(getattr(finding, "percent_of_window_cost", 0.0) or 0.0)
    cadence = ", ".join(
        f"{c.agent_id} every {c.median_gap_seconds / 3600:.1f}h across "
        f"{c.sessions} runs"
        for c in candidates[:5]
    )
    evidence = (
        f"{len(candidates)} workload(s) ran on a regular cadence with no human "
        f"turn after the first model call ({cadence}). They are "
        f"{percent:.0f}% of the window's spend, {_money(total)} (measured)."
    )
    if pricing_mode == "api":
        advise = (
            f"The Batch API bills a flat 50% of standard prices, so the same work "
            f"on the batch lane is {_money(saving)} less over this window "
            f"(estimated). {getattr(finding, 'friction', '')} Nothing here is "
            f"applied for you; the change lives in your own application code."
        )
        recoverable_usd: float | None = saving
    else:
        advise = (
            f"The Batch API's flat discount is an api-billed price lever, so no "
            f"dollar figure is shown for this plan. "
            f"{getattr(finding, 'friction', '')} Nothing here is "
            f"applied for you; the change lives in your own application code."
        )
        recoverable_usd = None
    return [CostProposal(
        kind="cost",
        analyzer="placement",
        signature="cost:placement:batch",
        title="Batch API candidates (unattended, cadence-regular workloads)",
        target_key={"agents": [c.agent_id for c in candidates], "placement": "batch"},
        evidence=evidence,
        baseline={
            "candidates": [
                {
                    "agent_id": c.agent_id, "sessions": c.sessions,
                    "median_gap_seconds": c.median_gap_seconds, "gap_cv": c.gap_cv,
                    "cost_usd": c.cost_usd, "tokens": c.tokens,
                    "estimated_batch_saving_usd": c.estimated_batch_saving_usd,
                }
                for c in candidates
            ],
            "candidate_cost_usd": total,
            "window_cost_usd": float(getattr(finding, "window_cost_usd", 0.0) or 0.0),
            "percent_of_window_cost": percent,
        },
        advise_text=advise,
        suggestion=agents,
        one_paste_fix=(
            "# " + fixes.fix_text("batch.submit_through_batch_api") + "\n"
            + "\n".join(f"#   {c.agent_id}" for c in candidates)
        ),
        past_overspend_usd=recoverable_usd,
        past_overspend_tokens=getattr(finding, "past_overspend_tokens", None),
        estimate_basis=str(getattr(finding, "estimate_basis", "") or ""),
        agent_id=candidates[0].agent_id if len(candidates) == 1 else "",
    )]


#: Where the summarize card's affordance routes to instead of an inline apply
#: — the ``tj summarize`` curate/diff screen (see ``_summarize_to_proposals``).
SUMMARIZE_REVIEW_HREF = "#/optimize/summarize"


def _summarize_to_proposals(finding: Any) -> list[CostProposal]:
    """One window-wide card for the ``summarize`` (oversized catalog prompt
    files) finding — see ``COST_ANALYZERS``'s docstring for why this is now a
    normal peer card rather than a link-only disclosure.

    One card, never one per file: ``finding.candidates`` can list several
    files, but the standing don't-fill-the-inbox constraint (#596) means this
    adds at most a single row regardless of how many files the scan flags.

    Deliberately never ``apply_capable``: the fix is a reviewed rewrite
    (structure kept, prose compressed, one file at a time) driven by
    ``core/summarize/``'s prepare/check/apply lifecycle, not a value this
    adapter can safely one-click. The card's copy and its UI affordance both
    route to that surface (``SUMMARIZE_REVIEW_HREF``) instead of offering an
    "Apply" button that would misrepresent a multi-step review as one click.
    """
    if finding is None:
        return []
    files = int(getattr(finding, "files", 0) or 0)
    usd = getattr(finding, "past_overspend_usd", None)
    tokens = getattr(finding, "past_overspend_tokens", None)
    if files <= 0 or (usd is None and tokens is None):
        # No candidates, or candidates with no observed load evidence to price
        # against (see `SummarizeFinding`'s "carries neither figure" case) —
        # a card with nothing to lead with would misstate "worth nothing" for
        # "not measured this window".
        return []
    candidates = list(getattr(finding, "candidates", []) or [])
    shown = ", ".join(c.path for c in candidates[:5])
    if len(candidates) > 5:
        shown += f", +{len(candidates) - 5} more"
    reduction = getattr(finding, "avg_reduction_pct", None)
    evidence = (
        f"{files} catalog prompt file(s) carry compressible prose"
        + (f" ({reduction}% average reduction)" if reduction is not None else "")
        + f": {shown}."
    )
    plural = "" if files == 1 else "s"
    headline = _money(usd) if usd is not None else f"~{tokens:,} tok"
    # The GROUNDING (how many files, in this window) is built here; the
    # instruction itself comes from the catalog. It used to be a ~330-character
    # paragraph hardcoded at this line, and the guard read green over it for
    # three independent reasons at once — see
    # `tests/unit/test_no_fix_prose_outside_the_catalog.py`.
    advise = (
        f"{files} oversized file{plural} in this window. "
        + fixes.fix_text("summarize.review_oversized_files")
    )
    return [CostProposal(
        kind="cost",
        analyzer="summarize",
        signature="cost:summarize",
        title=f"Review {files} oversized file{plural}, {headline}",
        target_key={"href": SUMMARIZE_REVIEW_HREF, "files": [c.path for c in candidates]},
        evidence=evidence,
        baseline={
            "files": files,
            "file_reduction_tokens": getattr(finding, "file_reduction_tokens", None),
            "sessions_examined": int(getattr(finding, "sessions_examined", 0) or 0),
            "calls_per_session": getattr(finding, "calls_per_session", None),
            "avg_reduction_pct": reduction,
            # The on-demand half of the figure's inputs, carried for the same
            # reason the always-on half's are: a skill/command/agent body is
            # priced by OBSERVED invocations, and a reader should be able to
            # see that count rather than take the total on trust.
            "invocations_observed": bool(
                getattr(finding, "invocations_observed", False),
            ),
            "invocations_total": int(getattr(finding, "invocations_total", 0) or 0),
        },
        advise_text=advise,
        past_overspend_usd=usd,
        past_overspend_tokens=tokens,
        estimate_basis=str(getattr(finding, "estimate_basis", "") or ""),
        caveat=str(getattr(finding, "caveat", "") or COST_CORRELATIONAL_CAVEAT),
    )]


# THE RELEARN AGGREGATE CARD IS GONE, and the deletion is the whole story.
#
# `_relearn_to_proposals` built one window-wide card whose ONLY figure was the
# total-observed-cost field (`past_overspend_usd` stayed None on it, deliberately:
# relearn's re-read tail is the same re-sent context `resend` already claims in
# full, and two analyzers claiming one span is CLAUDE.md rule 27). With that field
# deleted the card had no number to carry at all, and a card that renders "Not
# priced" where its headline belongs is worse than no card.
#
# It also had no reader. Measured on the live inbox before the purge: the card was
# absent from the rendered list entirely — the sub-$5 floor deliberately does not
# hide an unpriced item, so it was not filtered; it simply never earned a row, and
# no other surface read it either. Critical Rule 24 covers exactly this: a surface
# nothing links to is not a surface.
#
# relearn's measured cost is NOT lost. Every cluster still renders its own row in
# the Review inbox, on the canonical field, counted once — which is where the
# claim always lived. What is lost is relearn's presence in the cross-analyzer
# rollup, and that presence was only ever via the separate total this purge
# removes; it was never inside the headline avoidable figure. `relearn` stays in
# `COST_ANALYZERS` so the analyzer still runs and still feeds the write budget and
# its own per-cluster surface.
#
# Upstream reached the same place from the other side and got there first for its
# half: `c83ec77c` retired relearn's FORWARD claim from this card (the last
# dollar-field exception), dropping `relearn_claim_usd`/`_tokens` off `baseline`
# and the `estimate_basis` caption with them. That is step one of the retirement
# sequence the repo CLAUDE.md lays out; the purge above is step two. Neither
# decision is being reverted here — taken together they leave the card with no
# figure of either kind, which is why the card itself goes rather than shipping a
# headline slot reading "Not priced".


# --------------------------------------------------------------------------- #
# Persona-gated fix TEXT — the `cache` family (`cache` / `cache_thrash` /
# `cache-recommend`) only. Unrelated to `_persona_gated_write_fields` below:
# that helper picks between a workspace WRITE and a snippet fallback for
# script/reuse/verbosity. No cache proposal has a write to offer in the
# first place — every cache fix is an edit to a `cache_control` (or TTL)
# field on the raw Anthropic API request, code a Claude Code session never
# constructs itself (the harness builds that request on the user's behalf).
# There's no workspace surface to fall back to for that persona; the
# instruction itself is the thing they can't act on, so the only honest
# move is to say why instead of stating it as their fix (CLAUDE.md
# anti-pattern #22 — never hand out an instruction the reader can't
# follow). The measured finding underneath (efficacy percentage, token
# counts, evidence) is unaffected — it's true and useful regardless of who
# can act on the fix, so only `advise_text`/`suggestion` are ever gated,
# never `evidence`/`baseline`/the estimate fields.
#
# "unknown" stays on the actionable branch (unlike
# `_persona_gated_write_fields`, which groups "unknown" with "sdk" to keep a
# WRITE off by default): here the safe default is the opposite direction,
# since withholding is the risky move (a real fix denied) rather than
# over-offering one (an unusable write silently succeeding). "mixed" keeps
# the instruction too — the sdk share of a mixed window can act on it, and
# suppressing it would cost that share a real fix for no benefit to the
# claude-code share.
# --------------------------------------------------------------------------- #

# THE text lives in `core/fixes/registry.py`, where the lint can see that it
# says no action is needed and that the record is marked advisory accordingly.
CACHE_NO_LEVER_TEXT = fixes.fix_text("cache.no_claude_code_lever")


def _cache_breakpoint_fix(model: str | Any = "") -> str:
    """The caching instruction, once, named to the observed model when known.

    FOUR call sites in this module each wrote their own wording of this — "Add
    a stable cache prefix / enable prompt caching", "Add a cache_control
    breakpoint on this agent's stable prefix", "right after this prefix", and
    the resend card's own. They were one instruction, so they read from one
    record now; what actually differed between them was WHICH model or prefix
    was observed, and that is grounding, not a fifth wording.

    Degrades to the generic sentence when the model was not observed. An empty
    slot skips its substitution rather than rendering a plausible-looking
    guess, which is the same rule every other grounded fix follows.
    """
    from tokenjam.core.fixes.grounding import Evidence, ground

    record = fixes.fix_for("resend.sdk_cache_breakpoint")
    if record is None:                       # pragma: no cover - import-time bug
        return ""
    name = str(model or "").strip()
    return ground(record, Evidence(models=(name,)) if name else None)


def _persona_gated_cache_fields(
    persona: str, advise_text: str, suggestion: str = "",
) -> dict[str, Any]:
    """Decide, from the window's dominant persona, whether the cache_control
    instruction (and any snippet) is shown at all. See the module-section
    comment above for the reasoning."""
    if persona == "claude-code":
        return {"advise_text": CACHE_NO_LEVER_TEXT, "suggestion": ""}
    return {"advise_text": advise_text, "suggestion": suggestion}


def _cache_to_proposals(finding: Any, persona: str = "unknown") -> list[CostProposal]:
    """One proposal per flagged (provider, model) cache-efficacy row.

    Reduced by whatever the more specific per-agent root-cause cards (A1/A2/
    A3) already claim for that same (provider, model) — see
    ``_per_agent_cache_recoverable_by_model`` — so the rollup never sums the
    same underlying waste twice under two different signatures.

    The instruction is gated by ``persona`` — see
    ``_persona_gated_cache_fields``; a ``"claude-code"`` window gets the
    honest no-lever reason instead of a cache_control edit it can't make.
    """
    if finding is None:
        return []
    from tokenjam.core.optimize.analyzers.cache_efficacy import (
        estimate_cache_recoverable,
    )

    already_claimed = _per_agent_cache_recoverable_by_model(finding)

    proposals: list[CostProposal] = []
    for row in getattr(finding, "flagged", []) or []:
        usd, tokens = estimate_cache_recoverable([row])
        claimed_usd, claimed_tokens = already_claimed.get((row.provider, row.model), (0.0, 0))
        basis = str(getattr(finding, "estimate_basis", "") or "")
        if claimed_usd > 0 or claimed_tokens > 0:
            usd = round(max(0.0, (usd or 0.0) - claimed_usd), 6)
            tokens = max(0, (tokens or 0) - claimed_tokens)
            basis = (
                basis + (" " if basis else "")
                + f"Reduced by ${claimed_usd:.4f} already attributed to more "
                "specific per-agent cache proposals for this model, so the "
                "rollup does not double-count the same spend."
            )
        evidence = (
            f"{row.provider}/{row.model}: {row.efficacy * 100:.0f}% of input "
            f"tokens served from cache over {row.input_tokens:,} input tokens "
            f"(caching support: {row.support})."
        )
        proposals.append(CostProposal(
            kind="cost",
            analyzer="cache",
            signature=f"cost:cache:{row.provider}:{row.model}",
            title=f"Low cache efficacy on {row.model}",
            target_key={"provider": row.provider, "model": row.model},
            evidence=evidence,
            baseline={
                "provider": row.provider,
                "model": row.model,
                "input_tokens": int(row.input_tokens),
                "cache_tokens": int(row.cache_tokens),
                "efficacy": float(row.efficacy),
                "efficacy_ceiling": float(getattr(finding, "efficacy_ceiling", 0.80)),
            },
            past_overspend_usd=usd,
            past_overspend_tokens=tokens,
            estimate_basis=basis,
            **_persona_gated_cache_fields(
                persona,
                _cache_breakpoint_fix(row.model),
            ),
        ))
    return proposals


def _cache_uncached_to_proposals(finding: Any, persona: str = "unknown") -> list[CostProposal]:
    """One proposal per A1 uncached-agent candidate (see
    ``analyzers.cache_efficacy``): an agent group making cacheable calls with
    prompt caching never attempted. Scored through the same efficacy metric
    as ``_cache_to_proposals`` (agent-scoped).

    Persona-gated the same way as ``_cache_to_proposals`` — see
    ``_persona_gated_cache_fields``.
    """
    if finding is None:
        return []
    proposals: list[CostProposal] = []
    for c in getattr(finding, "uncached_agents", []) or []:
        evidence = (
            f"{c.agent_id}: {c.calls} calls on {c.model} with zero prompt "
            f"caching attempted (no cache reads, no cache writes) across "
            f"{c.sessions} session(s); assumed stable prefix "
            f"~{c.assumed_prefix_tokens:,} tokens (this agent's own p25 input size)."
        )
        proposals.append(CostProposal(
            kind="cost",
            analyzer="cache",
            signature=f"cost:cache-uncached:{c.agent_id}",
            title=f"Uncached agent: {c.agent_id}",
            target_key={"agent_id": c.agent_id, "provider": c.provider, "model": c.model},
            evidence=evidence,
            baseline={
                "agent_id": c.agent_id, "provider": c.provider, "model": c.model,
                "calls": c.calls, "sessions": c.sessions,
                "assumed_prefix_tokens": c.assumed_prefix_tokens,
            },
            past_overspend_usd=c.past_overspend_usd,
            past_overspend_tokens=c.past_overspend_tokens,
            estimate_basis=c.estimate_basis,
            agent_id=c.agent_id,
            **_persona_gated_cache_fields(
                persona,
                _cache_breakpoint_fix(getattr(c, "model", "")),
                c.cache_control_snippet,
            ),
        ))
    return proposals


def _cache_thrash_to_proposals(finding: Any, persona: str = "unknown") -> list[CostProposal]:
    """One proposal per A2 cache-thrash candidate. Card text branches on the
    detected root cause: a TTL-cadence card (honest break-even, which may say
    the switch isn't worth it) versus an instability checklist card.

    Persona-gated the same way as ``_cache_to_proposals`` — see
    ``_persona_gated_cache_fields``. The instability checklist is written for
    someone reading their own prompt-assembly code, so it's gated exactly
    like the TTL branch's cache_control edit — neither is something a Claude
    Code session can act on.
    """
    if finding is None:
        return []
    from tokenjam.core.optimize.analyzers.cache_efficacy import (
        SILENT_INVALIDATOR_CHECKLIST,
    )

    proposals: list[CostProposal] = []
    for c in getattr(finding, "thrash_agents", []) or []:
        evidence = (
            f"{c.agent_id}: caching attempted on {c.model} but read:write "
            f"ratio is {c.read_write_ratio:.2f} over {c.calls} calls "
            f"({c.cache_read_tokens:,} cache-read tokens vs "
            f"{c.cache_write_tokens:,} cache-write tokens); median inter-call "
            f"gap {c.inter_call_gap_p50_minutes:.1f} min."
        )
        if c.cause == "ttl":
            if c.ttl_worth_it:
                advise = (
                    "Calls land more than 5 minutes apart, so the default "
                    "5-minute cache write is expiring before it's reused. "
                    "Switching to the 1-hour cache TTL is estimated to pay "
                    "off at this cadence."
                )
            else:
                advise = (
                    "Calls land more than 5 minutes apart, so the default "
                    "5-minute cache write is expiring before it's reused. "
                    "The 1-hour TTL's write premium doesn't clear at this "
                    "cadence: caching not worth it at this cadence."
                )
        else:
            advise = (
                "Calls land close enough together that a TTL expiry doesn't "
                "explain the miss rate; the prefix itself is likely changing "
                "between calls. " + SILENT_INVALIDATOR_CHECKLIST
            )
        proposals.append(CostProposal(
            kind="cost",
            analyzer="cache_thrash",
            signature=f"cost:cache-thrash:{c.agent_id}",
            title=f"Cache thrash: {c.agent_id}",
            target_key={"agent_id": c.agent_id, "provider": c.provider, "model": c.model},
            evidence=evidence,
            baseline={
                "agent_id": c.agent_id, "provider": c.provider, "model": c.model,
                "calls": c.calls, "cache_write_tokens": c.cache_write_tokens,
                "cache_read_tokens": c.cache_read_tokens,
                "read_write_ratio": c.read_write_ratio, "cause": c.cause,
                "inter_call_gap_p50_minutes": c.inter_call_gap_p50_minutes,
                "ttl_worth_it": c.ttl_worth_it,
                "ttl_breakeven_usd": c.ttl_breakeven_usd,
            },
            past_overspend_usd=c.past_overspend_usd,
            past_overspend_tokens=c.past_overspend_tokens,
            estimate_basis=c.estimate_basis,
            agent_id=c.agent_id,
            **_persona_gated_cache_fields(persona, advise, c.cache_control_snippet),
        ))
    return proposals


def _cache_lookback_to_proposals(finding: Any, persona: str = "unknown") -> list[CostProposal]:
    """One proposal per A3 20-block-lookback-miss candidate. Weakest-
    confidence check of the three; the analyzer only classifies an agent here
    when A1/A2 don't already explain its cache waste.

    Persona-gated the same way as ``_cache_to_proposals`` — see
    ``_persona_gated_cache_fields``.
    """
    if finding is None:
        return []
    from tokenjam.core.optimize.analyzers.cache_efficacy import LOOKBACK_BLOCK_LIMIT

    proposals: list[CostProposal] = []
    for c in getattr(finding, "lookback_miss_agents", []) or []:
        evidence = (
            f"{c.agent_id}: {c.miss_count} cache miss(es) on {c.model}, each "
            f"directly following a turn with an estimated "
            f"{c.avg_prior_turn_blocks:.0f} content blocks (lookback limit: "
            f"{LOOKBACK_BLOCK_LIMIT})."
        )
        proposals.append(CostProposal(
            kind="cost",
            analyzer="cache",
            signature=f"cost:cache-lookback:{c.agent_id}",
            title=f"20-block lookback miss: {c.agent_id}",
            target_key={"agent_id": c.agent_id, "provider": c.provider, "model": c.model},
            evidence=evidence,
            baseline={
                "agent_id": c.agent_id, "provider": c.provider, "model": c.model,
                "miss_count": c.miss_count,
                "avg_prior_turn_blocks": c.avg_prior_turn_blocks,
            },
            past_overspend_usd=c.past_overspend_usd,
            past_overspend_tokens=c.past_overspend_tokens,
            estimate_basis=c.estimate_basis,
            agent_id=c.agent_id,
            **_persona_gated_cache_fields(
                persona,
                "Anthropic's cache breakpoint search looks back at most "
                f"{LOOKBACK_BLOCK_LIMIT} content blocks. Long tool-heavy "
                "turns push the prior breakpoint out of range; add an "
                "intermediate cache_control breakpoint every ~15 blocks in "
                "long tool-use turns.",
                c.cache_control_snippet,
            ),
        ))
    return proposals


def _cache_recommend_to_proposals(
    finding: Any, cache_finding: Any = None, persona: str = "unknown",
) -> list[CostProposal]:
    """One proposal per Anthropic prefix candidate ``cache-recommend`` flags —
    a placement recommendation for a `cache_control` breakpoint, carrying a
    ready-to-paste snippet (``CachePrefixCandidate.cache_control_snippet``,
    built in the analyzer, modelled on ``cache_efficacy``'s own snippet
    builders).

    Same class of finding as ``_cache_to_proposals``'s A1/A2/A3 root causes —
    an edit to a raw Anthropic API request a Claude Code session never
    constructs itself — so it's gated by the same rule; see
    ``_persona_gated_cache_fields``.

    Reduced by whatever ``cache``'s own per-agent root-cause cards (A1/A2/A3)
    already claim for the same model — the same dedup rule ``_cache_to_
    proposals`` applies against its own generic row, via the same
    ``_per_agent_cache_recoverable_by_model`` helper — so a prefix already
    counted as an uncached-agent / thrash / lookback-miss recovery doesn't
    also get counted here under a third signature. ``cache_finding`` is
    optional (the two analyzers are wired independently and either can be
    absent from a report); with none supplied, nothing is subtracted.
    """
    if finding is None or not getattr(finding, "enabled", False):
        return []
    candidates = list(getattr(finding, "candidates", []) or [])
    if not candidates:
        return []

    already_claimed = (
        _per_agent_cache_recoverable_by_model(cache_finding)
        if cache_finding is not None else {}
    )

    proposals: list[CostProposal] = []
    for c in candidates:
        usd = c.past_overspend_usd
        tokens = c.past_overspend_tokens
        basis = str(getattr(finding, "estimate_basis", "") or "")
        claimed_usd, claimed_tokens = already_claimed.get(("anthropic", c.model), (0.0, 0))
        if claimed_usd > 0 or claimed_tokens > 0:
            usd = round(max(0.0, (usd or 0.0) - claimed_usd), 6)
            tokens = max(0, (tokens or 0) - claimed_tokens)
            basis = (
                basis + (" " if basis else "")
                + f"Reduced by ${claimed_usd:.4f} already attributed to "
                "per-agent cache proposals for this model, so the rollup "
                "does not double-count the same spend."
            )
        preview = c.sample_chars[:60].replace("\n", " ").strip()
        evidence = (
            f'A stable prompt prefix (starting "{preview}...") recurred '
            f"across {c.occurrences} calls on {c.model or 'an unrecorded model'}, "
            f"averaging {c.avg_input_tokens:,.0f} input tokens per call; "
            f"~{c.estimated_cacheable_tokens:,} tokens of that prefix estimated "
            "cacheable."
        )
        proposals.append(CostProposal(
            kind="cost",
            analyzer="cache-recommend",
            signature=f"cost:cache-recommend:{c.prefix_hash}",
            title=f"Repeated prompt prefix on {c.model or 'unrecorded model'}",
            target_key={"prefix_hash": c.prefix_hash, "model": c.model},
            evidence=evidence,
            baseline={
                "prefix_hash": c.prefix_hash,
                "model": c.model,
                "occurrences": c.occurrences,
                "avg_input_tokens": c.avg_input_tokens,
                "estimated_cacheable_tokens": c.estimated_cacheable_tokens,
            },
            past_overspend_usd=usd,
            past_overspend_tokens=tokens,
            estimate_basis=basis,
            **_persona_gated_cache_fields(
                persona,
                _cache_breakpoint_fix(c.model),
                c.cache_control_snippet,
            ),
        ))
    return proposals


def _trim_to_proposals(finding: Any) -> list[CostProposal]:
    """One proposal per flagged agent/step (grouped from ``per_prompt``)."""
    if finding is None or not getattr(finding, "enabled", False):
        return []
    per_prompt = list(getattr(finding, "per_prompt", []) or [])
    if not per_prompt:
        return []

    # Group per_prompt by agent_id — the flagged "step". Sum bloat across an
    # agent's prompts so one card represents one step.
    by_agent: dict[str, dict[str, Any]] = {}
    for p in per_prompt:
        agent = str(getattr(p, "agent_id", "") or "unknown")
        acc = by_agent.setdefault(agent, {
            "bloat_chars": 0, "prompt_chars": 0, "token_reduction": 0, "prompts": 0,
        })
        acc["bloat_chars"] += int(getattr(p, "bloat_chars", 0) or 0)
        acc["prompt_chars"] += int(getattr(p, "prompt_chars", 0) or 0)
        acc["token_reduction"] += int(getattr(p, "estimated_token_reduction", 0) or 0)
        acc["prompts"] += 1

    # Prorate the finding-level dollar estimate across agents by bloat share, so
    # each card carries a coherent (labeled) slice rather than the whole figure.
    total_bloat = sum(a["bloat_chars"] for a in by_agent.values()) or 1
    finding_usd = getattr(finding, "past_overspend_usd", None)

    proposals: list[CostProposal] = []
    for agent, acc in sorted(by_agent.items()):
        if acc["bloat_chars"] <= 0:
            continue
        share = acc["bloat_chars"] / total_bloat
        usd = round(finding_usd * share, 6) if finding_usd is not None else None
        evidence = (
            f"{agent}: {acc['bloat_chars']:,} low-significance characters across "
            f"{acc['prompts']} prompt(s) (~{acc['token_reduction']:,} trimmable "
            f"input tokens)."
        )
        proposals.append(CostProposal(
            kind="cost",
            analyzer="trim",
            signature=f"cost:trim:{agent}",
            title=f"Prompt bloat in {agent}",
            target_key={"agent_id": agent},
            evidence=evidence,
            baseline={
                "agent_id": agent,
                "bloat_chars": acc["bloat_chars"],
                "prompt_chars": acc["prompt_chars"],
                "estimated_token_reduction": acc["token_reduction"],
            },
            advise_text=fixes.fix_text("trim.drop_low_significance_regions"),
            past_overspend_usd=usd,
            past_overspend_tokens=acc["token_reduction"] or None,
            estimate_basis=str(getattr(finding, "estimate_basis", "") or ""),
            agent_id=agent if agent != "unknown" else "",
        ))
    return proposals


#: A name only addresses an agent definition file (``.claude/agents/<name>.md``)
#: when it is a plain lowercase slug. Built-in dispatch types that carry no
#: definition file (``Explore``, ``Plan``) fail this on their capital letter,
#: which is correct — there is nothing on disk to edit for those.
#: CASE-INSENSITIVE ON PURPOSE. Claude Code's own built-in dispatch types
#: include capitalised names (`Explore`, `Plan`), and those are precisely the
#: ones the define-an-override fix targets — a user subagent named `Explore`
#: overrides the built-in and keeps its own `model`. A lowercase-only shape
#: check silently rejected them, so the fix that needs them most could never
#: reach its own branch. The dispatch-id guard below is what keeps this from
#: matching a per-dispatch id; the case rule was never doing that work.
_AGENT_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9]*(?:-[A-Za-z0-9]+)*$")

#: A Claude Code DISPATCH id — ``a`` + an optional caller-chosen instance label
#: + a hex suffix (``af8b26e872b7184a7``, ``aw-ratehistory-7e1dd2a1642d7c29``).
#: Minted per dispatch; names no file.
#:
#: WHY THIS IS NOT REDUNDANT with ``_AGENT_NAME_RE`` above, which it may look
#: like: a dispatch id is a lowercase letter followed by hex and dashes, which
#: IS a well-formed plain slug. So the slug check matches essentially every
#: dispatch id and rejects essentially none — it reads as a filter while
#: filtering nothing, and that is precisely how the agent-file lookup came to
#: spend its life hunting for a ``.claude/agents/a<hex>.md`` that cannot exist.
#: Without this note the stricter predicate below looks like defensive
#: over-engineering and the obvious "simplification" is to drop it, which
#: restores the bug. The real fix is that this module resolves against
#: ``sub_agent_type``, never a dispatch id; this pattern is the guard that
#: keeps a dispatch id from being mistaken for a definition name anyway.
_DISPATCH_ID_RE = re.compile(r"^a.*[0-9a-f]{16,17}$")

#: Cap on transcripts read to locate the repos a finding's sessions ran in.
#: Counted in DISTINCT sessions — the flagged rows are one per (session,
#: dispatch) and a session commonly fans out to a dozen-plus subagents, so
#: slicing the raw row list capped the scope decision at a small fraction of the
#: sessions it was meant to cover.
_MAX_SCOPE_SESSIONS = 20


def _names_agent_definition(name: str) -> bool:
    """Whether ``name`` could address a ``.claude/agents/<name>.md`` file.

    Rejects the empty string, anything that is not a plain lowercase slug, and
    any Claude Code per-dispatch id (which satisfies the slug shape but names no
    file). A True here is a shape check only — the caller still has to find the
    file on disk.
    """
    if not name or _DISPATCH_ID_RE.match(name):
        return False
    return bool(_AGENT_NAME_RE.match(name))


def _session_cwds(session_ids: list[str], config: Any) -> dict[str, str]:
    """``session_id -> repo cwd``, read from the sessions' own transcripts.

    Reuses the relearn detector's resolver rather than re-deriving a cwd from
    the encoded project directory name, which is unreliable. Best-effort: an
    unreadable transcript simply contributes no cwd.
    """
    from tokenjam.core.optimize.analyzers.relearn import _repo_cwd_map_for
    from tokenjam.core.transcript import resolve_projects_root

    override = getattr(getattr(config, "loop", None), "transcript_path", None)
    root = resolve_projects_root(override)
    # Dedupe BEFORE the cap: callers pass one entry per flagged ROW, and a
    # session that fanned out to N subagents contributes N identical ids, so
    # slicing the raw list spent the whole budget on a handful of sessions.
    unique = list(dict.fromkeys(session_ids))[:_MAX_SCOPE_SESSIONS]
    pairs = [(sid, sid) for sid in unique]
    return _repo_cwd_map_for(pairs, root)


def _agent_model_plumbing(over_powered: list[Any], config: Any) -> dict[str, Any]:
    """Whether a flagged subagent has a definition file whose model can be set.

    Keyed on the row's ``sub_agent_type`` — the agent TYPE that was dispatched,
    which is the only identity that names a file. ``sub_agent_id`` is minted per
    dispatch and cannot: it looks like a slug, so it used to pass the name check
    and send every lookup after a ``.claude/agents/a<hex>.md`` that can never
    exist, which is why this path resolved nothing at all.

    EXPECT THIS TO RETURN ``{}`` ON A CODING-AGENT WINDOW, AND DO NOT TREAT THAT
    AS A BUG IN THE LOOKUP. Measured on a real coding-agent corpus, every type
    actually dispatched was a Claude Code BUILT-IN (``general-purpose``,
    ``Explore``, ``fork`` and friends). A built-in has no
    ``.claude/agents/<name>.md`` — there is nothing on disk to open, whether or
    not the user has an agents directory at all — so the ``model:`` write branch
    below is unreachable in practice and the card degrades to the rubric that
    ``_subagent_to_proposals`` falls back to.

    That is a fact about how people USE the tool (they dispatch built-ins rather
    than authoring named agent definitions), not a defect in the resolution
    above: fed a user-defined type whose file exists, this resolves it. A reader
    who sees the empty result and starts debugging the lookup, loosening the
    name predicate, or widening the scope search is chasing a working mechanism.
    Recorded at more length in the persona matrix under product-state.

    Scope routing is relearn's: sessions concentrated in one repo write into
    that repo's ``.claude/agents/``, sessions spanning repos write into the
    user-global one. The flagged rows are cost-ordered, so the first subagent
    with a real definition file is the most expensive one that can be fixed
    outright. No file means the guidance block stays the fix — that covers the
    built-in dispatch types (``general-purpose``, ``Explore``) as well as a
    type whose definition lives outside the resolved scope.
    """
    from tokenjam.core.optimize.analyzers.model_downgrade import lookup_downgrade
    from tokenjam.core.optimize.analyzers.relearn import _scope_for
    from tokenjam.core.optimize.model_apply import (
        APPLY_KIND_AGENT_MODEL,
        default_agent_file_path,
    )

    named = [
        r for r in over_powered
        if _names_agent_definition(str(getattr(r, "sub_agent_type", "") or ""))
    ]
    if not named or config is None:
        return {}
    cwds = _session_cwds([str(r.session_id) for r in over_powered], config)
    repos = {Path(cwd).name for cwd in cwds.values() if cwd}
    scope = _scope_for(repos)
    repo_cwd = next(iter(cwds.values()), "") if len(repos) == 1 else ""

    # An EXISTING definition file is the strongest case: its `model:` key is a
    # value already written down, so the fix is a deterministic rewrite.
    for row in named:
        proposed = lookup_downgrade(str(row.provider), str(row.model))
        if not proposed:
            continue
        name = str(row.sub_agent_type)
        path = default_agent_file_path(scope, repo_cwd, name)
        if not path or not Path(path).is_file():
            continue
        return {
            "apply_kind": APPLY_KIND_AGENT_MODEL,
            "agent_name": name,
            "target_path": path,
            "scope": scope,
            "current_model": str(row.model),
            "proposed_model": proposed,
            "over_provisioned": "over_provisioned" in (getattr(row, "flags", None) or []),
            "creates_file": False,
        }

    # NO FILE IS NOT NO FIX. Every dispatch type Claude Code actually uses is a
    # BUILT-IN (`general-purpose`, `Explore`, `Plan`, `fork`), and a built-in
    # has no `.claude/agents/<name>.md` — so requiring the file to exist made
    # this branch unreachable on the dominant corpus and pushed the whole claim
    # down to a prose rubric. That read as "we cannot fix this", which is not
    # what the absence means.
    #
    # A user or project subagent defined with the SAME NAME as a built-in
    # overrides it and keeps its own `model` field, so the file can simply be
    # created. The product was not failing to find a file; it was declining to
    # create one. See `core/fixes/registry.SUBAGENT_DEFINE_BUILTIN` for the
    # user-facing statement, including the bit that explains the waste's origin:
    # `model` defaults to `inherit`, so an Opus-driven session hands every Task
    # dispatch an Opus worker unless something pins otherwise. Nobody decided
    # that — it is an unset default, which is why it goes unnoticed.
    for row in named:
        proposed = lookup_downgrade(str(row.provider), str(row.model))
        if not proposed:
            continue
        name = str(row.sub_agent_type)
        path = default_agent_file_path(scope, repo_cwd, name)
        if not path:
            continue
        return {
            "apply_kind": APPLY_KIND_AGENT_MODEL,
            "agent_name": name,
            "target_path": path,
            "scope": scope,
            "current_model": str(row.model),
            "proposed_model": proposed,
            "over_provisioned": "over_provisioned" in (getattr(row, "flags", None) or []),
            # The caller words the card differently for a create: "set its
            # model key" is wrong when the file does not exist yet, and a user
            # told to edit a file they do not have reads it as a bug.
            "creates_file": True,
        }
    return {}


def _subagent_to_proposals(finding: Any, config: Any = None) -> list[CostProposal]:
    """One proposal covering the subagent right-sizing finding.

    Unlike the three advise-only analyzers, this one is workspace-appliable for
    the common (CC-origin) case: the fan-out model choice is made by the
    orchestrating agent, which reads the workspace's CLAUDE.md — so a
    sizing rubric note IS a legitimate, reversible workspace fix. The subagent
    analyzer runs only over Claude Code data (``sub_agent_id`` is populated by
    the CC backfill; other runtimes carry NULL and are ignored), so a finding
    here is CC-origin, hence ``apply_capable``. If no oversized model is priced
    (nothing to key a delta on), the proposal degrades to advise-only.

    The delta-verify pass measures the fan-out model-mix cost delta across the
    over-powered models, so a single proposal listing them keeps the finding-
    level estimate coherent (mirrors the downsize adapter).
    """
    if finding is None:
        return []
    flagged = list(getattr(finding, "flagged", []) or [])
    over_powered = [r for r in flagged if "over_powered" in (getattr(r, "flags", []) or [])]
    if not over_powered:
        return []

    models = sorted({str(r.model) for r in over_powered})
    subagents = len({(r.session_id, r.sub_agent_id) for r in over_powered})
    pct = float(getattr(finding, "percent_of_cost", 0.0) or 0.0) * 100
    model_list = ", ".join(models)
    # State what the `over_powered` gate actually tests — `is_premium_tier(model)`
    # plus the cost floor, and NOTHING about output size or tool-call count (see
    # `subagent_rightsizing._flags_for`). The old wording ("did little work,
    # small output, few tool calls") described `over_provisioned`'s gate, not
    # this one, so a dispatch that ran hundreds of tool calls and returned a
    # large result was told it did little work — contradicting the numbers
    # rendered on the same card. `over_provisioned` legitimately keeps that
    # language on its own evidence (see `_derived_effort`); this sentence must
    # not borrow it.
    floor = float(getattr(finding, "min_flag_cost_usd", 0.0) or 0.0)
    evidence = (
        f"{subagents} subagent dispatch(es) ran on a premium-tier model "
        f"({model_list}), above the {_money(floor)} per-dispatch cost floor "
        f"this flag applies. Subagents are {pct:.0f}% of the window's cost."
    )
    # The trailing sentence here used to RESTATE the rubric — "route that shape
    # to the cheaper same-family model next time" is the rubric's own core
    # instruction in fewer words — so the artifact a user pastes into a
    # CLAUDE.md carried the same directive twice, a paragraph apart. The
    # pairwise catalog lint cannot see it, because only one of the two is a
    # record. What the sentence legitimately added is the OBSERVATION, which is
    # kept: the models are evidence, not a second copy of the rule.
    proposed_fix = (
        SUBAGENT_RUBRIC_INTRO
        + f"\n\nObserved oversized dispatches ran on: {model_list}."
    )
    # Apply-capable when we have a concrete model to name in the rubric; else
    # degrade to advise-only (no clean workspace surface to write).
    apply_capable = bool(models)
    # The stronger surface, when the flagged subagent has a definition file: set
    # its `model:` key outright instead of writing a rubric the orchestrator has
    # to read and honor. Falls back to that rubric when there is no file.
    try:
        agent_apply = _agent_model_plumbing(over_powered, config)
    except Exception:
        agent_apply = {}
    if agent_apply:
        creates = bool(agent_apply.get("creates_file"))
        if creates:
            # A built-in dispatch type has no definition file, and that is not
            # a dead end: a user or project subagent of the same name overrides
            # the built-in and keeps its own `model`. The card has to SAY that,
            # because "set its model key" reads as a bug to someone who does
            # not have the file.
            advise_extra = (
                f" {agent_apply['agent_name']} is a built-in, so it has no "
                f"definition file and inherits whatever model this session "
                f"runs on — `model` defaults to `inherit`, so an Opus-driven "
                f"session hands it an Opus worker unless something pins "
                f"otherwise. That is an unset default, not a decision. "
                f"Defining a subagent with the same name overrides the "
                f"built-in and keeps its own model, so tokenjam can create "
                f"{agent_apply['target_path']} pinned to "
                f"{agent_apply['proposed_model']}. Its next dispatch runs on "
                f"the new model, which is where measurement starts."
            )
        else:
            advise_extra = (
                f" {agent_apply['agent_name']} has its own definition file, so "
                f"tokenjam can set its model key to "
                f"{agent_apply['proposed_model']} directly. The change is committed "
                f"where the file is in a repo and reverts in one call. Its next "
                f"dispatch runs on the new model, which is where measurement starts."
            )
        return [CostProposal(
            kind="cost",
            analyzer="subagent",
            signature=f"cost:subagent:{agent_apply['agent_name']}",
            title=(
                f"Built-in subagent {agent_apply['agent_name']} inherits this "
                f"session's model (pin {agent_apply['proposed_model']})"
                if creates else
                f"Over-powered subagent {agent_apply['agent_name']} "
                f"({agent_apply['current_model']} to {agent_apply['proposed_model']})"
            ),
            target_key={
                "models": models, "subagent": True,
                "agent_name": agent_apply["agent_name"],
            },
            evidence=evidence,
            baseline={
                "flagged_subagents": subagents,
                "flagged_cost_usd": float(getattr(finding, "flagged_cost_usd", 0.0) or 0.0),
                "subagent_cost_usd": float(getattr(finding, "subagent_cost_usd", 0.0) or 0.0),
                "percent_of_cost": float(getattr(finding, "percent_of_cost", 0.0) or 0.0),
                "agent_name": agent_apply["agent_name"],
                "current_model": agent_apply["current_model"],
                "proposed_model": agent_apply["proposed_model"],
                "creates_file": creates,
            },
            advise_text=(
                "Lower the model tier for the flagged Task dispatches. "
                + str(getattr(finding, "caveat", "") or "") + advise_extra
            ).strip(),
            suggestion=f"model: {agent_apply['proposed_model']}",
            one_paste_fix=(
                (
                    f"# Create {agent_apply['target_path']}\n"
                    f"---\n"
                    f"name: {agent_apply['agent_name']}\n"
                    f"model: {agent_apply['proposed_model']}\n"
                    f"---"
                ) if creates else (
                    f"# In {agent_apply['target_path']}, frontmatter:\n"
                    f"model: {agent_apply['proposed_model']}"
                )
            ),
            past_overspend_usd=getattr(finding, "past_overspend_usd", None),
            past_overspend_tokens=getattr(finding, "past_overspend_tokens", None),
            estimate_basis=str(getattr(finding, "estimate_basis", "") or ""),
            advise_only=False,
            apply_capable=True,
            scope=agent_apply["scope"],
            apply_kind=agent_apply["apply_kind"],
            agent_name=agent_apply["agent_name"],
            target_path=agent_apply["target_path"],
            current_model=agent_apply["current_model"],
            proposed_model=agent_apply["proposed_model"],
        )]
    return [CostProposal(
        kind="cost",
        analyzer="subagent",
        signature="cost:subagent",
        title="Over-powered subagent dispatches (route the fan-out to a cheaper model)",
        target_key={"models": models, "subagent": True},
        evidence=evidence,
        baseline={
            "flagged_subagents": subagents,
            "flagged_cost_usd": float(getattr(finding, "flagged_cost_usd", 0.0) or 0.0),
            "subagent_cost_usd": float(getattr(finding, "subagent_cost_usd", 0.0) or 0.0),
            "percent_of_cost": float(getattr(finding, "percent_of_cost", 0.0) or 0.0),
        },
        advise_text=(
            "Lower the model tier for the flagged Task dispatches. On Claude Code "
            "this is a sizing rubric in your CLAUDE.md (apply it below) that the "
            "orchestrating agent reads before it spawns subagents. "
            + str(getattr(finding, "caveat", "") or "")
        ).strip(),
        past_overspend_usd=getattr(finding, "past_overspend_usd", None),
        past_overspend_tokens=getattr(finding, "past_overspend_tokens", None),
        estimate_basis=str(getattr(finding, "estimate_basis", "") or ""),
        advise_only=not apply_capable,
        apply_capable=apply_capable,
        delivery=DELIVERY_CLAUDE_MD_RULE if apply_capable else "",
        scope="project" if apply_capable else "",
        proposed_fix=proposed_fix if apply_capable else "",
    )]


def _mcp_remove_plumbing(server: Any) -> dict[str, Any]:
    """Whether ``server``'s config entry can be removed directly, and where.

    Unlike ``model_swap`` there is no search step: ``ConfiguredServer``
    already resolved the exact config file at detection time. This just
    re-verifies that still holds (the file can have moved, or a human can
    have already removed the entry by hand) at proposal-build time, so the
    card's pre-filled target is current, not stale analyzer-time data.
    """
    from tokenjam.core.optimize.analyzers.deadweight import (
        APPLY_KIND_MCP_REMOVE,
        mcp_remove_precheck,
    )

    check = mcp_remove_precheck(server.source, server.name)
    if not check["ok"]:
        return {"apply_capable": False, "apply_blocked_reason": check["reason"]}
    return {
        "apply_capable": True,
        "apply_kind": APPLY_KIND_MCP_REMOVE,
        "source_path": check["target_path"],
        "target_path": check["target_path"],
        "apply_blocked_reason": "",
    }


def _deadweight_to_proposals(finding: Any) -> list[CostProposal]:
    """One proposal per unused MCP server (Component C1).

    Reads ONLY ``DeadweightFinding.unused_servers`` — the C2 tax table (which
    lists every configured server, dead or alive, purely for ranked
    visibility) never feeds a proposal here, so a server's schema-injection
    tax is never counted both in the tax table AND a proposal (the same
    dedup guarantee ``compute_deadweight_finding`` itself enforces on
    ``past_overspend_tokens`` / ``past_overspend_usd``).

    ``past_overspend_usd`` is carried straight off the analyzer's own
    ``ServerDeadweight.estimated_tax_usd_window`` — a WINDOW-scoped figure
    with no projection folded in (#273: this used to be a fixed 90-day
    projection, which made it incomparable to every other analyzer's window
    figure and corrupted any sum across them). The analyzer already prices
    the token tax through ``core/pricing.py`` at the dominant model observed
    in that server's sessions (never a hardcoded rate; see
    ``deadweight._pricing_note``). Stays ``None`` when no priced model was
    observed for that server — this adapter never invents a rate itself.

    Apply-capable, like ``downsize``'s ``model_swap`` cards: the fix is a
    deterministic edit of a value already written down (the server's own
    ``mcpServers`` entry), so it routes through the same
    ``relearn_apply.apply_relearn_fix`` machinery under
    ``APPLY_KIND_MCP_REMOVE`` — reversible, git-committed where the config
    lives in a repo, one-step revert. Falls back to the one-paste ``claude
    mcp remove`` command, with the reason stated, when the precondition
    doesn't hold (file missing, malformed, or the entry already gone).
    """
    if finding is None:
        return []
    proposals: list[CostProposal] = []
    # Whole-window blind spot (a recorded session cwd that no longer exists
    # on disk, so this analyzer could not check that project's MCP config at
    # all), not per-server -- same string on every server's card below, since
    # deadweight emits N proposals (one per dead server) rather than the
    # single card `resend`/`relearn` attach their own `coverage_note` to.
    coverage_note = str(getattr(finding, "coverage_note", "") or "")
    # The MEASUREMENT blind spot rides on the same string. A card carrying a
    # priced total while other servers were excluded from it is showing a floor
    # as a total, and the Review inbox is exactly as capable of that as the
    # terminal was.
    measurement_note = str(getattr(finding, "measurement_note", "") or "")
    if measurement_note:
        coverage_note = " ".join(x for x in (coverage_note, measurement_note) if x)
    for server in getattr(finding, "unused_servers", []) or []:
        # A server measured to cost NOTHING has nothing to recover, so it gets
        # no card at all. This is also what keeps the two figures on one basis:
        # `tokens or None` coerces a measured zero to None while the dollar
        # figure stays a real 0.0, and a card with `tokens=None, usd=$0.00` is
        # the mixed-basis defect Critical Rule 28 forbids. Reachable, not
        # hypothetical — a server exposing zero tools measures to zero tokens.
        if not server.estimated_tax_tokens_window:
            continue
        evidence = (
            f"`{server.name}` MCP server ({server.scope} scope, configured at "
            f"{server.source}) made 0 tool calls across {server.sessions_present} "
            f"session(s) in the window."
        )
        if server.deferred_sessions:
            evidence += (
                f" ToolSearch deferred its schema in {server.deferred_sessions} "
                f"of those session(s)."
            )
        scope_flag = "user" if server.scope == "user" else "project"
        plumbing = _mcp_remove_plumbing(server)
        # "(or project-scoping)" is only a real second option for a
        # USER-scoped (global) server; a server already at project scope has
        # nothing left to narrow, so offering it there would be a no-op that
        # delivers none of the claim. This adapter used to hardcode the
        # alternative unconditionally, independently of `server.fix`'s own
        # copy of the same wording in deadweight.py -- the same bug existing
        # in two separate places is exactly what happens when the REASON
        # for a conditional is never written down next to it. Keep both
        # copies scope-conditional; do not let one drift back to a bare
        # string while the other stays conditional.
        reversible_note = (
            " Removing (or project-scoping) it is reversible and loses no "
            "data; it only stops the standing schema-injection tax on "
            "future sessions."
            if server.scope == "user" else
            " Removing it is reversible and loses no data; it only stops "
            "the standing schema-injection tax on future sessions."
        )
        advise = server.fix + reversible_note
        if plumbing.get("apply_capable"):
            advise += (
                f" tokenjam can remove this exact entry from "
                f"{plumbing['target_path']}, with the change committed and "
                f"revertable in one call."
            )
        elif plumbing.get("apply_blocked_reason"):
            advise += f" Applying it here is not on offer: {plumbing['apply_blocked_reason']}"
        if server.other_sources:
            # The one-paste `claude mcp remove` fallback (and, for that
            # matter, the deterministic auto-apply above) both edit exactly
            # ONE file. Neither reaches the other independently-declared
            # copies, so a reader must be told to repeat the command at each
            # of them rather than assuming one run closed out the claim.
            advise += (
                f" `claude mcp remove` (and the auto-apply above, if "
                f"offered) only edit {server.source}; the same command "
                f"needs to be run again from each of the "
                f"{len(server.other_sources)} other location(s) that "
                f"independently declare `{server.name}` to stop the rest "
                f"of the claimed tax."
            )
        proposals.append(CostProposal(
            kind="cost",
            analyzer="deadweight",
            signature=f"cost:deadweight:{server.name}",
            title=f"Unused MCP server: {server.name}",
            target_key={
                "server": server.name, "scope": server.scope, "source": server.source,
            },
            evidence=evidence,
            baseline={
                "sessions_present": server.sessions_present,
                "invocations": server.invocations,
                "deferred_sessions": server.deferred_sessions,
                "scope": server.scope,
                "source": server.source,
                "example_sessions": list(server.example_sessions),
                "priced_model": server.priced_model,
                "other_sources": list(server.other_sources),
                "primary_source_sessions": server.primary_source_sessions,
            },
            advise_text=advise,
            suggestion=f"claude mcp remove {server.name} --scope {scope_flag}",
            past_overspend_tokens=server.estimated_tax_tokens_window or None,
            past_overspend_usd=server.estimated_tax_usd_window,
            estimate_basis=server.tax_construction,
            coverage_note=coverage_note,
            advise_only=not plumbing.get("apply_capable", False),
            apply_capable=bool(plumbing.get("apply_capable")),
            apply_kind=str(plumbing.get("apply_kind", "")),
            agent_name=server.name,
            source_path=str(plumbing.get("source_path", "")),
            target_path=str(plumbing.get("target_path", "")),
            scope=server.scope,
            apply_blocked_reason=str(plumbing.get("apply_blocked_reason", "")),
        ))
    return proposals


def _deadweight_plugin_to_proposals(finding: Any) -> list[CostProposal]:
    """One proposal per unused plugin — reaches the Review inbox with the
    same lifecycle as an unused MCP server's card (Part D).

    Reads ONLY ``DeadweightFinding.unused_plugins``: a plugin with SOME
    components used and some not (``PluginDeadweight.partial_use_no_fix``)
    never gets a card here — the toggle is whole-plugin only, so there is
    genuinely no fix to offer, and a card dangling a number with no action
    behind it is worse than no card (root anti-pattern 22 in the meta-repo
    `CLAUDE.md`). A plugin with nothing priced (`estimated_tax_tokens_window
    == 0`) is skipped the same way `_deadweight_to_proposals` skips a
    zero-tax server — never a card that reads "$0, disable it".

    Never apply-capable. Unlike the MCP server's deterministic config-file
    splice, there is no file this adapter can edit on the user's behalf: the
    real fix is the CLI command `claude plugin disable <name>`, which the
    user runs themselves (see
    ``core/fixes/registry.DEADWEIGHT_DISABLE_PLUGIN``). Every card is
    advise-only with that command as its copy-pasteable ``suggestion`` —
    the same shape the MCP card falls back to when a direct splice isn't
    offered, just without the apply-capable branch, since no splice exists
    to attempt.
    """
    from tokenjam.core.optimize.analyzers.deadweight import UNUSED_RECENCY_WINDOW_DAYS

    if finding is None:
        return []
    proposals: list[CostProposal] = []
    for plugin in getattr(finding, "unused_plugins", []) or []:
        if not plugin.estimated_tax_tokens_window:
            continue
        component_names = ", ".join(
            f"{c.kind} `{c.name}`" for c in getattr(plugin, "components", []) or []
        )
        evidence = (
            f"`{plugin.name}` (enabled, resident): {component_names} — "
            f"nothing fired in {UNUSED_RECENCY_WINDOW_DAYS} days."
        )
        proposals.append(CostProposal(
            kind="cost",
            analyzer="deadweight",
            signature=f"cost:deadweight:plugin:{plugin.name}",
            title=f"Unused plugin: {plugin.name}",
            target_key={"plugin": plugin.name, "install_scope": plugin.install_scope},
            evidence=evidence,
            baseline={
                "skills": plugin.skills,
                "agents": plugin.agents,
                "components": [
                    {"kind": c.kind, "name": c.name}
                    for c in getattr(plugin, "components", []) or []
                ],
                "sessions_present": plugin.sessions_present,
                "priced_model": plugin.priced_model,
            },
            advise_text=plugin.fix,
            suggestion=f"claude plugin disable {plugin.name}",
            past_overspend_tokens=plugin.estimated_tax_tokens_window or None,
            past_overspend_usd=plugin.estimated_tax_usd_window,
            estimate_basis=plugin.tax_construction,
            advise_only=True,
            apply_capable=False,
            agent_name=plugin.name,
            scope=plugin.install_scope,
        ))
    return proposals


def _cluster_hash(value: Any) -> str:
    """Stable 12-hex-char identity for a cluster's structural key. Deterministic
    across runs over the same underlying signature (a JSON-serialisable
    structure), used only where the analyzer itself doesn't already hand back
    a cluster id (contrast ``ReuseCluster.cluster_id``, which does)."""
    encoded = json.dumps(value, sort_keys=False, default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:12]


# --------------------------------------------------------------------------- #
# Persona-gated fix modality — `script` / `reuse` only. (`verbosity` shares
# the same write SURFACE — see its own module comment above `_verbosity_to_
# proposals` — but never offers the write at all, for any persona: its
# cohort-scoped flag would become a global CLAUDE.md rule if written, which
# fails the no-quality-tax gate outright regardless of who reads it.)
#
# Both write the SAME class of artifact when apply-capable: a CLAUDE.md rule
# or a `.claude/skills/<slug>/SKILL.md` skill file (see
# `relearn_apply.default_target_path`). Nothing in an SDK-only service's
# request path ever reads a CLAUDE.md or a `.claude/skills/` note — those are
# read by an interactive coding-agent harness. Offering that write to an SDK
# caller is a write that visibly succeeds and changes nothing: a quiet lie in
# the user's favour (CLAUDE.md anti-pattern #22). The finding underneath is
# still true for an SDK caller; only the fix MODALITY is wrong, so this never
# drops the recommendation — it demotes the card to advise-only and carries
# the identical text as a copy-pasteable `suggestion` instead (the same
# ``CostProposal.suggestion`` field every advise-only card already renders as
# a first-class "the fix" block with a Copy button).
# --------------------------------------------------------------------------- #

def _persona_gated_write_fields(
    persona: str, proposed_fix: str, delivery: str, scope: str,
) -> dict[str, Any]:
    """Decide, from the window's dominant persona, whether the workspace write
    is offered — and fill in the ``CostProposal`` fields that
    follow from that decision.

    * ``"claude-code"`` — unchanged: the write is genuinely actionable, so it
      stays offered exactly as before.
    * ``"sdk"`` and ``"unknown"`` — no write offered. ``"unknown"`` means no
      session in the window carries an identifiable agent_id and no declared
      plan settles it either (`core.framing.dominant_persona`) — the exact
      shape of a pure-SDK caller who never ran ``tj onboard``. Grouping it
      with ``"sdk"`` here mirrors ``cmd_optimize._render_downgrade_cta``'s
      CTA, and avoids the one failure mode called out for this fix: silently
      offering a write to a persona that turns out to be SDK.
    * ``"mixed"`` — both audiences are meaningfully represented (same
      precedent as ``_render_downgrade_cta``, which renders both CTAs side
      by side rather than picking one), and a script/reuse/verbosity finding
      isn't attributable to one side of the mix or the other. The write
      stays on offer for the claude-code share; the identical recommendation
      is ALSO carried as ``suggestion`` so the sdk share of the mix isn't
      left with a card that looks actionable but silently isn't, for them.
    """
    write_offered = persona in {"claude-code", "mixed"}
    fields: dict[str, Any] = {
        "advise_only": not write_offered,
        "apply_capable": write_offered,
        "delivery": delivery if write_offered else "",
        "scope": scope if write_offered else "",
        "proposed_fix": proposed_fix if write_offered else "",
    }
    # Every persona except a clean claude-code window also gets the snippet
    # fallback — "mixed" needs it alongside the write (see above), and
    # "sdk"/"unknown" need it in place of the write.
    if persona != "claude-code":
        fields["suggestion"] = proposed_fix
    return fields


def _script_to_proposals(finding: Any, persona: str = "unknown") -> list[CostProposal]:
    """One proposal per flagged deterministic-tool-call cluster.

    Apply-capable as a skill: a note naming the repeated call pattern and
    recommending a script in its place. No agent-file/model-swap surface
    exists here (this isn't a model-routing finding), so unlike ``subagent``
    there is only the one apply shape. The skill's slug is derived from the
    title, which embeds the cluster's own hash so two clusters never collide
    on the same skill file (`relearn_apply`'s create-only guard would otherwise
    let a second cluster's apply silently overwrite the first's skill note).

    The write is only actually offered for a ``"claude-code"``/``"mixed"``
    ``persona`` — see ``_persona_gated_write_fields``. An ``"sdk"``/
    ``"unknown"`` window gets the identical recommendation as a
    copy-pasteable snippet instead; the skill note would sit unread by any
    SDK service's request path.
    """
    if finding is None:
        return []
    clusters = list(getattr(finding, "clusters", []) or [])
    if not clusters:
        return []
    degraded = bool(getattr(finding, "degraded", False))
    caveat = str(getattr(finding, "caveat", "") or "")

    proposals: list[CostProposal] = []
    for cluster in clusters:
        if cluster.total_cost_usd <= 0 and cluster.total_tokens <= 0:
            continue
        cluster_hash = _cluster_hash(cluster.signature)
        tool_names = [step.get("tool", "?") for step in cluster.signature]
        tool_list = " -> ".join(tool_names) or "(no tools recorded)"
        title = f"Deterministic tool pattern: {tool_list} ({cluster_hash})"
        evidence = (
            f"{cluster.instances} sessions ran the same tool-call structure "
            f"({tool_list}), averaging {cluster.avg_tokens:,} input+output "
            f"tokens and {_money(cluster.avg_cost_usd)} per session."
        )
        if degraded:
            evidence += (
                " Clustered on tool names only (enable [capture] tool_inputs "
                "in tj.toml for the finer argument-shape signature)."
            )
        advise = (
            _SCRIPT_SKILL_INTRO + " " + caveat
        ).strip()
        proposals.append(CostProposal(
            kind="cost",
            analyzer="script",
            signature=f"cost:script:{cluster_hash}",
            title=title,
            target_key={"signature": cluster.signature, "instances": cluster.instances},
            evidence=evidence,
            baseline={
                "instances": cluster.instances,
                "avg_cost_usd": cluster.avg_cost_usd,
                "avg_tokens": cluster.avg_tokens,
                "avg_duration_seconds": cluster.avg_duration_seconds,
                "example_session_id": cluster.example_session_id,
                "degraded": degraded,
                "apply_sessions": cluster.instances,
                "apply_examples": [
                    {"session_id": sid, "repo": "", "snippet": tool_list[:160]}
                    for sid in (cluster.example_session_ids or [])
                ],
            },
            advise_text=advise,
            past_overspend_usd=cluster.total_cost_usd or None,
            past_overspend_tokens=cluster.total_tokens or None,
            estimate_basis=str(getattr(finding, "estimate_basis", "") or ""),
            # `reuse` clusters this exact repeated-tool-sequence shape too
            # (see its own adapter below); carrying the full member set lets
            # `_net_cross_analyzer_session_overlap` catch the overlap instead
            # of both cards claiming the same sessions' cost.
            claimed_session_ids=tuple(cluster.member_session_ids),
            **_persona_gated_write_fields(
                persona, advise, delivery=DELIVERY_SKILL, scope="project",
            ),
        ))
    return proposals


def _reuse_to_proposals(finding: Any, persona: str = "unknown") -> list[CostProposal]:
    """One proposal per repeated planning-skeleton cluster.

    Apply-capable as a CLAUDE.md rule naming the recurring skeleton.
    Uses the finding's conservative ``cache_reuse_recoverable_*`` figure (you
    already paid for the plan once), not the ``script_replacement_*`` upper
    bound, matching ``ReuseFinding``'s own aggregate.

    The write is only actually offered for a ``"claude-code"``/``"mixed"``
    ``persona`` — see ``_persona_gated_write_fields``.
    """
    if finding is None:
        return []
    clusters = list(getattr(finding, "clusters", []) or [])
    if not clusters:
        return []

    proposals: list[CostProposal] = []
    for cluster in clusters:
        if cluster.cache_reuse_recoverable_usd <= 0 and cluster.cache_reuse_recoverable_tokens <= 0:
            continue
        tool_list = ", ".join(cluster.tool_signature) or "(no tool calls after the plan)"
        title = f"Repeated planning skeleton: {tool_list} ({cluster.cluster_id})"
        evidence = (
            f"{cluster.repetitions} sessions shared a planning-call skeleton "
            f"(tool sequence after the plan: {tool_list}), averaging "
            f"{cluster.avg_planning_tokens:,} planning tokens "
            f"({_money(cluster.avg_planning_cost_usd)} per call)."
        )
        advise = (
            _REUSE_NOTE_INTRO + " " + str(cluster.caveat or "")
        ).strip()
        proposals.append(CostProposal(
            kind="cost",
            analyzer="reuse",
            signature=f"cost:reuse:{cluster.cluster_id}",
            title=title,
            target_key={
                "cluster_id": cluster.cluster_id,
                "tool_signature": list(cluster.tool_signature),
            },
            evidence=evidence,
            baseline={
                "repetitions": cluster.repetitions,
                "avg_planning_tokens": cluster.avg_planning_tokens,
                "avg_planning_cost_usd": cluster.avg_planning_cost_usd,
                "skeleton_session_id": cluster.skeleton_session_id,
                "apply_sessions": cluster.repetitions,
                "apply_examples": [
                    {"session_id": sid, "repo": "", "snippet": tool_list[:160]}
                    for sid in (cluster.example_session_ids or [])
                ],
            },
            advise_text=advise,
            past_overspend_usd=cluster.cache_reuse_recoverable_usd or None,
            past_overspend_tokens=cluster.cache_reuse_recoverable_tokens or None,
            estimate_basis=str(getattr(finding, "estimate_basis", "") or ""),
            # See the matching comment on `_script_to_proposals` — `script`
            # clusters the identical shape, so this is what lets the two be
            # reconciled instead of both claiming the same sessions.
            claimed_session_ids=tuple(cluster.member_session_ids),
            **_persona_gated_write_fields(
                persona, advise, delivery=DELIVERY_CLAUDE_MD_RULE, scope="project",
            ),
        ))
    return proposals


def _verbosity_to_proposals(finding: Any, persona: str = "unknown") -> list[CostProposal]:
    """One proposal for the whole verbosity finding (unlike ``script``/
    ``reuse``, this is a single window-wide signal, not per-cluster).

    ALWAYS advise-only, regardless of ``persona`` — no CLAUDE.md write
    is ever offered here, unlike ``script``/``reuse`` which share the same
    write machinery. A cohort-scoped flag (this task shape ran long vs its
    OWN like-shaped peers) written as a global CLAUDE.md note would apply
    blanket terseness pressure to every future task, including legitimately
    verbose ones — the analyzer's own honesty caveat says output length is
    not waste, so an enforced-looking file note fails that gate outright.
    The recommendation is still carried as a copy-pasteable ``suggestion``
    for every persona (never silently dropped), it just never becomes a
    write. ``persona`` is accepted (and kept in the dispatcher's uniform
    ``(f, persona=persona)`` call shape) but intentionally unused — see
    above.
    """
    if finding is None:
        return []
    total_candidates = int(getattr(finding, "total_candidates", 0) or 0)
    if total_candidates <= 0:
        return []
    remedy = str(getattr(finding, "remedy_snippet", "") or "")
    max_tokens = getattr(finding, "suggested_max_tokens", None)
    caveat = str(getattr(finding, "caveat", "") or "")
    evidence = (
        f"{total_candidates} session(s) across "
        f"{int(getattr(finding, 'cohorts_examined', 0) or 0)} task-shape "
        f"cohort(s) ran output well above their cohort's median."
    )
    advise = remedy
    if max_tokens:
        # A DATA POINT, not an enforceable cap — "keep responses under N
        # tokens" reads as a rule once pasted anywhere; this states what
        # similar sessions happened to do instead.
        advise += (
            f" For reference, similar sessions in this cohort typically "
            f"finished under about {max_tokens:,} output tokens — a data "
            f"point to weigh, not a limit to enforce."
        )
    advise = (advise + " " + caveat).strip()
    return [CostProposal(
        kind="cost",
        analyzer="verbosity",
        signature="cost:verbosity",
        title="High-verbosity output vs cohort baseline",
        target_key={"total_candidates": total_candidates},
        evidence=evidence,
        baseline={
            "total_candidates": total_candidates,
            "sessions_examined": int(getattr(finding, "sessions_examined", 0) or 0),
            "cohorts_examined": int(getattr(finding, "cohorts_examined", 0) or 0),
            "suggested_max_tokens": max_tokens,
            "apply_sessions": total_candidates,
        },
        advise_text=advise,
        suggestion=advise,
        past_overspend_usd=getattr(finding, "past_overspend_usd", None),
        past_overspend_tokens=getattr(finding, "past_overspend_tokens", None),
        estimate_basis=str(getattr(finding, "estimate_basis", "") or ""),
    )]


def _rightsize_target(subagent_finding: Any, config: Any) -> dict[str, Any]:
    """The concrete agent file the compound offload card should name, if one
    exists: the most expensive over-powered subagent that has its own
    definition file, plus the cheaper model to pin in it.

    Reuses ``_agent_model_plumbing`` — the same resolver the ``subagent`` card
    already uses — rather than re-deriving a path, so the two cards can never
    name different files for the same agent. Empty dict when the subagent
    analyzer didn't run, nothing was flagged, or no flagged subagent has a
    definition file; the card then states the right-sizing lever generically
    instead of inventing an agent name.
    """
    if subagent_finding is None:
        return {}
    flagged = list(getattr(subagent_finding, "flagged", []) or [])
    over_powered = [r for r in flagged if "over_powered" in (getattr(r, "flags", []) or [])]
    if not over_powered:
        return {}
    try:
        return _agent_model_plumbing(over_powered, config)
    except Exception:
        return {}


def compound_offload_fix(rightsize: dict[str, Any], fix_offload: str, fix_rightsize: str) -> str:
    """The single CLAUDE.md rule that carries BOTH halves of the compound lever.

    PUBLIC because the CLI renders the same finding and must lead with the same
    fix. It used to lead with `/compact` while the inbox card led with this —
    two surfaces showing different fixes for one finding, and the CLI showing
    the weaker one, which `COMPACTION_FIX`'s own text disclaims as "never fixes
    the pattern going forward".

    One artifact, not two: the offload directive decides where context-heavy
    work runs, the right-sizing directive decides what it runs on, and they
    compound. Writing them as one block is what keeps this a consolidation of
    the resend / subagent / downsize recommendations rather than a third card
    on top of them.
    """
    parts = [fix_offload, fix_rightsize]
    if rightsize:
        parts.append(
            f"Concretely: {rightsize['agent_name']} already runs oversized for "
            f"the work it does — pin `model: {rightsize['proposed_model']}` in "
            f"{rightsize['target_path']} and set its reasoning effort to match "
            f"the task rather than inheriting the parent's."
        )
    return "\n\n".join(p for p in parts if p)


#: The effort levels Claude Code's subagent frontmatter accepts. The KEY is
#: ``effort``. This snippet used to emit ``reasoning_effort``, which is not a
#: field Claude Code reads at all — so the user pasted it, believed effort was
#: pinned, and nothing changed. A silently-ignored key is worse than no line:
#: it converts an unfixed problem into one the user believes is fixed.
AGENT_EFFORT_LEVELS = ("low", "medium", "high", "xhigh", "max")


def _derived_effort(rightsize: dict[str, Any]) -> str | None:
    """The effort to pin, or ``None`` when the observation does not say.

    The old snippet hardcoded ``low`` for every dispatch. That is a guess, and
    on this analyzer's population it is usually the WRONG guess: the flagged
    rows are the large, tool-heavy, long-output dispatches (the `output_tokens`
    / `tool_calls` gate was deleted precisely because it excluded them), and
    telling a user to pin those to low effort recommends making the expensive
    work worse rather than cheaper.

    So effort is emitted only where the observation actually supports it — an
    ``over_provisioned`` dispatch, which by construction was handed a large
    context and produced little output — and omitted otherwise. Omitting a line
    we cannot derive is the honest default; a plausible-looking guess in a
    frontmatter block is indistinguishable from a measurement to the reader.
    """
    if not rightsize.get("over_provisioned"):
        return None
    return "low"


def _rightsize_frontmatter_snippet(rightsize: dict[str, Any]) -> str:
    """The copyable agent-file frontmatter for the right-sizing half.

    Both keys live in the same ``.claude/agents/<name>.md`` frontmatter block,
    so the second half of the compound fix is one paste, not two — but the
    effort line appears only when the observation supports a value for it (see
    :func:`_derived_effort`).
    """
    name = rightsize.get("agent_name") or "<subagent-name>"
    model = rightsize.get("proposed_model") or "<cheaper-same-family-model>"
    path = rightsize.get("target_path") or f".claude/agents/{name}.md"
    effort = _derived_effort(rightsize)
    effort_line = f"effort: {effort}\n" if effort else ""
    return (
        f"# {path} — frontmatter\n"
        f"---\n"
        f"name: {name}\n"
        f"model: {model}\n"
        f"{effort_line}"
        f"---"
    )


def _resend_to_proposals(
    finding: Any, persona: str = "unknown",
    subagent_finding: Any = None, config: Any = None,
) -> list[CostProposal]:
    """One window-wide card for the ``resend`` (context re-send) finding.

    This card is COMPOUND by design: it consolidates what resend and subagent
    right-sizing would otherwise say on separate cards, because the two are one
    behavioural change (offload context-heavy work to a subagent, and size that
    subagent to the work) applied through one CLAUDE.md rule. Consolidating rather
    than adding is deliberate — the Review inbox does not grow.

    Persona-gated like every other lever-bearing adapter. Three levers exist
    and they are NOT interchangeable:

    * a CLAUDE.md rule instructing offload of context-heavy sub-tasks
      to subagents (``fix_subagent_offload``) is the DURABLE claude-code
      lever: it persists across sessions and stops the repeated volume from
      accumulating on the main thread in the first place. Apply-capable via
      the same ``_persona_gated_write_fields`` machinery ``script`` /
      ``reuse`` / ``verbosity`` already use, for a ``claude-code``/``mixed``
      persona — see that helper.
    * ``/compact`` / a fresh session is a MANUAL, per-session, transient
      action available to everyone, but it fixes nothing going forward (a
      real CC user who feels a session getting too full typically abandons
      it and starts fresh anyway rather than compacting). It is carried only
      as a secondary, immediate-relief note for an already-full session —
      never the headline fix.
    * an SDK-side ``cache_control`` breakpoint is the SDK/API developer's
      ADDITIONAL lever. A Claude Code (or other agent-harness) window never
      constructs the request itself, so that snippet is not theirs to paste —
      showing it to them is the same wrong-audience defect the cache family
      already avoids via ``_persona_gated_cache_fields``. Suppressed for a
      ``claude-code`` persona, unchanged from before.

    ``ResendFinding.estimate_basis`` documents the discounted derivation; this
    adapter carries it verbatim. The caveat is carried ONCE via ``caveat=`` and
    deliberately NOT folded into ``advise_text`` — doing both printed it twice
    on the card (the description and the caveat line rendered the same sentence).
    """
    if finding is None:
        return []
    repeat_share = getattr(finding, "repeat_share", None)
    if repeat_share is None:
        return []
    sessions_examined = int(getattr(finding, "sessions_examined", 0) or 0)
    repeat_tokens = int(getattr(finding, "repeat_tokens", 0) or 0)
    evidence = (
        f"{float(repeat_share) * 100:.0f}% of prompt tokens across "
        f"{sessions_examined} session(s) were context already sent in an "
        f"earlier turn (conservative lower bound; independent of whether "
        f"caching is enabled)."
    )
    # No second cost sentence here. This used to point at a total-cost figure the
    # card rendered beside the avoidable one; that figure is gone (founder
    # decision — a field two analyzers emit does not earn a place in the
    # contract), so a sentence promising "reported below as cost" would point at
    # nothing. What the avoidable figure does and does not cover is still stated,
    # in `coverage_note`, which is where it belongs.
    fix_compaction = str(getattr(finding, "fix_compaction", "") or "")
    fix_cache_control = str(getattr(finding, "fix_cache_control", "") or "")
    fix_subagent_offload = str(getattr(finding, "fix_subagent_offload", "") or "")
    fix_rightsize = str(getattr(finding, "fix_rightsize", "") or "")
    # The cache_control snippet is the SDK lever only; a claude-code window
    # can't paste it, so suppress it there — unchanged from before.
    cache_snippet = "" if persona == "claude-code" else fix_cache_control
    rightsize = _rightsize_target(subagent_finding, config)

    if persona in {"claude-code", "mixed"} and fix_subagent_offload:
        compound_fix = compound_offload_fix(rightsize, fix_subagent_offload, fix_rightsize)
        advise = compound_fix
        if fix_compaction:
            advise = advise + " Immediate relief in an already-full session: " + fix_compaction
        write_fields = _persona_gated_write_fields(
            persona, compound_fix, delivery=DELIVERY_CLAUDE_MD_RULE,
            scope=rightsize.get("scope") or "project",
        )
        # resend's `suggestion` slot is reserved for the SDK cache_control
        # snippet above, not the write-fallback text the helper would add
        # for a "mixed" persona — drop it so the two don't collide.
        write_fields.pop("suggestion", None)
        # The second half of the compound fix: the agent-file frontmatter that
        # pins model AND reasoning effort. Carried as the one-paste artifact
        # because the write lands in CLAUDE.md, and the apply machinery
        # writes exactly one target per apply.
        one_paste_fix = _rightsize_frontmatter_snippet(rightsize)
    elif persona == "sdk":
        # `/compact` is a Claude Code interactive command an SDK caller has
        # no access to, so it must never be the advise text here — see
        # RESEND_SDK_TRIM_FIX's docstring in context_resend.py. Lead with the
        # cache_control snippet when a priced example produced one; fall
        # back to a persona-neutral, call-site instruction otherwise.
        from tokenjam.core.optimize.analyzers.context_resend import (
            RESEND_SDK_TRIM_FIX,
        )
        advise = _cache_breakpoint_fix("") if cache_snippet else RESEND_SDK_TRIM_FIX
        write_fields = {
            "advise_only": True, "apply_capable": False,
            "delivery": "", "scope": "", "proposed_fix": "",
        }
        one_paste_fix = cache_snippet or RESEND_SDK_TRIM_FIX
    else:
        advise = fix_compaction
        write_fields = {
            "advise_only": True, "apply_capable": False,
            "delivery": "", "scope": "", "proposed_fix": "",
        }
        one_paste_fix = cache_snippet or fix_compaction

    return [CostProposal(
        kind="cost",
        analyzer="resend",
        signature="cost:resend",
        title="Repeated context re-sent every turn",
        target_key={"repeat_share": float(repeat_share)},
        evidence=evidence,
        baseline={
            "sessions_examined": sessions_examined,
            "repeat_tokens": repeat_tokens,
            "repeat_share": float(repeat_share),
            "repeat_share_median": getattr(finding, "repeat_share_median", None),
            "repeat_share_p90": getattr(finding, "repeat_share_p90", None),
            "offloadable_share": getattr(finding, "offloadable_share", None),
            "offloadable_share_sessions": getattr(finding, "offloadable_share_sessions", 0),
            "offloadable_share_sessions_total": getattr(finding, "offloadable_share_sessions_total", 0),
            "offloadable_share_median": getattr(finding, "offloadable_share_median", None),
            "offload_recoverable_usd": getattr(finding, "offload_recoverable_usd", None),
            "rightsize_recoverable_usd": getattr(finding, "rightsize_recoverable_usd", None),
            # The token counts those two dollar terms were priced over, carried
            # so the per-term implied rate is auditable off the card itself
            # (Critical Rule 28), and the compaction lever's separate, wider
            # token estimate — which is deliberately NOT part of
            # `past_overspend_tokens` and must never be summed into a rollup.
            "offload_recoverable_tokens": getattr(finding, "offload_recoverable_tokens", None),
            "rightsize_recoverable_tokens": getattr(finding, "rightsize_recoverable_tokens", None),
            "compaction_avoidable_tokens": getattr(finding, "compaction_avoidable_tokens", None),
            "rightsize_agent_name": rightsize.get("agent_name", ""),
            "rightsize_target_path": rightsize.get("target_path", ""),
            # The two figures' differing populations, carried machine-readable
            # alongside the prose `coverage_note` so the split is auditable.
            "cost_in_scope_usd": getattr(finding, "cost_in_scope_usd", None),
            "cost_driver_role_usd": getattr(finding, "cost_driver_role_usd", None),
            "cost_no_lever_usd": getattr(finding, "cost_no_lever_usd", None),
            "offload_ceiling_usd": getattr(finding, "offload_ceiling_usd", None),
            "sessions_in_scope": getattr(finding, "sessions_in_scope", 0),
            "sessions_no_lever": getattr(finding, "sessions_no_lever", 0),
            "driver_role_sessions": getattr(finding, "driver_role_sessions", 0),
        },
        advise_text=advise,
        suggestion=cache_snippet,
        one_paste_fix=one_paste_fix,
        past_overspend_usd=getattr(finding, "past_overspend_usd", None),
        past_overspend_tokens=getattr(finding, "past_overspend_tokens", None),
        coverage_note=str(getattr(finding, "coverage_note", "") or ""),
        estimate_basis=str(getattr(finding, "estimate_basis", "") or ""),
        caveat=str(getattr(finding, "caveat", "") or COST_CORRELATIONAL_CAVEAT),
        **write_fields,
    )]


#: LAST RESORT ONLY — the look-back to use when neither the config nor the
#: store can be consulted. It is not a default in the sense of "what a normal
#: run uses"; ``cost_window_days_for`` below is that, and every production
#: caller goes through it.
#:
#: Re-exported from ``core/optimize/report_window`` rather than defined here:
#: the stored analyzer report falls back to the same number, and two fallbacks
#: would put the two surfaces back on two windows in exactly the degraded case
#: where nobody can check. How far back a normal run looks is
#: ``[optimize] scan_window_days``, bounded by the chosen analysis span and by
#: the history the store actually holds — see that module.
FALLBACK_COST_WINDOW_DAYS = _REPORT_FALLBACK_WINDOW_DAYS


def _as_anchor(value: Any) -> Any | None:
    """A caller-supplied window anchor as a usable datetime, or ``None``.

    Defensive because the anchor crosses a thread boundary from the scan cycle:
    anything that is not a datetime is discarded rather than raised on, so a
    malformed anchor degrades to "this pass owns its own" instead of sinking a
    background recompute that would otherwise have succeeded.
    """
    from datetime import datetime as _datetime

    return value if isinstance(value, _datetime) else None


def cost_window_days_for(config: Any, conn: Any) -> int:
    """The span past-overspend may accumulate over.

    Delegated to ``core/optimize/report_window`` — the ONE seam the stored
    analyzer report resolves its window through too. This function used to call
    ``analysis_span.window_days_for`` directly, i.e. the chosen span bounded by
    the measured history, while the Dashboard's tiles came off a report scoped
    to ``[optimize] scan_window_days``. Both published ``past_overspend_usd``
    and the two windows were free to disagree; on a real corpus they did (69
    against 30), so the Review inbox headline and the tile row could not be
    compared even though they name the same metric. Read that module's
    docstring before changing what a window means here, and do not restore a
    second derivation.
    """
    from tokenjam.core.optimize.report_window import report_window_days

    return report_window_days(config, conn)


#: Mirrors ``relearn_store``'s own ``_LOCK``/``_COMPUTING`` pair, kept local to
#: this module since a cost-proposals recompute and a relearn recompute are
#: independent jobs that must each be able to run without waiting on the
#: other — only two cost-proposals recomputes (a scheduled tick racing a
#: manual "Rescan now") should ever serialize against each other.
_COST_LOCK = threading.Lock()
_COST_COMPUTING = threading.Event()


def is_computing_cost_proposals() -> bool:
    return _COST_COMPUTING.is_set()


#: The Review inbox's cross-reference for waste a caller has decided NOT to
#: sum as a peer card. Generic infrastructure (see ``estimated_recoverable_
#: rollup``'s ``excluded`` parameter and the UI's ``ExcludedWasteNote``) with
#: no current occupant: ``summarize`` used to be the one entry here (issue
#: #326) until summarize got a real peer card instead (see
#: ``_summarize_to_proposals`` and ``COST_ANALYZERS``'s docstring) — kept
#: rather than deleted because the shape (state the total, link to where it
#: lives, never silently drop it) is the right move for a FUTURE analyzer
#: whose fix has no representable inbox card, should one appear.


def _adapter_failure_entries(failures: dict[str, str]) -> dict[str, Any]:
    """``excluded`` entries for analyzers whose adapter raised.

    Same channel the rollup already uses for money a caller deliberately did
    not sum in — here the money is not withheld but UNKNOWN, so the figure
    fields are ``None`` rather than ``0``. Absent is never zero: a surface
    reading this states that the analyzer is missing from the total, which is
    the one honest thing to say when its contribution could not be built.
    """
    return {
        name: {
            "past_overspend_usd": None,
            "past_overspend_tokens": None,
            "label": name,
            "note": (
                f"The {name} analyzer could not be turned into review rows on "
                f"this refresh, so none of its money is in the total above. "
                f"The figure is unknown, not zero."
            ),
            "error": message,
        }
        for name, message in sorted(failures.items())
    }


#: The three outcomes a recompute call can have. A caller that publishes one
#: must publish WHICH — see :class:`CostRecomputeResult`.
RECOMPUTE_READY = "ready"
RECOMPUTE_DECLINED = "declined"
RECOMPUTE_FAILED = "failed"

#: Machine-readable ``reason`` tokens for the two ways a call declines. Both
#: are NORMAL: something else is already measuring, and two measurements of one
#: window is the defect this module exists to avoid.
DECLINED_SCAN_CYCLE_IN_FLIGHT = "scan_cycle_in_flight"
DECLINED_RECOMPUTE_IN_FLIGHT = "recompute_in_flight"


@dataclass(frozen=True)
class CostRecomputeResult:
    """What a :func:`recompute_cost_proposals` call actually did.

    This used to be a bare ``list[CostProposal]``, and ``[]`` meant FOUR
    different things — a cycle was in flight, the lock was held, the build
    raised, or there was genuinely nothing to propose. Callers could not tell
    them apart, so ``POST /relearn/cost-proposals/refresh`` reported a declined
    refresh as ``{"status": "ready", "proposals": 0}`` while the store still
    held a full, good set: a surface asserting more than its data supports
    (root anti-pattern 22 in the workspace ``CLAUDE.md``).

    ``proposals``
        THIS call's own build. Empty unless ``status`` is ``"ready"`` — a
        declined or failed call built nothing, and returning someone else's
        proposals under this field would be the same conflation again.
    ``served_count`` / ``served_computed_at``
        what a reader of the proposal store would see RIGHT NOW, and when it
        was built. On ``"ready"`` that is this call's fresh write; otherwise it
        is the last-good stored set, which is exactly what a decline leaves up.
        ``served_computed_at`` is ``None`` when nothing has ever been stored.
    ``fresh``
        whether ``served_count`` describes THIS call's measurement. ``False``
        is the honest signal that a surface is republishing a previous result.
    ``reason`` / ``detail``
        a machine token and one human sentence, ``None`` on ``"ready"``.
    """

    status: str
    proposals: list[CostProposal]
    served_count: int
    served_computed_at: str | None
    fresh: bool
    reason: str | None = None
    detail: str | None = None

    def __post_init__(self) -> None:
        if self.status not in (RECOMPUTE_READY, RECOMPUTE_DECLINED, RECOMPUTE_FAILED):
            raise ValueError(f"unknown CostRecomputeResult.status {self.status!r}")


def _serving_last_good(
    status: str, *, config: Any, reason: str, detail: str,
) -> CostRecomputeResult:
    """A non-fresh result describing whatever the proposal store already holds.

    One store read, here, so every caller reports the same count for the same
    moment instead of each deciding whether to look. A store that has never
    been written reads as ``0``/``None`` — "nothing has ever been computed",
    which is a different claim from "this refresh found nothing".
    """
    from tokenjam.core.optimize import relearn_store

    try:
        block = relearn_store.read_cost_proposals(config=config)
    except Exception:
        block = None
    stored = list((block or {}).get("cost_proposals") or [])
    return CostRecomputeResult(
        status=status,
        proposals=[],
        served_count=len(stored),
        served_computed_at=(block or {}).get("cost_computed_at"),
        fresh=False,
        reason=reason,
        detail=detail,
    )


def recompute_cost_proposals(
    db: Any,
    config: Any,
    *,
    window_days: int | None = None,
    agent_id: str | None = None,
    until: Any | None = None,
    report: Any | None = None,
    provenance: Any | None = None,
) -> CostRecomputeResult:
    """Build an ``OptimizeReport`` over the last ``window_days``, adapt the
    cost findings into proposals, and write them into the shared proposal
    store.

    Returns a :class:`CostRecomputeResult`, never a bare list: a caller that
    publishes this outcome has to be able to say WHICH of ready / declined /
    failed it got, and a declined call still reports the last-good stored set
    through ``served_count`` rather than a zero it never measured.

    This is the "daemon path produces findings -> same proposal store" entry
    point the Review-inbox refresh calls; ``tj optimize`` can call it too so a
    manual run also refreshes the inbox. Locked against concurrent recomputes
    (the scheduled job and a manual "Rescan now" can otherwise overlap and
    race each other's cache write) — a caller that hits the lock DECLINES
    rather than blocking.

    A failure is never silent: the exception is recorded via
    ``relearn_store.write_cost_proposals_error`` (behavioral requirement #5)
    so the Review inbox can show a "last refresh failed" warning instead of
    reading a permanently-empty tab as "nothing to report." A SUCCESSFUL
    recompute clears any previously-recorded error.

    ``provenance`` is the CYCLE's record (``core/optimize/cycle_provenance.py``).
    When it is present — the daemon path, where the report leg already minted
    and sealed it — the window, the anchor and the persona come OFF it instead
    of being resolved again here, and it is stored beside the proposals so this
    artifact and the report carry the same ``cycle_id``. ``None`` means a lone
    refresh (``tj optimize``, a direct call), which mints its own from the
    values it resolves itself. An ``agent_id`` scope always resolves its own
    persona: the cycle's label is window-wide, and a per-agent recompute is a
    different population.

    A LONE REFRESH DECLINES WHILE A SCAN CYCLE IS IN FLIGHT, and that is the
    other half of the one-measurement invariant, not a nicety. ``_COST_LOCK``
    only serializes two COST recomputes; the cycle is a three-leg pass (report
    store, then relearn, then this) and only its LAST leg ever touches that
    lock. So a manual refresh landing while a cycle sits between its report leg
    and its cost leg used to take the lock uncontested, build its OWN report
    over a corpus ingestion has moved on, mint its OWN ``CycleProvenance``, and
    write the cost store under a cycle id nothing else carries — and then the
    cycle's own cost leg found the lock held, returned ``[]``, and dropped its
    cost work with no error recorded anywhere. The result is precisely the torn
    artifact ``cycle_provenance`` exists to make impossible: a report store from
    cycle N sitting beside a cost store from a standalone pass at a different
    anchor, with no way for a surface to tell. Declining is the same contract
    this function already offers on a held lock — the caller is told it declined
    and the last-good proposals stay up — except now the cycle's coherent write
    is the one that survives. The cycle's own cost leg is never blocked by this:
    it passes ``provenance``, which is what distinguishes "a leg of the pass in
    flight" from "a second, competing pass".

    A DECLINE IS NOT RECORDED IN ``excluded``, deliberately. That channel is
    keyed by ANALYZER and states that a named analyzer's money is missing from
    an otherwise-fresh total (see :func:`_adapter_failure_entries`). A declined
    call withheld no analyzer — it declined to take a measurement at all, and
    the store keeps the previous cycle's proposals AND that cycle's provenance
    record, which is what a surface compares. Stamping an ``excluded`` entry
    would assert a partial fresh total that was never built, and recording a
    ``cost_proposals_error`` would flag the tab degraded over an outcome that is
    normal and expected. The decline is reported to the CALLER instead, which is
    the only party positioned to say anything true about it.
    """
    from datetime import timedelta

    from tokenjam.core.framing import (
        agent_persona_mix,
        config_declared_plan,
        dominant_persona,
        dominant_plan,
        plan_tier_mix,
        pricing_mode_for,
    )
    from tokenjam.core.optimize import relearn_store
    from tokenjam.core.optimize.cycle_provenance import (
        UNKNOWN as _PERSONA_UNKNOWN,
        CycleProvenance,
        begin_cycle,
    )
    from tokenjam.core.optimize.runner import build_report
    from tokenjam.utils.time_parse import utcnow

    # Checked BEFORE the lock, because the thing being guarded against is not
    # another cost recompute (the lock handles that) but a cycle that has not
    # reached its cost leg yet and therefore holds nothing. `provenance` is the
    # discriminator: a leg OF the cycle carries the cycle's record, a competing
    # standalone pass carries none. `report` counts too — a caller handing in a
    # report is reusing someone else's measurement, not starting a rival one.
    if provenance is None and report is None:
        from tokenjam.core.optimize import scan_cycle

        if scan_cycle.is_cycle_computing():
            return _serving_last_good(
                RECOMPUTE_DECLINED, config=config,
                reason=DECLINED_SCAN_CYCLE_IN_FLIGHT,
                detail=(
                    "A full analyzer scan cycle is in flight, so this refresh "
                    "declined rather than mint a second measurement of the same "
                    "window. The proposals below are the previous cycle's."
                ),
            )
    if not _COST_LOCK.acquire(blocking=False):
        return _serving_last_good(
            RECOMPUTE_DECLINED, config=config,
            reason=DECLINED_RECOMPUTE_IN_FLIGHT,
            detail=(
                "Another cost-proposal recompute is already running, so this "
                "refresh declined rather than race its write. The proposals "
                "below are the last completed result."
            ),
        )
    _COST_COMPUTING.set()
    try:
        try:
            # THE CYCLE'S RECORD, when there is one. It already carries the
            # window, the anchor and the persona this pass resolved, so taking
            # them off it is what makes this artifact and the report describe
            # one measurement rather than two that happen to agree.
            record = provenance if isinstance(provenance, CycleProvenance) else None
            cycle_days = record.window_days if record is not None else None
            # None means "derive it" — the normal path. An explicit value is
            # a caller that already knows its own window (`tj optimize --since`),
            # and is honoured as given.
            effective_window_days = (
                max(1, window_days) if window_days is not None
                else int(cycle_days) if cycle_days
                else cost_window_days_for(config, getattr(db, "conn", None))
            )
            # `until` is the anchor the trailing window is subtracted from.
            # A caller that is refreshing SEVERAL stores in one cycle passes
            # ONE instant for all of them (`core/optimize/scan_cycle.py`), so
            # two surfaces publishing the same metric cannot end up covering
            # windows offset from each other. `None` means this is a lone
            # refresh and owns its own anchor.
            until = _as_anchor(until) or (
                record.until_dt if record is not None else None
            ) or utcnow()
            # The record's own bound, EXCEPT when this caller overrode the
            # length — an explicit `window_days` and the cycle's `since` would
            # otherwise describe two different spans on one artifact.
            since = (
                record.since_dt
                if record is not None and window_days is None
                and record.since_dt is not None
                else until - timedelta(days=effective_window_days)
            )
            conn = getattr(db, "conn", None)
            # Persona decides which cost analyzers are worth running at all —
            # one with no fix this persona can apply is dropped BEFORE the
            # report is built, so it never queries and never reaches the
            # inbox. `build_report` applies the same gate internally (it is
            # the choke point); selecting the persona-scoped list here keeps
            # this surface honest on its own terms rather than relying on the
            # callee to undo an over-broad request.
            #
            # THE CYCLE'S SEALED PERSONA WHEN THERE IS ONE. `build_report`
            # already classified this window once and the record carries that
            # verdict, so a window-wide cycle recompute reads it instead of
            # running the classification again over the same connection and
            # window. An `agent_id` scope is a different population and always
            # resolves its own, and an unsealed record falls through to the
            # derivation below rather than gating on "unknown".
            cycle_persona = record.persona if record is not None else None
            if agent_id is None and cycle_persona and cycle_persona != _PERSONA_UNKNOWN:
                persona = cycle_persona
            else:
                persona = dominant_persona(
                    agent_persona_mix(conn, since, until, agent_id=agent_id)
                    if conn is not None else {},
                    declared_plan=config_declared_plan(config),
                )
            # A REPORT THE CALLER ALREADY BUILT, when there is one. This
            # function used to always build its own, which meant every scan
            # cycle ran `build_report` TWICE over the same window — so an
            # analyzer like `subagent` was computed twice, by two separate
            # scans of a database that ingestion keeps writing to, and the two
            # results were stored separately and published side by side. Same
            # window and same anchor could not make them agree, because they
            # read the corpus at different moments. One pass, two views: the
            # adapters below are a pure transformation of a report, so reusing
            # the cycle's report makes the two surfaces identical by
            # construction rather than by timing.
            #
            # The persona gate is NOT lost by reusing it: `build_report` applies
            # the gate internally (it is the choke point), and the adapters only
            # read findings they know how to adapt, so an analyzer outside
            # `COST_ANALYZERS` present on a full report contributes nothing.
            if report is None:
                report = build_report(
                    db, config, since, until, agent_id=agent_id,
                    # `summarize` IS a COST_ANALYZER now and would already
                    # be selected by `cost_analyzers_for_persona`; this is the
                    # PERSONA-SCOPED list, not the raw one — the skip gate still
                    # decides which cost analyzers run for this window.
                    findings=list(cost_analyzers_for_persona(persona)),
                )
            # Same plan-tier -> pricing-mode resolution `tj optimize` uses, so
            # the web Review inbox suppresses the same dollar figures the CLI
            # does (placement's batch-lever dollars, currently the only card
            # this gates — see `_placement_to_proposals`).
            plan_mix = plan_tier_mix(conn, since, until, agent_id) if conn is not None else {}
            pricing_mode = pricing_mode_for(dominant_plan(plan_mix))
            # An analyzer whose adapter raised contributed NOTHING to the
            # proposals about to be stored, so the headline summed over them
            # is short by that analyzer's whole figure. Recorded as an
            # `excluded` entry with NO number: what it would have contributed
            # is precisely what could not be computed, and inventing a zero
            # there would restate the bug as a fact.
            adapter_failures: dict[str, str] = {}

            def _record_adapter_failure(name: str, exc: BaseException) -> None:
                adapter_failures[name] = f"{type(exc).__name__}: {exc}"

            proposals = cost_proposals_from_report(
                report, config=config, pricing_mode=pricing_mode,
                window_days=float(effective_window_days),
                on_adapter_error=_record_adapter_failure,
            )
            excluded = _adapter_failure_entries(adapter_failures)
            # THE PER-PERSONA LEDGER. The adapters are a pure transformation of
            # a report, so one scoped report in gives one scoped proposal list
            # out — no second measurement, no extra corpus pass. What makes
            # this necessary rather than a convenience is that a dollar figure
            # summed over a mixed corpus cannot be narrowed on read: the
            # `persona` parameter `/optimize` gets to honour by slicing an
            # analyzer set has no equivalent here, so the narrowing has to have
            # happened at compute time or it cannot happen at all.
            #
            # Only a caller whose report carries per-persona passes (the scan
            # cycle — see `core/optimize/report_store.py`) can produce this. A
            # lone refresh writes an unscoped ledger and says so, rather than
            # relabelling one corpus-wide list as every persona's.
            by_persona: dict[str, list[CostProposal]] = {}
            # Relearn reaches the inbox headline through
            # `inbox_contribution.gather_rollup_population`, NOT through an
            # adapter tuple, so its money never appears in `by_persona` above.
            # A persona-scoped rollup that folded in the whole-corpus relearn
            # cache would put two populations into one total — so each scoped
            # pass's own lane-partitioned finding is stored beside its
            # proposals for the route to use instead.
            relearn_by_persona: dict[str, Any] = {}
            for scope_persona, sub in (
                getattr(report, "persona_reports", None) or {}
            ).items():
                try:
                    from tokenjam.core.optimize.runner import report_from_dict

                    scoped_report = (
                        report_from_dict(sub) if isinstance(sub, dict) else sub
                    )
                    by_persona[scope_persona] = cost_proposals_from_report(
                        scoped_report, config=config, pricing_mode=pricing_mode,
                        window_days=float(effective_window_days),
                        # Adapter failures are recorded from the BASE pass
                        # only. A per-persona pass failing the same adapter
                        # would stamp the same `excluded` entry twice, and the
                        # entry describes the analyzer, not the persona.
                    )
                    scoped_relearn = (scoped_report.findings or {}).get("relearn")
                    if scoped_relearn is not None:
                        from dataclasses import asdict as _asdict, is_dataclass

                        relearn_by_persona[scope_persona] = (
                            _asdict(scoped_relearn)
                            if is_dataclass(scoped_relearn)
                            and not isinstance(scoped_relearn, type)
                            else scoped_relearn
                        )
                except Exception:
                    # One persona's adaptation failing must not cost the other
                    # its proposals, nor sink the whole recompute. The missing
                    # persona simply has no entry, which the read side already
                    # treats as "cannot answer for this persona".
                    continue
        except Exception as exc:
            try:
                relearn_store.write_cost_proposals_error(str(exc), config=config)
            except Exception:
                pass
            # The build failed, so this call has no measurement of its own — but
            # `write_cost_proposals_error` deliberately preserves the last GOOD
            # block, so there is usually still something up. Report both facts:
            # `failed` (abnormal, unlike a decline) AND what a reader is looking
            # at while it stays failed.
            return _serving_last_good(
                RECOMPUTE_FAILED, config=config,
                reason=f"{type(exc).__name__}: {exc}",
                detail=(
                    "The cost-proposal recompute failed, so nothing fresher was "
                    "produced. Any proposals below are the last result that "
                    "completed."
                ),
            )

        stored_at: str | None = None
        try:
            written = relearn_store.write_cost_proposals(
                proposals, config=config,
                by_persona=by_persona or None,
                relearn_by_persona=relearn_by_persona or None,
                window_days=effective_window_days,
                excluded=excluded or None,
                # The RESOLVED bounds, not just the length. A day count alone
                # cannot be compared against the analyzer report's own
                # scan_since/scan_until, which is what made a per-analyzer
                # disagreement between the two surfaces undiagnosable. Both
                # spellings now come off ONE record — see
                # `core/optimize/cycle_provenance.py`.
                since=since.isoformat(), until=until.isoformat(),
                # The cycle's record, or this lone refresh's own, so the stored
                # proposals name the pass and the build that produced them.
                provenance=record if record is not None else begin_cycle(
                    config, conn=conn, anchor=until, since=since,
                    window_days=effective_window_days, persona=persona,
                ),
            )
            stored_at = written.get("cost_computed_at")
            relearn_store.clear_cost_proposals_error(config=config)
        except Exception:
            # Best-effort, exactly as before: the build SUCCEEDED, so the caller
            # gets its proposals. `served_computed_at` stays None, which is the
            # honest reading — the store may not have taken this write.
            pass
        return CostRecomputeResult(
            status=RECOMPUTE_READY,
            proposals=proposals,
            served_count=len(proposals),
            served_computed_at=stored_at,
            fresh=True,
        )
    finally:
        _COST_COMPUTING.clear()
        _COST_LOCK.release()


def trigger_background_cost_recompute(
    backend_factory: Callable[[], Any],
    *,
    config: Any | None = None,
    window_days: int | None = None,
    until: Any | None = None,
    report: Any | None = None,
) -> bool:
    """Fire-and-forget a cost-proposals recompute on a daemon thread — the
    Cost-advisories-tab equivalent of ``relearn_store.
    trigger_background_recompute``, so ``tj serve`` can keep this tab warm on
    a schedule (behavioral requirement #5) without blocking startup or a
    request thread on the analyzer run.

    ``backend_factory`` builds a FRESH ``StorageBackend`` (e.g. ``lambda:
    DuckDBBackend(config.storage)``) — never the caller's live request
    connection. The backend is closed when the job finishes. Returns
    ``False`` (no-op) if a recompute is already running.
    """
    if is_computing_cost_proposals():
        return False

    def _job() -> None:
        backend = None
        try:
            backend = backend_factory()
            recompute_cost_proposals(
                backend, config, window_days=window_days, until=until,
                report=report,
            )
        except Exception:
            # Best-effort background job — never crash the scheduler/thread.
            pass
        finally:
            close = getattr(backend, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass

    threading.Thread(target=_job, name="cost-proposals-recompute", daemon=True).start()
    return True


# --------------------------------------------------------------------------- #
# PACING IS GONE FROM THIS MODULE, DELIBERATELY.
#
# `compute_projection_ratio()` used to live here: `r = clamp(30/active_days,
# 1.0, 3.0)`, the one shared multiplier turning each analyzer's window figure
# into an `estimated_monthly_*` forecast. Both are deleted. The product no
# longer makes a forward per-analyzer claim, so the only thing `r` still did
# was produce a THIRD dollar quantity (measured 7.14% larger than the window
# figure on the reference corpus) that a surface could render by mistake —
# which is exactly what happened.
#
# Do not reintroduce a ratio here. See the field contract in the repo
# `CLAUDE.md`.
# --------------------------------------------------------------------------- #


def cost_proposals_from_report(
    report: Any, config: Any = None, *, pricing_mode: str = "api",
    window_days: float = 30.0,
    on_adapter_error: Any = None,
) -> list[CostProposal]:
    """Every cost proposal derivable from an already-built ``OptimizeReport``.

    Reads the ``downsize`` finding off the typed ``report.downgrade`` slot and
    the ``cache`` / ``cache-recommend`` / ``trim`` / ``subagent`` /
    ``placement`` / ``deadweight`` / ``script`` / ``reuse`` / ``verbosity`` /
    ``resend`` / ``summarize`` findings off ``report.findings``. Missing
    findings (analyzer not run, no candidates) contribute nothing. Never
    raises — a malformed finding is skipped so one bad analyzer can't sink
    the inbox.

    **A skipped analyzer is a hole in the headline, so it is never silent.**
    Swallowing the exception keeps the inbox alive, which is right; swallowing
    it WITHOUT A TRACE published a smaller total that looked complete, which is
    the failure this argument exists to end. A whole analyzer once vanished
    this way — a stored report rehydrated its per-agent rows as plain dicts,
    every ``row.delta_usd`` raised, and the inbox quietly dropped the entire
    ``downsize`` contribution while the Dashboard tile went on showing it, so
    the two surfaces disagreed by that analyzer's full figure with no error
    anywhere. ``on_adapter_error(analyzer_name, exc)`` is called for each
    adapter that raises; a caller that persists the result routes those into
    the ``excluded`` channel so the surface states "this analyzer is missing"
    rather than implying its money is zero. ``None`` keeps the bare skip, for
    callers with nowhere to put the disclosure.

    ``config`` is optional and used for one thing: looking up the local source
    path a user registered for an agent, which decides whether the downsize card
    can offer the gated model-id swap or falls back to its one-paste artifact.
    Without it every card is advise-only.

    ``pricing_mode`` gates the ``placement`` card's dollar figure — the Batch
    API discount is an api-billed lever, so a subscription/local caller gets
    the advise text without a number, same as the CLI. Defaults to ``"api"``
    so existing callers that don't know their caller's plan keep today's
    behaviour.

    ``script`` / ``reuse`` read ``report.persona`` (set once by ``runner.
    build_report`` — see ``AnalyzerContext.persona``) to decide whether their
    workspace write is offered at all — see
    ``_persona_gated_write_fields``. ``verbosity`` reads the same finding
    shape but never offers a write for ANY persona (see ``_verbosity_to_
    proposals``) — a cohort-scoped flag written as a global CLAUDE.md rule
    fails the no-quality-tax gate regardless of who would read it. The whole
    ``cache`` family (``cache`` / ``cache_thrash`` / ``cache-recommend``)
    reads the same ``persona`` to decide whether their cache_control
    instruction is shown at all — see ``_persona_gated_cache_fields``; unlike
    script/reuse there is no write to gate, only the instruction text.
    ``downsize`` reads the same ``persona`` to swap its CTA for the
    claude-code-actionable levers (``tj route export`` / ``tj optimize
    subagent`` / ``/compact``) instead of a raw model-swap instruction that
    persona can't act on — see ``_DOWNSIZE_CC_LEVER``. ``placement`` reads it
    to suppress the card outright for a ``"claude-code"`` window: the Batch
    API is a lever in the user's own application code, which an interactive
    session doesn't have. A report built without a
    ``persona`` field (e.g. hand-constructed in a test) defaults to
    ``"unknown"``, which keeps the script/reuse write off (that helper's
    conservative default) while leaving the cache family's instruction on
    (that helper's conservative default runs the other way — see its own
    docstring) — neither ever assumes ``"claude-code"``.

    ``window_days`` is accepted for call-site compatibility but no longer
    used: every card's ``past_overspend_usd``/``_tokens`` is its analyzer's own
    observation of THIS window, carried through unscaled — there is no
    projection or netting step here any more.
    """
    proposals = _adapt_report(
        report, config=config, pricing_mode=pricing_mode,
        on_adapter_error=on_adapter_error,
    )
    proposals = _apply_placement(proposals, report, config)
    return [_with_past_overspend(p) for p in proposals]


def _adapt_report(
    report: Any, *, config: Any = None, pricing_mode: str = "api",
    on_adapter_error: Any = None,
) -> list[CostProposal]:
    """Run every cost adapter over one report. PURE, no placement.

    Split out of :func:`cost_proposals_from_report` so placement can run over
    this producer's write-bearing cards separately from the adapters that
    build them. Same adapters, same report either way.
    """
    findings = getattr(report, "findings", {}) or {}
    # THE SCOPE FIRST, the corpus's own persona second. A report built for one
    # side of the picker over a mixed corpus carries `persona="mixed"` — which
    # gates nothing — beside rows that are entirely one persona's. Reading
    # `persona` alone there adapts a card for a lever the reader does not have
    # and prices it off their own sessions, which is a more convincing wrong
    # answer than the unscoped version was. `persona_scope` is `None` on an
    # unscoped report, so this is exactly the old behaviour for that case.
    persona = str(
        getattr(report, "persona_scope", None)
        or getattr(report, "persona", "")
        or "unknown"
    )
    proposals: list[CostProposal] = []

    # Second half of the persona skip gate. `build_report` already refuses to
    # RUN an analyzer with no fix for this persona, so on the live path these
    # findings are absent anyway; `_pick` makes the inbox correct by its own
    # construction rather than by trusting how the report was built — a report
    # assembled elsewhere (a cached payload, a wider selection, a test) can
    # still carry the finding, and a card the user cannot apply must not
    # appear in Review either way. A `None` finding contributes nothing.
    disabled = _disabled_analyzers(persona)

    def _pick(name: str) -> Any:
        return None if name in disabled else findings.get(name)

    # NAMED, because the name is what a failure has to be reported AS: an
    # adapter that raises contributes nothing, and "nothing" is only
    # distinguishable from "this analyzer found nothing" if the skip can say
    # which analyzer it was. See `on_adapter_error` in the docstring.
    adapters = (
        (
            "downsize",
            # Needs resend's driver-role figures for the reciprocal
            # `coverage_note` (#613), read here like `resend` already reads
            # `subagent` below, so neither analyzer depends on the other.
            lambda f: _downsize_to_proposal(
                f, config, persona=persona, resend_finding=_pick("resend"),
            ),
            getattr(report, "downgrade", None),
        ),
        ("cache", lambda f: _cache_to_proposals(f, persona=persona), _pick("cache")),
        ("cache", lambda f: _cache_uncached_to_proposals(f, persona=persona), _pick("cache")),
        ("cache", lambda f: _cache_thrash_to_proposals(f, persona=persona), _pick("cache")),
        ("cache", lambda f: _cache_lookback_to_proposals(f, persona=persona), _pick("cache")),
        (
            "cache-recommend",
            lambda f: _cache_recommend_to_proposals(f, _pick("cache"), persona=persona),
            _pick("cache-recommend"),
        ),
        ("trim", _trim_to_proposals, _pick("trim")),
        ("subagent", lambda f: _subagent_to_proposals(f, config), _pick("subagent")),
        (
            "placement",
            lambda f: _placement_to_proposals(f, pricing_mode=pricing_mode, persona=persona),
            _pick("placement"),
        ),
        ("deadweight", _deadweight_to_proposals, _pick("deadweight")),
        ("deadweight", _deadweight_plugin_to_proposals, _pick("deadweight")),
        ("script", lambda f: _script_to_proposals(f, persona=persona), _pick("script")),
        ("reuse", lambda f: _reuse_to_proposals(f, persona=persona), _pick("reuse")),
        ("verbosity", lambda f: _verbosity_to_proposals(f, persona=persona), _pick("verbosity")),
        (
            "resend",
            # The resend card is compound: it names the concrete over-powered
            # subagent (from the `subagent` finding) that the offload rule
            # should also right-size, so the two levers land as one card rather
            # than two. Reading the sibling finding here, not in the analyzer,
            # keeps `context_resend.py` free of a cross-analyzer dependency.
            lambda f: _resend_to_proposals(
                f, persona=persona, subagent_finding=_pick("subagent"), config=config,
            ),
            _pick("resend"),
        ),
        ("summarize", _summarize_to_proposals, _pick("summarize")),
    )
    for name, adapter, finding in adapters:
        try:
            proposals.extend(adapter(finding))
        except Exception as exc:
            if on_adapter_error is not None:
                try:
                    on_adapter_error(name, exc)
                except Exception:
                    pass   # a broken reporter must not sink the inbox either
            continue
    return _net_cross_analyzer_session_overlap(proposals)


# Lower number wins the overlap and keeps its claim unchanged; a higher
# number is the one netted down for whatever share of ITS sessions the
# winner already claimed. `reuse` is listed ahead of `script` because its
# premise is the one the product already designated conservative/canonical
# (`_reuse_to_proposals`'s docstring: "the finding's conservative
# cache_reuse_recoverable_* figure ... matching ReuseFinding's own
# aggregate" — `script`'s figure is explicitly the upper-bound alternative to
# that same number, per `ReuseCluster`'s own field comment). Analyzers not
# listed here don't participate even if they happen to carry
# `claimed_session_ids` in the future — adding a pair is a deliberate edit,
# same as the per-pair fixes Critical Rule 27 already requires, but the
# ARITHMETIC below is shared instead of hand-copied per pair.
_SESSION_CLAIM_PRIORITY: dict[str, int] = {
    "reuse": 0,
    "script": 1,
}


def _net_cross_analyzer_session_overlap(
    proposals: list[CostProposal],
) -> list[CostProposal]:
    """Net a proposal's ``past_overspend_*`` down when another, higher-
    priority analyzer already claimed some of the SAME sessions.

    CLAUDE.md Critical Rule 27 (`.claude/rules/optimize-cost-figures.md`):
    two analyzers that both claim ``past_overspend_*`` must draw from
    disjoint spans, and names three remedies — disjoint at the source,
    subtract what a more specific card claimed, or partition the population
    behind a shared predicate. This is remedy (b), made mechanical instead of
    hand-written per pair: any analyzer whose adapter populates
    ``claimed_session_ids`` with its cluster's FULL session population (not
    just the 3 rendered examples) is automatically checked against every
    other such analyzer, and the lower-priority claim is reduced by the
    PROPORTION of its own sessions the higher-priority one already claimed —
    proportional because a cluster is one card pricing many sessions
    together, so a partial overlap must shrink the card partially, not zero
    it out or leave it untouched.

    Only ``reuse``/``script`` opt in today (see ``_SESSION_CLAIM_PRIORITY``);
    every other proposal has an empty ``claimed_session_ids`` and passes
    through completely unchanged, exactly as before this function existed.
    """
    eligible = [
        p for p in proposals
        if p.claimed_session_ids and p.analyzer in _SESSION_CLAIM_PRIORITY
    ]
    if len(eligible) < 2:
        return proposals

    already_claimed: set[str] = set()
    shares: dict[int, tuple[float, str]] = {}   # id(p) -> (share, winner name)
    for p in sorted(eligible, key=lambda p: _SESSION_CLAIM_PRIORITY[p.analyzer]):
        own = set(p.claimed_session_ids)
        overlap = own & already_claimed
        if overlap:
            shares[id(p)] = (len(overlap) / len(own), p.analyzer)
        already_claimed |= own

    if not shares:
        return proposals

    out: list[CostProposal] = []
    for p in proposals:
        entry = shares.get(id(p))
        if entry is None:
            out.append(p)
            continue
        share, _ = entry
        keep = max(0.0, 1.0 - share)
        new_usd = (
            round(p.past_overspend_usd * keep, 6)
            if p.past_overspend_usd is not None else None
        )
        new_tokens = (
            int(round(p.past_overspend_tokens * keep))
            if p.past_overspend_tokens is not None else None
        )
        note = (
            f"{round(share * 100)}% of this cluster's sessions are already "
            f"priced on another card pricing the identical repeated shape; "
            f"reduced to avoid claiming them twice."
        )
        out.append(replace(
            p,
            past_overspend_usd=new_usd,
            past_overspend_tokens=new_tokens,
            coverage_note=(p.coverage_note + " " + note).strip(),
        ))
    return out


#: Cap on the sessions whose transcript is opened to resolve a destination.
#: Distinct sessions, and deliberately far above ``_MAX_SCOPE_SESSIONS``: that
#: cap answers "which ONE repo does this agent definition live in", where 20
#: samples settle it, while placement is answering "which of the user's
#: projects incurred this and in what proportion", where a truncated sample
#: silently concentrates the whole finding into whichever repos happened to
#: sort first. The read itself is cheap — the leading records of each
#: transcript, through the shared parse cache — so the cap is a bound on a
#: pathological corpus, not a sampling decision.
_MAX_PLACEMENT_SESSIONS = 400


def _placement_weights(analyzer: str, report: Any) -> dict[str, int]:
    """``session_id -> attribution weight`` for one rule-writing analyzer.

    Each of the three cost-lane rule writers already knows which sessions its
    claim came from; none of them published it in a form placement could read
    until now. The weights are TOKENS in every case, and they are a breakdown
    of the analyzer's own ``past_overspend_tokens`` rather than a new
    measurement — see the field notes on ``DowngradeFinding.driver_session_tokens``
    and ``ResendFinding.session_weights``.

    An analyzer with no per-session breakdown returns ``{}``, which places its
    rule in the user-global file exactly as before. That is a real answer, not
    a failure: a rule with no evidence about WHERE belongs in the file every
    session loads.
    """
    findings = getattr(report, "findings", {}) or {}
    if analyzer == "downsize":
        finding = getattr(report, "downgrade", None)
        return {
            str(k): int(v)
            for k, v in (getattr(finding, "driver_session_tokens", {}) or {}).items()
            if k and int(v or 0) > 0
        }
    if analyzer == "resend":
        finding = findings.get("resend")
        return {
            str(k): int(v)
            for k, v in (getattr(finding, "session_weights", {}) or {}).items()
            if k and int(v or 0) > 0
        }
    if analyzer == "subagent":
        finding = findings.get("subagent")
        weights: dict[str, int] = {}
        for row in (getattr(finding, "flagged", None) or []):
            if "over_powered" not in (getattr(row, "flags", None) or []):
                continue
            sid = str(getattr(row, "session_id", "") or "")
            if not sid:
                continue
            # The dispatch's own billed volume. Claude Code files a subagent's
            # turns under its PARENT session id (Critical Rule 34), which is
            # exactly what placement wants: the parent session is the one whose
            # working directory names the project the rule belongs in.
            weights[sid] = weights.get(sid, 0) + sum(int(
                getattr(row, field_name, 0) or 0
            ) for field_name in (
                "input_tokens", "output_tokens", "cache_tokens", "cache_write_tokens",
            ))
        return {k: v for k, v in weights.items() if v > 0}
    return {}


def _evidence_for(proposal: CostProposal, report: Any, plan: Any) -> Any:
    """What this proposal observed, in the terms its rule can be written in.

    Assembled from what the analyzers ALREADY computed — the repos placement
    resolved, the models and dispatch types the finding flagged. Nothing here
    is a new measurement and nothing is inferred: an unobserved field stays
    empty, and the substitution that needs it is skipped so the generic wording
    survives (see ``core/fixes/grounding``).
    """
    from pathlib import Path as _Path

    from tokenjam.core.fixes.grounding import Evidence

    findings = getattr(report, "findings", {}) or {}
    repos = tuple(
        _Path(d.root).name for d in getattr(plan, "destinations", ()) if d.root
    )
    agents: tuple[str, ...] = ()
    models: tuple[str, ...] = ()
    subagent = findings.get("subagent")
    if subagent is not None and proposal.analyzer in {"subagent", "resend"}:
        flagged = [
            r for r in (getattr(subagent, "flagged", None) or [])
            if "over_powered" in (getattr(r, "flags", None) or [])
        ]
        agents = tuple(dict.fromkeys(
            str(getattr(r, "sub_agent_type", "") or "") for r in flagged
        ).keys() - {""})
        models = tuple(dict.fromkeys(
            str(getattr(r, "model", "") or "") for r in flagged
        ).keys() - {""})
    return Evidence(repos=repos, agents=tuple(sorted(agents)), models=tuple(sorted(models)))


def _placement_for(
    proposal: CostProposal, report: Any, config: Any,
) -> Any | None:
    """Where ``proposal``'s rule should be written, or ``None``.

    ``None`` means "no placement evidence" — the caller then keeps the
    single-destination behaviour. Never raises: placement reads the live
    filesystem and the live transcript tree, and a hiccup there must degrade to
    the historical answer rather than sink the inbox.
    """
    from tokenjam.core.optimize import rule_placement as rp

    weights = _placement_weights(proposal.analyzer, report)
    if not weights:
        return None
    ranked = sorted(weights.items(), key=lambda kv: -kv[1])[:_MAX_PLACEMENT_SESSIONS]
    from tokenjam.core.optimize.scope import _claude_home_for
    from tokenjam.core.transcript import resolve_projects_root, session_cwd_map

    override = getattr(getattr(config, "loop", None), "transcript_path", None)
    projects_root = resolve_projects_root(override)
    cwds = session_cwd_map([sid for sid, _ in ranked], projects_root)
    return rp.build_placement_plan(
        [rp.SessionShare(session_id=sid, weight=weight) for sid, weight in ranked],
        cwds,
        total_tokens=int(proposal.past_overspend_tokens or 0),
        total_usd=proposal.past_overspend_usd,
        # The user-global fallback has to name the SAME machine the transcripts
        # came from. Derived from the resolved projects root rather than from
        # `Path.home()`, so a review served against a throwaway `--db` cannot
        # offer to write into the operator's real `~/.claude/CLAUDE.md` — the
        # scope-agreement requirement `relearn_apply.default_target_path`
        # takes this argument for in the first place.
        claude_home=_claude_home_for(projects_root),
    )


def _placements_for(
    writers: list[CostProposal], report: Any, config: Any,
) -> dict[str, Any]:
    """WHERE each rule goes. A rule confined to the projects that actually
    exhibited the behaviour is re-sent in those projects only, so its standing
    cost falls by the ratio of their sessions to the window's — cheaper to
    keep, never a reason it would or wouldn't be offered.
    """
    from tokenjam.core.optimize import rule_placement
    from tokenjam.core.rulewrite.delivery import standing_tokens_per_session

    placements: dict[str, Any] = {}
    for p in writers:
        try:
            plan = _placement_for(p, report, config)
        except Exception:
            plan = None
        if plan is None:
            continue
        choice = rule_placement.choose_placement(
            plan,
            standing_tokens_per_session=standing_tokens_per_session(
                p.delivery, p.proposed_fix,
            ),
            total_sessions=int(getattr(getattr(report, "window", None), "sessions", 0) or 0),
        )
        placements[p.signature] = (plan, choice)
    return placements


def write_bearing(proposals: list[CostProposal]) -> list[CostProposal]:
    """The cards that actually write something: ``apply_capable`` with a
    write-bearing ``proposed_fix``. Everything else (the advise-only majority,
    the model-id swaps, the MCP-server removals) writes no standing prompt
    text and has no destination to place."""
    return [p for p in proposals if p.apply_capable and p.delivery and p.proposed_fix]


def _apply_placement(
    proposals: list[CostProposal], report: Any, config: Any = None,
) -> list[CostProposal]:
    """Fill in WHERE each write-bearing card's rule would land.

    Every ``apply_capable`` card is already offered — this only resolves and
    stamps placement's ``placement_*`` fields; it never withdraws an offer.
    """
    writers = write_bearing(proposals)
    if not writers:
        return proposals

    placements = _placements_for(writers, report, config)
    if not placements:
        return proposals

    out: list[CostProposal] = []
    for p in proposals:
        placed = placements.get(p.signature)
        if placed is None:
            out.append(p)
            continue
        plan, choice = placed
        out.append(replace(
            p,
            placement_scope=choice.scope,
            placement_paths=[d.path for d in choice.destinations],
            placement_sessions=[d.sessions for d in choice.destinations],
            placement_standing_tokens=choice.standing_tokens,
            placement_alternative_standing_tokens=(
                choice.alternative_standing_tokens
            ),
            placement_footprint_tokens=choice.footprint_tokens,
            placement_basis=choice.basis,
            # The placement gap rides on its OWN note rather than being
            # merged into the analyzer's `coverage_note`: that field states
            # what the analyzer's FIGURE covers, and this states which
            # sessions the rule could be PLACED for. Two different
            # populations, so merging them would recreate exactly the
            # ratio-of-two-populations defect Critical Rule 30 is about.
            placement_coverage_note=plan.coverage_note,
        ))
    return out



#: Suffix appended to every stamped ``past_overspend_basis`` so the figure can
#: never be read off a payload without the tense being stated with it.
PAST_OVERSPEND_OBSERVED_NOTE = (
    "Observed over the analyzed window: what this behaviour already cost, "
    "priced at the rates it actually billed at. Not a projection and not a "
    "claim about what a fix returns."
)

def _with_past_overspend(proposal: CostProposal) -> CostProposal:
    """Stamp the tense-bearing ``past_overspend_basis`` onto a finished proposal.

    ONE place decides the wording for every analyzer: two surfaces reading two
    differently-worded "what did this cost me" figures is how the Dashboard
    hero and the Review inbox headline silently come to disagree.

    It no longer MOVES a number between fields. It used to copy
    ``estimated_recoverable_usd`` onto ``past_overspend_usd``, which is what
    left two names live for one quantity — the ambiguity this collapse removed.
    The adapter sets ``past_overspend_usd``/``_tokens`` directly, from its
    analyzer's field of the same name, and nothing rewrites it here.

    **The headline is the AVOIDABLE figure for every analyzer**, because waste
    is only ever the avoidable portion. There is no second figure any more: the
    total-observed-cost pair this function used to stamp is deleted, so a card
    carries one number or none, and what the number does not cover is stated in
    words by ``coverage_note``.

    Never applies a projection ratio — there is none left to apply. Never
    mutates: returns a new proposal.
    """
    if proposal.past_overspend_usd is None and proposal.past_overspend_tokens is None:
        # Nothing observed — there is no basis to describe.
        return proposal
    return replace(
        proposal,
        past_overspend_basis=" ".join(
            x for x in (proposal.estimate_basis, PAST_OVERSPEND_OBSERVED_NOTE) if x
        ),
    )


#: The pre-collapse names a cached cost-proposal dict may still carry, mapped
#: onto the one canonical field. ``estimated_recoverable_*`` was the SAME
#: quantity under a forward-framed name, so it migrates straight across.
#: ``estimated_monthly_*`` deliberately does NOT appear here: it was that
#: quantity multiplied by a pace ratio, and reviving a paced figure as the
#: past-tense headline is exactly the mistake this collapse exists to prevent —
#: a legacy entry carrying only the monthly key renders no dollar figure until
#: the next recompute (at most 6h), which is the honest degradation.
_LEGACY_PROPOSAL_FIELD_ALIASES = (
    ("past_overspend_usd", "estimated_recoverable_usd"),
    ("past_overspend_tokens", "estimated_recoverable_tokens"),
)

#: Keys a warm cache may still carry for figures that no longer exist. Unlike the
#: aliases above there is nothing to migrate them ONTO: `estimated_monthly_*` was
#: a paced variant of the canonical figure, and the two `*cost*` trios were the
#: total-observed-cost pair (analyzer-side input, then the published field). Both
#: are deleted, so the only correct read-time action is to drop them — a renderer
#: that still branched on one would resurrect a retired figure from a stale entry.
_RETIRED_PROPOSAL_KEYS = (
    "estimated_monthly_usd", "estimated_monthly_tokens",
    "cost_of_waste_usd", "cost_of_waste_tokens", "cost_of_waste_basis",
    "observed_cost_usd", "observed_cost_tokens", "observed_cost_basis",
)


def _without_retired_keys(proposal: dict[str, Any]) -> dict[str, Any]:
    """A copy of ``proposal`` with every retired figure key removed."""
    return {k: v for k, v in proposal.items() if k not in _RETIRED_PROPOSAL_KEYS}


def backfill_legacy_past_overspend_fields(proposal: dict[str, Any]) -> dict[str, Any]:
    """Read-time backward compat for a cost-proposal dict cached before the
    per-analyzer dollar fields were collapsed onto ``past_overspend_*``.

    A cache written by the CURRENT build always carries them (the adapters set
    them and ``_with_past_overspend`` stamps the basis), so this only fires for
    an entry that predates the collapse — which would otherwise render an em
    dash where the page's headline number belongs until the next scheduled
    recompute, up to 6h away. Renames the legacy keys, applies the SAME basis
    derivation ``_with_past_overspend`` does, and never invents a figure: a
    legacy entry with no observed figure at all stays empty.

    Dropping the RETIRED keys is unconditional, unlike the rename. A cache
    written by any build between the field-collapse and the total-cost purge
    carries ``past_overspend_basis`` already, so it takes the early return —
    which is why the retired-key strip has to happen before that return, not
    after it. Otherwise a deleted figure keeps rendering off a warm cache for up
    to the recompute interval.
    """
    stamped = _without_retired_keys(proposal)
    if "past_overspend_basis" in stamped:
        return stamped
    for canonical, legacy in _LEGACY_PROPOSAL_FIELD_ALIASES:
        # Always PRESENT, even when there is nothing to carry: a renderer that
        # indexes the canonical key must not blow up on a legacy entry, and an
        # explicit `None` reads as "not measured" rather than "worth nothing".
        stamped[canonical] = (
            stamped.get(canonical)
            if stamped.get(canonical) is not None
            else proposal.get(legacy)
        )
        stamped.pop(legacy, None)
    stamped["past_overspend_basis"] = " ".join(
        x for x in (proposal.get("estimate_basis") or "", PAST_OVERSPEND_OBSERVED_NOTE) if x
    )
    return stamped


# --------------------------------------------------------------------------- #
# THE rollup — the one aggregate every surface reads. There is no second one.
#
# There used to be two (`estimated_recoverable_rollup` beside this), summing
# two per-analyzer fields that had become the same quantity under two names,
# plus a third paced figure derived from them. A driver session comparing them
# reported "identical for 6 of 7 analyzers" while the Review inbox rendered the
# paced third value, 7.14% larger. Both extra fields and the second rollup are
# gone; if you find yourself adding a rollup here, you are re-creating that bug.
# --------------------------------------------------------------------------- #

def past_overspend_rollup(
    proposals: list[Any],
    *,
    window_days: int = FALLBACK_COST_WINDOW_DAYS,
    excluded: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Sum ``past_overspend_usd``/``past_overspend_tokens`` across
    ``proposals``, deduplicated by ``signature`` (a proposal's stable identity —
    see the ``CostProposal`` docstring) so a stale or duplicate cache entry is
    never double-counted.

    Generic over ``analyzer``: reads only the shared ``CostProposal`` fields, so
    a new analyzer's cards are picked up automatically with no change here (the
    dedup rule for overlapping CLAIMS lives one layer down, in each analyzer's
    adapter — see CLAUDE.md rule 27).

    **EVERY ROW OF THE REVIEW INBOX, THROUGH THE SAME DOOR.** The inbox is one
    list fed by more than one producer, and this total has to cover all of it or
    every sentence derived from the list (the collapsed tail's combined figure,
    the below-floor "still counted in the total above" note) is false for the
    part it cannot see. So a producer that is not the cost pipeline contributes
    by handing ROWS to this function on the canonical field — see
    ``core/optimize/inbox_contribution.py``, which builds relearn's clusters into
    exactly that shape on the window this rollup is labelled with. This is
    deliberately the ONLY way in: the retired mechanism was a relearn-only
    ``relearn_clusters=`` parameter that computed a figure INSIDE here on a
    second time basis and published it in its own key, and that parameter is
    still guarded against by ``test_cost_proposals.
    test_the_rollup_has_no_per_analyzer_side_channel``. No per-analyzer parameter
    belongs on this signature, ever.

    ``proposal_count`` / ``deduplicated_proposal_count`` / ``by_analyzer``
    therefore count inbox ROWS, which is what they have always counted: they
    describe the population this total was summed over, so a contributing row
    that came from a non-cost producer is counted and named like any other. A
    contribution that is NOT reflected in these counts would be a delta a reader
    cannot attribute, which is the failure mode the ``observed_cost_usd`` note
    below records.

    This is the AVOIDABLE portion of what the flagged behaviours already cost
    over the analyzed window, every token priced at the rate it genuinely
    billed at. Waste is only ever what could have been avoided, so this — not
    the larger total-cost figure — is the number any surface using
    waste/overspend wording may show. Nothing here is projected, discounted, or
    claimed:

    * **No pacing.** There is no ratio in this module any more, and this
      function does not accept ``active_days``/``n_sessions``, so there is
      nothing to project FROM. The whole point of the figure is that it is an
      observation of a window that already happened.
    * **Only a proposal carrying a numeric figure contributes** — to the sum
      AND to ``proposal_count``. A card with no figure still renders
      individually in the inbox; counting it here without a contribution would
      silently understate the average the headline implies.
    * The token sum is counted INDEPENDENTLY of the dollar sum (a different,
      often overlapping, never identical set of proposals carries each). A
      renderer leading with the token figure must quote
      ``token_proposal_count`` against ``deduplicated_proposal_count`` and say
      the figure is a floor, not a total.

    **Every figure this block publishes covers the SAME set of proposals.** It
    used to publish a second total, ``observed_cost_usd``, summed over whichever
    proposals happened to carry a total-cost figure — 2 of them, against the 13
    the headline summed — under a disclosure asserting "the avoidable figure is a
    subset of it". That was false as published: roughly $5,754 of the $6,163
    avoidable total came from proposals reporting no observed cost at all, so the
    headline was mostly OUTSIDE the figure it was described as a subset of. Two
    figures over two different populations cannot be shown as two views of one
    quantity, and a reader computes the ratio anyway. The second total and its
    disclosure are deleted rather than re-worded, because the defect was
    structural: any second total published here would be summed over its own
    population and invite the same inference.

    ``excluded`` is a passthrough for waste a caller decided NOT to sum in as a
    peer proposal (an analyzer with its own review surface and no representable
    inbox card). Never summed into any total here; carried on the result
    unchanged so a caller can render "$X more available in <analyzer> → review
    it" instead of silently omitting a real figure. ``None`` becomes ``{}`` —
    "nothing known to be excluded", not "excluded total is zero".
    """
    seen: dict[str, dict[str, Any]] = {}
    for p in proposals:
        row = asdict(p) if is_dataclass(p) and not isinstance(p, type) else dict(p)
        sig = str(row.get("signature") or "")
        if not sig or sig in seen:
            continue
        seen[sig] = row

    total_usd = 0.0
    total_tokens = 0
    usd_count = 0
    token_count = 0
    by_analyzer: dict[str, dict[str, Any]] = {}
    for row in seen.values():
        analyzer = str(row.get("analyzer") or "unknown")
        # An analyzer earns a breakdown entry only by CONTRIBUTING a figure.
        # A card carrying nothing still renders individually in the inbox, but
        # listing it here (and in the "contributing analyzers:" basis string)
        # would name it as a contributor to a total it added nothing to.
        if (
            row.get("past_overspend_usd") is None
            and row.get("past_overspend_tokens") is None
        ):
            continue
        entry = by_analyzer.setdefault(
            analyzer,
            {"analyzer": analyzer, "count": 0, "usd": 0.0, "tokens": 0},
        )
        entry["count"] += 1

        tokens = row.get("past_overspend_tokens")
        if tokens is not None:
            total_tokens += int(tokens)
            token_count += 1
            entry["tokens"] = int(entry["tokens"]) + int(tokens)

        usd = row.get("past_overspend_usd")
        if usd is None:
            continue
        usd = float(usd)
        total_usd += usd
        usd_count += 1
        entry["usd"] = round(float(entry["usd"]) + usd, 6)

    deduplicated_proposal_count = len(seen)
    if usd_count == 0:
        basis = (
            f"no open (not yet applied) Review inbox row currently carries an "
            f"observed dollar figure for the last {window_days} days."
        )
    else:
        breakdown = "; ".join(
            f"{a['analyzer']} ({a['count']})"
            for a in sorted(by_analyzer.values(), key=lambda x: x["analyzer"])
        )
        basis = (
            f"sum of the AVOIDABLE window figure (past_overspend_usd) across "
            f"{usd_count} of {deduplicated_proposal_count} open (not yet "
            f"applied), deduplicated-by-signature Review inbox row(s), observed "
            f"over the last {window_days} days; contributing analyzers: "
            f"{breakdown}."
        )
    if token_count:
        basis += (
            f" Token figure: sum of past_overspend_tokens across "
            f"{token_count} of {deduplicated_proposal_count} row(s); the "
            f"rest carry no token figure, so it is a floor, not a total."
        )
    return {
        "past_overspend_usd": round(total_usd, 6),
        "past_overspend_tokens": total_tokens,
        "proposal_count": usd_count,
        "token_proposal_count": token_count,
        "deduplicated_proposal_count": deduplicated_proposal_count,
        "window_days": window_days,
        "by_analyzer": sorted(by_analyzer.values(), key=lambda x: x["analyzer"]),
        "excluded": excluded or {},
        "basis": basis,
        "disclosure": PAST_OVERSPEND_OBSERVED_NOTE,
    }
