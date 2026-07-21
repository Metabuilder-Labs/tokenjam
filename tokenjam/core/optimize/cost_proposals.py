"""Adapt cost-analyzer findings into Review-inbox proposals ("advisories").

The self-improve loop's relearn detector already produces
``RelearnCluster`` proposals that the Lens Improve inbox renders and that a
user can mark and apply. The *cost* analyzers (``downsize`` model
over-sizing, ``cache`` efficacy, ``trim`` prompt bloat, ``subagent``
right-sizing, ``deadweight`` dead MCP servers, ``script`` deterministic
workflows, ``reuse`` repeated planning skeletons, ``verbosity`` high-output
outliers) produce findings of a different shape. This module adapts each
finding into a ``CostProposal`` so the inbox can list them BESIDE the relearn
proposals, typed by a distinct ``kind`` field.

Two structural facts carry over from the relearn ``advise_only`` lane and are
NOT optional here:

  * **Advise-only by default, apply-capable where a real workspace surface
    exists.** Most cost fixes live in the user's own application code (a
    model-routing decision, a cache-prefix change, a prompt edit) that
    tokenjam cannot write into — those cards have NO apply path, exactly like
    an ``advise_only`` ``RelearnCluster`` (empty ``suggested_target``). A
    minority (``subagent``, the per-agent slice of ``downsize``, ``script``,
    ``reuse``, ``verbosity``) DO have a workspace surface an orchestrating
    agent reads before acting (a CLAUDE.md rubric, a model-id key, a new
    skill note) and route through the same rung-gated
    ``relearn_apply.apply_relearn_fix`` machinery the relearn lane uses
    (``apply_capable=True``, ``rung``, ``scope``, ``proposed_fix``). Every
    other card carries a recommendation and, where sensible, a copyable
    config/code suggestion; the user applies it themselves.
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
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any

# House-style label strings. Kept verbatim on every cost proposal so no channel
# can surface a savings figure without the honesty framing (Rule 14).
COST_ESTIMATE_CONFIDENCE = "estimated"
COST_CORRELATIONAL_CAVEAT = (
    "Estimated, correlational figure; not a causal savings claim. The "
    "recommendation lives in your own application code. Review the evidence "
    "before changing anything."
)

#: The analyzers this wiring covers, by registration name.
COST_ANALYZERS = (
    "downsize", "cache", "trim", "subagent", "deadweight", "script", "reuse",
    "verbosity",
)

# The rung-1/rung-2 apply notes below all route through the SAME workspace-note
# machinery `subagent` already uses (`relearn_apply.apply_relearn_fix`, rung 1
# = CLAUDE.md note, rung 2 = a new .claude/skills/<slug>/SKILL.md). None of
# these three analyzers has a workspace file it can edit outright the way a
# model-routing swap does — the fix is behavioral (an orchestrator or the model
# itself reading guidance), same class of surface as the subagent rubric.

# The rung-2 skill note a `script` proposal writes: the observed tool-call
# pattern is deterministic enough that a script could run it directly instead
# of dispatching a full agent turn.
_SCRIPT_SKILL_INTRO = (
    "This tool-call pattern repeated across many sessions with the same "
    "structural shape (same tools, same argument types, different values). "
    "Consider replacing it with a deterministic script that runs these calls "
    "directly, and reserve the agent turn for the parts that actually need a "
    "model's judgment."
)

# The rung-1 note a `reuse` proposal writes: the planning skeleton recurs.
_REUSE_NOTE_INTRO = (
    "This class of task shares a planning skeleton: the same tool sequence "
    "follows the first planning call, session after session, with only the "
    "argument values differing (dates, versions, paths). Consider templating "
    "the plan for this shape instead of re-planning it from scratch each "
    "time. Review before reusing: a skeleton match is a candidate, not proof "
    "the plan is identical."
)

# The rung-1 sizing-rubric note a CC-origin subagent proposal writes into the
# workspace CLAUDE.md when applied. A shape-based default, not a per-subagent
# edit — it names the observed oversized dispatches and states the routing rule.
SUBAGENT_RUBRIC_INTRO = (
    "Right-size Task-dispatched subagents: default a subagent to the cheapest "
    "same-family model that fits its shape, and only reach for a premium-tier "
    "model (Opus / Fable) when the subtask genuinely needs deep reasoning. A "
    "subagent that does little tool work and returns a short result rarely needs "
    "the premium tier."
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
    # Estimated recoverable saving, carried straight from the finding and
    # labeled. ``None`` when the finding produced no estimate for this item.
    estimated_recoverable_usd:    float | None = None
    estimated_recoverable_tokens: int | None   = None
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
    # — a rung-1 sizing rubric note in the workspace's CLAUDE.md — so its card
    # can route an actual, reversible, human-gated write through the existing
    # relearn apply path (``relearn_apply.apply_relearn_fix``). The adapter (not
    # the analyzer) supplies these; ``apply_capable`` gates the apply action, and
    # a proposal with no clean workspace surface degrades to advise-only like the
    # other three (``apply_capable=False``, ``advise_only=True``).
    apply_capable:        bool = False
    rung:                 int  = 0
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
    # Why the direct apply is not on offer, when it is not. Rendered on the card
    # next to the one-paste fix so a fallback is never silent.
    apply_blocked_reason: str  = ""
    # The exact fix, with this agent's own measured values already substituted
    # in. Every advise-only card carries one.
    one_paste_fix:        str  = ""


# --------------------------------------------------------------------------- #
# Per-analyzer adapters. Each reads ONE finding dataclass and returns 0..N
# proposals. All tolerate a None/empty finding (returns []).
# --------------------------------------------------------------------------- #

def _downsize_to_proposal(finding: Any, config: Any = None) -> list[CostProposal]:
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
    """
    if finding is None or getattr(finding, "candidate_sessions", 0) <= 0:
        return []
    per_agent = _downsize_agent_proposals(finding, config)
    if per_agent:
        return per_agent
    suggestions: dict[str, str] = dict(getattr(finding, "suggestions", {}) or {})
    if not suggestions:
        return []
    models = sorted(suggestions.keys())
    model_list = ", ".join(models)
    evidence = (
        f"{finding.candidate_sessions} of {finding.total_sessions} sessions "
        f"({finding.percent_of_sessions:.0f}%) ran on a larger-than-needed model "
        f"({model_list}); candidate sessions are {finding.percent_of_tokens:.0f}% "
        f"of the window's tokens."
    )
    advise = (
        "Route the flagged structural-shaped work to the cheaper same-family "
        "model before it runs. Suggested swaps: "
        + "; ".join(f"{m} → {alt}" for m, alt in sorted(suggestions.items()))
        + ". " + str(getattr(finding, "caveat", "") or "")
    ).strip()
    suggestion = "\n".join(f"{m} -> {alt}" for m, alt in sorted(suggestions.items()))
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
        estimated_recoverable_usd=getattr(finding, "estimated_recoverable_usd", None),
        estimated_recoverable_tokens=getattr(finding, "estimated_recoverable_tokens", None),
        estimate_basis=str(getattr(finding, "estimate_basis", "") or ""),
    )]


def _per_agent_cache_recoverable_by_model(finding: Any) -> dict[tuple[str, str], tuple[float, int]]:
    """Sum of ``estimated_recoverable_usd``/``estimated_recoverable_tokens``
    already claimed by the root-caused per-agent cards (A1 uncached / A2
    thrash / A3 lookback), keyed by (provider, model).

    The generic per-(provider, model) efficacy row and these per-agent checks
    both read from the SAME underlying spans — a flagged agent's own calls
    are part of the aggregate the generic row's efficacy is computed over. So
    the dollars a per-agent card claims must be subtracted from the generic
    row's figure before it's surfaced, or the Review-inbox rollup (which sums
    every open card's ``estimated_recoverable_usd`` with no analyzer
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
            usd = getattr(c, "estimated_recoverable_usd", None) or 0.0
            tokens = getattr(c, "estimated_recoverable_tokens", None) or 0
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


def _model_swap_plumbing(row: Any, config: Any) -> dict[str, Any]:
    """Whether this agent's swap can be written directly, and where.

    The direct write is offered only when the user registered a local source
    path for the agent and every precondition in
    ``model_apply.model_swap_precheck`` holds. Otherwise the card keeps its
    one-paste artifact and states the reason.
    """
    from tokenjam.core.optimize.model_apply import (
        APPLY_KIND_MODEL_SWAP,
        model_swap_precheck,
    )

    agents = getattr(config, "agents", None) or {}
    agent_cfg = agents.get(row.agent_id) if hasattr(agents, "get") else None
    source_path = str(getattr(agent_cfg, "source_path", "") or "")
    check = model_swap_precheck(source_path, row.model)
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


def _downsize_agent_proposals(finding: Any, config: Any) -> list[CostProposal]:
    """One card per agent, carrying that agent's own price arithmetic.

    Replaces the window-wide card when per-agent rows exist, so one source of
    over-sized spend produces exactly one card rather than an aggregate plus its
    own parts.
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
        one_paste = (
            f"{row.model} -> {row.alt_model}\n"
            f"# Set this agent's model id to {row.alt_model} where it is "
            f"configured, then redeploy or restart the agent."
        )
        advise = (
            f"Route {row.agent_id}'s flagged structural-shaped work from "
            f"{row.model} to {row.alt_model}. The price difference above is "
            f"arithmetic on this agent's measured tokens, given the switch; "
            f"whether the cheaper model answers as well is not measured here, "
            f"so review the example sessions first."
        )
        if plumbing.get("apply_capable"):
            advise += (
                f" tokenjam can make this exact substitution in "
                f"{plumbing['target_path']}, with the change committed and "
                f"revertable in one call. After it is applied you must redeploy "
                f"or restart the agent: measurement starts at the first call "
                f"that runs on {row.alt_model}, not at the moment of the write."
            )
        elif plumbing.get("apply_blocked_reason"):
            advise += f" Applying it here is not on offer: {plumbing['apply_blocked_reason']}"
        proposals.append(CostProposal(
            kind="cost",
            analyzer="downsize",
            signature=f"cost:downsize:{row.agent_id}",
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
            estimated_recoverable_usd=row.delta_usd,
            estimated_recoverable_tokens=row.total_tokens,
            estimate_basis=row.estimate_basis,
            agent_id=row.agent_id if row.agent_id != "unknown" else "",
            advise_only=not plumbing.get("apply_capable", False),
            apply_capable=bool(plumbing.get("apply_capable")),
            apply_kind=str(plumbing.get("apply_kind", "")),
            source_path=str(plumbing.get("source_path", "")),
            target_path=str(plumbing.get("target_path", "")),
            current_model=str(plumbing.get("current_model", "")),
            proposed_model=str(plumbing.get("proposed_model", "")),
            apply_blocked_reason=str(plumbing.get("apply_blocked_reason", "")),
        ))
    return proposals


def _placement_to_proposals(
    finding: Any, *, pricing_mode: str = "api",
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
    """
    if finding is None:
        return []
    candidates = list(getattr(finding, "candidates", []) or [])
    if not candidates:
        return []
    agents = ", ".join(c.agent_id for c in candidates[:5])
    total = float(getattr(finding, "candidate_cost_usd", 0.0) or 0.0)
    saving = float(getattr(finding, "estimated_recoverable_usd", 0.0) or 0.0)
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
            "# Submit these workloads through the Batch API instead of the "
            "synchronous endpoint:\n"
            + "\n".join(f"#   {c.agent_id}" for c in candidates)
        ),
        estimated_recoverable_usd=recoverable_usd,
        estimated_recoverable_tokens=getattr(finding, "estimated_recoverable_tokens", None),
        estimate_basis=str(getattr(finding, "estimate_basis", "") or ""),
        agent_id=candidates[0].agent_id if len(candidates) == 1 else "",
    )]


def _cache_to_proposals(finding: Any) -> list[CostProposal]:
    """One proposal per flagged (provider, model) cache-efficacy row.

    Reduced by whatever the more specific per-agent root-cause cards (A1/A2/
    A3) already claim for that same (provider, model) — see
    ``_per_agent_cache_recoverable_by_model`` — so the rollup never sums the
    same underlying waste twice under two different signatures.
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
            advise_text=(
                "Add a stable cache prefix / enable prompt caching for this model "
                "so repeated context is served from cache instead of re-billed as "
                "fresh input."
            ),
            estimated_recoverable_usd=usd,
            estimated_recoverable_tokens=tokens,
            estimate_basis=basis,
        ))
    return proposals


def _cache_uncached_to_proposals(finding: Any) -> list[CostProposal]:
    """One proposal per A1 uncached-agent candidate (see
    ``analyzers.cache_efficacy``): an agent group making cacheable calls with
    prompt caching never attempted. Scored through the same efficacy metric
    as ``_cache_to_proposals`` (agent-scoped)."""
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
            advise_text=(
                "Add a cache_control breakpoint on this agent's stable prefix "
                "(system prompt / tool definitions) so repeated calls read "
                "from cache instead of paying full input price every time."
            ),
            suggestion=c.cache_control_snippet,
            estimated_recoverable_usd=c.estimated_recoverable_usd,
            estimated_recoverable_tokens=c.estimated_recoverable_tokens,
            estimate_basis=c.estimate_basis,
            agent_id=c.agent_id,
        ))
    return proposals


def _cache_thrash_to_proposals(finding: Any) -> list[CostProposal]:
    """One proposal per A2 cache-thrash candidate. Card text branches on the
    detected root cause: a TTL-cadence card (honest break-even, which may say
    the switch isn't worth it) versus an instability checklist card."""
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
            advise_text=advise,
            suggestion=c.cache_control_snippet,
            estimated_recoverable_usd=c.estimated_recoverable_usd,
            estimated_recoverable_tokens=c.estimated_recoverable_tokens,
            estimate_basis=c.estimate_basis,
            agent_id=c.agent_id,
        ))
    return proposals


def _cache_lookback_to_proposals(finding: Any) -> list[CostProposal]:
    """One proposal per A3 20-block-lookback-miss candidate. Weakest-
    confidence check of the three; the analyzer only classifies an agent here
    when A1/A2 don't already explain its cache waste."""
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
            advise_text=(
                "Anthropic's cache breakpoint search looks back at most "
                f"{LOOKBACK_BLOCK_LIMIT} content blocks. Long tool-heavy "
                "turns push the prior breakpoint out of range; add an "
                "intermediate cache_control breakpoint every ~15 blocks in "
                "long tool-use turns."
            ),
            suggestion=c.cache_control_snippet,
            estimated_recoverable_usd=c.estimated_recoverable_usd,
            estimated_recoverable_tokens=c.estimated_recoverable_tokens,
            estimate_basis=c.estimate_basis,
            agent_id=c.agent_id,
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
    finding_usd = getattr(finding, "estimated_recoverable_usd", None)

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
            advise_text=(
                "Trim the low-significance regions from this step's prompt "
                "template (boilerplate, repeated instructions, dead context) so "
                "every call carries fewer input tokens."
            ),
            estimated_recoverable_usd=usd,
            estimated_recoverable_tokens=acc["token_reduction"] or None,
            estimate_basis=str(getattr(finding, "estimate_basis", "") or ""),
            agent_id=agent if agent != "unknown" else "",
        ))
    return proposals


#: A ``sub_agent_id`` only names an agent definition when it is a plain slug.
#: Claude Code stamps a UUID for inline Task dispatches, and there is no file to
#: edit for those: that is the guidance-block fallback case, not a lookup to
#: guess at.
_AGENT_NAME_RE = re.compile(r"^[a-z][a-z0-9]*(?:-[a-z0-9]+)*$")

#: Cap on transcripts read to locate the repos a finding's sessions ran in.
_MAX_SCOPE_SESSIONS = 20


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
    pairs = [(sid, sid) for sid in session_ids[:_MAX_SCOPE_SESSIONS]]
    return _repo_cwd_map_for(pairs, root)


def _agent_model_plumbing(over_powered: list[Any], config: Any) -> dict[str, Any]:
    """Whether a flagged subagent has a definition file whose model can be set.

    Scope routing is relearn's: sessions concentrated in one repo write into
    that repo's ``.claude/agents/``, sessions spanning repos write into the
    user-global one. The flagged rows are cost-ordered, so the first subagent
    with a real definition file is the most expensive one that can be fixed
    outright. No file means the guidance block stays the fix, which is the
    inline Task-tool case.
    """
    from tokenjam.core.optimize.analyzers.model_downgrade import lookup_downgrade
    from tokenjam.core.optimize.analyzers.relearn import _scope_for
    from tokenjam.core.optimize.model_apply import (
        APPLY_KIND_AGENT_MODEL,
        default_agent_file_path,
    )

    named = [
        r for r in over_powered
        if _AGENT_NAME_RE.match(str(getattr(r, "sub_agent_id", "") or ""))
    ]
    if not named or config is None:
        return {}
    cwds = _session_cwds([str(r.session_id) for r in over_powered], config)
    repos = {Path(cwd).name for cwd in cwds.values() if cwd}
    scope = _scope_for(repos)
    repo_cwd = next(iter(cwds.values()), "") if len(repos) == 1 else ""

    for row in named:
        proposed = lookup_downgrade(str(row.provider), str(row.model))
        if not proposed:
            continue
        name = str(row.sub_agent_id)
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
        }
    return {}


def _subagent_to_proposals(finding: Any, config: Any = None) -> list[CostProposal]:
    """One proposal covering the subagent right-sizing finding.

    Unlike the three advise-only analyzers, this one is workspace-appliable for
    the common (CC-origin) case: the fan-out model choice is made by the
    orchestrating agent, which reads the workspace's CLAUDE.md — so a rung-1
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
    evidence = (
        f"{subagents} subagent dispatch(es) ran on a premium-tier model "
        f"({model_list}) but did little work (small output, few tool calls). "
        f"Subagents are {pct:.0f}% of the window's cost."
    )
    proposed_fix = (
        SUBAGENT_RUBRIC_INTRO
        + f"\n\nObserved oversized dispatches ran on: {model_list}. Route that "
        "shape to the cheaper same-family model next time."
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
            },
            advise_text=(
                "Lower the model tier for the flagged Task dispatches. "
                + str(getattr(finding, "caveat", "") or "") + advise_extra
            ).strip(),
            suggestion=f"model: {agent_apply['proposed_model']}",
            one_paste_fix=(
                f"# In {agent_apply['target_path']}, frontmatter:\n"
                f"model: {agent_apply['proposed_model']}"
            ),
            estimated_recoverable_usd=getattr(finding, "estimated_recoverable_usd", None),
            estimated_recoverable_tokens=getattr(finding, "estimated_recoverable_tokens", None),
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
        estimated_recoverable_usd=getattr(finding, "estimated_recoverable_usd", None),
        estimated_recoverable_tokens=getattr(finding, "estimated_recoverable_tokens", None),
        estimate_basis=str(getattr(finding, "estimate_basis", "") or ""),
        advise_only=not apply_capable,
        apply_capable=apply_capable,
        rung=1 if apply_capable else 0,
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
    """One proposal per dead-weight MCP server (Component C1).

    Reads ONLY ``DeadweightFinding.dead_servers`` — the C2 tax table (which
    lists every configured server, dead or alive, purely for ranked
    visibility) never feeds a proposal here, so a server's schema-injection
    tax is never counted both in the tax table AND a proposal (the same
    dedup guarantee ``compute_deadweight_finding`` itself enforces on
    ``estimated_recoverable_tokens`` / ``estimated_recoverable_usd``).

    ``estimated_recoverable_usd`` is carried straight off the analyzer's own
    ``ServerDeadweight.estimated_tax_usd_90d`` — the analyzer already prices
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
    for server in getattr(finding, "dead_servers", []) or []:
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
        advise = (
            server.fix + " Removing (or project-scoping) it is reversible "
            "and loses no data; it only stops the standing schema-injection "
            "tax on future sessions."
        )
        if plumbing.get("apply_capable"):
            advise += (
                f" tokenjam can remove this exact entry from "
                f"{plumbing['target_path']}, with the change committed and "
                f"revertable in one call."
            )
        elif plumbing.get("apply_blocked_reason"):
            advise += f" Applying it here is not on offer: {plumbing['apply_blocked_reason']}"
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
            },
            advise_text=advise,
            suggestion=f"claude mcp remove {server.name} --scope {scope_flag}",
            estimated_recoverable_tokens=server.estimated_tax_tokens_90d or None,
            estimated_recoverable_usd=server.estimated_tax_usd_90d,
            estimate_basis=(
                server.tax_construction
                + " Projected over a 90-day window from the sessions observed."
            ),
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


def _cluster_hash(value: Any) -> str:
    """Stable 12-hex-char identity for a cluster's structural key. Deterministic
    across runs over the same underlying signature (a JSON-serialisable
    structure), used only where the analyzer itself doesn't already hand back
    a cluster id (contrast ``ReuseCluster.cluster_id``, which does)."""
    encoded = json.dumps(value, sort_keys=False, default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:12]


# --------------------------------------------------------------------------- #
# Persona-gated fix modality — `script` / `reuse` / `verbosity` only.
#
# All three write the SAME class of artifact when apply-capable: a rung-1
# CLAUDE.md note or rung-2 `.claude/skills/<slug>/SKILL.md` file (see
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
    persona: str, proposed_fix: str, rung: int, scope: str,
) -> dict[str, Any]:
    """Decide, from the window's dominant persona, whether the rung-1/rung-2
    workspace write is offered — and fill in the ``CostProposal`` fields that
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
        "rung": rung if write_offered else 0,
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

    Apply-capable at rung 2: a skill note naming the repeated call pattern and
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
            estimated_recoverable_usd=cluster.total_cost_usd or None,
            estimated_recoverable_tokens=cluster.total_tokens or None,
            estimate_basis=str(getattr(finding, "estimate_basis", "") or ""),
            **_persona_gated_write_fields(persona, advise, rung=2, scope="project"),
        ))
    return proposals


def _reuse_to_proposals(finding: Any, persona: str = "unknown") -> list[CostProposal]:
    """One proposal per repeated planning-skeleton cluster.

    Apply-capable at rung 1: a CLAUDE.md note naming the recurring skeleton.
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
            estimated_recoverable_usd=cluster.cache_reuse_recoverable_usd or None,
            estimated_recoverable_tokens=cluster.cache_reuse_recoverable_tokens or None,
            estimate_basis=str(getattr(finding, "estimate_basis", "") or ""),
            **_persona_gated_write_fields(persona, advise, rung=1, scope="project"),
        ))
    return proposals


def _verbosity_to_proposals(finding: Any, persona: str = "unknown") -> list[CostProposal]:
    """One proposal for the whole verbosity finding (unlike ``script``/
    ``reuse``, this is a single window-wide signal, not per-cluster).

    Apply-capable at rung 1: a CLAUDE.md note carrying the finding's own
    ``remedy_snippet`` plus its suggested ``max_tokens`` cap. This is the
    least-grounded of the three analyzers by design (output length is not
    waste), so the note leans on the finding's own honesty caveat rather than
    asserting a saving. Left un-degraded to advise-only would strand the one
    lever this analyzer already computes (the remedy snippet) with nowhere to
    go; a workspace note is the same class of soft, orchestrator-read surface
    the ``subagent`` rubric already uses, so this reuses that precedent
    rather than inventing a new one.

    The write is only actually offered for a ``"claude-code"``/``"mixed"``
    ``persona`` — see ``_persona_gated_write_fields``. This is the sharpest
    case of the three: the lever itself (``max_tokens``) is genuinely
    SDK-actionable — an SDK caller can cap it in their own request — it is
    only the CLAUDE.md *artifact* that is the wrong surface for them. Gating
    still carries the cap as a ``suggestion`` so that lever isn't lost.
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
        advise += (
            f" Suggested cap for this shape of task: keep responses under "
            f"about {max_tokens:,} output tokens (the cohort baseline most "
            f"sessions already sit under)."
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
        estimated_recoverable_usd=getattr(finding, "estimated_recoverable_usd", None),
        estimated_recoverable_tokens=getattr(finding, "estimated_recoverable_tokens", None),
        estimate_basis=str(getattr(finding, "estimate_basis", "") or ""),
        **_persona_gated_write_fields(persona, advise, rung=1, scope="project"),
    )]


#: Default look-back for the daemon/CLI cost-proposal recompute. Matches the
#: monthly framing the cost analyzers project against.
DEFAULT_COST_WINDOW_DAYS = 30


def recompute_cost_proposals(
    db: Any,
    config: Any,
    *,
    window_days: int = DEFAULT_COST_WINDOW_DAYS,
    agent_id: str | None = None,
) -> list[CostProposal]:
    """Build an ``OptimizeReport`` over the last ``window_days``, adapt the
    three cost findings into proposals, and write them into the shared proposal
    store. Returns the proposals. Never raises — a build failure yields ``[]``
    (the inbox shows its empty state), never a crash on the refresh path.

    This is the "daemon path produces findings -> same proposal store" entry
    point the Review-inbox refresh calls; ``tj optimize`` can call it too so a
    manual run also refreshes the inbox.
    """
    from datetime import timedelta

    from tokenjam.core.framing import dominant_plan, plan_tier_mix, pricing_mode_for
    from tokenjam.core.optimize import relearn_store
    from tokenjam.core.optimize.runner import build_report
    from tokenjam.utils.time_parse import utcnow

    try:
        until = utcnow()
        since = until - timedelta(days=max(1, window_days))
        report = build_report(
            db, config, since, until, agent_id=agent_id,
            findings=list(COST_ANALYZERS),
        )
        # Same plan-tier -> pricing-mode resolution `tj optimize` uses, so the
        # web Review inbox suppresses the same dollar figures the CLI does
        # (placement's batch-lever dollars, currently the only card this
        # gates — see `_placement_to_proposals`).
        conn = getattr(db, "conn", None)
        plan_mix = plan_tier_mix(conn, since, until, agent_id) if conn is not None else {}
        pricing_mode = pricing_mode_for(dominant_plan(plan_mix))
        proposals = cost_proposals_from_report(
            report, config=config, pricing_mode=pricing_mode,
        )
    except Exception:
        return []

    try:
        relearn_store.write_cost_proposals(proposals, config=config)
    except Exception:
        pass
    return proposals


def cost_proposals_from_report(
    report: Any, config: Any = None, *, pricing_mode: str = "api",
) -> list[CostProposal]:
    """Every cost proposal derivable from an already-built ``OptimizeReport``.

    Reads the ``downsize`` finding off the typed ``report.downgrade`` slot and
    the ``cache`` / ``trim`` / ``subagent`` / ``placement`` / ``deadweight`` /
    ``script`` / ``reuse`` / ``verbosity`` findings off ``report.findings``.
    Missing findings (analyzer not run, no candidates) contribute nothing.
    Never raises — a malformed finding is skipped so one bad analyzer can't
    sink the inbox.

    ``config`` is optional and used for one thing: looking up the local source
    path a user registered for an agent, which decides whether the downsize card
    can offer the gated model-id swap or falls back to its one-paste artifact.
    Without it every card is advise-only.

    ``pricing_mode`` gates the ``placement`` card's dollar figure — the Batch
    API discount is an api-billed lever, so a subscription/local caller gets
    the advise text without a number, same as the CLI. Defaults to ``"api"``
    so existing callers that don't know their caller's plan keep today's
    behaviour.

    ``script`` / ``reuse`` / ``verbosity`` read ``report.persona`` (set once
    by ``runner.build_report`` — see ``AnalyzerContext.persona``) to decide
    whether their rung-1/rung-2 workspace write is offered at all — see
    ``_persona_gated_write_fields``. A report built without that field (e.g.
    hand-constructed in a test) defaults to ``"unknown"``, which keeps the
    write off rather than assuming ``"claude-code"``.
    """
    findings = getattr(report, "findings", {}) or {}
    persona = str(getattr(report, "persona", "") or "unknown")
    proposals: list[CostProposal] = []
    adapters = (
        (lambda f: _downsize_to_proposal(f, config), getattr(report, "downgrade", None)),
        (_cache_to_proposals, findings.get("cache")),
        (_cache_uncached_to_proposals, findings.get("cache")),
        (_cache_thrash_to_proposals, findings.get("cache")),
        (_cache_lookback_to_proposals, findings.get("cache")),
        (_trim_to_proposals, findings.get("trim")),
        (lambda f: _subagent_to_proposals(f, config), findings.get("subagent")),
        (lambda f: _placement_to_proposals(f, pricing_mode=pricing_mode), findings.get("placement")),
        (_deadweight_to_proposals, findings.get("deadweight")),
        (lambda f: _script_to_proposals(f, persona=persona), findings.get("script")),
        (lambda f: _reuse_to_proposals(f, persona=persona), findings.get("reuse")),
        (lambda f: _verbosity_to_proposals(f, persona=persona), findings.get("verbosity")),
    )
    for adapter, finding in adapters:
        try:
            proposals.extend(adapter(finding))
        except Exception:
            continue
    return proposals


# --------------------------------------------------------------------------- #
# Component E — the Review inbox's single "estimated recoverable" headline.
# Pure arithmetic over whatever proposals the caller hands it (the API route
# is what narrows that set to "open" — not yet applied — before calling
# this); kept separate so the sum/dedup logic is unit-testable on its own and
# never entangled with the applied-ledger lookup.
# --------------------------------------------------------------------------- #

def estimated_recoverable_rollup(
    proposals: list[Any],
    *,
    window_days: int = DEFAULT_COST_WINDOW_DAYS,
) -> dict[str, Any]:
    """Sum ``estimated_recoverable_usd`` across ``proposals``, deduplicated by
    ``signature`` (a proposal's stable identity — see the ``CostProposal``
    docstring) so a stale or duplicate cache entry is never double-counted.

    Generic over ``analyzer``: reads only the shared ``CostProposal`` fields,
    so a new analyzer's cards are picked up automatically with no changes
    here (§3's dedup rule already lives one layer down, in each analyzer's
    adapter — one underlying waste source becomes one card before it ever
    reaches this function).

    Only a proposal carrying a numeric estimate contributes to the sum AND to
    ``proposal_count`` — a card with no estimate yet still renders
    individually in the inbox, it just isn't folded into this aggregate
    (counting it in "N proposals" without a dollar contribution would
    silently understate the average the headline implies).

    ``estimated_recoverable_tokens`` is summed independently, over whichever
    proposals carry a token estimate — a different (often overlapping but never
    identical) set from the dollar-bearing ones. Renderers that lead with the
    token figure (subscription users, where dollars cover only the API-billed
    slice) must therefore quote ``token_proposal_count`` rather than
    ``proposal_count``, and say so against ``deduplicated_proposal_count`` when
    coverage is partial: the token sum is a floor, not a total.

    Tagged ``estimated`` — this is a heuristic figure, never a measured one.
    """
    seen: dict[str, dict[str, Any]] = {}
    for p in proposals:
        row = asdict(p) if is_dataclass(p) and not isinstance(p, type) else dict(p)
        sig = str(row.get("signature") or "")
        if not sig or sig in seen:
            continue  # empty/duplicate signature never counts twice
        seen[sig] = row

    contributing: list[dict[str, Any]] = []
    by_analyzer: dict[str, dict[str, Any]] = {}
    total_usd = 0.0
    total_tokens = 0
    token_proposal_count = 0
    for row in seen.values():
        # The token sum is counted INDEPENDENTLY of the dollar sum: the two
        # estimates are populated by different analyzers and a proposal can
        # carry either one alone. Folding tokens in only where a dollar
        # estimate also exists would silently understate the token headline
        # the suppressed-dollars rendering path leads with.
        tokens = row.get("estimated_recoverable_tokens")
        if tokens is not None:
            total_tokens += int(tokens)
            token_proposal_count += 1

        usd = row.get("estimated_recoverable_usd")
        if usd is None:
            continue
        usd = float(usd)
        total_usd += usd
        analyzer = str(row.get("analyzer") or "unknown")
        entry = by_analyzer.setdefault(analyzer, {"analyzer": analyzer, "count": 0, "usd": 0.0})
        entry["count"] += 1
        entry["usd"] = round(entry["usd"] + usd, 6)
        contributing.append({
            "signature": row.get("signature"), "analyzer": analyzer,
            "title": row.get("title"), "usd": round(usd, 6),
        })

    proposal_count = len(contributing)
    # Denominator for BOTH coverage claims: every open, deduplicated proposal,
    # including the ones carrying neither estimate. A renderer that says "across
    # N proposals" without this can't tell the reader its figure is partial.
    deduplicated_proposal_count = len(seen)
    if proposal_count == 0:
        basis = (
            "no open (not yet applied) cost proposal currently carries a "
            "dollar estimate. Estimated, correlational; never mixed with "
            "the measured verified-saved figure."
        )
    else:
        breakdown = "; ".join(
            f"{a['analyzer']} ({a['count']})"
            for a in sorted(by_analyzer.values(), key=lambda x: x["analyzer"])
        )
        basis = (
            f"sum of estimated_recoverable_usd across {proposal_count} of "
            f"{deduplicated_proposal_count} open (not yet applied), "
            f"deduplicated-by-signature cost proposal(s) over the last "
            f"{window_days}d; contributing analyzers: {breakdown}. "
            "Estimated, correlational; never mixed with the measured "
            "verified-saved figure."
        )
    if token_proposal_count:
        basis += (
            f" Token figure: sum of estimated_recoverable_tokens across "
            f"{token_proposal_count} of {deduplicated_proposal_count} "
            f"proposal(s); the rest carry no token estimate, so it is a "
            f"floor, not a total."
        )

    return {
        "estimated_recoverable_usd": round(total_usd, 6),
        "estimated_recoverable_tokens": total_tokens,
        "proposal_count": proposal_count,
        "token_proposal_count": token_proposal_count,
        "deduplicated_proposal_count": deduplicated_proposal_count,
        "window_days": window_days,
        "by_analyzer": sorted(by_analyzer.values(), key=lambda x: x["analyzer"]),
        "contributing": contributing,
        "estimate_confidence": COST_ESTIMATE_CONFIDENCE,
        "estimate_basis": basis,
    }
