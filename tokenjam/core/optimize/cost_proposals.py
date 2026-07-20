"""Adapt cost-analyzer findings into Review-inbox proposals ("advisories with
receipts").

The self-improve loop's relearn detector already produces
``RelearnCluster`` proposals that the Lens Improve inbox renders and that a
user can mark, apply, and verify. Three *cost* analyzers — ``downsize``
(model over-sizing), ``cache`` (cache efficacy), ``trim`` (prompt bloat) —
produce findings of a different shape. This module adapts each finding into a
``CostProposal`` so the inbox can list them BESIDE the relearn proposals, typed
by a distinct ``kind`` field.

Two structural facts carry over from the relearn ``advise_only`` lane and are
NOT optional here:

  * **Advise-only everywhere.** A cost fix lives in the user's own application
    code (a model-routing decision, a cache-prefix change, a prompt edit), not
    a workspace tokenjam may write into. So a cost proposal has NO apply path —
    exactly like an ``advise_only`` ``RelearnCluster`` (empty
    ``suggested_target``). The card carries a recommendation and, where
    sensible, a copyable config/code suggestion; the user applies it themselves.
  * **Estimated / correlational, never causal.** Every saving figure a cost
    finding carries is a heuristic ESTIMATE (house style, CLAUDE.md Rule 14).
    The adapter preserves the finding's own ``estimate_basis`` and labels the
    figure ``estimated``; the later realized delta (see ``cost_verify``) is
    correlational with the user's change, never proof tokenjam's advice caused
    it.

The adapter is pure: it reads an already-built ``OptimizeReport`` and returns
proposals. It never touches the DB, the store, or the network.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
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

#: The three analyzers this wiring covers, by registration name.
COST_ANALYZERS = ("downsize", "cache", "trim", "subagent")

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
    evidence, an estimate with its basis, ``advise_only``) plus the cost-
    specific ``target_key`` the delta-verify pass re-measures against.
    """
    kind:      str                     # always "cost" — the inbox discriminator
    analyzer:  str                     # "downsize" | "cache" | "trim" | "subagent"
    signature: str                     # stable identity for dedup + verify keying
    title:     str
    # WHICH thing is flagged, machine-readable — the key ``cost_verify`` uses to
    # re-measure a delta over spans after the user marks the proposal applied
    # (downsize: the oversized model(s); cache: a provider/model; trim: an
    # agent/step).
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


def _placement_to_proposals(finding: Any) -> list[CostProposal]:
    """One card for the batch-placement candidates (advise-only).

    Advise-only is not a formality here: moving a workload to the batch lane is
    an architectural change in the user's own application, and the card says so
    beside the number.
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
    advise = (
        f"The Batch API bills a flat 50% of standard prices, so the same work "
        f"on the batch lane is {_money(saving)} less over this window "
        f"(estimated). {getattr(finding, 'friction', '')} Nothing here is "
        f"applied for you; the change lives in your own application code."
    )
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
        estimated_recoverable_usd=saving,
        estimated_recoverable_tokens=getattr(finding, "estimated_recoverable_tokens", None),
        estimate_basis=str(getattr(finding, "estimate_basis", "") or ""),
        agent_id=candidates[0].agent_id if len(candidates) == 1 else "",
    )]


def _cache_to_proposals(finding: Any) -> list[CostProposal]:
    """One proposal per flagged (provider, model) cache-efficacy row."""
    if finding is None:
        return []
    from tokenjam.core.optimize.analyzers.cache_efficacy import (
        estimate_cache_recoverable,
    )

    proposals: list[CostProposal] = []
    for row in getattr(finding, "flagged", []) or []:
        usd, tokens = estimate_cache_recoverable([row])
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
            estimate_basis=str(getattr(finding, "estimate_basis", "") or ""),
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
        proposals = cost_proposals_from_report(report, config=config)
    except Exception:
        return []

    try:
        relearn_store.write_cost_proposals(proposals, config=config)
    except Exception:
        pass
    return proposals


def cost_proposals_from_report(report: Any, config: Any = None) -> list[CostProposal]:
    """Every cost proposal derivable from an already-built ``OptimizeReport``.

    Reads the ``downsize`` finding off the typed ``report.downgrade`` slot and
    the ``cache`` / ``trim`` / ``subagent`` / ``placement`` findings off
    ``report.findings``. Missing findings (analyzer not run, no candidates)
    contribute nothing. Never raises — a malformed finding is skipped so one bad
    analyzer can't sink the inbox.

    ``config`` is optional and used for one thing: looking up the local source
    path a user registered for an agent, which decides whether the downsize card
    can offer the gated model-id swap or falls back to its one-paste artifact.
    Without it every card is advise-only.
    """
    findings = getattr(report, "findings", {}) or {}
    proposals: list[CostProposal] = []
    adapters = (
        (lambda f: _downsize_to_proposal(f, config), getattr(report, "downgrade", None)),
        (_cache_to_proposals, findings.get("cache")),
        (_trim_to_proposals, findings.get("trim")),
        (lambda f: _subagent_to_proposals(f, config), findings.get("subagent")),
        (_placement_to_proposals, findings.get("placement")),
    )
    for adapter, finding in adapters:
        try:
            proposals.extend(adapter(finding))
        except Exception:
            continue
    return proposals
