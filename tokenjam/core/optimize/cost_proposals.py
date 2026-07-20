"""Adapt cost-analyzer findings into Review-inbox proposals ("advisories with
receipts").

The self-improve loop's pothole detector already produces
``PotholeCluster`` proposals that the Lens Improve inbox renders and that a
user can mark, apply, and verify. Three *cost* analyzers — ``downsize``
(model over-sizing), ``cache`` (cache efficacy), ``trim`` (prompt bloat) —
produce findings of a different shape. This module adapts each finding into a
``CostProposal`` so the inbox can list them BESIDE the pothole proposals, typed
by a distinct ``kind`` field.

Two structural facts carry over from the pothole ``advise_only`` lane and are
NOT optional here:

  * **Advise-only everywhere.** A cost fix lives in the user's own application
    code (a model-routing decision, a cache-prefix change, a prompt edit), not
    a workspace tokenjam may write into. So a cost proposal has NO apply path —
    exactly like an ``advise_only`` ``PotholeCluster`` (empty
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

from dataclasses import dataclass
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

    Mirrors the fields the inbox already reads off a ``PotholeCluster`` (title,
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
    # pothole apply path (``pothole_apply.apply_pothole_fix``). The adapter (not
    # the analyzer) supplies these; ``apply_capable`` gates the apply action, and
    # a proposal with no clean workspace surface degrades to advise-only like the
    # other three (``apply_capable=False``, ``advise_only=True``).
    apply_capable:        bool = False
    rung:                 int  = 0
    scope:                str  = ""
    proposed_fix:         str  = ""


# --------------------------------------------------------------------------- #
# Per-analyzer adapters. Each reads ONE finding dataclass and returns 0..N
# proposals. All tolerate a None/empty finding (returns []).
# --------------------------------------------------------------------------- #

def _downsize_to_proposal(finding: Any) -> list[CostProposal]:
    """One proposal covering the model-over-sizing finding.

    ``DowngradeFinding.suggestions`` maps each oversized model to its cheaper
    same-family alternative. The delta-verify pass later measures the model-mix
    cost delta across ALL flagged models, so a single proposal (listing them)
    keeps the estimate — which is a finding-level aggregate — coherent.
    """
    if finding is None or getattr(finding, "candidate_sessions", 0) <= 0:
        return []
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
        estimated_recoverable_usd=getattr(finding, "estimated_recoverable_usd", None),
        estimated_recoverable_tokens=getattr(finding, "estimated_recoverable_tokens", None),
        estimate_basis=str(getattr(finding, "estimate_basis", "") or ""),
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


def _subagent_to_proposals(finding: Any) -> list[CostProposal]:
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

    from tokenjam.core.optimize import pothole_store
    from tokenjam.core.optimize.runner import build_report
    from tokenjam.utils.time_parse import utcnow

    try:
        until = utcnow()
        since = until - timedelta(days=max(1, window_days))
        report = build_report(
            db, config, since, until, agent_id=agent_id,
            findings=list(COST_ANALYZERS),
        )
        proposals = cost_proposals_from_report(report)
    except Exception:
        return []

    try:
        pothole_store.write_cost_proposals(proposals, config=config)
    except Exception:
        pass
    return proposals


def cost_proposals_from_report(report: Any) -> list[CostProposal]:
    """Every cost proposal derivable from an already-built ``OptimizeReport``.

    Reads the ``downsize`` finding off the typed ``report.downgrade`` slot and
    the ``cache`` / ``trim`` / ``subagent`` findings off ``report.findings``.
    Missing findings (analyzer not run, no candidates) contribute nothing. Never
    raises — a malformed finding is skipped so one bad analyzer can't sink the
    inbox.
    """
    findings = getattr(report, "findings", {}) or {}
    proposals: list[CostProposal] = []
    adapters = (
        (_downsize_to_proposal, getattr(report, "downgrade", None)),
        (_cache_to_proposals, findings.get("cache")),
        (_trim_to_proposals, findings.get("trim")),
        (_subagent_to_proposals, findings.get("subagent")),
    )
    for adapter, finding in adapters:
        try:
            proposals.extend(adapter(finding))
        except Exception:
            continue
    return proposals
