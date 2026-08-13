"""
Subagent right-sizing analyzer.

Claude Code spawns subagents (the Task tool); each subagent's turns are stored
under the parent session with a `sub_agent_id` (set by backfill from the
record's `agentId` / `isSidechain`). A single research session routinely spawns
dozens of subagents that, folded into one parent total, hide where the tokens
actually went — on a real session we measured 66% of spend across ~147
subagents, invisible above the DB.

This analyzer breaks a window's cost down per subagent and flags two structural
right-sizing candidates (honesty discipline, CLAUDE.md Rule 14 — candidate
flags only, never a quality judgment):

  * over_powered     — ran on a premium-tier model (Fable or Opus, via the
                       shared model_tiers predicate); a cheaper same-family
                       model is worth a look, REGARDLESS of how much output it
                       produced or how many tool calls it made. This gate used
                       to also require output_tokens < 2,000 AND tool_calls <=
                       5 — but a Claude Code Task subagent is a full agent loop
                       (100-400 LLM calls, cache-read in the hundreds of
                       millions of tokens across a session), not one dispatch/
                       one answer, so cost compounds with output length AND
                       tool-call count. Requiring both to stay SMALL made the
                       worst offenders (the ones that did the most work) the
                       LEAST eligible to be flagged — the exact gate inversion
                       CLAUDE.md Critical Rule 29 warns about. Measured on a
                       real corpus: of premium-tier subagent spend, only 6.5%
                       cleared BOTH clauses; loosening either threshold instead
                       of dropping them (a sensitivity sweep up to output <
                       25,000 with no tool-call limit) still only reached 18%
                       of subagent spend. Dropping both clauses and pricing the
                       swap over the full premium population is the fix.
  * over_provisioned — was handed a large context (input + cache reads) yet
                       produced little output; the prompt it was dispatched
                       with is likely larger than the task needed.

It reads aggregate token counts only — no content capture required. The
`sub_agent_id` column is populated by the Claude Code backfill path; spans from
other runtimes carry NULL and are ignored here.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from tokenjam.core.model_tiers import is_premium_tier
from tokenjam.core.optimize.analyzers.model_downgrade import lookup_downgrade
from tokenjam.core.optimize.registry import register
from tokenjam.core.optimize.span_pricing import price_span, rates_at, span_instant
from tokenjam.core.optimize.types import AnalyzerContext
from tokenjam.core.persona_scope import add_persona_clause

# "Produced little": total output tokens below this look like a small task.
# Used by over_provisioned only — over_powered dropped this clause (see module
# docstring; CLAUDE.md Critical Rule 29's gate-inversion note).
SMALL_OUTPUT_TOKENS = 2_000

# "Handed a large context": input + cache-read tokens at or above this.
CONTEXT_HEAVY_TOKENS = 50_000

# Noise floor: don't flag subagents whose spend is trivially small, regardless
# of shape — the absolute saving isn't worth a recommendation.
MIN_FLAG_COST_USD = 0.05

# A dispatch cohort (same calling agent + same model — the "like-shaped
# dispatches" peer group) needs at least this many subagents before its
# median context size is a meaningful baseline. Mirrors output_verbosity.py's
# MIN_COHORT_SESSIONS (same config knob, OptimizeConfig.min_cohort_sessions —
# kept as a literal default here too, same convention as every other
# analyzer's module-local default constant).
MIN_COHORT_SESSIONS = 5

# Cap on how many rows the finding payload carries (a single session can spawn
# 100+ subagents). Aggregates below are computed over ALL rows; only the
# rendered/serialised lists are capped, top-by-cost.
TOP_N = 25

# Honesty caveat (CLAUDE.md Rule 14). Surfaced verbatim next to the flags.
SUBAGENT_HONESTY_CAVEAT = (
    "Candidate-flagging heuristic, not a quality judgment. Review the flagged "
    "subagents before changing how you dispatch them or which model they use."
)

# estimate_basis for the savings contract (#111). Two quantified components:
# over_powered is the token-cost delta of routing each flagged subagent (ANY
# premium-tier subagent above the noise floor — see module docstring for why
# this gate no longer restricts by output size or tool-call count) to its
# one-tier-down swap target (see SUBAGENT_DOWNGRADE_TARGET) over the SAME
# tokens — a model-swap counterfactual with no token-count change and no
# quality claim. over_provisioned is the context EXCESS over its dispatch
# cohort's own median (same calling agent + model — the like-shaped peer
# group, mirrors output_verbosity.py's per-task-shape median), priced at the
# cache-read rate (context arrives overwhelmingly as cache reads). Neither
# invents a target size or a rate: a subagent with no cheaper alternative, no
# pricing data, or no cohort baseline (fewer than min_cohort_sessions
# like-shaped peers) contributes nothing to that component.
SUBAGENT_ESTIMATE_BASIS = (
    "over_powered subagents (any premium-tier subagent above the noise floor) are priced at "
    "claude-sonnet-5, one tier down. That is not model_downgrade's two-tier opus-to-haiku "
    "jump. The price uses the same tokens: a model-swap delta, structural fit only.\n\n"
    "over_provisioned subagents are priced on their context excess over their dispatch "
    "cohort's own median (same calling agent + model). That excess is priced at the "
    "cache-read rate. There is no quality validation: review before re-dispatching. "
    "No guaranteed saving."
)


@dataclass
class SubagentRow:
    """One (session, subagent) group's usage + structural flags."""
    session_id:         str
    sub_agent_id:       str
    model:              str
    llm_calls:          int
    tool_calls:         int
    input_tokens:       int
    output_tokens:      int
    cache_tokens:       int
    cache_write_tokens: int
    cost_usd:           float
    # Provider is captured so the over_powered estimate can price the cheaper
    # same-family alternative; defaulted for round-trip of pre-provider payloads.
    provider:           str = "unknown"
    flags:              list[str] = field(default_factory=list)
    # The dispatching agent — with `model`, defines the "like-shaped dispatch"
    # cohort the over_provisioned estimate baselines against (see
    # `_dispatch_cohort_key`). Defaulted for round-trip of older payloads.
    agent_id:           str = ""
    # The STABLE identity: the agent TYPE that was dispatched. `sub_agent_id`
    # above is per-DISPATCH, so it belongs to one session by construction and
    # can neither form a cross-session cohort nor name an agent definition;
    # this is the name that addresses `.claude/agents/<name>.md`. Both are kept
    # because they answer different questions — grouping stays on
    # `(session_id, sub_agent_id)` so two concurrent dispatches of the SAME
    # type remain separate rows, while cohorting and file resolution key on the
    # type. Empty when the span carries no type: history ingested before the
    # column existed, or a deliberate non-recording (see
    # `backfill._PER_DISPATCH_TASK_KINDS`). Defaulted for round-trip of older
    # payloads.
    sub_agent_type:     str = ""
    # When this subagent's earliest span ran. Carried so the row can be priced
    # at the rate that actually billed it rather than at today's list price —
    # the pricing table has a time axis, and a row is an aggregate with no
    # instant of its own otherwise. Defaulted for round-trip of older payloads;
    # `span_pricing.span_instant` supplies the window start when it is absent.
    started_at:         datetime | None = None


@dataclass
class SubagentRightsizingFinding:
    """Per-subagent cost breakdown for the window, plus flagged candidates."""
    sessions_with_subagents: int = 0
    total_subagents:         int = 0
    subagent_cost_usd:       float = 0.0
    subagent_tokens:         int = 0
    window_cost_usd:         float = 0.0
    percent_of_cost:         float = 0.0
    flagged_cost_usd:        float = 0.0
    rows:        list[SubagentRow] = field(default_factory=list)  # top-by-cost
    flagged:     list[SubagentRow] = field(default_factory=list)  # candidates
    confidence:  str = "structural"
    caveat:      str = SUBAGENT_HONESTY_CAVEAT
    # Recoverable-savings contract (#111): the quantified sum of the
    # over_powered model-swap delta plus the over_provisioned context-excess
    # estimate (see SUBAGENT_ESTIMATE_BASIS). None only when neither component
    # priced anything (e.g. no flagged subagent had a priced alternative or a
    # usable cohort baseline) — flagged_cost_usd above still reports the raw
    # spend sitting in every flagged row regardless.
    past_overspend_usd:    float | None = None
    past_overspend_tokens: int | None   = None
    estimate_basis:               str          = SUBAGENT_ESTIMATE_BASIS
    # The effective noise floor this run applied (config-overridable, see
    # core.config.OptimizeConfig.min_flag_cost_usd) — carried on the finding
    # so a renderer never hardcodes a number that could be stale against the
    # user's own config.
    min_flag_cost_usd:            float        = MIN_FLAG_COST_USD
    estimate_confidence:          str          = "heuristic"


def _flags_for(
    *, model: str, output_tokens: int,
    input_tokens: int, cache_tokens: int, cost_usd: float,
    min_flag_cost_usd: float = MIN_FLAG_COST_USD,
) -> list[str]:
    """Structural right-sizing flags for one subagent (candidate-only)."""
    if cost_usd < min_flag_cost_usd:
        return []
    flags: list[str] = []
    if is_premium_tier(model):
        flags.append("over_powered")
    if (input_tokens + cache_tokens) >= CONTEXT_HEAVY_TOKENS and output_tokens < SMALL_OUTPUT_TOKENS:
        flags.append("over_provisioned")
    return flags


def _compute_rows(
    conn, since, until, agent_id: str | None,
    *, persona_scope: str | None = None,
    min_flag_cost_usd: float = MIN_FLAG_COST_USD,
) -> list[SubagentRow]:
    """Aggregate per (session_id, sub_agent_id) for real subagents in window."""
    clauses = [
        "start_time >= $1", "start_time < $2",
        "sub_agent_id IS NOT NULL",
    ]
    params: list[Any] = [since, until]
    if agent_id:
        clauses.append(f"agent_id = ${len(params) + 1}")
        params.append(agent_id)
    # The persona POPULATION scope. Without it this analyzer's dollar figure is
    # computed over the whole mixed corpus and then published under whichever
    # persona the reader picked. See `core/persona_scope.py`.
    add_persona_clause(clauses, persona_scope)
    where = " AND ".join(clauses)
    rows = conn.execute(
        f"SELECT session_id, sub_agent_id, "
        f"FIRST(agent_id) AS agent_id, "
        # Constant within a (session, dispatch) group — backfill stamps it once
        # per transcript — so MAX just picks it while ignoring NULLs from any
        # pre-migration-19 span in the same group.
        f"MAX(sub_agent_type) AS sub_agent_type, "
        f"arg_max(model, COALESCE(input_tokens, 0) + COALESCE(output_tokens, 0)) AS model, "
        f"arg_max(provider, COALESCE(input_tokens, 0) + COALESCE(output_tokens, 0)) AS provider, "
        f"COUNT(*) FILTER (WHERE name = 'gen_ai.llm.call') AS llm_calls, "
        f"COUNT(*) FILTER (WHERE tool_name IS NOT NULL) AS tool_calls, "
        f"COALESCE(SUM(input_tokens), 0) AS in_tok, "
        f"COALESCE(SUM(output_tokens), 0) AS out_tok, "
        f"COALESCE(SUM(cache_tokens), 0) AS cache_tok, "
        f"COALESCE(SUM(cache_write_tokens), 0) AS cache_w_tok, "
        f"COALESCE(SUM(cost_usd), 0.0) AS cost, "
        f"MIN(start_time) AS started_at "
        f"FROM spans WHERE {where} "
        f"GROUP BY session_id, sub_agent_id "
        f"ORDER BY cost DESC",
        params,
    ).fetchall()

    result: list[SubagentRow] = []
    for (sid, said, row_agent_id, said_type, model, provider, llm_calls,
         tool_calls, in_tok, out_tok, cache_tok, cache_w_tok, cost,
         started_at) in rows:
        in_tok = int(in_tok or 0)
        out_tok = int(out_tok or 0)
        cache_tok = int(cache_tok or 0)
        cache_w_tok = int(cache_w_tok or 0)
        cost = float(cost or 0.0)
        model = str(model or "unknown")
        provider = str(provider or "unknown")
        flags = _flags_for(
            model=model, output_tokens=out_tok,
            input_tokens=in_tok, cache_tokens=cache_tok, cost_usd=cost,
            min_flag_cost_usd=min_flag_cost_usd,
        )
        result.append(SubagentRow(
            session_id=str(sid),
            sub_agent_id=str(said),
            agent_id=str(row_agent_id or ""),
            sub_agent_type=str(said_type or ""),
            model=model,
            llm_calls=int(llm_calls or 0),
            tool_calls=int(tool_calls or 0),
            input_tokens=in_tok,
            output_tokens=out_tok,
            cache_tokens=cache_tok,
            cache_write_tokens=cache_w_tok,
            cost_usd=round(cost, 8),
            provider=provider,
            flags=flags,
            started_at=started_at,
        ))
    return result


# Swap target this analyzer prices over_powered subagents against: one tier
# down from the premium tier, NOT model_downgrade.lookup_downgrade's opus ->
# haiku (a two-tier jump). That ladder is tuned for analyze_model_downgrade's
# whole-session premium quota audit, which only ever flags structurally
# Sonnet-shaped sessions (tiny input/output/tool-call counts) and can afford
# an aggressive drop. A subagent flagged here can be a full agent loop that
# did real, extensive work, just possibly on too expensive a model — so it
# earns the narrower, defensible one-tier-down step instead: claude-sonnet-5,
# $2/$10 per Mtok, one tier below Opus/Fable, current-generation pricing that
# survives an audit. Anthropic-only for now; other providers fall back to
# model_downgrade's shared ladder (openai/google), which this analyzer has no
# reason to override.
SUBAGENT_DOWNGRADE_TARGET: dict[str, str] = {
    "anthropic": "claude-sonnet-5",
}


def _subagent_downgrade_target(provider: str, model: str) -> str | None:
    """The swap target priced for an over_powered subagent on ``(provider,
    model)``. See :data:`SUBAGENT_DOWNGRADE_TARGET` for why this is an
    explicit one-tier-down choice rather than a passthrough to
    ``model_downgrade.lookup_downgrade``.

    Note what the override does to the ``model`` argument on a provider that
    HAS an entry: nothing. It is ignored, and every model on that provider maps
    to the one target. A consequence worth knowing before reasoning about the
    two analyzers together: this analyzer therefore never consults
    ``lookup_downgrade``'s table for such a provider, and so is immune to that
    table's coverage. A premium model with no row there still gets a target
    here, and ADDING a row there changes this analyzer's output not at all —
    it moves ``downsize``'s numbers only. Reading a downgrade-table change as
    though it flowed through to both is the easy mistake; the fallthrough on
    the last line is reached only by providers absent from the override.
    """
    target = SUBAGENT_DOWNGRADE_TARGET.get(provider)
    if target:
        return target
    return lookup_downgrade(provider, model)


def _alt_cost_for_row(
    r: SubagentRow, alt_model: str, *, window_start: datetime,
) -> float | None:
    """Cost of ``r``'s exact token mix priced at ``alt_model``, or ``None`` when
    the alternative has no pricing data. Routes through
    :func:`tokenjam.core.optimize.span_pricing.price_span`, and so through
    :func:`tokenjam.core.cost.calculate_cost` — the ONE place that prices all
    four token classes (input + output + cache read + cache write) — rather
    than hand-rolling the arithmetic here a second time (a second hand-rolled
    pricer is exactly how `model_downgrade._alt_unit_cost` silently dropped
    cache-write from its alternative side while pricing it on the actual
    side). Every token class the original was billed for is priced on the
    alternative side too, so the delta reflects only the rate difference —
    never an artificial saving from dropping a token class the cheaper model
    would still be charged for.

    Priced at the row's OWN start, matching the rate that billed
    ``r.cost_usd``. Pricing the alternative at today's rate while the observed
    side is historical would put a rate change into the delta and call it a
    model-swap saving."""
    return price_span(
        r.provider, alt_model, at=span_instant(r.started_at, window_start=window_start),
        input_tokens=r.input_tokens, output_tokens=r.output_tokens,
        cache_read_tokens=r.cache_tokens, cache_write_tokens=r.cache_write_tokens,
    )


def _over_powered_recoverable(
    rows: list[SubagentRow], *, window_start: datetime,
) -> tuple[float, int]:
    """Conservative recoverable estimate over the over_powered subagents.

    For each subagent flagged ``over_powered`` (ran on a premium-tier model,
    above the noise floor), price its one-tier-down swap target (see
    :func:`_subagent_downgrade_target`) over the identical token mix and take
    the POSITIVE cost delta — a pure model-swap
    counterfactual, no token-count change and no quality claim. A subagent with
    no cheaper alternative or no pricing data contributes nothing (we refuse to
    invent a number). ``over_provisioned`` is priced separately, by
    :func:`_over_provisioned_recoverable` — see that function for why it needs
    the full row population, not just the flagged rows.

    Returns ``(recoverable_usd, recoverable_tokens)`` where the token figure is
    the quota (input + output + cache) sitting in the priced over_powered
    subagents — the share that routing to a cheaper model frees against a plan
    cap, mirroring the model-downgrade analyzer's token framing so the Overview
    tiles and the ranked-report share stay comparable.
    """
    recoverable_usd = 0.0
    recoverable_tokens = 0
    for r in rows:
        if "over_powered" not in r.flags:
            continue
        alt = _subagent_downgrade_target(r.provider, r.model)
        if not alt:
            continue
        alt_cost = _alt_cost_for_row(r, alt, window_start=window_start)
        if alt_cost is None:
            continue
        delta = r.cost_usd - alt_cost
        if delta <= 0:
            continue
        recoverable_usd += delta
        recoverable_tokens += (
            r.input_tokens + r.output_tokens
            + r.cache_tokens + r.cache_write_tokens
        )
    return recoverable_usd, recoverable_tokens


def _dispatch_cohort_key(r: SubagentRow) -> tuple[str, str]:
    """The "like-shaped dispatch" peer group for a subagent: same calling
    agent + same model. Mirrors output_verbosity.py's per-task-shape cohort —
    the closest defensible like-for-like grouping available here (a subagent
    dispatch carries no tool-call-sequence signature the way a whole session
    does, so agent + model is the shape signal this analyzer already has)."""
    return (r.agent_id, r.model)


def _over_provisioned_recoverable(
    rows: list[SubagentRow], *, window_start: datetime,
    min_cohort_sessions: int = MIN_COHORT_SESSIONS,
) -> tuple[float, int]:
    """Recoverable estimate over the over_provisioned subagents.

    Mirrors output_verbosity.py's cohort-median shape. Groups ALL subagents
    (not just the flagged ones — a flagged row's baseline is built from its
    unflagged peers too) into "like-shaped dispatch" cohorts (same calling
    agent + same model, see :func:`_dispatch_cohort_key`), takes each
    cohort's MEDIAN received context (input + cache tokens — the same
    aggregate the ``over_provisioned`` flag itself gates on), and for every
    flagged row with a usable cohort baseline (at least ``min_cohort_sessions``
    members, median > 0) prices ``received_context - cohort_median`` at the
    CACHE-READ rate: context arrives overwhelmingly as cache reads (~95% of
    window token volume, measured), never the full input rate a fresh prompt
    would cost. Never prices the excess against zero context (every dispatch
    needs some baseline) and never invents a target size: a cohort below the
    size bar, a flagged row already at or below its cohort's median, or a
    model with no pricing data contributes nothing (we refuse to invent a
    number).

    Returns ``(recoverable_usd, recoverable_tokens)`` — the token figure is
    the context excess itself, comparable to the model-downgrade/over_powered
    token framing above.
    """
    contexts_by_cohort: dict[tuple[str, str], list[int]] = {}
    for r in rows:
        contexts_by_cohort.setdefault(_dispatch_cohort_key(r), []).append(
            r.input_tokens + r.cache_tokens
        )

    cohort_medians: dict[tuple[str, str], float] = {
        key: statistics.median(contexts)
        for key, contexts in contexts_by_cohort.items()
        if len(contexts) >= min_cohort_sessions
    }

    recoverable_usd = 0.0
    recoverable_tokens = 0
    for r in rows:
        if "over_provisioned" not in r.flags:
            continue
        median = cohort_medians.get(_dispatch_cohort_key(r))
        if median is None or median <= 0:
            continue
        excess = int(round((r.input_tokens + r.cache_tokens) - median))
        if excess <= 0:
            continue
        rates = rates_at(
            r.provider, r.model, span_instant(r.started_at, window_start=window_start),
        )
        if rates is None:
            continue
        recoverable_usd += excess / 1_000_000 * rates.cache_read_per_mtok
        recoverable_tokens += excess
    return recoverable_usd, recoverable_tokens


@register("subagent")
def run(ctx: AnalyzerContext) -> None:
    """Registry entry point. Attaches the finding to ctx.report.findings."""
    optimize_cfg = getattr(ctx.config, "optimize", None)
    min_flag_cost_usd = getattr(
        optimize_cfg, "min_flag_cost_usd", MIN_FLAG_COST_USD,
    )
    min_cohort_sessions = getattr(
        optimize_cfg, "min_cohort_sessions", MIN_COHORT_SESSIONS,
    )
    rows = _compute_rows(
        ctx.conn, ctx.since, ctx.until, ctx.agent_id,
        persona_scope=ctx.persona_scope,
        min_flag_cost_usd=min_flag_cost_usd,
    )
    if not rows:
        return

    subagent_cost = sum(r.cost_usd for r in rows)
    subagent_tokens = sum(
        r.input_tokens + r.output_tokens + r.cache_tokens + r.cache_write_tokens
        for r in rows
    )
    window_cost = ctx.summary.total_cost_usd or 0.0
    flagged = sorted(
        (r for r in rows if r.flags), key=lambda r: r.cost_usd, reverse=True
    )

    # Quantified recoverable estimate (#101 / savings contract #111): the
    # model-swap delta over the over_powered subagents (computed over ALL
    # flagged rows, not just the TOP_N rendered, so the ranked-report share
    # reflects the full window) plus the context-excess-over-cohort-median
    # delta over the over_provisioned subagents (needs the FULL row
    # population, not just flagged, to build each dispatch cohort's
    # baseline — see `_over_provisioned_recoverable`). None when neither
    # component priced anything — the finding then renders unranked (honest
    # empty estimate), exactly like it did before either could quantify.
    op_usd, op_tokens = _over_powered_recoverable(flagged, window_start=ctx.since)
    ctx_usd, ctx_tokens = _over_provisioned_recoverable(
        rows, window_start=ctx.since, min_cohort_sessions=min_cohort_sessions,
    )
    recoverable_usd = op_usd + ctx_usd
    recoverable_tokens = op_tokens + ctx_tokens
    est_usd = round(recoverable_usd, 8) if recoverable_tokens > 0 else None
    est_tokens = recoverable_tokens if recoverable_tokens > 0 else None

    ctx.report.findings["subagent"] = SubagentRightsizingFinding(
        sessions_with_subagents=len({r.session_id for r in rows}),
        total_subagents=len(rows),
        subagent_cost_usd=round(subagent_cost, 8),
        subagent_tokens=subagent_tokens,
        window_cost_usd=round(window_cost, 8),
        percent_of_cost=round(subagent_cost / window_cost, 4) if window_cost > 0 else 0.0,
        flagged_cost_usd=round(sum(r.cost_usd for r in flagged), 8),
        rows=rows[:TOP_N],
        flagged=flagged[:TOP_N],
        past_overspend_usd=est_usd,
        past_overspend_tokens=est_tokens,
        min_flag_cost_usd=min_flag_cost_usd,
    )
