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
                       shared model_tiers predicate) but produced little output
                       and made few tool calls; a cheaper same-family model is
                       worth a look (mirrors the downsize heuristic, scoped to
                       one subagent).
  * over_provisioned — was handed a large context (input + cache reads) yet
                       produced little output; the prompt it was dispatched
                       with is likely larger than the task needed.

It reads aggregate token counts only — no content capture required. The
`sub_agent_id` column is populated by the Claude Code backfill path; spans from
other runtimes carry NULL and are ignored here.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from tokenjam.core.model_tiers import is_premium_tier
from tokenjam.core.optimize.registry import register
from tokenjam.core.optimize.types import AnalyzerContext

# "Produced little": total output tokens below this look like a small task.
SMALL_OUTPUT_TOKENS = 2_000

# "Did little tool work": at or below this many tool calls.
FEW_TOOL_CALLS = 5

# "Handed a large context": input + cache-read tokens at or above this.
CONTEXT_HEAVY_TOKENS = 50_000

# Noise floor: don't flag subagents whose spend is trivially small, regardless
# of shape — the absolute saving isn't worth a recommendation.
MIN_FLAG_COST_USD = 0.05

# Cap on how many rows the finding payload carries (a single session can spawn
# 100+ subagents). Aggregates below are computed over ALL rows; only the
# rendered/serialised lists are capped, top-by-cost.
TOP_N = 25

# Honesty caveat (CLAUDE.md Rule 14). Surfaced verbatim next to the flags.
SUBAGENT_HONESTY_CAVEAT = (
    "Candidate-flagging heuristic, not a quality judgment. Review the flagged "
    "subagents before changing how you dispatch them or which model they use."
)

# estimate_basis for the savings contract (#111). Candidate-only in v1 — we
# surface the spend concentrated in flagged subagents, not a guaranteed saving.
SUBAGENT_ESTIMATE_BASIS = (
    "spend concentrated in structurally-flagged subagents (premium model with "
    "little output, or large context with little output); review before "
    "re-dispatching — no guaranteed saving"
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
    flags:              list[str] = field(default_factory=list)


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
    # Recoverable-savings contract (#111). Candidate-only in v1: we report the
    # spend sitting in flagged subagents (flagged_cost_usd) rather than assert a
    # guaranteed recovery, so the precise estimate stays None.
    estimated_recoverable_usd:    float | None = None
    estimated_recoverable_tokens: int | None   = None
    estimate_basis:               str          = SUBAGENT_ESTIMATE_BASIS
    estimate_confidence:          str          = "heuristic"


def _flags_for(
    *, model: str, output_tokens: int, tool_calls: int,
    input_tokens: int, cache_tokens: int, cost_usd: float,
) -> list[str]:
    """Structural right-sizing flags for one subagent (candidate-only)."""
    if cost_usd < MIN_FLAG_COST_USD:
        return []
    flags: list[str] = []
    is_premium = is_premium_tier(model)
    if is_premium and output_tokens < SMALL_OUTPUT_TOKENS and tool_calls <= FEW_TOOL_CALLS:
        flags.append("over_powered")
    if (input_tokens + cache_tokens) >= CONTEXT_HEAVY_TOKENS and output_tokens < SMALL_OUTPUT_TOKENS:
        flags.append("over_provisioned")
    return flags


def _compute_rows(conn, since, until, agent_id: str | None) -> list[SubagentRow]:
    """Aggregate per (session_id, sub_agent_id) for real subagents in window."""
    clauses = [
        "start_time >= $1", "start_time < $2",
        "sub_agent_id IS NOT NULL",
    ]
    params: list[Any] = [since, until]
    if agent_id:
        clauses.append(f"agent_id = ${len(params) + 1}")
        params.append(agent_id)
    where = " AND ".join(clauses)
    rows = conn.execute(
        f"SELECT session_id, sub_agent_id, "
        f"arg_max(model, COALESCE(input_tokens, 0) + COALESCE(output_tokens, 0)) AS model, "
        f"COUNT(*) FILTER (WHERE name = 'gen_ai.llm.call') AS llm_calls, "
        f"COUNT(*) FILTER (WHERE tool_name IS NOT NULL) AS tool_calls, "
        f"COALESCE(SUM(input_tokens), 0) AS in_tok, "
        f"COALESCE(SUM(output_tokens), 0) AS out_tok, "
        f"COALESCE(SUM(cache_tokens), 0) AS cache_tok, "
        f"COALESCE(SUM(cache_write_tokens), 0) AS cache_w_tok, "
        f"COALESCE(SUM(cost_usd), 0.0) AS cost "
        f"FROM spans WHERE {where} "
        f"GROUP BY session_id, sub_agent_id "
        f"ORDER BY cost DESC",
        params,
    ).fetchall()

    result: list[SubagentRow] = []
    for (sid, said, model, llm_calls, tool_calls,
         in_tok, out_tok, cache_tok, cache_w_tok, cost) in rows:
        in_tok = int(in_tok or 0)
        out_tok = int(out_tok or 0)
        cache_tok = int(cache_tok or 0)
        cache_w_tok = int(cache_w_tok or 0)
        cost = float(cost or 0.0)
        model = str(model or "unknown")
        flags = _flags_for(
            model=model, output_tokens=out_tok, tool_calls=int(tool_calls or 0),
            input_tokens=in_tok, cache_tokens=cache_tok, cost_usd=cost,
        )
        result.append(SubagentRow(
            session_id=str(sid),
            sub_agent_id=str(said),
            model=model,
            llm_calls=int(llm_calls or 0),
            tool_calls=int(tool_calls or 0),
            input_tokens=in_tok,
            output_tokens=out_tok,
            cache_tokens=cache_tok,
            cache_write_tokens=cache_w_tok,
            cost_usd=round(cost, 8),
            flags=flags,
        ))
    return result


@register("subagent")
def run(ctx: AnalyzerContext) -> None:
    """Registry entry point. Attaches the finding to ctx.report.findings."""
    rows = _compute_rows(ctx.conn, ctx.since, ctx.until, ctx.agent_id)
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
    )
