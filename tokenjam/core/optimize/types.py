"""Dataclasses used by tj optimize analyzers and the runner."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

# Mandatory caveat string. Every channel that surfaces the downsize
# finding must include this verbatim; spec rule #2 is non-negotiable.
MODEL_DOWNGRADE_CAVEAT = (
    "Candidate-flagging heuristic, not a quality judgment. "
    "Review the example sessions before changing models."
)

# Mandatory caveat string for the Opus quota audit (issue #5). Honesty
# discipline (CLAUDE.md Rule 14): the audit flags Opus sessions whose STRUCTURE
# matches Sonnet-shaped work — it is an accountability list to spot-check, never
# a claim that the cheaper model would have produced the same answer. Surfaced
# verbatim next to every "% of Opus quota reclaimable" headline.
OPUS_QUOTA_AUDIT_CAVEAT = (
    "Candidates to spot-check, not a verdict. Each flagged session merely has "
    "the structural shape (small input/output, few tool calls) of work a "
    "smaller model often handles — review the example sessions before changing "
    "your routing. Never \"safe to downgrade.\""
)

# Mandatory caveat string for the Reuse analyzer. Honesty discipline
# (CLAUDE.md Rule 14): structural detection only, never a claim of
# interchangeability. Surfaced verbatim next to every recoverable figure.
REUSE_HONESTY_CAVEAT = (
    "Structural skeleton match, not a guarantee the plans were "
    "interchangeable. Review the templates before reusing them."
)

# Required `estimate_basis` for ReuseFinding (issue #115 AC8 / savings
# contract). Must contain the word "review".
REUSE_ESTIMATE_BASIS = (
    "structurally repeated planning calls — cache-reuse number assumes future "
    "re-plans skip the LLM call entirely; review templates before reusing"
)


@dataclass
class WindowSummary:
    since:       datetime
    until:       datetime
    days:        float
    sessions:    int
    spans:       int
    total_tokens: int
    total_cost_usd: float
    thin_data:   bool


@dataclass
class DowngradeExample:
    trace_id:   str
    session_id: str | None
    model:      str
    tool_calls: int
    duration_seconds: float | None
    cost_usd:   float


@dataclass
class DowngradeFinding:
    candidate_sessions: int
    total_sessions:     int
    actual_cost_usd:    float
    alternative_cost_usd: float
    monthly_savings_usd: float
    percent_of_sessions: float
    examples:           list[DowngradeExample]
    suggestions:        dict[str, str]
    caveat:             str = MODEL_DOWNGRADE_CAVEAT
    bench_command:      str | None = None
    # Token-share fields. Same model swap doesn't reduce token count, but for
    # subscription users (who pay a flat fee) the meaningful framing is
    # "candidate sessions are X% of your cycle's tokens — routing those to a
    # cheaper model frees that share against your plan cap."
    candidate_tokens:           int   = 0  # input + output + cache, candidates only
    window_total_tokens:        int   = 0  # input + output + cache, all sessions
    percent_of_tokens:          float = 0.0
    monthly_tokens_in_candidates: int = 0  # projected to a 30-day month
    # Recoverable-savings contract (#111). estimated_recoverable_usd is for
    # api-billed framing; estimated_recoverable_tokens for subscription / local.
    # None means "no estimate available for this finding state". estimate_basis
    # is the one-line heuristic explanation surfaced behind the "estimated" tag.
    # estimate_confidence is the estimate's confidence (distinct from any
    # structural `confidence` on wave-2 findings); always "heuristic" in v1.
    estimated_recoverable_usd:    float | None = None
    estimated_recoverable_tokens: int | None   = None
    estimate_basis:               str          = ""
    estimate_confidence:          str          = "heuristic"
    # Sampling confidence (#308). `n_sessions` is the candidate-session sample
    # the projection rests on; `ci_low`/`ci_high` are the 95% bootstrap interval
    # on `monthly_savings_usd`, so a 5-session estimate shows a visibly wider
    # band than a 500-session one. This is SAMPLING confidence on the projection,
    # NOT a claim the model swap preserves quality — the MODEL_DOWNGRADE_CAVEAT
    # still governs that. ci_low/ci_high are None when n < 2 (no spread to
    # estimate from a single point).
    n_sessions:                   int          = 0
    ci_low:                       float | None = None
    ci_high:                      float | None = None


@dataclass
class OpusAuditExample:
    """One Opus session flagged as a Sonnet-shaped quota-reclaim candidate."""
    trace_id:   str
    session_id: str | None
    model:      str
    alt_model:  str
    input_tokens:  int
    output_tokens: int
    cache_tokens:  int
    tool_calls: int
    duration_seconds: float | None
    cost_usd:   float


@dataclass
class OpusQuotaAudit:
    """Retroactive Opus quota audit (issue #5).

    Reframes the structural downsize heuristic as an *accountability* audit
    scoped to Opus sessions: how much of your Opus quota was spent on sessions
    whose shape matches Sonnet-shaped work. The headline figure is
    ``percent_quota_reclaimable`` (candidate Opus tokens / total Opus tokens) —
    quota language, never a dollar "saving" (the subscription majority is on a
    flat fee; dollar framing mis-targets them). Dollar fields are a best-effort
    SECONDARY signal for API users only.
    """
    window_days: float = 0.0
    opus_sessions: int = 0
    opus_tokens: int = 0  # input + output + cache across all Opus sessions
    candidate_sessions: int = 0
    candidate_tokens: int = 0  # input + output + cache, candidates only
    # The headline: % of Opus quota (tokens) held by Sonnet-shaped sessions.
    percent_quota_reclaimable: float = 0.0
    percent_sessions: float = 0.0
    # model -> cheaper-alternative suggestions observed among the candidates.
    suggestions: dict[str, str] = field(default_factory=dict)
    examples: list[OpusAuditExample] = field(default_factory=list)
    # Secondary, API-only calibration figures (never the headline).
    actual_cost_usd: float = 0.0
    alternative_cost_usd: float = 0.0
    caveat: str = OPUS_QUOTA_AUDIT_CAVEAT

    @property
    def has_opus(self) -> bool:
        return self.opus_sessions > 0


@dataclass
class BudgetProjection:
    provider:               str
    budget_usd:             float
    cycle_start_day:        int
    cycle_start:            datetime
    cycle_end:              datetime
    days_into_cycle:        float
    days_remaining:         float
    window_spend_usd:       float
    daily_run_rate_usd:     float
    monthly_run_rate_usd:   float
    projected_cycle_total:  float
    projected_overage_usd:  float
    exhaustion_date:        datetime | None
    days_until_exhaustion:  float | None
    over_budget:            bool
    applies_to_services:    list[str]
    downgrade_run_rate_usd: float | None = None


@dataclass
class ReuseCluster:
    """One cluster of sessions sharing the same planning skeleton."""
    cluster_id:        str                 # deterministic hash of the cluster key
    tool_signature:    tuple[str, ...]     # ordered tool names after the planner
    prompt_prefix_hash: str | None         # None when capture.prompts is off
    repetitions:       int                 # number of sessions in the cluster
    avg_planning_tokens: int               # mean tokens of the planning LLM call
    avg_planning_cost_usd: float           # mean USD cost of the planning LLM call
    # Two recoverable framings (savings contract). Cache-reuse is the
    # conservative number (you already paid once); script-replacement is the
    # upper bound (replace every planning call with a deterministic template).
    cache_reuse_recoverable_usd:        float
    script_replacement_recoverable_usd: float
    cache_reuse_recoverable_tokens:        int
    script_replacement_recoverable_tokens: int
    example_session_ids: list[str]         # top 3, ordered by recency
    skeleton_session_id: str               # which session's plan to render
    caveat:            str = REUSE_HONESTY_CAVEAT


@dataclass
class ReuseFinding:
    """Clusters of sessions with structurally repeated planning calls."""
    clusters:      list[ReuseCluster] = field(default_factory=list)
    capture_mode:  Literal["tool_sequence_only", "with_prompt_prefix"] = (
        "tool_sequence_only"
    )
    # Recoverable-savings contract (#111). The aggregate uses the conservative
    # cache-reuse number — the front-page tile shows what's recoverable going
    # forward, not the script-replacement upper bound. None when no cluster
    # cleared the thresholds.
    estimated_recoverable_usd:    float | None = None
    estimated_recoverable_tokens: int | None   = None
    estimate_basis:    str = REUSE_ESTIMATE_BASIS
    confidence:        Literal["heuristic"] = "heuristic"
    # Populated in Mode 1 (capture.prompts off) to nudge the richer mode.
    hint:              str = ""


@dataclass
class OptimizeReport:
    window:    WindowSummary
    downgrade: DowngradeFinding | None = None
    budgets:   list[BudgetProjection] = field(default_factory=list)
    notes:     list[str] = field(default_factory=list)
    # Generic findings dict keyed by analyzer registration name. Wave 2
    # analyzers (cache, cache-recommend, trim,
    # script) attach their results here so adding a new
    # analyzer doesn't require a typed slot on this dataclass.
    # Existing analyzers (downsize, budget-projection) keep their
    # typed slots above for backwards-compat with cmd_optimize and mcp.
    findings:  dict = field(default_factory=dict)


@dataclass
class AnalyzerContext:
    """
    Shared state passed to each analyzer. Analyzers read from `conn`, `config`,
    `since`, `until`, `agent_id`, and `summary`; they write findings into
    `report` (mutating in place).

    Cross-analyzer dependencies (e.g. budget-projection reads the downgrade
    finding via `report.downgrade`) are expressed by ordering analyzers in
    `tokenjam.core.optimize.runner.ANALYZER_ORDER`.
    """
    conn:                   Any
    config:                 Any            # TjConfig (avoid circular import)
    since:                  datetime
    until:                  datetime
    agent_id:               str | None
    window_days:            float
    summary:                WindowSummary
    report:                 OptimizeReport
    # Budget-analyzer flow control:
    budget_provider_filter: str | None    = None
    budget_usd_override:    float | None  = None
