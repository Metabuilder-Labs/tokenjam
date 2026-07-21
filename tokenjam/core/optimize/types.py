"""Dataclasses used by tj optimize analyzers and the runner."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
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
# verbatim next to every "% of premium quota misallocated" headline.
OPUS_QUOTA_AUDIT_CAVEAT = (
    "Candidates to spot-check, not a verdict. Each flagged stretch merely has "
    "the structural shape (small new input/output, low tool fan-out, no "
    "delegation) of work a smaller model often handles — review the example "
    "sessions before changing your routing. Segment percentages flag stretches "
    "whose shape looks mechanical; the surrounding session context is not "
    "evaluated and may have justified the larger model. Never \"safe to "
    "downgrade.\""
)

# Confidence label for the segment-level misallocation estimate. The headline is
# a per-turn heuristic over contiguous mechanical stretches with no quality
# validation, so it is surfaced as an explicit "estimate" (with a wide bootstrap
# interval), never as a settled figure (CLAUDE.md Rule 14).
SEGMENT_ESTIMATE_CONFIDENCE = "estimate"

# The `estimate_basis` string surfaced behind the "estimate" label (Rule 14).
SEGMENT_ESTIMATE_BASIS = (
    "contiguous turn-stretches whose per-turn shape (small new input/output, low "
    "tool fan-out, no delegation) looks mechanical — surrounding session context "
    "not evaluated; no quality validation"
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
    ``percent_quota_misallocated`` (candidate Opus tokens / total Opus tokens) —
    retrospective quota is already SPENT, so this is a behaviour mirror (how much
    premium quota went to Sonnet-shaped sessions), never a claim it can be
    "reclaimed". Quota language, never a dollar "saving" (the subscription
    majority is on a flat fee; dollar framing mis-targets them). Dollar fields
    are a best-effort SECONDARY signal for API users only.
    """
    window_days: float = 0.0
    opus_sessions: int = 0
    # Quota-weighted premium token-equivalents (cache reads at 0.1x, output at
    # 5x — the #119 weighting) attributed per-turn to the turn's OWN model, so a
    # Sonnet turn inside an Opus session never lands in this premium total. This
    # is the denominator of the headline share.
    opus_tokens: int = 0
    candidate_sessions: int = 0
    # Quota-weighted premium token-equivalents inside flagged cheap segments —
    # the numerator. Segment-inclusive: a mechanical stretch inside an otherwise
    # hard session counts, which the old whole-session audit structurally missed.
    candidate_tokens: int = 0
    # THE headline (founder decision D1): ONE segment-inclusive misallocation
    # figure — the share of premium quota that went to Sonnet-shaped work,
    # computed on the corrected per-turn attribution. It is a labelled estimate
    # (segment_estimate_confidence + the bootstrap CI below), not two numbers.
    percent_quota_misallocated: float = 0.0
    percent_sessions: float = 0.0
    # Segment accounting + confidence (design §5.2 / §6, founder D3). The number
    # is a heuristic estimate, shown with an explicit label + a WIDE bootstrap
    # interval that resamples SEGMENTS (not sessions), so the band widens honestly
    # when few segments carry the estimate. ci low/high are None below 2 segments
    # (a single point has no spread — the estimate is inherently wide).
    segment_count: int = 0
    segment_estimate_confidence: str = SEGMENT_ESTIMATE_CONFIDENCE
    estimate_basis: str = SEGMENT_ESTIMATE_BASIS
    segment_ci_low: float | None = None
    segment_ci_high: float | None = None
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
    # The effective recurrence bar this run applied (config-overridable, see
    # core.config.OptimizeConfig.min_reuse_repetitions) — carried on the
    # finding so a renderer's empty-state message never hardcodes a number
    # that could be stale against the user's own config. Mirrors
    # analyzers.plan_reuse.MIN_REPETITIONS's default (kept as a literal here
    # to avoid a types -> analyzers import).
    min_repetitions:   int = 3


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


# ---------------------------------------------------------------------------
# OpusQuotaAudit (de)serialization — the round-trip pair for the daemon path
# ---------------------------------------------------------------------------
# `tj quota-audit` reads per-session token/model metadata the API shim can't
# expose at this grain, so when `tj serve` holds the DuckDB write lock the
# daemon computes the audit and returns `audit_to_dict(audit)` (mirroring
# `context_diagnostic.diagnostic_to_dict`). The CLI rebuilds the dataclass with
# `audit_from_dict` and renders identically. These are a genuine inverse pair —
# every field `audit_to_dict` emits, `audit_from_dict` reconstructs — so the
# serve path never silently drops a field the CLI renders.


def audit_to_dict(audit: OpusQuotaAudit) -> dict[str, Any]:
    """JSON-serialisable view of an :class:`OpusQuotaAudit` (round-trips)."""
    return {
        "window_days": audit.window_days,
        "opus_sessions": audit.opus_sessions,
        "opus_tokens": audit.opus_tokens,
        "candidate_sessions": audit.candidate_sessions,
        "candidate_tokens": audit.candidate_tokens,
        "percent_quota_misallocated": audit.percent_quota_misallocated,
        # DEPRECATED alias — kept one release (through 0.6.x) because
        # ``percent_quota_reclaimable`` shipped publicly in 0.5.4. Consumers
        # should read ``percent_quota_misallocated``; this mirror will be
        # removed. Emitted on BOTH the direct and serve paths so the parity
        # contract (byte-identical audit_to_dict output) still holds.
        "percent_quota_reclaimable": audit.percent_quota_misallocated,
        "percent_sessions": audit.percent_sessions,
        # Segment accounting + confidence (design §5.2 / §6). Every key here must
        # survive the round-trip below, or the data-access parity test fails —
        # the silent-drift guard for a new DB-computed field.
        "segment_count": audit.segment_count,
        "segment_estimate_confidence": audit.segment_estimate_confidence,
        "estimate_basis": audit.estimate_basis,
        "segment_ci_low": audit.segment_ci_low,
        "segment_ci_high": audit.segment_ci_high,
        "suggestions": dict(audit.suggestions),
        "examples": [asdict(ex) for ex in audit.examples],
        "actual_cost_usd": audit.actual_cost_usd,
        "alternative_cost_usd": audit.alternative_cost_usd,
        "caveat": audit.caveat,
    }


def audit_from_dict(data: dict[str, Any]) -> OpusQuotaAudit:
    """Reconstruct an :class:`OpusQuotaAudit` from :func:`audit_to_dict`.

    Missing keys fall back to the dataclass defaults so a server-side schema
    drift degrades gracefully rather than raising.
    """
    examples = [
        OpusAuditExample(
            trace_id=str(ex.get("trace_id", "")),
            session_id=ex.get("session_id"),
            model=str(ex.get("model", "")),
            alt_model=str(ex.get("alt_model", "")),
            input_tokens=int(ex.get("input_tokens", 0) or 0),
            output_tokens=int(ex.get("output_tokens", 0) or 0),
            cache_tokens=int(ex.get("cache_tokens", 0) or 0),
            tool_calls=int(ex.get("tool_calls", 0) or 0),
            duration_seconds=ex.get("duration_seconds"),
            cost_usd=float(ex.get("cost_usd", 0.0) or 0.0),
        )
        for ex in data.get("examples", []) or []
    ]
    return OpusQuotaAudit(
        window_days=float(data.get("window_days", 0.0) or 0.0),
        opus_sessions=int(data.get("opus_sessions", 0) or 0),
        opus_tokens=int(data.get("opus_tokens", 0) or 0),
        candidate_sessions=int(data.get("candidate_sessions", 0) or 0),
        candidate_tokens=int(data.get("candidate_tokens", 0) or 0),
        # Prefer the new key; fall back to the deprecated alias so a payload
        # produced by a pre-rename (0.5.4) daemon still reconstructs.
        percent_quota_misallocated=float(
            data.get("percent_quota_misallocated",
                     data.get("percent_quota_reclaimable", 0.0)) or 0.0
        ),
        percent_sessions=float(data.get("percent_sessions", 0.0) or 0.0),
        segment_count=int(data.get("segment_count", 0) or 0),
        segment_estimate_confidence=str(
            data.get("segment_estimate_confidence", SEGMENT_ESTIMATE_CONFIDENCE)
        ),
        estimate_basis=str(data.get("estimate_basis", SEGMENT_ESTIMATE_BASIS)),
        segment_ci_low=_opt_float(data.get("segment_ci_low")),
        segment_ci_high=_opt_float(data.get("segment_ci_high")),
        suggestions=dict(data.get("suggestions", {}) or {}),
        examples=examples,
        actual_cost_usd=float(data.get("actual_cost_usd", 0.0) or 0.0),
        alternative_cost_usd=float(data.get("alternative_cost_usd", 0.0) or 0.0),
        caveat=str(data.get("caveat", OPUS_QUOTA_AUDIT_CAVEAT)),
    )


def _opt_float(raw: Any) -> float | None:
    """Coerce a JSON value to ``float`` while preserving ``None`` (the CI bounds
    are ``None`` below 2 segments — that must round-trip as ``None``, not 0.0)."""
    return None if raw is None else float(raw)
