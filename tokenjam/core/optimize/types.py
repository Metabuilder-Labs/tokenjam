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
    "structurally repeated planning calls — the headline prices the "
    "cache-reuse premise (a template only removes the RE-plan delta: avg cost "
    "x (reps - 1)), because the first planning call in a cluster had to "
    "happen: nothing existed to reuse yet. The script-replacement premise "
    "(a template/skill removes every planning call: avg cost x reps) stays "
    "available on every cluster as script_replacement_recoverable_usd/"
    "_tokens, but it is never the headline: charging the user for the one "
    "instance that was necessary work states a figure they can disprove "
    "(a 2-repetition cluster would be a 2x overclaim); review templates "
    "before reusing"
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
    #: Distinct calendar days within the window with >=1 session — the
    #: user's own "active day" pace, the `D_active` term of the 30-day
    #: projection basis in `core/optimize/projection.py`. It does NOT pace any
    #: per-analyzer dollar figure — no such figure exists any more (see the
    #: field contract in the repo `CLAUDE.md`). Defaults to 0 for a hand-built
    #: `WindowSummary` (tests, older callers, a cached report predating the
    #: field).
    active_days: int = 0


@dataclass
class DowngradeExample:
    trace_id:   str
    session_id: str | None
    model:      str
    tool_calls: int
    duration_seconds: float | None
    cost_usd:   float


@dataclass
class DriverRoleExample:
    """One session where a premium model drove undelegated, tool-heavy work.

    A spot-check row, not the aggregate: `offload_usd` + `tier_usd` are this
    session's own two halves of the inline-vs-routed counterfactual, so a user
    can open the session and check the claim against what actually happened.
    """
    session_id:      str
    agent_id:        str
    model:           str
    alt_model:       str
    turns:           int   # main-thread turns in the session
    stretch_turns:   int   # of those, turns inside a tool-driven stretch
    tool_calls:      int   # main-thread tool calls
    tail_tokens:     int   # tokens re-read purely because work stayed inline
    offload_usd:     float
    tier_usd:        float
    recoverable_usd: float


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
    # Recoverable-savings contract (#111). past_overspend_usd is for
    # api-billed framing; past_overspend_tokens for subscription / local.
    # None means "no estimate available for this finding state". estimate_basis
    # is the one-line heuristic explanation surfaced behind the "estimated" tag.
    # estimate_confidence is the estimate's confidence (distinct from any
    # structural `confidence` on wave-2 findings); always "heuristic" in v1.
    past_overspend_usd:    float | None = None
    past_overspend_tokens: int | None   = None
    estimate_basis:               str          = ""
    estimate_confidence:          str          = "heuristic"
    # Sampling confidence (#308). `n_sessions` is the candidate-session sample
    # the projection rests on — BOTH cases combined (`candidate_sessions +
    # driver_sessions`), since `past_overspend_usd` is their sum too; `ci_low`/
    # `ci_high` are the 95% bootstrap interval on `monthly_savings_usd`, so a
    # 5-session estimate shows a visibly wider band than a 500-session one.
    # This is SAMPLING confidence on the projection, NOT a claim the model
    # swap preserves quality — the MODEL_DOWNGRADE_CAVEAT still governs that.
    # ci_low/ci_high are None when n < 2 (no spread to estimate from a single
    # point).
    n_sessions:                   int          = 0
    ci_low:                       float | None = None
    ci_high:                      float | None = None
    # `n_sessions / total_sessions`, as a percent — the population this finding's
    # ONE `past_overspend_usd` actually covers (both cases). `percent_of_sessions`
    # above covers the tiny-session case ALONE, so a surface that renders the
    # dollar tile beside a "N of M sessions" stat must use THIS pair
    # (`n_sessions`/`percent_of_all_sessions`), never `candidate_sessions`/
    # `percent_of_sessions` — a window where the driver-role case carries the
    # whole figure has `candidate_sessions == 0`, which would render a real
    # dollar amount beside "0%, 0 of N" (two figures over two different
    # populations shown together).
    percent_of_all_sessions:      float        = 0.0
    # Per-agent price arithmetic for the proposed swap (one
    # `analyzers.downsize_agents.AgentPriceRow` per agent/model group over the
    # candidate sessions): exact per-type tokens at the current model's rates
    # versus the proposed model's, over the window and projected to 30 days.
    # Typed loosely to keep `types` free of an analyzer import; empty when no
    # candidate group had pricing data for both sides. `list[Any]` costs the
    # round trip its TYPE, though: `hydrate_dataclass` has nothing to dispatch
    # on, so a stored report came back with plain dicts here and every
    # consumer reading `row.delta_usd` raised — silently, wherever an adapter
    # caught broadly. The element type is therefore declared as DATA, resolved
    # lazily at hydration time (see `runner._hydrate_target`), which keeps this
    # module analyzer-free while making the round trip lossless in type as
    # well as in value.
    per_agent:                    list[Any]    = field(
        default_factory=list,
        metadata={
            "hydrate": "tokenjam.core.optimize.analyzers.downsize_agents:AgentPriceRow",
        },
    )
    # --- The PRIMARY case: a premium model in the driver role ---------------
    # Sessions where a premium-tier model drove long, undelegated, tool-heavy
    # work inline instead of routing it to cheap workers. Every field here
    # describes ONLY that case; the fields above describe ONLY the secondary
    # tiny-session case. The two populations are disjoint (a driver-flagged
    # session is excluded from the tiny-session walk), which is what makes it
    # safe for `past_overspend_usd` to be their sum.
    driver_sessions:          int   = 0
    driver_recoverable_usd:   float = 0.0
    # The two halves of the driver-role claim, kept visible so the headline is
    # never a black box. `driver_offload_usd` is the re-read tail that stops
    # happening once the work runs in a worker's own context;
    # `driver_tier_usd` is the same turns' own work repriced at the worker
    # tier. They compound (where the work runs vs what it runs on) and sum to
    # `driver_recoverable_usd`.
    driver_offload_usd:       float = 0.0
    driver_tier_usd:          float = 0.0
    driver_tokens:            int   = 0
    driver_tail_tokens:       int   = 0
    # Premium driver model -> the worker tier its work would have routed to.
    # Named on the card: a counterfactual whose substitute is unstated cannot
    # be inspected.
    driver_substitutes:       dict[str, str] = field(default_factory=dict)
    driver_examples:          list[DriverRoleExample] = field(default_factory=list)
    driver_estimate_basis:    str   = ""
    #: ``session_id -> that session's own driver-role tokens``. A BREAKDOWN of
    #: `driver_tokens`, not a second quantity: the values sum to it. It exists
    #: so `core/optimize/rule_placement` can answer "which projects incurred
    #: this, and in what proportion" — the fix for this card is a CLAUDE.md
    #: rule, and a rule written into the projects that exhibited the behaviour
    #: is re-sent in those projects only, rather than in every session of every
    #: project the user has. Weights are TOKENS and are the single attribution
    #: weight for both the token and the dollar split, so every destination's
    #: implied per-token rate equals the finding's own (Critical Rule 28).
    #: Empty when the driver-role case did not fire; placement then falls back
    #: to the user-global file, which is the historical behaviour.
    driver_session_tokens:    dict[str, int] = field(default_factory=dict)


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
    # Segment accounting + confidence (design §5.2 / §6, D3). The number
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
    # EVERY session in the cluster, not just the 3 examples above — the
    # population `cost_proposals._net_cross_analyzer_session_overlap` needs to
    # tell whether this cluster's claim overlaps another analyzer's (`script`
    # clusters the identical repeated-tool-sequence shape and, absent this,
    # claimed the same sessions a second time — see CLAUDE.md Critical Rule 27
    # in `.claude/rules/optimize-cost-figures.md`). Defaulted for round-trip
    # with older serialized reports.
    member_session_ids: tuple[str, ...] = ()


#: Capture modes whose clusters were built, wholly or partly, on tool
#: signature alone. Every surface that shows a reuse finding must warn on ALL
#: of these — branching on `== "tool_sequence_only"` is how the partial case
#: silently rendered as the confident, unqualified path.
DEGRADED_CAPTURE_MODES = frozenset({"tool_sequence_only", "mixed_prompt_prefix"})


@dataclass
class ReuseFinding:
    """Clusters of sessions with structurally repeated planning calls."""
    clusters:      list[ReuseCluster] = field(default_factory=list)
    # What clustering ACTUALLY ran on, derived from the analyzed window — never
    # from the `[capture] prompts` toggle. Echoing the toggle let a report
    # declare "with_prompt_prefix" while every cluster member's
    # `prompt_prefix_hash` was None, i.e. advertise content matching while
    # silently degrading to tool-signature-only clustering.
    # `mixed_prompt_prefix` is the partially-degraded middle: SOME planning
    # calls carried prompt text and the rest were clustered on tool signature
    # alone, so the window's clusters do not share one basis. Collapsing it
    # into `with_prompt_prefix` (the old behavior — any nonzero coverage) made
    # a partly tool-signature-only result advertise full content matching,
    # while the basis string on the same finding said the opposite.
    capture_mode:  Literal[
        "tool_sequence_only", "mixed_prompt_prefix", "with_prompt_prefix"
    ] = "tool_sequence_only"
    # Measured share of the window's planning calls that carried prompt text,
    # 0.0-1.0. `None` means "no planning call to measure", never 0.0 — the
    # difference between an unanswered question and a measured absence.
    prompt_capture_coverage: float | None = None
    # Recoverable-savings contract (#111). The aggregate uses the conservative
    # cache-reuse number (avg cost x (reps - 1)), never the script-replacement
    # upper bound (avg cost x reps): the first planning call in a cluster was
    # necessary work, so a figure that includes it is one a user can disprove.
    # None when no cluster cleared the thresholds.
    past_overspend_usd:    float | None = None
    past_overspend_tokens: int | None   = None
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
    #: Analyzers that RAISED, keyed by name, with the exception. An analyzer
    #: whose failure is swallowed and not recorded reads as one that found
    #: nothing, which is a positive claim the run has no evidence for — so the
    #: dispatch loop isolates failures (one analyzer must not destroy twelve
    #: others' findings) and records them here rather than merely omitting them.
    #: Every surface that renders a report must be able to say "we did not get
    #: an answer" separately from "the answer was none".
    analyzer_errors: dict = field(default_factory=dict)
    # Dominant user persona for the window ("claude-code" | "sdk" | "mixed" |
    # "unknown") — see `tokenjam.core.framing.dominant_persona`. Computed once
    # by `runner.build_report` (mirrors `AnalyzerContext.persona` below) and
    # carried on the report so a consumer working from the built report alone
    # (e.g. `cost_proposals.cost_proposals_from_report`) never has to
    # recompute it from a bare `conn`.
    persona:   str = "unknown"
    #: The analyzer names this pass actually DISPATCHED. Without it a reader
    #: cannot tell "this analyzer ran and found nothing" from "this analyzer was
    #: never invoked", and that distinction becomes load-bearing the moment a
    #: report is served for a persona other than :attr:`persona`: an analyzer
    #: the requested persona has a lever for, which this pass never ran, is
    #: NOT-YET-KNOWN and must render as such rather than as an empty result
    #: (root anti-pattern 22). Empty means "this report predates the field", not
    #: "nothing ran" — a consumer must degrade to "unknown", never to "none".
    computed_analyzers: list = field(default_factory=list)
    #: The personas whose gates were honored when :attr:`computed_analyzers` was
    #: selected. A pass run for several personas (the daemon's, which computes
    #: the union so one stored artifact can answer for either side of the
    #: "Viewing as" picker) lists them all; a single-persona pass lists just its
    #: own. This is what tells a route whether it may answer a request for a
    #: persona other than :attr:`persona` at all.
    computed_for_personas: list = field(default_factory=list)
    #: The persona this report's ROWS were restricted to, or ``None`` for the
    #: whole corpus. :attr:`persona` says what the window is; this says what was
    #: looked at. A consumer publishing a figure under a persona label must
    #: check that this MATCHES that persona — a report computed over everything
    #: cannot answer "how much is my SDK traffic costing me", and rendering it
    #: as though it could is the whole-corpus-number-under-a-persona-label bug.
    #: ``None`` on a report deserialized from an artifact written before this
    #: field existed, which is indistinguishable from "unscoped" and is treated
    #: as exactly that.
    persona_scope: str | None = None
    #: One fully-scoped report per persona the pass was asked to answer for,
    #: keyed by persona and shaped like this one (``report_to_dict`` of a nested
    #: :class:`OptimizeReport`). The dispatch gate could be widened to a union
    #: and sliced on read because it only decides which analyzers run; a
    #: POPULATION cannot be unioned — one number cannot cover two populations —
    #: so the pass runs once per persona and stores each result whole. Empty
    #: means the artifact predates per-persona scoping: a reader asking for a
    #: specific persona then has NOT-YET-KNOWN, never this report's figures.
    persona_reports: dict = field(default_factory=dict)
    # Why the filesystem-reading analyzers (deadweight, relearn, summarize)
    # scanned nothing, when they scanned nothing — see
    # `core/optimize/scope.py`. `None` means they DID scan, which is a
    # different statement from "scanned and found nothing" and must render
    # differently (root anti-pattern 22): an empty deadweight finding under a
    # suppressed scope reads as "no dead MCP servers" when the truth is that
    # no config was ever looked at. One field on the report rather than one
    # per finding, so every surface reads the same answer.
    filesystem_scan_skipped_reason: str | None = None


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
    # Dominant user persona for the window — see `OptimizeReport.persona`
    # above. Computed once in `runner.build_report` via
    # `tokenjam.core.framing.agent_persona_mix` / `dominant_persona` (the
    # same functions the CLI's own persona-dependent CTA already uses), not
    # a second classifier. Analyzers that need to know whether they're
    # looking at an SDK caller (e.g. deciding fix modality) read it here
    # instead of re-deriving it from `conn`.
    persona:                str            = "unknown"
    # The persona whose POPULATION this pass is scoped to, or `None` for the
    # whole corpus. NOT the same field as `persona` above, which records what
    # the window IS. This one records what the pass was asked to LOOK AT, and
    # it is what makes a dollar figure published under a persona label actually
    # be that persona's money: the dispatch gate only decides which analyzers
    # run, so without this every survivor still aggregates the whole mixed
    # corpus. Analyzers apply it through
    # `core/optimize/persona_scope.add_persona_clause` — never a hand-written
    # prefix test, which would be a second bucketing rule free to drift from
    # `alerts.is_interactive_coding_agent`.
    persona_scope:          str | None     = None

    @property
    def effective_persona(self) -> str:
        """THE persona a gate in this pass must key off.

        `persona_scope` when the pass is scoped to one, else the window's own
        dominant `persona`. Any gate that reads `persona` directly is wrong for
        a scoped pass over a MIXED corpus: the window resolves to `mixed`, which
        disables nothing, so a `claude-code`-scoped pass still attaches findings
        for levers a Claude Code user does not have — measured over Claude Code
        rows and then labelled with their persona, which is worse than the
        unscoped version of the same card.
        """
        return self.persona_scope or self.persona
    # Which filesystem the filesystem-reading analyzers (deadweight, relearn,
    # summarize) may read, and why. Resolved once in `runner.build_report` via
    # `core.optimize.scope.resolve_analyzer_scope` — an analyzer must never
    # re-derive a root from `Path.home()` or the env var itself, or `--db`
    # stops isolating it. `None` only in a hand-built context (tests): treat it
    # as the unscoped default. See `core/optimize/scope.py` for the contract.
    scope:                  Any            = None
    # The owning `DuckDBBackend`'s re-entrant write lock, or `None`. An
    # analyzer that writes through `conn` directly must take it — see
    # `.claude/rules/core-architecture.md` and `core/agent_config.store_for`.
    write_lock:             Any            = None


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
        percent_quota_misallocated=float(
            data.get("percent_quota_misallocated", 0.0) or 0.0
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
