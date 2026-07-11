"""
Optimize orchestrator. Builds OptimizeReport by running selected analyzers
from the registry in a deterministic order.

Analyzer ordering matters: budget-projection reads ctx.report.downgrade,
so downsize must run first when both are selected.
"""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from typing import Any

from tokenjam.core.config import TjConfig
from tokenjam.utils.time_parse import utcnow
from tokenjam.core.optimize.registry import ANALYZER_REGISTRY
from tokenjam.core.optimize.types import (
    AnalyzerContext,
    OptimizeReport,
    WindowSummary,
)

# Ensure analyzers are imported (triggers @register side effects).
# Auto-discovery in analyzers/__init__.py walks the directory.
from tokenjam.core.optimize import analyzers as _analyzers  # noqa: F401

# Deterministic order. Adding a new analyzer? Append to this list. Analyzers
# requested via positional are filtered against ANALYZER_REGISTRY but executed
# in the order defined here, so cross-analyzer dependencies stay stable.
ANALYZER_ORDER: list[str] = [
    "downsize",
    "budget-projection",
    "cache",
    "cache-recommend",
    "script",
    "reuse",
    "trim",
    "subagent",
    "summarize",
]

THIN_DATA_DAYS = 7


def _utcnow() -> datetime:
    # Canonical timezone-aware UTC (CLAUDE.md Rule 9).
    return utcnow()


def summarize_window(
    conn,
    since: datetime,
    until: datetime,
    agent_id: str | None = None,
) -> WindowSummary:
    clauses = ["start_time >= $1", "start_time < $2", "model IS NOT NULL"]
    params: list[Any] = [since, until]
    if agent_id:
        clauses.append(f"agent_id = ${len(params) + 1}")
        params.append(agent_id)
    where = " AND ".join(clauses)
    row = conn.execute(
        f"SELECT COUNT(*) AS spans, "
        f"COUNT(DISTINCT session_id) AS sessions, "
        f"COALESCE(SUM(COALESCE(input_tokens,0) + COALESCE(output_tokens,0) + COALESCE(cache_tokens,0) + COALESCE(cache_write_tokens,0)), 0) AS tokens, "
        f"COALESCE(SUM(cost_usd), 0.0) AS cost "
        f"FROM spans WHERE {where}",
        params,
    ).fetchone()
    spans = int(row[0] or 0)
    sessions = int(row[1] or 0)
    tokens = int(row[2] or 0)
    cost = float(row[3] or 0.0)
    days = max((until - since).total_seconds() / 86400.0, 0.0)
    return WindowSummary(
        since=since,
        until=until,
        days=days,
        sessions=sessions,
        spans=spans,
        total_tokens=tokens,
        total_cost_usd=cost,
        thin_data=days < THIN_DATA_DAYS or sessions < 3,
    )


def build_report(
    db,
    config: TjConfig,
    since: datetime,
    until: datetime | None = None,
    agent_id: str | None = None,
    findings: list[str] | None = None,
    budget_provider_filter: str | None = None,
    budget_usd_override: float | None = None,
) -> OptimizeReport:
    """
    Build a complete OptimizeReport.

    `findings`:
      - None  -> run all registered analyzers in ANALYZER_ORDER
      - list  -> run only the named analyzers (must be keys in ANALYZER_REGISTRY)

    Analyzers are executed in ANALYZER_ORDER, never in caller-supplied order,
    so dependent analyzers (e.g. budget-projection reading the downgrade
    finding) work correctly regardless of how the caller lists them.
    """
    until = until or _utcnow()
    if until <= since:
        raise ValueError("until must be after since")

    conn = getattr(db, "conn", None)
    if conn is None:
        raise RuntimeError("optimize requires a direct DuckDB connection")

    summary = summarize_window(conn, since, until, agent_id=agent_id)
    window_days = max(summary.days, 1.0 / 86400.0)

    report = OptimizeReport(window=summary)
    if summary.thin_data:
        report.notes.append(
            "Window contains less than ~1 week of activity — projections shown "
            "below should be treated as preliminary."
        )

    ctx = AnalyzerContext(
        conn=conn,
        config=config,
        since=since,
        until=until,
        agent_id=agent_id,
        window_days=window_days,
        summary=summary,
        report=report,
        budget_provider_filter=budget_provider_filter,
        budget_usd_override=budget_usd_override,
    )

    selected = set(findings) if findings is not None else set(ANALYZER_REGISTRY.keys())
    # Validate against registry; raise on unknown names so typos surface early.
    unknown = selected - set(ANALYZER_REGISTRY.keys())
    if unknown:
        raise ValueError(
            f"Unknown finding(s): {sorted(unknown)}. "
            f"Available: {sorted(ANALYZER_REGISTRY.keys())}"
        )

    for name in ANALYZER_ORDER:
        if name in selected and name in ANALYZER_REGISTRY:
            ANALYZER_REGISTRY[name](ctx)

    # Analyzers not in ANALYZER_ORDER (future ones, registered but not yet
    # explicitly ordered) run last in arbitrary order. Maintainers should add
    # new analyzers to ANALYZER_ORDER when they land.
    for name, analyzer in ANALYZER_REGISTRY.items():
        if name in selected and name not in ANALYZER_ORDER:
            analyzer(ctx)

    return report


def report_to_dict(report: OptimizeReport) -> dict:
    """Convert OptimizeReport to a JSON-serialisable dict."""
    def _serialise(o: Any) -> Any:
        if isinstance(o, datetime):
            return o.isoformat()
        if hasattr(o, "__dataclass_fields__"):
            return {k: _serialise(v) for k, v in asdict(o).items()}
        if isinstance(o, list):
            return [_serialise(x) for x in o]
        if isinstance(o, dict):
            return {k: _serialise(v) for k, v in o.items()}
        return o
    return _serialise(report)


def _parse_dt(value: Any) -> datetime | None:
    """Parse an ISO-8601 string back into datetime; tolerate None / already-dt."""
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        s = value.replace("Z", "+00:00") if value.endswith("Z") else value
        try:
            return datetime.fromisoformat(s)
        except ValueError:
            return None
    return None


def report_from_dict(d: dict) -> OptimizeReport:
    """
    Reconstruct an OptimizeReport from the dict produced by `report_to_dict`.

    Symmetric with `report_to_dict`. Used by the CLI when fetching an
    optimize report from a running `tj serve` via /api/v1/optimize — the
    daemon serialises the report and the CLI deserialises here so the same
    rendering path works for both local and HTTP-fetched reports
    (issue #68 §12).

    Wave-2 analyzer findings live under `d["findings"]` keyed by analyzer
    name; each registered analyzer module's `from_dict` helper is invoked
    via the registry. Unknown finding names are dropped silently — this
    keeps the CLI forward-compatible if the daemon advertises a finding
    the local install doesn't know how to render yet.
    """
    from tokenjam.core.optimize.types import (
        BudgetProjection,
        DowngradeExample,
        DowngradeFinding,
        WindowSummary,
    )

    w = d.get("window") or {}
    window = WindowSummary(
        since=_parse_dt(w.get("since")) or _utcnow(),
        until=_parse_dt(w.get("until")) or _utcnow(),
        days=float(w.get("days", 0.0)),
        sessions=int(w.get("sessions", 0)),
        spans=int(w.get("spans", 0)),
        total_tokens=int(w.get("total_tokens", 0)),
        total_cost_usd=float(w.get("total_cost_usd", 0.0)),
        thin_data=bool(w.get("thin_data", False)),
    )

    downgrade = None
    if d.get("downgrade"):
        dd = dict(d["downgrade"])
        examples = [
            DowngradeExample(
                trace_id=str(ex.get("trace_id", "")),
                session_id=ex.get("session_id"),
                model=str(ex.get("model", "")),
                tool_calls=int(ex.get("tool_calls", 0)),
                duration_seconds=ex.get("duration_seconds"),
                cost_usd=float(ex.get("cost_usd", 0.0)),
            )
            for ex in (dd.get("examples") or [])
        ]
        downgrade = DowngradeFinding(
            candidate_sessions=int(dd.get("candidate_sessions", 0)),
            total_sessions=int(dd.get("total_sessions", 0)),
            actual_cost_usd=float(dd.get("actual_cost_usd", 0.0)),
            alternative_cost_usd=float(dd.get("alternative_cost_usd", 0.0)),
            monthly_savings_usd=float(dd.get("monthly_savings_usd", 0.0)),
            percent_of_sessions=float(dd.get("percent_of_sessions", 0.0)),
            examples=examples,
            suggestions=dict(dd.get("suggestions") or {}),
            caveat=str(dd.get("caveat", "")),
            bench_command=dd.get("bench_command"),
            candidate_tokens=int(dd.get("candidate_tokens", 0)),
            window_total_tokens=int(dd.get("window_total_tokens", 0)),
            percent_of_tokens=float(dd.get("percent_of_tokens", 0.0)),
            monthly_tokens_in_candidates=int(dd.get("monthly_tokens_in_candidates", 0)),
            estimated_recoverable_usd=dd.get("estimated_recoverable_usd"),
            estimated_recoverable_tokens=dd.get("estimated_recoverable_tokens"),
            estimate_basis=str(dd.get("estimate_basis", "")),
            estimate_confidence=str(dd.get("estimate_confidence", "heuristic")),
            # Sampling confidence (#308) — round-trip n + the CI bounds.
            n_sessions=int(dd.get("n_sessions", 0)),
            ci_low=dd.get("ci_low"),
            ci_high=dd.get("ci_high"),
        )

    budgets = []
    for b in d.get("budgets") or []:
        bb = dict(b)
        budgets.append(BudgetProjection(
            provider=str(bb.get("provider", "")),
            budget_usd=float(bb.get("budget_usd", 0.0)),
            cycle_start_day=int(bb.get("cycle_start_day", 1)),
            cycle_start=_parse_dt(bb.get("cycle_start")) or _utcnow(),
            cycle_end=_parse_dt(bb.get("cycle_end")) or _utcnow(),
            days_into_cycle=float(bb.get("days_into_cycle", 0.0)),
            days_remaining=float(bb.get("days_remaining", 0.0)),
            window_spend_usd=float(bb.get("window_spend_usd", 0.0)),
            daily_run_rate_usd=float(bb.get("daily_run_rate_usd", 0.0)),
            monthly_run_rate_usd=float(bb.get("monthly_run_rate_usd", 0.0)),
            projected_cycle_total=float(bb.get("projected_cycle_total", 0.0)),
            projected_overage_usd=float(bb.get("projected_overage_usd", 0.0)),
            exhaustion_date=_parse_dt(bb.get("exhaustion_date")),
            days_until_exhaustion=bb.get("days_until_exhaustion"),
            over_budget=bool(bb.get("over_budget", False)),
            applies_to_services=list(bb.get("applies_to_services") or []),
            downgrade_run_rate_usd=bb.get("downgrade_run_rate_usd"),
        ))

    findings = {}
    for name, payload in (d.get("findings") or {}).items():
        constructor = _finding_constructor_for(name)
        if constructor is None:
            # Forward-compatible: ignore unknown findings rather than crash.
            continue
        try:
            findings[name] = constructor(payload)
        except Exception:
            # Don't let one malformed finding break the whole report.
            continue

    return OptimizeReport(
        window=window,
        downgrade=downgrade,
        budgets=budgets,
        notes=list(d.get("notes") or []),
        findings=findings,
    )


# Dispatch table: finding-registration-name -> (dict) -> dataclass constructor.
# Filled lazily so importing runner.py doesn't require every analyzer module
# to be imported (analyzers self-register via auto-discovery; the order
# matters during package init).
def _build_finding_constructors() -> dict:
    from tokenjam.core.optimize.analyzers.cache_efficacy import (
        CacheEfficacyFinding,
        CacheEfficacyRow,
    )
    from tokenjam.core.optimize.analyzers.cache_recommend import (
        CachePrefixCandidate,
        CacheRecommendFinding,
    )
    from tokenjam.core.optimize.analyzers.prompt_bloat import (
        BloatPrompt,
        BloatRegion,
        PromptBloatFinding,
    )
    from tokenjam.core.optimize.analyzers.workflow_restructure import (
        WorkflowCluster,
        WorkflowRestructureFinding,
    )
    from tokenjam.core.optimize.analyzers.subagent_rightsizing import (
        SubagentRightsizingFinding,
        SubagentRow,
    )
    from tokenjam.core.optimize.analyzers.summarize import (
        SUMMARIZE_HONESTY_CAVEAT,
        SummarizeCandidate,
        SummarizeFinding,
    )
    from tokenjam.core.optimize.types import ReuseCluster, ReuseFinding

    def _cache_efficacy(d: dict) -> CacheEfficacyFinding:
        rows = [CacheEfficacyRow(**r) for r in d.get("rows") or []]
        flagged = [CacheEfficacyRow(**r) for r in d.get("flagged") or []]
        return CacheEfficacyFinding(
            rows=rows, flagged=flagged,
            confidence=d.get("confidence", "structural"),
            efficacy_ceiling=d.get("efficacy_ceiling", 0.80),
            estimated_recoverable_usd=d.get("estimated_recoverable_usd"),
            estimated_recoverable_tokens=d.get("estimated_recoverable_tokens"),
            estimate_basis=d.get("estimate_basis", ""),
            estimate_confidence=d.get("estimate_confidence", "heuristic"),
        )

    def _cache_recommend(d: dict) -> CacheRecommendFinding:
        candidates = [
            CachePrefixCandidate(**c) for c in d.get("candidates") or []
        ]
        return CacheRecommendFinding(
            enabled=bool(d.get("enabled", False)),
            candidates=candidates,
            skipped_provider_count=int(d.get("skipped_provider_count", 0)),
            confidence=d.get("confidence", "structural"),
            hint=d.get("hint"),
        )

    def _workflow_restructure(d: dict) -> WorkflowRestructureFinding:
        clusters = [WorkflowCluster(**c) for c in d.get("clusters") or []]
        return WorkflowRestructureFinding(
            clusters=clusters,
            sessions_examined=int(d.get("sessions_examined", 0)),
            degraded=bool(d.get("degraded", False)),
            confidence=d.get("confidence", "structural"),
            caveat=d.get("caveat", ""),
            estimated_recoverable_usd=d.get("estimated_recoverable_usd"),
            estimated_recoverable_tokens=d.get("estimated_recoverable_tokens"),
            estimate_basis=d.get("estimate_basis", ""),
            estimate_confidence=d.get("estimate_confidence", "heuristic"),
        )

    def _prompt_bloat(d: dict) -> PromptBloatFinding:
        per_prompt = []
        for p in d.get("per_prompt") or []:
            regions = [BloatRegion(**r) for r in p.get("regions") or []]
            pp = dict(p)
            pp["regions"] = regions
            per_prompt.append(BloatPrompt(**pp))
        return PromptBloatFinding(
            enabled=bool(d.get("enabled", False)),
            prompts_scored=int(d.get("prompts_scored", 0)),
            prompts_skipped=int(d.get("prompts_skipped", 0)),
            total_bloat_chars=int(d.get("total_bloat_chars", 0)),
            total_chars=int(d.get("total_chars", 0)),
            per_prompt=per_prompt,
            confidence=d.get("confidence", "structural"),
            hint=d.get("hint"),
            estimated_recoverable_usd=d.get("estimated_recoverable_usd"),
            estimated_recoverable_tokens=d.get("estimated_recoverable_tokens"),
            estimate_basis=d.get("estimate_basis", ""),
            estimate_confidence=d.get("estimate_confidence", "heuristic"),
        )

    def _reuse(d: dict) -> ReuseFinding:
        clusters = []
        for c in d.get("clusters") or []:
            cc = dict(c)
            # asdict() serialised the tuple to a list; restore the tuple so the
            # dataclass field type holds across the round-trip.
            cc["tool_signature"] = tuple(cc.get("tool_signature") or ())
            clusters.append(ReuseCluster(**cc))
        return ReuseFinding(
            clusters=clusters,
            capture_mode=d.get("capture_mode", "tool_sequence_only"),
            estimated_recoverable_usd=d.get("estimated_recoverable_usd"),
            estimated_recoverable_tokens=d.get("estimated_recoverable_tokens"),
            estimate_basis=d.get("estimate_basis", ""),
            confidence=d.get("confidence", "heuristic"),
            hint=d.get("hint", ""),
        )

    def _subagent(d: dict) -> SubagentRightsizingFinding:
        rows = [SubagentRow(**r) for r in d.get("rows") or []]
        flagged = [SubagentRow(**r) for r in d.get("flagged") or []]
        return SubagentRightsizingFinding(
            sessions_with_subagents=int(d.get("sessions_with_subagents", 0)),
            total_subagents=int(d.get("total_subagents", 0)),
            subagent_cost_usd=float(d.get("subagent_cost_usd", 0.0)),
            subagent_tokens=int(d.get("subagent_tokens", 0)),
            window_cost_usd=float(d.get("window_cost_usd", 0.0)),
            percent_of_cost=float(d.get("percent_of_cost", 0.0)),
            flagged_cost_usd=float(d.get("flagged_cost_usd", 0.0)),
            rows=rows,
            flagged=flagged,
            confidence=d.get("confidence", "structural"),
            caveat=d.get("caveat", ""),
            estimated_recoverable_usd=d.get("estimated_recoverable_usd"),
            estimated_recoverable_tokens=d.get("estimated_recoverable_tokens"),
            estimate_basis=d.get("estimate_basis", ""),
            estimate_confidence=d.get("estimate_confidence", "heuristic"),
        )

    def _summarize(d: dict) -> SummarizeFinding:
        cands = [SummarizeCandidate(**c) for c in d.get("candidates") or []]
        return SummarizeFinding(
            candidates=cands,
            files=int(d.get("files", 0)),
            estimated_recoverable_usd=d.get("estimated_recoverable_usd"),
            estimated_recoverable_tokens=d.get("estimated_recoverable_tokens"),
            estimate_basis=d.get("estimate_basis", ""),
            estimate_confidence=d.get("estimate_confidence", "heuristic"),
            caveat=d.get("caveat", SUMMARIZE_HONESTY_CAVEAT),
            reduction_pct=d.get("reduction_pct"),
            avg_reduction_pct=d.get("avg_reduction_pct"),
        )

    return {
        "cache": _cache_efficacy,
        "cache-recommend": _cache_recommend,
        "script": _workflow_restructure,
        "reuse": _reuse,
        "trim": _prompt_bloat,
        "subagent": _subagent,
        "summarize": _summarize,
    }


_FINDING_CONSTRUCTORS: dict = {}


def _ensure_constructors_loaded() -> dict:
    """Lazy-load the finding constructors on first use."""
    global _FINDING_CONSTRUCTORS
    if not _FINDING_CONSTRUCTORS:
        _FINDING_CONSTRUCTORS = _build_finding_constructors()
    return _FINDING_CONSTRUCTORS


# Wire the lazy loader into report_from_dict above.
def _finding_constructor_for(name: str):
    return _ensure_constructors_loaded().get(name)
