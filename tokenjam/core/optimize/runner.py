"""
Optimize orchestrator. Builds OptimizeReport by running selected analyzers
from the registry in a deterministic order.

Analyzer ordering matters: budget-projection reads ctx.report.downgrade,
so model-downgrade must run first when both are selected.
"""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any

from tokenjam.core.config import TjConfig
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
# requested via --finding are filtered against ANALYZER_REGISTRY but executed
# in the order defined here, so cross-analyzer dependencies stay stable.
ANALYZER_ORDER: list[str] = [
    "model-downgrade",
    "budget-projection",
    "cache-efficacy",
    "cache-recommend",
    "workflow-restructure",
    "prompt-bloat",
]

THIN_DATA_DAYS = 7


def _utcnow() -> datetime:
    return datetime.now(tz=timezone.utc)


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
        f"COALESCE(SUM(COALESCE(input_tokens,0) + COALESCE(output_tokens,0)), 0) AS tokens, "
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
