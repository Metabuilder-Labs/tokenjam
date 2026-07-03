"""
tj optimize analyzers.

Public API:

  - build_report(...) -> OptimizeReport: orchestrator
  - report_to_dict(report) -> dict: JSON-serialisable
  - WindowSummary, DowngradeExample, DowngradeFinding, BudgetProjection,
    OptimizeReport: result dataclasses
  - MODEL_DOWNGRADE_CAVEAT: mandatory caveat string
  - ANALYZER_REGISTRY: dict of name -> analyzer callable (populated by
    auto-discovery in analyzers/__init__.py)

Adding a new analyzer:

  1. Drop a .py file under analyzers/ with a function decorated with
     @register("name") taking (ctx: AnalyzerContext) -> None
  2. Add the name to ANALYZER_ORDER in runner.py if it depends on or is
     depended upon by another analyzer
  3. Nothing else needs editing — cmd_optimize's positional choices read
     from ANALYZER_REGISTRY directly

See README.md in this package for details.
"""
from __future__ import annotations

# Re-export the public surface used by cmd_optimize.py, mcp/server.py, tests.
from tokenjam.core.optimize.registry import ANALYZER_REGISTRY, register
from tokenjam.core.optimize.runner import (
    ANALYZER_ORDER,
    build_report,
    report_from_dict,
    report_to_dict,
    summarize_window,
)
from tokenjam.core.optimize.types import (
    MODEL_DOWNGRADE_CAVEAT,
    OPUS_QUOTA_AUDIT_CAVEAT,
    REUSE_HONESTY_CAVEAT,
    AnalyzerContext,
    BudgetProjection,
    DowngradeExample,
    DowngradeFinding,
    OpusAuditExample,
    OpusQuotaAudit,
    OptimizeReport,
    ReuseCluster,
    ReuseFinding,
    WindowSummary,
)

# Re-export the existing analyzer functions so legacy importers (tests,
# mcp/server.py) continue to find them at the package level.
from tokenjam.core.optimize.analyzers.budget_projection import (
    _cycle_bounds,
    project_budget,
)
from tokenjam.core.optimize.analyzers.model_downgrade import (
    DOWNGRADE_CANDIDATES,
    analyze_model_downgrade,
    audit_opus_quota,
)

__all__ = [
    "ANALYZER_ORDER",
    "ANALYZER_REGISTRY",
    "AnalyzerContext",
    "BudgetProjection",
    "DOWNGRADE_CANDIDATES",
    "DowngradeExample",
    "DowngradeFinding",
    "MODEL_DOWNGRADE_CAVEAT",
    "OPUS_QUOTA_AUDIT_CAVEAT",
    "OpusAuditExample",
    "OpusQuotaAudit",
    "REUSE_HONESTY_CAVEAT",
    "OptimizeReport",
    "ReuseCluster",
    "ReuseFinding",
    "WindowSummary",
    "_cycle_bounds",
    "analyze_model_downgrade",
    "audit_opus_quota",
    "build_report",
    "project_budget",
    "register",
    "report_from_dict",
    "report_to_dict",
    "summarize_window",
]
