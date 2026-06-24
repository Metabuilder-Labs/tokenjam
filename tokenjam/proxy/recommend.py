"""Policy status + suggestion helpers for the MCP surface (#223).

Pure (HTTP-free) functions that assemble the read-only views the MCP tools and
the `/api/v1/policy/*` routes expose:

- :func:`policy_status` — the defined `[[policies]]` + recent persisted decisions.
- :func:`suggest_policies` — recommend policies from the user's ACTUAL usage
  (a ``budget_cap`` ceiling per api-billed provider with cycle spend but no
  ceiling yet).

Honesty (Critical Rule 14): everything here is suggest-mode and `unvalidated`.
Suggestions are framed as starting points to review, NEVER as validated-safe;
the savings meter (in `audit.reconcile_savings`) is estimated-recoverable, never
"saved". Dollar figures are api-only.
"""
from __future__ import annotations

import math
from typing import Any

from tokenjam.core.framing import provider_pricing_mode
from tokenjam.core.models import PolicyDecisionFilters
from tokenjam.proxy.audit import UNVALIDATED_LABEL, decision_to_display_dict

SUGGEST_MODE_NOTE = (
    "Suggest mode — policies record what they WOULD do; nothing is enforced. "
    "OSS policies run unvalidated (there is no certification engine in the open "
    "tree), so a suggestion is never implied to be validated as safe."
)


def _defined_policies(config: Any) -> list[dict]:
    out: list[dict] = []
    for p in getattr(config, "policies", None) or []:
        out.append({
            "name": getattr(p, "name", ""),
            "kind": getattr(p, "kind", ""),
            "mode": getattr(p, "mode", "suggest"),
            "enabled": getattr(p, "enabled", True),
            "target_provider": getattr(p, "target_provider", None),
            "target_agent": getattr(p, "target_agent", None),
            "label": UNVALIDATED_LABEL,
        })
    return out


def policy_status(db: Any, config: Any, *, limit: int = 20) -> dict:
    """Defined policies + recent persisted decisions — suggest-mode, unvalidated."""
    decisions = []
    try:
        recs = db.get_policy_decisions(PolicyDecisionFilters(limit=limit))
        decisions = [decision_to_display_dict(r) for r in recs]
    except Exception:  # noqa: BLE001 — read-only view; never raise into the agent
        decisions = []
    return {
        "suggest_mode": True,
        "enforced": False,
        "label": UNVALIDATED_LABEL,
        "policies": _defined_policies(config),
        "recent_decisions": decisions,
        "note": SUGGEST_MODE_NOTE,
    }


def _round_up_ceiling(spend: float) -> float:
    """A round-number starting-point ceiling comfortably above observed spend."""
    target = max(spend * 1.5, spend + 10.0)
    if target <= 0:
        return 10.0
    # Round up to a "nice" magnitude (next 10 / 50 / 100 depending on size).
    step = 10.0 if target < 100 else (50.0 if target < 500 else 100.0)
    return math.ceil(target / step) * step


def _provider_cycle_spend(db: Any, config: Any) -> dict[str, float]:
    """Per-provider USD spend over each provider's current billing cycle.

    Mirrors budget_cap's window (cycle bounds + cost_usd), so a suggested ceiling
    lines up with what budget_cap would actually meter.
    """
    conn = getattr(db, "conn", None)
    if conn is None:
        return {}
    from tokenjam.core.cycle import cycle_bounds
    from tokenjam.utils.time_parse import utcnow

    out: dict[str, float] = {}
    try:
        providers = [r[0] for r in conn.execute(
            "SELECT DISTINCT provider FROM spans WHERE provider IS NOT NULL"
        ).fetchall()]
    except Exception:  # noqa: BLE001
        return {}
    now = utcnow()
    budgets = getattr(config, "budgets", None) or {}
    for provider in providers:
        budget = budgets.get(provider)
        start_day = getattr(budget, "cycle_start_day", 1) if budget is not None else 1
        cs, ce = cycle_bounds(now, start_day)
        try:
            row = conn.execute(
                "SELECT COALESCE(SUM(cost_usd), 0.0) FROM spans "
                "WHERE start_time >= $1 AND start_time < $2 AND provider = $3",
                [cs, ce, provider],
            ).fetchone()
        except Exception:  # noqa: BLE001
            continue
        spend = float(row[0] or 0.0) if row else 0.0
        if spend > 0:
            out[provider] = spend
    return out


def suggest_policies(db: Any, config: Any) -> dict:
    """Suggest budget_cap ceilings from observed api-billed cycle spend.

    A suggestion is offered for each api-billed provider that has spend this
    cycle but no `[budget.<provider>] usd` ceiling configured yet. Dollars are
    api-only (subscription/local providers are skipped — no dollar suggestion).
    """
    budgets = getattr(config, "budgets", None) or {}
    spend_by_provider = _provider_cycle_spend(db, config)
    suggestions: list[dict] = []

    for provider, spend in sorted(spend_by_provider.items()):
        _, pricing_mode = provider_pricing_mode(config, provider)
        if pricing_mode != "api":
            continue  # dollar ceilings only make sense for usage-billed traffic
        existing = budgets.get(provider)
        if existing is not None and getattr(existing, "usd", None):
            continue  # already has a ceiling — nothing to suggest
        ceiling = _round_up_ceiling(spend)
        name = f"{provider}-budget-cap"
        suggestions.append({
            "kind": "budget_cap",
            "provider": provider,
            "observed_cycle_spend_usd": round(spend, 4),
            "suggested_ceiling_usd": ceiling,
            "rationale": (
                f"{provider} has ${spend:.2f} of api spend this cycle and no "
                f"[budget.{provider}] usd ceiling. A budget_cap policy would, in "
                f"suggest mode, flag when cycle spend approaches/exceeds the "
                f"ceiling (it enforces nothing)."
            ),
            "toml": (
                f"[[policies]]\n"
                f'name = "{name}"\n'
                f'kind = "budget_cap"\n'
                f'mode = "suggest"\n'
                f'target_provider = "{provider}"\n\n'
                f"[budget.{provider}]\n"
                f"usd = {ceiling:g}\n"
            ),
            "label": UNVALIDATED_LABEL,
        })

    return {
        "suggest_mode": True,
        "label": UNVALIDATED_LABEL,
        "suggestions": suggestions,
        "note": (
            "These are SUGGESTIONS derived from your usage, in suggest mode. The "
            "suggested ceiling is a starting point to review against your budget "
            "— it is not validated as safe, and adding a policy enforces nothing "
            "(suggest mode only)."
        ),
    }
