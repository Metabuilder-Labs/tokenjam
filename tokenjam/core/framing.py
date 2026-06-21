"""Shared plan-tier-aware framing for dollar / token figures.

Single source of truth for the rules that decide whether a dollar figure is
shown verbatim, suppressed in favour of token-share framing, or annotated with
a qualifier. Consumed by the CLI commands (`tj cost` / `tj optimize` /
`tj tokenmaxx`) and emitted into REST API responses as a ``framing`` block so
the local web UI renders identical numbers without re-deriving the rules in
JavaScript.

This module *reads* plan-tier / pricing-mode; it does not define them. The
canonical derivation lives on ``SessionRecord.pricing_mode`` and the
``SUBSCRIPTION_PLAN_TIERS`` frozenset in :mod:`tokenjam.otel.semconv`.

Honesty discipline (see CLAUDE.md Critical Rule 14): subscription users never
see a raw dollar figure that includes subscription traffic without
qualification; local users see tokens only; unknown-plan users see dollars with
an "may overstate" qualifier.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from tokenjam.otel.semconv import SUBSCRIPTION_PLAN_TIERS
from tokenjam.utils.formatting import format_tokens

# Subscription plan label + flat monthly fee. Keys must match
# SessionRecord.plan_tier values. Plans whose fee is contract-priced
# (team / enterprise) have no fee here — callers skip the multiplier in that
# case. This is the single copy; cmd_optimize / cmd_tokenmaxx import it.
PLAN_LABEL_AND_FEE: dict[str, tuple[str, float | None]] = {
    "pro":        ("Pro plan",          20.0),
    "max_5x":     ("Max 5x plan",      100.0),
    "max_20x":    ("Max 20x plan",     200.0),
    "plus":       ("ChatGPT Plus",      20.0),
    "team":       ("ChatGPT Team",       None),
    "enterprise": ("ChatGPT Enterprise", None),
}

# Human-readable labels for api / local tiers (subscription labels live above).
PLAN_DISPLAY_LABEL: dict[str, str] = {
    "api":   "API billing",
    "local": "Local inference",
}

# display_rule values — the UI / CLI reads this to pick a rendering path.
DISPLAY_SHOW_DOLLARS = "show_dollars"
DISPLAY_SHOW_DOLLARS_WITH_QUALIFIER = "show_dollars_with_qualifier"
DISPLAY_SUPPRESS_SUBSCRIPTION = "suppress_dollars_for_subscription_share"
DISPLAY_TOKENS_ONLY = "tokens_only"
DISPLAY_SUPPRESS_UNKNOWN = "suppress_dollars_unknown"

# Shown when sessions lack plan_tier and config has no declared plan.
RECONFIGURE_HINT = (
    "Run `tj onboard --claude-code --reconfigure` (or `--codex`)."
)


def pricing_mode_for(plan_tier: str) -> str:
    """Mirror ``SessionRecord.pricing_mode`` without needing an instance."""
    if plan_tier == "local":
        return "local"
    if plan_tier in SUBSCRIPTION_PLAN_TIERS:
        return "subscription"
    if plan_tier == "api":
        return "api"
    return "unknown"


def dominant_plan(plan_mix: dict[str, int]) -> str:
    """Pick the rendering plan tier from a plan-tier session-count mix.

    - Empty mix (e.g. spans inserted without sessions in test fixtures) →
      ``"api"``, the historical rendering mode. Real users always have a
      populated sessions table.
    - Any non-unknown plan_tier present → the most common one.
    - Otherwise → ``"unknown"`` (the caller suppresses dollars).
    """
    if not plan_mix:
        return "api"
    known = {k: v for k, v in plan_mix.items() if k != "unknown"}
    if not known:
        return "unknown"
    return max(known.items(), key=lambda kv: kv[1])[0]


def plan_label_for(plan_tier: str, provider: str | None = None) -> str | None:
    """Human-readable plan label, optionally scoped to a billing provider."""
    label, _ = PLAN_LABEL_AND_FEE.get(plan_tier, (None, None))
    if label is None:
        label = PLAN_DISPLAY_LABEL.get(plan_tier)
    if not label:
        return None
    if provider:
        return f"{label} ({provider})"
    return label


def _declared_budget_plans(config: Any) -> list[tuple[str, str]]:
    """Return ``(provider, plan_tier)`` pairs from active config, then global.

    Mirrors the global-config fallback in :func:`config_declared_plan` so
    session stamping stays aligned with UI framing when a project-local
    ``.tj/config.toml`` omits ``[budget]`` sections (issue #106).
    """
    budgets = getattr(config, "budgets", None) or {}
    entries: list[tuple[str, str]] = []
    for provider in sorted(budgets.keys()):
        plan = getattr(budgets[provider], "plan", None)
        if plan:
            entries.append((str(provider), str(plan)))
    if entries:
        return entries
    try:
        import sys
        from pathlib import Path
        if sys.version_info >= (3, 11):
            import tomllib
        else:
            import tomli as tomllib  # type: ignore[no-redef]
        global_path = Path.home() / ".config" / "tj" / "config.toml"
        if not global_path.exists():
            return []
        with open(global_path, "rb") as f:
            raw = tomllib.load(f)
        budget_block = raw.get("budget") or {}
        for provider in sorted(budget_block.keys()):
            plan = (budget_block[provider] or {}).get("plan")
            if plan:
                entries.append((str(provider), str(plan)))
    except Exception:  # noqa: BLE001
        return []
    return entries


def config_declared_plan_labels(config: Any) -> list[str]:
    """All declared ``[budget.<provider>].plan`` labels from config.

    When multiple providers declare plans, each label is suffixed with the
    provider key (e.g. ``API billing (anthropic)``) so Claude Code + Codex
    setups are distinguishable in the UI.
    """
    entries = _declared_budget_plans(config)

    multi = len(entries) > 1
    labels: list[str] = []
    seen: set[str] = set()
    for provider, tier in entries:
        text = plan_label_for(tier, provider if multi else None)
        if text and text not in seen:
            labels.append(text)
            seen.add(text)
    return labels


def config_declared_plan(config: Any) -> str | None:
    """Return the user's declared plan tier from config.

    Checks the active config first; if no ``[budget.<provider>].plan`` is set
    (common when running from a project dir whose ``.tj/config.toml`` has no
    ``[budget]`` section), falls back to peeking at the global config at
    ``~/.config/tj/config.toml``. Without this fallback, framing silently
    rendered api-pricing in subdirectories even when the user had set their
    plan globally via ``tj onboard`` (issue #106). When multiple providers
    declare a plan, the first in sorted order wins (deterministic).
    """
    budgets = getattr(config, "budgets", None) or {}
    for provider in sorted(budgets.keys()):
        plan = getattr(budgets[provider], "plan", None)
        if plan:
            return str(plan)

    # Active config has no plan — peek at the global config file directly.
    try:
        import sys
        from pathlib import Path
        if sys.version_info >= (3, 11):
            import tomllib
        else:
            import tomli as tomllib  # type: ignore[no-redef]
        global_path = Path.home() / ".config" / "tj" / "config.toml"
        if not global_path.exists():
            return None
        with open(global_path, "rb") as f:
            raw = tomllib.load(f)
        budget_block = raw.get("budget") or {}
        for provider in sorted(budget_block.keys()):
            plan = (budget_block[provider] or {}).get("plan")
            if plan:
                return str(plan)
    except Exception:  # noqa: BLE001 — best-effort fallback, never fatal
        return None
    return None


def apply_declared_plans_to_sessions(
    conn: Any,
    config: Any,
    *,
    reconcile: bool = False,
) -> int:
    """Apply ``[budget.<provider>].plan`` to matching sessions.

    By default only promotes ``plan_tier`` from ``unknown``. When *reconcile*
    is True (explicit plan change via ``--reconfigure``), all sessions with
    spans for that provider are updated to the declared plan.
    """
    if conn is None:
        return 0
    updated = 0
    for provider, plan in _declared_budget_plans(config):
        if reconcile:
            count_row = conn.execute(
                "SELECT COUNT(*) FROM sessions "
                "WHERE session_id IN ("
                "  SELECT DISTINCT session_id FROM spans "
                "  WHERE billing_account = $1"
                ")",
                [str(provider)],
            ).fetchone()
            to_update = int(count_row[0]) if count_row else 0
            if to_update == 0:
                continue
            conn.execute(
                "UPDATE sessions SET plan_tier = $1 "
                "WHERE session_id IN ("
                "  SELECT DISTINCT session_id FROM spans "
                "  WHERE billing_account = $2"
                ")",
                [str(plan), str(provider)],
            )
            updated += to_update
            continue
        count_row = conn.execute(
            "SELECT COUNT(*) FROM sessions "
            "WHERE COALESCE(plan_tier, 'unknown') = 'unknown' "
            "AND session_id IN ("
            "  SELECT DISTINCT session_id FROM spans "
            "  WHERE billing_account = $1"
            ")",
            [str(provider)],
        ).fetchone()
        to_update = int(count_row[0]) if count_row else 0
        if to_update == 0:
            continue
        conn.execute(
            "UPDATE sessions SET plan_tier = $1 "
            "WHERE COALESCE(plan_tier, 'unknown') = 'unknown' "
            "AND session_id IN ("
            "  SELECT DISTINCT session_id FROM spans "
            "  WHERE billing_account = $2"
            ")",
            [str(plan), str(provider)],
        )
        updated += to_update
    return updated


@dataclass
class WindowSummary:
    """Minimal window aggregate that :func:`compute_framing` consumes.

    Callers can also pass any object/dict exposing these attributes (e.g. an
    ``OptimizeReport.window``); :func:`compute_framing` reads them defensively.
    """
    total_cost_usd: float = 0.0
    total_tokens: int = 0
    sessions: int = 0
    plan_tier_mix: dict[str, int] = field(default_factory=dict)


@dataclass
class Framing:
    """Plan-tier framing decision for a window of usage."""
    pricing_mode: str = "unknown"
    plan_tier: str = "unknown"
    plan_label: str | None = None
    plan_labels: list[str] = field(default_factory=list)
    plan_monthly_usd: float | None = None
    subscription_share_pct: float = 0.0
    api_share_pct: float = 0.0
    display_rule: str = DISPLAY_SHOW_DOLLARS
    qualifier_text: str | None = None
    # Window totals carried so renderers can compute token-share without a
    # second query (used by render_savings in subscription mode).
    window_total_tokens: int = 0
    window_total_cost_usd: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _attr(obj: Any, name: str, default: Any) -> Any:
    """Read ``name`` from an object attr or a dict key, with a default."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def compute_framing(
    config: Any,
    window_summary: Any,
    by_provider_breakdown: Any = None,
) -> Framing:
    """Compute the plan-tier framing for a window.

    ``window_summary`` is a :class:`WindowSummary`, an ``OptimizeReport.window``,
    or any object/dict exposing ``total_cost_usd``, ``total_tokens``,
    ``sessions`` and ``plan_tier_mix`` (session counts by plan tier).

    ``by_provider_breakdown`` is accepted for forward-compatibility (refining
    the API-only dollar split in mixed windows); the core decision does not
    require it.
    """
    mix: dict[str, int] = dict(_attr(window_summary, "plan_tier_mix", {}) or {})
    total_cost = float(_attr(window_summary, "total_cost_usd", 0.0) or 0.0)
    total_tokens = int(_attr(window_summary, "total_tokens", 0) or 0)

    total_sessions = sum(mix.values())
    sub_sessions = sum(v for k, v in mix.items() if k in SUBSCRIPTION_PLAN_TIERS)
    api_sessions = int(mix.get("api", 0))
    unknown_sessions = int(mix.get("unknown", 0))

    dom = dominant_plan(mix)
    # When the window carries no plan-tier data at all, prefer the user's
    # declared plan so framing-only contexts (e.g. /api/v1/budget) still reflect
    # a subscription user. This deliberately diverges from the CLI's data-driven
    # `dominant_plan` (which the CLI calls directly) — compute_framing is the
    # UI/API path and benefits from the config fallback.
    declared = config_declared_plan(config)
    if not mix and declared:
        dom = declared

    sub_pct = (100.0 * sub_sessions / total_sessions) if total_sessions else 0.0
    api_pct = (100.0 * api_sessions / total_sessions) if total_sessions else 0.0
    all_unknown = total_sessions > 0 and unknown_sessions == total_sessions
    inferred_from_config = False
    # When every session is unknown but config declares a plan, trust config for
    # framing so the UI reflects the user's onboard choice without requiring a
    # DB backfill (belt-and-suspenders alongside apply_declared_plans_to_sessions).
    if all_unknown and declared:
        dom = declared
        all_unknown = False
        inferred_from_config = True

    mode = pricing_mode_for(dom)
    label, fee = PLAN_LABEL_AND_FEE.get(dom, (None, None))
    if label is None:
        label = PLAN_DISPLAY_LABEL.get(dom)

    display_rule = DISPLAY_SHOW_DOLLARS
    qualifier: str | None = None

    if all_unknown:
        display_rule = DISPLAY_SUPPRESS_UNKNOWN
        qualifier = (
            "Plan tier unknown — figures may overstate actual cost. "
            + RECONFIGURE_HINT
        )
    elif mode == "subscription":
        display_rule = DISPLAY_SUPPRESS_SUBSCRIPTION
        if api_sessions > 0:
            qualifier = (
                f"{sub_pct:.1f}% of this window was subscription-billed; "
                f"dollar figures reflect API traffic only."
            )
    elif mode == "local":
        display_rule = DISPLAY_TOKENS_ONLY
        qualifier = "Local inference — no marginal cost."
    elif mode == "api":
        if unknown_sessions > 0 and not inferred_from_config:
            display_rule = DISPLAY_SHOW_DOLLARS_WITH_QUALIFIER
            qualifier = (
                f"{unknown_sessions} of {total_sessions} sessions have unknown "
                f"plan tier; dollar figures may overstate actual cost. "
                + RECONFIGURE_HINT
            )

    declared_labels = config_declared_plan_labels(config)

    return Framing(
        pricing_mode=mode,
        plan_tier=dom,
        plan_label=label,
        plan_labels=declared_labels or ([label] if label else []),
        plan_monthly_usd=fee,
        subscription_share_pct=round(sub_pct, 1),
        api_share_pct=round(api_pct, 1),
        display_rule=display_rule,
        qualifier_text=qualifier,
        window_total_tokens=total_tokens,
        window_total_cost_usd=total_cost,
    )


def _fmt_usd(value: float) -> str:
    """Compact USD for tiles: whole dollars when round, else 2dp."""
    value = round(value, 2)
    if value == int(value):
        return f"${int(value):,}"
    return f"${value:,.2f}"


def render_dollar(value: float | None, framing: Framing) -> str:
    """Render a single dollar value framed for the pricing mode.

    Returns e.g. ``"$148"`` (api), ``"12.4% of cycle"`` (subscription with a
    known plan fee), or ``"—"`` (local, or no value).
    """
    if value is None:
        return "—"
    mode = framing.pricing_mode
    if mode == "local":
        return "—"
    if mode == "subscription":
        if framing.plan_monthly_usd:
            pct = 100.0 * value / framing.plan_monthly_usd
            return f"{pct:.1f}% of cycle"
        return "—"
    # api / unknown — dollars shown (qualifier carried separately on Framing)
    return _fmt_usd(value)


def render_savings(
    value_usd: float | None,
    value_tokens: int | None,
    framing: Framing,
) -> str:
    """Render a savings / recoverable figure framed for the pricing mode.

    - api / unknown: dollars (``"$148"``)
    - subscription: token-share of the cycle (``"12.4% of cycle tokens"``)
    - local: token count (``"1.2M tokens"``)

    Returns ``"—"`` when the relevant figure is unavailable.
    """
    mode = framing.pricing_mode
    if mode == "subscription":
        if value_tokens is None:
            return "—"
        if framing.window_total_tokens > 0:
            pct = 100.0 * value_tokens / framing.window_total_tokens
            return f"{pct:.1f}% of cycle tokens"
        return f"{format_tokens(value_tokens)} tokens"
    if mode == "local":
        if value_tokens is None:
            return "—"
        return f"{format_tokens(value_tokens)} tokens"
    # api / unknown
    if value_usd is None:
        return "—"
    return _fmt_usd(value_usd)


def plan_tier_mix(
    conn: Any,
    since: Any = None,
    until: Any = None,
    agent_id: str | None = None,
) -> dict[str, int]:
    """Count sessions by plan_tier inside an optional window.

    Single home for the ``SELECT plan_tier, COUNT(*) FROM sessions`` query used
    by ``cmd_optimize`` and the ``/api/v1/optimize`` + ``/api/v1/cost`` routes.
    """
    # Built by concatenating static fragments (never f-string SQL, per CLAUDE.md
    # Critical Rule 7). Placeholder indices are appended as the params list grows;
    # every value is bound through a $N parameter, never interpolated.
    clauses: list[str] = []
    params: list[Any] = []
    if since is not None:
        params.append(since)
        clauses.append("started_at >= $" + str(len(params)))
    if until is not None:
        params.append(until)
        clauses.append("started_at < $" + str(len(params)))
    if agent_id:
        params.append(agent_id)
        clauses.append("agent_id = $" + str(len(params)))
    where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
    sql = (
        "SELECT COALESCE(plan_tier, 'unknown'), COUNT(*) FROM sessions"
        + where
        + " GROUP BY 1"
    )
    rows = conn.execute(sql, params).fetchall()
    return {str(r[0]): int(r[1]) for r in rows}
