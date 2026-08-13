"""GET /api/v1/budget — read budgets. POST /api/v1/budget — update budgets.

Three genuinely distinct budget concepts live here:

  - Coding-tool GROUP daily caps (`config.defaults.coding_budget` /
    `config.coding_agents[group_id].budget`, `GroupBudgetConfig.daily_usd`).
    One row per coding TOOL (claude-code / codex — see
    `tokenjam.core.agent_kind`), not per project and not per session: a group
    cap ceilings the SUM of today's spend across every member agent_id
    (e.g. every `claude-code-<project>` variant). Daily-only — a per-session
    cap has no meaning once many sessions across many projects share one row.
    Enforced in `AlertEngine._check_coding_group_daily_budget`, checked at
    SESSION END (not the moment spend crosses the line — see module note
    on the UI copy this feeds).

  - SDK-workflow per-agent caps (`config.defaults.budget` /
    `config.agents[id].budget`, `BudgetConfig.daily_usd` / `session_usd`).
    One row per SDK-declared agent_id, unchanged from the original per-agent
    design. `session_usd` is kept fully functional for backward
    compatibility with already-configured caps (`AlertEngine._check_cost_budgets`
    still fires COST_BUDGET_SESSION), but this screen no longer offers a way
    to set a NEW one — a per-session cap on a heterogeneous, human-driven
    coding session isn't a coherent product decision, and SDK workflows have
    always been able to express what they need with `daily_usd` alone.
    Enforced in `AlertEngine._check_agent_daily_budget` (daily) and the
    unchanged session_usd path (session).

  - Per-provider spend FORECASTS (`config.budgets[provider]`, `ProviderBudget.usd`,
    TOML `[budget.<provider>]`). A recurring monthly ceiling the Optimize
    "Budget projection" analyzer projects the current run rate against. It is
    read-only advisory (a forecast, never an alert) and scoped to a provider's
    whole cycle, not a single agent/day. See
    `tokenjam/core/optimize/analyzers/budget_projection.py`.

All three are returned by GET /budget. `POST /budget` writes the first two
(coding-group and SDK-workflow caps, both daily-only from this endpoint's
perspective — session_usd is read-only here, never accepted on a write, so a
save can never invent a NEW per-session cap even though an old one keeps
being honoured if already present); `POST /budget/provider` writes the
per-provider forecast ceiling.
"""
from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from tokenjam.api.deps import require_api_key
from tokenjam.core.agent_kind import (
    CODING_AGENT_GROUPS,
    group_agent_ids,
    present_coding_groups,
    sdk_agent_ids,
)
from tokenjam.core.alerts import is_interactive_coding_agent
from tokenjam.core.framing import PERSONAS, WindowSummary, compute_framing
from tokenjam.core.config import (
    AgentConfig,
    BudgetConfig,
    CodingGroupConfig,
    GroupBudgetConfig,
    ProviderBudget,
    active_config_path,
    resolve_config_path,
    resolve_effective_budget,
    resolve_group_budget,
    validate_budget_value,
    validate_cycle_start_day,
    write_config,
)

router = APIRouter(dependencies=[Depends(require_api_key)])

# Scope prefix for POST /budget targeting a coding-tool group cap, e.g.
# "group:claude-code". Distinguishes a group scope from an SDK agent_id scope
# without needing a second request field — see BudgetUpdate.scope.
_GROUP_SCOPE_PREFIX = "group:"
_DEFAULTS_SCOPE = "defaults"
_DEFAULTS_CODING_SCOPE = "defaults_coding"


def _write_target(config) -> "str | Path | None":
    """Where a budget mutation of ``app.state.config`` must be persisted.

    The live config is the one the daemon was started with — possibly from an
    explicit `tj serve --config PATH`, which no rediscovery in this process can
    see. Re-deriving the path here would serialize the mutated config over an
    unrelated file while leaving the config the daemon actually reads
    unchanged. Ask the config itself; fall back to discovery only when it did
    not come from a file at all.
    """
    return active_config_path(config) or resolve_config_path()


def _provider_budget_dict(pb: ProviderBudget) -> dict:
    return {
        "usd": pb.usd,
        "cycle_start_day": pb.cycle_start_day,
        "applies_to_services": list(pb.applies_to_services),
        "plan": pb.plan,
    }


def _b(b) -> dict:
    """Serialise a BudgetConfig (daily_usd + session_usd)."""
    return {"daily_usd": b.daily_usd, "session_usd": b.session_usd}


def _gb(gb) -> dict:
    """Serialise a GroupBudgetConfig (daily_usd only — no session concept at
    group scope)."""
    return {"daily_usd": gb.daily_usd}


def _budget_payload(config, agent_ids: list[str], *, persona: str | None = None) -> dict:
    """Two budget zones, per the Budget-page redesign:

    - `coding`: one row per coding TOOL (claude-code / codex), grouping every
      member agent_id (e.g. every claude-code-<project> variant) under one
      daily-only cap that ceilings their SUMMED spend. Only tools actually
      present in `agent_ids`, plus any explicitly configured in
      `config.coding_agents` even if not yet present in the data (so a
      pre-set cap survives before the first session lands).
    - `sdk`: one row per SDK-declared workflow agent_id, unchanged from the
      original per-agent design (daily + session, session read-only here —
      see module docstring).
    """
    # PERSONA SCOPE. A budget is a spend cap on a deployed workload, which is an
    # SDK-workflow concept: the Budget surface is reachable only from the
    # Sessions screen's SDK-services zone. Scoping HERE rather than in the view
    # is what makes that real — the coding groups and the legacy flat map are
    # both emptied of coding agents before they reach the wire, so a client that
    # renders either one cannot show a coding-agent cap it was never meant to.
    # `None` (no parameter) keeps the full payload for the CLI and for any
    # caller that predates the scope.
    sdk_only = persona == "sdk"

    present_groups = set() if sdk_only else set(present_coding_groups(agent_ids))
    all_groups = sorted(
        present_groups | (set() if sdk_only else set(config.coding_agents))
    )
    # Stable, deterministic order: claude-code before codex before any custom
    # future group name, alphabetically.
    ordered_groups = [g for g in CODING_AGENT_GROUPS if g in all_groups] + sorted(
        g for g in all_groups if g not in CODING_AGENT_GROUPS
    )

    groups = {}
    for gid in ordered_groups:
        group_cfg = config.coding_agents.get(gid)
        configured = _gb(group_cfg.budget) if group_cfg else _gb(GroupBudgetConfig())
        effective = _gb(resolve_group_budget(gid, config))
        groups[gid] = {
            "members": group_agent_ids(agent_ids, gid),
            "configured": configured,
            "effective": effective,
        }

    sdk_ids = sorted(sdk_agent_ids(agent_ids))
    if sdk_only:
        # TWO classifiers disagree here, deliberately, and this scope takes the
        # SAFER of the two rather than reconciling them.
        #
        # `agent_kind.sdk_agent_ids` (used just above) matches codex on the
        # EXACT id `codex_exec`, because Codex hardcodes that service name and
        # never varies it — that tightness is correct for budget GROUPING and
        # its module docstring explains why retightening the older predicate
        # would change five unrelated call sites. Consequence: a `codex-`
        # prefixed id that is not exactly `codex_exec` classifies as SDK there,
        # while `alerts.is_interactive_coding_agent` (what the persona picker,
        # the Sessions zones and the analyzer gate all use) calls it coding.
        #
        # For "may this row appear on a surface that must show no coding-agent
        # budgets", the question is not which classifier is more precise — it is
        # which failure is worse. Excluding an SDK row shows less than it could;
        # including a coding row breaks the decision this scope exists to
        # enforce. So the broader predicate wins for exclusion only, and neither
        # classifier's own rules are touched.
        sdk_ids = [a for a in sdk_ids if not is_interactive_coding_agent(a)]
    sdk_agents = {}
    for aid in sdk_ids:
        agent_cfg = config.agents.get(aid)
        raw = _b(agent_cfg.budget) if agent_cfg else _b(BudgetConfig())
        eff = _b(resolve_effective_budget(aid, config))
        sdk_agents[aid] = {"configured": raw, "effective": eff}

    # Plan-tier framing block (#110). The budget surface has no time window, so
    # framing falls back to the user's declared plan (compute_framing reads it
    # from config when the window mix is empty).
    framing = compute_framing(config, WindowSummary())

    # LEGACY top-level `defaults` / `agents` keys — the pre-redesign shape,
    # computed exactly as `_budget_payload` used to (one flat row per
    # agent_id, coding and SDK alike, no grouping). Kept as a STRICT SUPERSET
    # alongside `coding`/`sdk` so the still-committed old BudgetView (reads
    # `data.defaults.daily_usd` unguarded, and `data.agents`) keeps rendering
    # and saving correctly while the UI rewrite is in flight on a branch that
    # cannot yet touch this file's own PR. Remove only once that rewrite
    # lands and nothing reads these two keys anymore.
    legacy_agents = {}
    for aid in (sdk_ids if sdk_only else agent_ids):
        agent_cfg = config.agents.get(aid)
        raw = _b(agent_cfg.budget) if agent_cfg else _b(BudgetConfig())
        eff = _b(resolve_effective_budget(aid, config))
        legacy_agents[aid] = {"configured": raw, "effective": eff}

    return {
        "coding": {
            "defaults": _gb(config.defaults.coding_budget),
            "groups": groups,
        },
        "sdk": {
            "defaults": _b(config.defaults.budget),
            "agents": sdk_agents,
        },
        # -- legacy superset (see comment above) --
        "defaults": _b(config.defaults.budget),
        "agents": legacy_agents,
        # The provider spend-forecast ceilings Optimize's Budget projection
        # reads — surfaced here so they're visible/editable where users
        # otherwise only see the enforcement caps above.
        "provider_budgets": {
            provider: _provider_budget_dict(pb) for provider, pb in config.budgets.items()
        },
        "framing": framing.to_dict(),
    }


def _all_agent_ids(request: Request, config) -> list[str]:
    # Merge: config agents + DB-observed agents
    db = request.app.state.db
    db_agent_ids: set[str] = set()
    if hasattr(db, "conn"):
        rows = db.conn.execute(
            "SELECT DISTINCT agent_id FROM sessions ORDER BY agent_id"
        ).fetchall()
        db_agent_ids = {r[0] for r in rows}
    return sorted(set(config.agents) | db_agent_ids)


@router.get("/budget")
async def get_budget(request: Request, persona: str | None = None) -> dict:
    """Budget caps, optionally scoped to one persona.

    `persona=sdk` returns the SDK slice only: no coding-tool groups, and the
    legacy flat `agents` map restricted to SDK agent ids. Budgets are an
    SDK-workflow feature — the surface lives in the Sessions screen's
    SDK-services zone — so a coding-agent cap has nowhere honest to render.
    Omitting the parameter keeps the full payload, which the CLI still reads.
    """
    config = request.app.state.config
    if persona is not None and persona not in PERSONAS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown persona {persona!r}. Expected one of {sorted(PERSONAS)}.",
        )
    return _budget_payload(config, _all_agent_ids(request, config), persona=persona)


class BudgetUpdate(BaseModel):
    # "defaults" (SDK-zone default) | "defaults_coding" (coding-zone default)
    # | "group:<group_id>" (a coding-tool group cap) | an SDK agent_id.
    scope: str
    daily_usd: float | None = None
    # Read-only from this endpoint's perspective: accepted ONLY for SDK-agent
    # / "defaults" scopes (backward compat with an already-configured cap);
    # rejected outright for a coding-group scope, which has no per-session
    # concept. The redesigned UI never sends this field on a new save.
    session_usd: float | None = None


@router.post("/budget")
async def post_budget(request: Request, body: BudgetUpdate):
    config = request.app.state.config
    config_path_str = _write_target(config)
    if config_path_str is None:
        return JSONResponse(status_code=400, content={"error": "No config file found"})

    is_group_scope = body.scope == _DEFAULTS_CODING_SCOPE or body.scope.startswith(_GROUP_SCOPE_PREFIX)
    if is_group_scope and body.session_usd is not None:
        return JSONResponse(
            status_code=400,
            content={"error": "session_usd is not applicable to a coding-tool group cap"},
        )

    if body.scope == _DEFAULTS_SCOPE:
        budget = config.defaults.budget
    elif body.scope == _DEFAULTS_CODING_SCOPE:
        budget = config.defaults.coding_budget
    elif body.scope.startswith(_GROUP_SCOPE_PREFIX):
        group_id = body.scope[len(_GROUP_SCOPE_PREFIX):]
        if not group_id:
            return JSONResponse(status_code=400, content={"error": "group id must not be empty"})
        if group_id not in config.coding_agents:
            config.coding_agents[group_id] = CodingGroupConfig()
        budget = config.coding_agents[group_id].budget
    else:
        if body.scope not in config.agents:
            config.agents[body.scope] = AgentConfig()
        budget = config.agents[body.scope].budget

    try:
        if body.daily_usd is not None:
            budget.daily_usd = validate_budget_value(body.daily_usd, "daily_usd")
        if body.session_usd is not None:
            budget.session_usd = validate_budget_value(body.session_usd, "session_usd")
    except ValueError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

    write_config(config, Path(config_path_str))
    return _budget_payload(config, _all_agent_ids(request, config))


class ProviderBudgetUpdate(BaseModel):
    provider: str
    usd: float | None = None
    cycle_start_day: int | None = None


@router.post("/budget/provider")
async def post_provider_budget(request: Request, body: ProviderBudgetUpdate):
    """Update the per-provider spend-forecast ceiling (`[budget.<provider>]`)
    that Optimize's Budget projection analyzer reads. Distinct from
    `POST /budget` above, which edits the per-agent alert caps."""
    config = request.app.state.config
    config_path_str = _write_target(config)
    if config_path_str is None:
        return JSONResponse(status_code=400, content={"error": "No config file found"})

    provider = body.provider.strip()
    if not provider:
        return JSONResponse(status_code=400, content={"error": "provider must not be empty"})

    existing = config.budgets.get(provider)
    usd = existing.usd if existing else None
    cycle_start_day = existing.cycle_start_day if existing else 1
    applies_to_services = list(existing.applies_to_services) if existing else []
    plan = existing.plan if existing else None

    try:
        if body.usd is not None:
            usd = validate_budget_value(body.usd, "usd")
        if body.cycle_start_day is not None:
            cycle_start_day = validate_cycle_start_day(body.cycle_start_day)
    except ValueError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

    config.budgets[provider] = ProviderBudget(
        usd=usd,
        cycle_start_day=cycle_start_day,
        applies_to_services=applies_to_services,
        plan=plan,
    )
    # usd=0/None with no other fields ever set means "nothing configured" —
    # drop the section instead of writing an inert [budget.<provider>] stanza.
    if usd is None and not applies_to_services and plan is None and cycle_start_day == 1:
        del config.budgets[provider]

    write_config(config, Path(config_path_str))
    return _budget_payload(config, _all_agent_ids(request, config))
