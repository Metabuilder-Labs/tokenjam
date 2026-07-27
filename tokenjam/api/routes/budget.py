"""GET /api/v1/budget — read budgets. POST /api/v1/budget — update budgets.

Two genuinely distinct budget concepts live here, both surfaced on this one
screen so a user setting a budget can see both ceilings in one place:

  - Per-agent ENFORCEMENT caps (`config.defaults.budget` / `config.agents[id].budget`,
    `BudgetConfig.daily_usd` / `session_usd`). Checked synchronously on every
    ingested span (`AlertEngine._check_cost_budgets`) and fire a real-time alert
    the moment an agent's spend crosses the line. Scope: one agent, one day/session.

  - Per-provider spend FORECASTS (`config.budgets[provider]`, `ProviderBudget.usd`,
    TOML `[budget.<provider>]`). A recurring monthly ceiling the Optimize
    "Budget projection" analyzer projects the current run rate against. It is
    read-only advisory (a forecast, never an alert) and scoped to a provider's
    whole cycle, not a single agent/day. See
    `tokenjam/core/optimize/analyzers/budget_projection.py`.

These were previously two disconnected objects: this screen showed only the
per-agent caps, while the Optimize projection read a provider ceiling this
screen never displayed or let you edit. Both are now returned by GET /budget
and both are writable here — `POST /budget` for the per-agent caps (unchanged
shape), `POST /budget/provider` for the per-provider forecast ceiling.
"""
from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from tokenjam.api.deps import require_api_key
from tokenjam.core.framing import WindowSummary, compute_framing
from tokenjam.core.config import (
    AgentConfig,
    BudgetConfig,
    ProviderBudget,
    active_config_path,
    resolve_config_path,
    resolve_effective_budget,
    validate_budget_value,
    validate_cycle_start_day,
    write_config,
)

router = APIRouter(dependencies=[Depends(require_api_key)])


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


def _budget_payload(config, agent_ids: list[str]) -> dict:
    def _b(b):
        return {"daily_usd": b.daily_usd, "session_usd": b.session_usd}

    agents = {}
    for aid in agent_ids:
        agent_cfg = config.agents.get(aid)
        raw = _b(agent_cfg.budget) if agent_cfg else _b(BudgetConfig())
        eff = _b(resolve_effective_budget(aid, config))
        agents[aid] = {"configured": raw, "effective": eff}

    # Plan-tier framing block (#110). The budget surface has no time window, so
    # framing falls back to the user's declared plan (compute_framing reads it
    # from config when the window mix is empty).
    framing = compute_framing(config, WindowSummary())

    return {
        "defaults": _b(config.defaults.budget),
        "agents": agents,
        # The provider spend-forecast ceilings Optimize's Budget projection
        # reads — surfaced here so they're visible/editable where users
        # otherwise only see the per-agent enforcement caps above.
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
async def get_budget(request: Request) -> dict:
    config = request.app.state.config
    return _budget_payload(config, _all_agent_ids(request, config))


class BudgetUpdate(BaseModel):
    scope: str           # "defaults" or an agent_id
    daily_usd: float | None = None
    session_usd: float | None = None


@router.post("/budget")
async def post_budget(request: Request, body: BudgetUpdate):
    config = request.app.state.config
    config_path_str = _write_target(config)
    if config_path_str is None:
        return JSONResponse(status_code=400, content={"error": "No config file found"})

    if body.scope == "defaults":
        budget = config.defaults.budget
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
