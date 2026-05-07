"""GET /api/v1/budget — read budgets. POST /api/v1/budget — update budgets."""
from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from tj.api.deps import require_api_key
from tj.core.config import (
    AgentConfig,
    BudgetConfig,
    find_config_file,
    resolve_effective_budget,
    validate_budget_value,
    write_config,
)

router = APIRouter(dependencies=[Depends(require_api_key)])


def _budget_payload(config, agent_ids: list[str]) -> dict:
    def _b(b):
        return {"daily_usd": b.daily_usd, "session_usd": b.session_usd}

    agents = {}
    for aid in agent_ids:
        agent_cfg = config.agents.get(aid)
        raw = _b(agent_cfg.budget) if agent_cfg else _b(BudgetConfig())
        eff = _b(resolve_effective_budget(aid, config))
        agents[aid] = {"configured": raw, "effective": eff}

    return {
        "defaults": _b(config.defaults.budget),
        "agents": agents,
    }


@router.get("/budget")
async def get_budget(request: Request) -> dict:
    config = request.app.state.config
    db = request.app.state.db

    # Merge: config agents + DB-observed agents
    db_agent_ids: set[str] = set()
    if hasattr(db, "conn"):
        rows = db.conn.execute(
            "SELECT DISTINCT agent_id FROM sessions ORDER BY agent_id"
        ).fetchall()
        db_agent_ids = {r[0] for r in rows}

    all_agent_ids = sorted(set(config.agents) | db_agent_ids)
    return _budget_payload(config, all_agent_ids)


class BudgetUpdate(BaseModel):
    scope: str           # "defaults" or an agent_id
    daily_usd: float | None = None
    session_usd: float | None = None


@router.post("/budget")
async def post_budget(request: Request, body: BudgetUpdate):
    config = request.app.state.config
    config_path_str = find_config_file()
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

    # Return updated payload with all known agents
    db = request.app.state.db
    db_agent_ids: set[str] = set()
    if hasattr(db, "conn"):
        rows = db.conn.execute(
            "SELECT DISTINCT agent_id FROM sessions ORDER BY agent_id"
        ).fetchall()
        db_agent_ids = {r[0] for r in rows}
    all_agent_ids = sorted(set(config.agents) | db_agent_ids)
    return _budget_payload(config, all_agent_ids)
