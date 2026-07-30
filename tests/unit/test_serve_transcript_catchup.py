"""`tj serve` must schedule continuous transcript ingestion.

Before this, the daemon scheduled retention cleanup, the relearn recompute, the
cost-proposals recompute and a plan-stamping pass — and NOT a single job that
looked at `~/.claude/projects`. A session dropped by the live OTLP path (dead
endpoint, missing env vars, daemon down) was never reconciled by anything, so
completeness depended on a human running `tj backfill claude-code`.
"""
from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from tokenjam.cli.cmd_serve import cmd_serve
from tokenjam.core.config import IngestConfig, TjConfig


def _ctx_obj(ingest: IngestConfig | None = None) -> dict:
    config = TjConfig(version="1")
    if ingest is not None:
        config.ingest = ingest
    return {"config": config, "db": MagicMock()}


def _boot(ctx_obj: dict):
    """Invoke `tj serve` far enough to build the scheduler + lifespan, without
    ever binding a port or starting uvicorn. Returns the scheduler mock, the
    created-app kwargs, and the CLI result."""
    scheduler = MagicMock()
    created: dict = {}

    def _create_app(*args, **kwargs):
        created.update(kwargs)
        return MagicMock()

    with patch("tokenjam.cli.cmd_serve._port_in_use", return_value=False), \
         patch("uvicorn.run"), \
         patch("apscheduler.schedulers.background.BackgroundScheduler",
               return_value=scheduler), \
         patch("tokenjam.core.ingest.build_default_pipeline",
               return_value=MagicMock()), \
         patch("tokenjam.api.app.create_app", side_effect=_create_app):
        result = CliRunner().invoke(cmd_serve, [], obj=ctx_obj)
    return scheduler, created, result


def _interval_jobs(scheduler: MagicMock) -> list[dict]:
    """Every `scheduler.add_job(fn, "interval", ...)` call, as kwargs dicts."""
    jobs = []
    for call in scheduler.add_job.call_args_list:
        if len(call.args) >= 2 and call.args[1] == "interval":
            jobs.append(call.kwargs)
    return jobs


def test_serve_schedules_a_recurring_transcript_catch_up() -> None:
    scheduler, _, result = _boot(_ctx_obj())

    assert result.exit_code == 0, result.output
    minute_jobs = [j for j in _interval_jobs(scheduler) if "minutes" in j]
    assert minute_jobs, "no minute-cadence job scheduled — catch-up is missing"
    assert minute_jobs[0]["minutes"] == IngestConfig.interval_minutes


def test_catch_up_cadence_follows_config() -> None:
    scheduler, _, result = _boot(_ctx_obj(IngestConfig(interval_minutes=5)))

    assert result.exit_code == 0, result.output
    minute_jobs = [j for j in _interval_jobs(scheduler) if "minutes" in j]
    assert minute_jobs[0]["minutes"] == 5


def test_catch_up_can_be_turned_off() -> None:
    scheduler, _, result = _boot(_ctx_obj(IngestConfig(auto_catch_up=False)))

    assert result.exit_code == 0, result.output
    assert not [j for j in _interval_jobs(scheduler) if "minutes" in j]
    assert "Transcript catch-up" not in result.output


def test_serve_announces_the_catch_up_cadence() -> None:
    _, _, result = _boot(_ctx_obj(IngestConfig(interval_minutes=15)))

    assert "Transcript catch-up" in result.output
    assert "15m" in result.output


def test_startup_runs_a_catch_up_with_the_wider_startup_window() -> None:
    """Downtime must self-heal: the startup pass covers however long the daemon
    was off, so it uses `startup_lookback_days`, not the interval lookback."""
    ctx_obj = _ctx_obj(IngestConfig(startup_lookback_days=9))
    _, created, result = _boot(ctx_obj)
    assert result.exit_code == 0, result.output

    lifespan = created["lifespan"]
    seen: list = []

    def _capture(db_factory, config=None, root=None, lookback=None, on_done=None):
        seen.append(lookback)
        return MagicMock()

    async def _drive() -> None:
        async with lifespan(MagicMock()):
            pass

    with patch("tokenjam.core.transcript_sync.start_catch_up", side_effect=_capture), \
         patch("tokenjam.core.optimize.relearn_store.trigger_background_recompute"), \
         patch("tokenjam.core.optimize.cost_proposals."
               "trigger_background_cost_recompute"):
        asyncio.run(_drive())

    assert seen, "startup did not kick a catch-up pass"
    assert seen[0].days == 9
