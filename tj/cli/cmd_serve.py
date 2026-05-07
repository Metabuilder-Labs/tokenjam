from __future__ import annotations

import click
from pathlib import Path

from tj.utils.formatting import console


@click.command("serve")
@click.option("--host", default=None, help="Bind host (default: from config)")
@click.option("--port", default=None, type=int, help="Bind port (default: from config)")
@click.option("--reload", is_flag=True, help="Enable auto-reload for development")
@click.pass_context
def cmd_serve(ctx: click.Context, host: str | None, port: int | None,
              reload: bool) -> None:
    """Start the tj API server."""
    config = ctx.obj["config"]
    bind_host = host or config.api.host
    bind_port = port or config.api.port

    import uvicorn
    from tj.api.app import create_app
    from tj.core.ingest import build_default_pipeline

    db = ctx.obj["db"]
    pipeline = build_default_pipeline(db, config)
    app = create_app(config, db, pipeline)

    # Schedule retention cleanup using a separate DB connection per run
    # to avoid concurrent write conflicts with uvicorn worker threads.
    from apscheduler.schedulers.background import BackgroundScheduler
    from tj.core.retention import run_retention_cleanup
    from tj.core.db import DuckDBBackend

    def _retention_job() -> None:
        retention_db = DuckDBBackend(config.storage)
        try:
            run_retention_cleanup(retention_db, config.storage)
        finally:
            retention_db.close()

    scheduler = BackgroundScheduler()
    scheduler.add_job(
        _retention_job,
        "cron",
        hour=0,
        minute=0,
    )
    scheduler.start()

    @app.on_event("shutdown")
    async def _shutdown_scheduler() -> None:
        scheduler.shutdown(wait=False)

    # Write the resolved config path so other subcommands (e.g. onboard --codex)
    # can find the secret this server is using regardless of CWD. Defer the
    # write to a FastAPI startup event so it only fires after uvicorn binds
    # the port — otherwise a failed-to-bind serve clobbers the state file
    # of the running daemon (D2).
    import json as _json
    _state_path = Path.home() / ".local" / "share" / "tj" / "server.state"

    @app.on_event("startup")
    async def _write_server_state() -> None:
        _state_path.parent.mkdir(parents=True, exist_ok=True)
        _state_path.write_text(
            _json.dumps({
                "config_path": str(config.config_path) if config.config_path else None,
                "port": bind_port,
                "pid": __import__("os").getpid(),
            })
        )

    console.print(f"[bold]tj serve[/bold] starting on http://{bind_host}:{bind_port}")
    console.print(f"  API docs:    http://{bind_host}:{bind_port}/docs")
    if config.export.prometheus.enabled:
        console.print(f"  Metrics:     http://{bind_host}:{bind_port}/metrics")
    console.print()

    if reload:
        console.print(
            "[yellow]Warning: --reload requires an import string, not an app instance. "
            "Reload mode is not supported with injected db/config — ignoring --reload.[/yellow]"
        )
    uvicorn.run(app, host=bind_host, port=bind_port)
