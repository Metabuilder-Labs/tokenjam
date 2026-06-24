from __future__ import annotations

import click
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from tokenjam.utils.formatting import console


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
    from fastapi import FastAPI
    from tokenjam.api.app import create_app
    from tokenjam.core.ingest import build_default_pipeline

    db = ctx.obj["db"]
    pipeline = build_default_pipeline(db, config)

    # Schedule retention cleanup using a separate DB connection per run
    # to avoid concurrent write conflicts with uvicorn worker threads.
    from apscheduler.schedulers.background import BackgroundScheduler
    from tokenjam.core.retention import run_retention_cleanup
    from tokenjam.core.db import DuckDBBackend

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

    # ~/.local/share/tj/server.state lets other subcommands (e.g. `tj onboard
    # --codex`) find the config this server is using regardless of CWD. We
    # write it from the lifespan so it only happens after uvicorn binds the
    # port — a failed bind must NOT clobber the running daemon's state file.
    # Same reasoning for `scheduler.start()`: don't fire off a background
    # thread for a server that's about to exit with EADDRINUSE.
    import json as _json
    _state_path = Path.home() / ".local" / "share" / "tj" / "server.state"

    # Optional enforcement-plane proxy (#219) — a second in-process listener on
    # config.proxy.port, started/stopped with the server's lifespan. Suggest
    # mode only; the pricing-mode gate forwards subscription/unknown unmodified.
    proxy_runner = None
    if config.proxy.enabled:
        from tokenjam.proxy.server import ProxyRunner
        # Pass the serve DB so in-process policies (budget_cap, #222) can read
        # current-cycle spend AND policy decisions + the savings ledger are
        # persisted (#221) — all over the same per-thread-cursor connection (#124).
        proxy_runner = ProxyRunner(config, db=db)

    @asynccontextmanager
    async def _lifespan(_app: FastAPI) -> AsyncIterator[None]:
        # startup
        scheduler.start()
        if proxy_runner is not None:
            proxy_runner.start()
        # Stamp unknown sessions from declared [budget.*].plan on startup so
        # historical/backfilled rows match config without a separate onboard pass.
        from tokenjam.core.framing import apply_declared_plans_to_sessions

        conn = getattr(db, "conn", None)
        if conn is not None:
            try:
                apply_declared_plans_to_sessions(conn, config)
            except Exception:
                pass
        _state_path.parent.mkdir(parents=True, exist_ok=True)
        _state_path.write_text(
            _json.dumps({
                "config_path": str(config.config_path) if config.config_path else None,
                "port": bind_port,
                "pid": __import__("os").getpid(),
            })
        )
        try:
            yield
        finally:
            # shutdown
            if proxy_runner is not None:
                await proxy_runner.stop()
            scheduler.shutdown(wait=False)

    app = create_app(config, db, pipeline, lifespan=_lifespan)

    console.print(f"[bold]tj serve[/bold] starting on http://{bind_host}:{bind_port}")
    console.print(f"  API docs:    http://{bind_host}:{bind_port}/docs")
    if config.export.prometheus.enabled:
        console.print(f"  Metrics:     http://{bind_host}:{bind_port}/metrics")
    if config.proxy.enabled:
        _ks = " [yellow](killswitch: pass-through)[/yellow]" if config.proxy.killswitch else ""
        console.print(
            f"  Proxy:       http://{config.proxy.host}:{config.proxy.port} "
            f"(suggest mode){_ks}"
        )
    console.print()

    if reload:
        console.print(
            "[yellow]Warning: --reload requires an import string, not an app instance. "
            "Reload mode is not supported with injected db/config — ignoring --reload.[/yellow]"
        )
    uvicorn.run(app, host=bind_host, port=bind_port)
