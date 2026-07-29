from __future__ import annotations

import click
import socket
from contextlib import asynccontextmanager
from typing import AsyncIterator

from tokenjam.core.server_state import server_state_path
from tokenjam.utils.formatting import console


def _port_in_use(host: str, port: int) -> bool:
    """Return True if *port* on *host* is already bound.

    Pre-flight check so `tj serve` can fail fast with a clear message instead
    of the confusing "Application startup complete" -> EADDRINUSE sequence
    uvicorn prints when it can't bind the port (issue #509). We attempt the
    bind ourselves and release it immediately; a subsequent uvicorn bind on
    the same address is racy in theory but the window is negligible and the
    goal here is a clear diagnostic, not a lock.
    """
    # Pick the address family from the host so an IPv6 bind_host (e.g. "::" or
    # "::1") doesn't raise a bogus "invalid argument" OSError against an IPv4
    # socket and get misread as a port conflict.
    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    with socket.socket(family, socket.SOCK_STREAM) as sock:
        # Match uvicorn's own bind semantics, which set SO_REUSEADDR. Without
        # it, a port left in TIME_WAIT by a just-Ctrl-C'd server reads as
        # bound here even though uvicorn WOULD bind it — so the pre-flight
        # check, added to give a CLEARER error (issue #509), was stricter than
        # the real bind and manufactured a false "already in use" that blocked
        # a legitimate quick restart. A genuinely live listener still fails to
        # bind (SO_REUSEADDR does not let two sockets both LISTEN on one port),
        # so real conflicts are still caught.
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, port))
        except OSError:
            return True
    return False


@click.command("serve")
@click.option("--host", default=None, help="Bind host (default: from config)")
@click.option("--port", default=None, type=int, help="Bind port (default: from config)")
@click.option("--reload", is_flag=True, help="Enable auto-reload for development")
@click.pass_context
def cmd_serve(ctx: click.Context, host: str | None, port: int | None,
              reload: bool) -> None:
    """Run the local web UI (Lens)."""
    config = ctx.obj["config"]
    bind_host = host or config.api.host
    bind_port = port or config.api.port

    # Fail fast with a clear message if the port is already bound, rather than
    # letting uvicorn print "Application startup complete" and THEN a bind
    # error that reads like a crash-after-boot (issue #509).
    if _port_in_use(bind_host, bind_port):
        console.print(
            f"[red]Port {bind_port} is already in use[/red] — is [bold]tj serve[/bold] "
            f"already running?"
        )
        console.print(
            "  Run [bold]tj stop[/bold] to stop it, or [bold]tj serve --port <n>[/bold] "
            "to use a different port."
        )
        raise SystemExit(1)

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

    # ONE analyzer scan cycle, for every store the surfaces read.
    #
    # There used to be three jobs here — relearn 6h, cost proposals 6h, the full
    # report on `scan_interval_hours` — each with its own startup kick, plus two
    # on-demand endpoints that each refreshed a DIFFERENT subset. So "Rescan"
    # meant something different depending on which screen you pressed it from,
    # and the Dashboard's waste tiles could be hours fresher than the Review
    # inbox headline they get compared against, with nothing disclosing it.
    #
    # `core/optimize/scan_cycle.py` owns the cycle and the reasoning; this is
    # the schedule and the startup kick for it. `POST /optimize/rescan` fires
    # the same cycle, so every trigger refreshes the same three stores.
    from tokenjam.core.db import DuckDBBackend as _DuckDBBackend
    from tokenjam.core.optimize import scan_cycle

    def _scan_cycle_job() -> None:
        # Resolved through the module (not a from-import) so the scheduled and
        # startup passes share one patchable seam.
        scan_cycle.trigger_scan_cycle(
            lambda: _DuckDBBackend(config.storage), config,
        )

    # `scan_enabled = false` is the kill switch for automatic scanning, and it
    # now gates every store rather than only the report — it is documented as
    # "keeps the daemon from ever scanning on its own", and relearn and the cost
    # proposals are scans. An explicit rescan from a surface is a human asking
    # and stays available; nothing re-enables inline compute on a request.
    if scan_cycle.scan_enabled(config):
        scheduler.add_job(
            _scan_cycle_job, "interval", hours=config.optimize.scan_interval_hours,
        )

    # Continuous transcript ingestion. Claude Code's OTLP exporter has no retry
    # and no buffer, so the live path silently drops any session whose shell
    # lacked the telemetry env vars, or that ran while this daemon was down or
    # while the shell's baked-in endpoint pointed at a dead port. Until now
    # nothing ever reconciled that: the ONLY remedy was a human running
    # `tj backfill claude-code`, and every un-ingested session becomes
    # unrecoverable when Claude Code prunes its transcript ~30 days later.
    #
    # Backfill is idempotent (deterministic span ids + a batched anti-join), so
    # a re-run over an overlapping window costs a re-parse and inserts only the
    # genuinely-missing spans.
    from datetime import timedelta as _timedelta
    from tokenjam.core import transcript_sync

    ingest_cfg = config.ingest

    def _catch_up_job(lookback) -> None:
        # Resolved through the module (not a from-import) so the scheduled and
        # startup passes share one patchable seam.
        transcript_sync.start_catch_up(
            lambda: DuckDBBackend(config.storage), config=config, lookback=lookback,
        )

    if ingest_cfg.auto_catch_up:
        scheduler.add_job(
            lambda: _catch_up_job(_timedelta(hours=ingest_cfg.lookback_hours)),
            "interval",
            minutes=ingest_cfg.interval_minutes,
        )

    # ~/.local/share/tj/server.state lets other subcommands (e.g. `tj onboard
    # --codex`) find the config this server is using regardless of CWD. We
    # write it from the lifespan so it only happens after uvicorn binds the
    # port — a failed bind must NOT clobber the running daemon's state file.
    # Same reasoning for `scheduler.start()`: don't fire off a background
    # thread for a server that's about to exit with EADDRINUSE.
    import json as _json
    _state_path = server_state_path()

    # Optional enforcement-plane proxy (#219) — a second in-process listener on
    # config.proxy.port, started/stopped with the server's lifespan. Suggest
    # mode only; the pricing-mode gate forwards subscription/unknown unmodified.
    proxy_runner = None
    if config.proxy.enabled:
        from tokenjam.proxy.server import ProxyRunner
        # Pass the serve DB so in-process policies (budget_cap, #222) can read
        # current-cycle spend AND policy decisions + the savings ledger are
        # persisted (#221) — all over the same per-thread-cursor connection (#124).
        # Pass the pipeline so the policy self-observation span (#223) flows
        # through the ingest hooks like any other span.
        proxy_runner = ProxyRunner(config, db=db, pipeline=pipeline)

    @asynccontextmanager
    async def _lifespan(_app: FastAPI) -> AsyncIterator[None]:
        # startup
        scheduler.start()
        if proxy_runner is not None:
            proxy_runner.start()
        # One startup kick for the whole cycle. The interval's own first fire
        # is hours out, so without this a fresh `tj serve` would serve
        # "never_run" on every analyzer surface until then — correctly saying
        # "not computed yet", but for hours. Own connections, own threads: it
        # never blocks the bind/startup path.
        if scan_cycle.scan_enabled(config):
            _scan_cycle_job()
        # Catch up on anything the live path missed while this daemon was down.
        # Wider window than the interval job: a startup pass has to cover
        # however long we were off, not just one interval. Runs on its own
        # thread with its own connection, so it never delays the bind.
        if ingest_cfg.auto_catch_up:
            _catch_up_job(_timedelta(days=ingest_cfg.startup_lookback_days))
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
            # Drain any queued async hooks before tearing down the worker so no
            # alert is lost on shutdown. close() also flushes internally, but we
            # call flush() first to keep the "flush then close" contract explicit
            # and to drain even if a future close() changes behavior.
            if hasattr(pipeline, "flush"):
                pipeline.flush()
            if hasattr(pipeline, "close"):
                pipeline.close()

    app = create_app(config, db, pipeline, lifespan=_lifespan)

    console.print(f"[bold]tj serve[/bold] starting on http://{bind_host}:{bind_port}")
    console.print(f"  API docs:    http://{bind_host}:{bind_port}/docs")
    if ingest_cfg.auto_catch_up:
        console.print(
            f"  Transcript catch-up: on startup, then every "
            f"{ingest_cfg.interval_minutes}m"
        )
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
