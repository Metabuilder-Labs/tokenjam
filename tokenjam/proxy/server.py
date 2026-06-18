"""Run the proxy listener in-process inside ``tj serve`` (#219).

``tj serve`` binds the main API on ``api.port``; when ``proxy.enabled`` it also
runs this second listener on ``proxy.port`` in the SAME event loop, started and
stopped from the server's lifespan. Keeping it in-process (rather than a
separate daemon) means the proxy shares the server's lifecycle exactly — it can
never outlive ``tj serve`` or be orphaned.
"""
from __future__ import annotations

import asyncio
import contextlib
import logging
from typing import Any

import uvicorn

from tokenjam.proxy.app import build_proxy_app
from tokenjam.proxy.observer import ProxyObserver

logger = logging.getLogger("tokenjam.proxy")


class ProxyRunner:
    """Owns the proxy's uvicorn server + its lifecycle within an event loop."""

    def __init__(self, config: Any, observer: ProxyObserver | None = None,
                 db: Any = None, pipeline: Any = None) -> None:
        self.config = config
        # Shared tj-serve DB (optional). Used three ways: in-process policies
        # (budget_cap, #222) read current-cycle spend from it, the audit sink
        # (#221) persists decisions + the savings ledger through it, and the
        # self-observation span (#223) is emitted through `pipeline` (or the db).
        self.db = db
        self.pipeline = pipeline
        if observer is None:
            sink = None
            if db is not None:
                from tokenjam.proxy.audit import AuditSink
                sink = AuditSink(db, pipeline=pipeline)
            observer = ProxyObserver(sink=sink)
        self.observer = observer
        self._server: uvicorn.Server | None = None
        self._task: asyncio.Task | None = None

    def start(self) -> None:
        """Schedule the proxy server on the running event loop (non-blocking)."""
        app = build_proxy_app(self.config, observer=self.observer, db=self.db)
        uconfig = uvicorn.Config(
            app,
            host=self.config.proxy.host,
            port=self.config.proxy.port,
            log_level="warning",
            # The parent tj-serve process already installs signal handlers; the
            # embedded server must not steal them or it would race shutdown.
            lifespan="on",
        )
        self._server = uvicorn.Server(uconfig)
        # The parent `tj serve` process owns the process signal handlers. The
        # embedded proxy server must NOT install or capture them, or it would
        # override the parent's SIGINT/SIGTERM handling for the lifetime of the
        # proxy task. Neutralise both the (older) method and the (current)
        # capture_signals context manager so this stays version-robust.
        self._server.install_signal_handlers = lambda: None  # type: ignore[assignment]
        self._server.capture_signals = contextlib.nullcontext  # type: ignore[assignment]
        self._task = asyncio.create_task(self._server.serve())
        logger.info(
            "tj proxy listening on http://%s:%d (mode=%s, killswitch=%s)",
            self.config.proxy.host, self.config.proxy.port,
            self.config.proxy.mode, self.config.proxy.killswitch,
        )

    async def stop(self) -> None:
        if self._server is not None:
            self._server.should_exit = True
        if self._task is not None:
            try:
                await self._task
            except Exception:  # noqa: BLE001
                logger.exception("tj proxy server task ended with error")
