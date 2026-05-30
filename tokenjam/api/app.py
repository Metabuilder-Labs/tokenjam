"""FastAPI application factory. Called by `tj serve`."""
from __future__ import annotations

from html import escape as html_escape
from pathlib import Path
from typing import TYPE_CHECKING, Any, AsyncContextManager, Callable

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from tokenjam.api.middleware import IngestAuthMiddleware
from tokenjam.core.config import TjConfig

if TYPE_CHECKING:
    from tokenjam.core.db import StorageBackend
    from tokenjam.core.ingest import IngestPipeline

_UI_DIR = Path(__file__).resolve().parent.parent / "ui"


def create_app(
    config: TjConfig,
    db: StorageBackend,
    ingest_pipeline: IngestPipeline,
    lifespan: Callable[[FastAPI], AsyncContextManager[Any]] | None = None,
) -> FastAPI:
    """
    Build and return the FastAPI app.

    db and ingest_pipeline are passed in (not imported globally) so tests
    can inject mocks easily.

    `lifespan`, if provided, is a FastAPI lifespan context manager — used by
    `tj serve` to start/stop the retention scheduler and write server.state
    only after uvicorn has bound the port (so a failed bind can't clobber a
    running daemon's state file).
    """
    app = FastAPI(
        title="TokenJam",
        version="0.1.0",
        docs_url="/docs",
        redoc_url=None,
        lifespan=lifespan,
    )

    # CORS — local only by default
    app.add_middleware(
        CORSMiddleware,
        allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$",
        allow_methods=["GET", "POST", "PATCH"],
        allow_headers=["Authorization", "Content-Type"],
    )

    # Ingest auth middleware
    app.add_middleware(IngestAuthMiddleware)

    # Shared state for routes
    app.state.config = config
    app.state.db = db
    app.state.pipeline = ingest_pipeline

    # Register routers
    from tokenjam.api.routes.spans import router as spans_router
    from tokenjam.api.routes.traces import router as traces_router
    from tokenjam.api.routes.cost import router as cost_router
    from tokenjam.api.routes.tools import router as tools_router
    from tokenjam.api.routes.alerts import router as alerts_router
    from tokenjam.api.routes.drift import router as drift_router
    from tokenjam.api.routes.metrics import router as metrics_router
    from tokenjam.api.routes.status import router as status_router
    from tokenjam.api.routes.sessions import router as sessions_router
    from tokenjam.api.routes.otlp import router as otlp_router
    from tokenjam.api.routes.budget import router as budget_router
    from tokenjam.api.routes.agents import router as agents_router
    from tokenjam.api.routes.optimize import router as optimize_router
    from tokenjam.api.routes.cost_compare import router as cost_compare_router

    app.include_router(spans_router, prefix="/api/v1")
    app.include_router(traces_router, prefix="/api/v1")
    app.include_router(cost_router, prefix="/api/v1")
    app.include_router(tools_router, prefix="/api/v1")
    app.include_router(alerts_router, prefix="/api/v1")
    app.include_router(drift_router, prefix="/api/v1")
    app.include_router(status_router, prefix="/api/v1")
    app.include_router(sessions_router, prefix="/api/v1")
    app.include_router(budget_router, prefix="/api/v1")
    app.include_router(agents_router, prefix="/api/v1")
    app.include_router(optimize_router, prefix="/api/v1")
    app.include_router(cost_compare_router, prefix="/api/v1")
    app.include_router(metrics_router)  # /metrics — no prefix
    app.include_router(otlp_router)  # /v1/traces, /v1/metrics, /v1/logs — no prefix

    # --- Web UI ---
    _index_html = ""
    index_path = _UI_DIR / "index.html"
    if index_path.exists():
        _index_html = index_path.read_text()

    def _serve_ui() -> HTMLResponse:
        html = _index_html
        if config.api.auth.enabled and config.api.auth.api_key:
            html = html.replace(
                "</head>",
                f'<meta name="tj-api-key" content="{html_escape(config.api.auth.api_key, quote=True)}">\n</head>',
            )
        return HTMLResponse(html)

    @app.get("/", include_in_schema=False)
    async def ui_root():
        return _serve_ui()

    @app.get("/ui/{path:path}", include_in_schema=False)
    async def ui_catchall(path: str):
        return _serve_ui()

    return app
