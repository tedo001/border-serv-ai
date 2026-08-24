"""FastAPI application factory."""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from ibvap import __version__
from ibvap.api.routers import auth, cameras, events, system, watchlists
from ibvap.api.state import AppState
from ibvap.core.config import Settings, load_settings
from ibvap.core.errors import AuthError, ConfigError, IbvapError, StorageError
from ibvap.core.logging import configure_logging, get_logger

log = get_logger(__name__)

API_PREFIX = "/api/v1"

DESCRIPTION = """
**IBVAP** turns existing IP CCTV infrastructure into an intelligent surveillance
network - human and vehicle detection and tracking, face recognition, ANPR,
virtual fencing, behavioural analytics and camera tamper detection - without
dedicated FRS/ANPR appliances or smart cameras.

### Authentication

Humans authenticate at `POST /api/v1/auth/login` and present
`Authorization: Bearer <token>`. Machines present `X-API-Key: <key>`.

### Roles

| Role | May |
|------|-----|
| `viewer` | read events, view live video |
| `operator` | acknowledge alerts, manage the vehicle watchlist |
| `supervisor` | configure cameras, zones and rules; enrol faces |
| `admin` | manage users, API keys and system settings |
"""


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Bring the node up and tear it down with the server."""
    state: AppState = app.state.ibvap
    try:
        await state.startup()
    except Exception:
        # Release whatever did come up; leaving a half-started node running is
        # worse than failing loudly, because it looks healthy from outside.
        log.error("startup_failed", exc_info=True)
        await state.shutdown()
        raise
    try:
        yield
    finally:
        await state.shutdown()


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the IBVAP application."""
    settings = settings or load_settings()
    configure_logging(
        settings.telemetry.log_level,
        settings.telemetry.log_format,
        settings.telemetry.log_file,
    )
    settings.ensure_secret()

    if not settings.security.require_auth:
        log.warning(
            "authentication_disabled",
            detail="every endpoint is open; never run this way outside an isolated bench",
        )

    app = FastAPI(
        title="IBVAP - Intelligent Border Video Analytics Platform",
        description=DESCRIPTION,
        version=__version__,
        lifespan=lifespan,
        docs_url="/api/docs",
        redoc_url="/api/redoc",
        openapi_url="/api/openapi.json",
    )
    app.state.ibvap = AppState(settings)

    _configure_middleware(app, settings)
    _configure_errors(app)

    for router in (auth.router, events.router, cameras.router, watchlists.router):
        app.include_router(router, prefix=API_PREFIX)
    # Health and metrics sit at the root: monitoring systems and orchestrators
    # expect them there, and versioning a liveness probe serves nobody.
    app.include_router(system.router)
    app.include_router(system.router, prefix=API_PREFIX, include_in_schema=False)

    _configure_ui(app, settings)
    return app


def _configure_middleware(app: FastAPI, settings: Settings) -> None:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.security.cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type", "X-API-Key"],
        expose_headers=["X-IBVAP-SHA256"],
    )

    @app.middleware("http")
    async def add_security_headers(request: Request, call_next):
        """Apply hardening headers and record request latency."""
        started = time.perf_counter()
        response = await call_next(request)
        elapsed = time.perf_counter() - started

        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "SAMEORIGIN"
        response.headers["Referrer-Policy"] = "same-origin"
        response.headers["X-Response-Time-Ms"] = f"{elapsed * 1000:.1f}"

        # The console is entirely self-contained, so a strict CSP costs nothing
        # and removes script injection as a route into an operator's session.
        # 'unsafe-inline' for styles only - the console ships inline CSS but no
        # inline scripts.
        if not request.url.path.startswith("/api/"):
            response.headers["Content-Security-Policy"] = (
                "default-src 'self'; img-src 'self' data: blob:; "
                "style-src 'self' 'unsafe-inline'; script-src 'self'; "
                "connect-src 'self' ws: wss:; frame-ancestors 'self'"
            )
        return response


def _configure_errors(app: FastAPI) -> None:
    """Map domain exceptions onto HTTP responses."""

    @app.exception_handler(AuthError)
    async def _auth_error(_request: Request, exc: AuthError) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content={"detail": str(exc)},
            headers={"WWW-Authenticate": "Bearer"},
        )

    @app.exception_handler(ConfigError)
    async def _config_error(_request: Request, exc: ConfigError) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST, content={"detail": str(exc)}
        )

    @app.exception_handler(StorageError)
    async def _storage_error(_request: Request, exc: StorageError) -> JSONResponse:
        log.error("storage_error", error=str(exc))
        return JSONResponse(
            status_code=status.HTTP_507_INSUFFICIENT_STORAGE, content={"detail": str(exc)}
        )

    @app.exception_handler(IbvapError)
    async def _domain_error(_request: Request, exc: IbvapError) -> JSONResponse:
        log.error("domain_error", error=str(exc), type=type(exc).__name__)
        # Recoverable faults are transient by definition, so they map to 503
        # and tell a caller it is worth retrying; anything else is a 500.
        code = (
            status.HTTP_503_SERVICE_UNAVAILABLE if exc.recoverable
            else status.HTTP_500_INTERNAL_SERVER_ERROR
        )
        return JSONResponse(status_code=code, content={"detail": str(exc)})


def _configure_ui(app: FastAPI, settings: Settings) -> None:
    """Serve the bundled browser console, when enabled and present."""
    if not settings.api.serve_ui:
        @app.get("/", include_in_schema=False)
        async def _root() -> RedirectResponse:
            return RedirectResponse("/api/docs")
        return

    ui_directory = Path(__file__).resolve().parent.parent / "ui"
    if not (ui_directory / "index.html").is_file():
        log.warning("ui_assets_missing", path=str(ui_directory))

        @app.get("/", include_in_schema=False)
        async def _root_fallback() -> RedirectResponse:
            return RedirectResponse("/api/docs")
        return

    app.mount("/static", StaticFiles(directory=str(ui_directory)), name="static")

    @app.get("/", include_in_schema=False)
    async def _console() -> FileResponse:
        return FileResponse(ui_directory / "index.html")


def run() -> None:  # pragma: no cover - process entry point
    """Run the API server. Used by the console script and container image."""
    import uvicorn

    settings = load_settings()
    uvicorn.run(
        create_app(settings),
        host=settings.api.host,
        port=settings.api.port,
        log_config=None,  # structlog already owns logging
        access_log=False,
    )
