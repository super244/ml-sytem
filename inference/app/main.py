"""AI-Factory API server — v0.3.0.

Improvements over v0.2:
- Lifespan context manager (replaces deprecated on_event hooks).
- Structured problem+json error responses for 4xx/5xx.
- Request-ID middleware for distributed tracing.
- Compression middleware (GZip) for large JSON responses.
- Explicit router mount order (health first for fast probes).
"""

from __future__ import annotations

import logging
import sqlite3
import time
import uuid
from collections.abc import AsyncIterator, Iterable
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse

from inference.app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


# ── Lifespan ──────────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Run startup work before yielding; teardown on exit."""
    logger.info("AI-Factory API starting up (v%s)", settings.version)
    from ai_factory.core.platform.container import build_platform_container

    try:
        repo_root = Path(settings.repo_root).resolve()
        container = build_platform_container(
            repo_root=str(repo_root),
            artifacts_dir=settings.artifacts_dir,
        )
        container.orchestration.monitoring_summary()
        app.state.platform_container = container
        logger.info("Orchestration layer warm-up complete.")
    except (OSError, sqlite3.Error) as exc:
        logger.warning("Orchestration warm-up skipped (environment): %s", exc)
    yield
    logger.info("AI-Factory API shut down cleanly.")


# ── Application factory ───────────────────────────────────────────────────────


app = FastAPI(
    title=settings.title,
    version=settings.version,
    description="Unified AI Operating System — training, evaluation, and inference API.",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)


# ── Middleware (order matters — outermost = first to see request) ──────────────

app.add_middleware(GZipMiddleware, minimum_size=1024)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials="*" not in settings.cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID", "X-Response-Time"],
)


@app.middleware("http")
async def request_id_and_timing(request: Request, call_next):
    """Attach a unique request-ID and measure response latency."""
    request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    start = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start) * 1000
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Response-Time"] = f"{elapsed_ms:.1f}ms"
    return response


# ── Router registration ───────────────────────────────────────────────────────


def _register_v1_routers(application: FastAPI) -> None:
    from inference.app.routers.agents import router as agents_router
    from inference.app.routers.automl import router as automl_router
    from inference.app.routers.autonomous import router as autonomous_router
    from inference.app.routers.cluster import router as cluster_router
    from inference.app.routers.datasets import router as datasets_router
    from inference.app.routers.generation import router as generation_router
    from inference.app.routers.health import router as health_router
    from inference.app.routers.instances import router as instances_router
    from inference.app.routers.lab import router as lab_router
    from inference.app.routers.metadata import router as metadata_router
    from inference.app.routers.openai import router as openai_router
    from inference.app.routers.orchestration import router as orchestration_router
    from inference.app.routers.telemetry import router as telemetry_router
    from inference.app.routers.titan import router as titan_router
    from inference.app.routers.workspace import router as workspace_router

    # Health first — fastest startup check, no dependencies.
    routers: Iterable = (
        health_router,
        workspace_router,
        metadata_router,
        generation_router,
        instances_router,
        orchestration_router,
        datasets_router,
        agents_router,
        autonomous_router,
        automl_router,
        cluster_router,
        telemetry_router,
        lab_router,
        openai_router,
        titan_router,
    )
    for router in routers:
        application.include_router(router, prefix="/v1")


_register_v1_routers(app)


# ── Root endpoint ─────────────────────────────────────────────────────────────


@app.get("/", tags=["root"])
async def root() -> dict[str, str]:
    return {
        "message": "AI-Factory API",
        "version": settings.version,
        "status": "running",
        "docs": "/docs",
    }


# ── Exception handlers ────────────────────────────────────────────────────────


try:
    from inference.app.services.openai_service import OpenAIError

    @app.exception_handler(OpenAIError)
    async def openai_error_handler(_, exc: OpenAIError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "message": exc.message,
                    "type": exc.error_type,
                    "param": exc.param,
                    "code": exc.code,
                }
            },
            headers=exc.headers,
        )

except ImportError:
    pass


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """RFC 9457 Problem Details responses for all HTTP errors."""
    problem: dict[str, object] = {
        "type": "about:blank",
        "title": _status_title(exc.status_code),
        "status": exc.status_code,
        "detail": exc.detail,
        "instance": str(request.url),
    }
    headers = dict(exc.headers or {})
    headers["Content-Type"] = "application/problem+json"
    return JSONResponse(
        status_code=exc.status_code,
        content=problem,
        headers=headers,
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch-all so unhandled 500s also return structured JSON."""
    logger.exception("Unhandled exception for %s", request.url)
    return JSONResponse(
        status_code=500,
        content={
            "type": "about:blank",
            "title": "Internal Server Error",
            "status": 500,
            "detail": "An unexpected error occurred. Check server logs for details.",
            "instance": str(request.url),
        },
        headers={"Content-Type": "application/problem+json"},
    )


def _status_title(code: int) -> str:
    titles = {
        400: "Bad Request",
        401: "Unauthorized",
        403: "Forbidden",
        404: "Not Found",
        405: "Method Not Allowed",
        409: "Conflict",
        422: "Unprocessable Entity",
        429: "Too Many Requests",
        500: "Internal Server Error",
        502: "Bad Gateway",
        503: "Service Unavailable",
        504: "Gateway Timeout",
    }
    return titles.get(code, "Error")
