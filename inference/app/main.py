from __future__ import annotations

from collections.abc import Iterable

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from inference.app.config import get_settings

settings = get_settings()
app = FastAPI(title=settings.title, version=settings.version)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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


@app.get("/")
async def root() -> dict[str, str]:
    return {"message": "AI-Factory API", "status": "running"}


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
