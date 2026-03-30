from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from inference.app.config import get_settings
from inference.app.routers.agents import router as agents_router
from inference.app.routers.automl import router as automl_router
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
from inference.app.routers.workspace import router as workspace_router
from inference.app.services.openai_service import OpenAIError

settings = get_settings()
app = FastAPI(title=settings.title, version=settings.version)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router, prefix="/v1")
app.include_router(instances_router, prefix="/v1")
app.include_router(orchestration_router, prefix="/v1")
app.include_router(metadata_router, prefix="/v1")
app.include_router(workspace_router, prefix="/v1")
app.include_router(generation_router, prefix="/v1")
app.include_router(openai_router, prefix="/v1")
app.include_router(telemetry_router, prefix="/v1")
app.include_router(cluster_router, prefix="/v1")
app.include_router(datasets_router, prefix="/v1")
app.include_router(agents_router, prefix="/v1")
app.include_router(automl_router, prefix="/v1")
app.include_router(lab_router, prefix="/v1")

# Backward-compatible aliases.
app.include_router(health_router)
app.include_router(instances_router)
app.include_router(orchestration_router)
app.include_router(metadata_router)
app.include_router(workspace_router)
app.include_router(generation_router)


@app.exception_handler(OpenAIError)
async def openai_error_handler(_, exc: OpenAIError):
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
