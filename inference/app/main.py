from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from inference.app.config import get_settings
from inference.app.routers.generation import router as generation_router
from inference.app.routers.health import router as health_router
from inference.app.routers.instances import router as instances_router
from inference.app.routers.metadata import router as metadata_router
from inference.app.routers.workspace import router as workspace_router

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
app.include_router(metadata_router, prefix="/v1")
app.include_router(workspace_router, prefix="/v1")
app.include_router(generation_router, prefix="/v1")

# Backward-compatible aliases.
app.include_router(health_router)
app.include_router(instances_router)
app.include_router(metadata_router)
app.include_router(workspace_router)
app.include_router(generation_router)
