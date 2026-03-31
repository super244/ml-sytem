from __future__ import annotations

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

# Essential minimal endpoints
@app.get("/")
async def root():
    return {"message": "AI-Factory API", "status": "running"}

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.get("/status")
async def status():
    return {
        "title": settings.title,
        "version": settings.version,
        "status": "running",
        "uptime": "0s"
    }

# Load routers lazily - only import when endpoint is called
_router_loaded = {}

def load_router_if_needed(router_name: str):
    """Load router only on first access."""
    if router_name in _router_loaded:
        return _router_loaded[router_name]
    
    try:
        if router_name == "workspace":
            from inference.app.routers.workspace import router
            app.include_router(router, prefix="/v1")
        elif router_name == "metadata":
            from inference.app.routers.metadata import router
            app.include_router(router, prefix="/v1")
        elif router_name == "generation":
            from inference.app.routers.generation import router
            app.include_router(router, prefix="/v1")
        elif router_name == "instances":
            from inference.app.routers.instances import router
            app.include_router(router, prefix="/v1")
        elif router_name == "orchestration":
            from inference.app.routers.orchestration import router
            app.include_router(router, prefix="/v1")
        elif router_name == "datasets":
            from inference.app.routers.datasets import router
            app.include_router(router, prefix="/v1")
        elif router_name == "agents":
            from inference.app.routers.agents import router
            app.include_router(router, prefix="/v1")
        elif router_name == "automl":
            from inference.app.routers.automl import router
            app.include_router(router, prefix="/v1")
        elif router_name == "cluster":
            from inference.app.routers.cluster import router
            app.include_router(router, prefix="/v1")
        elif router_name == "telemetry":
            from inference.app.routers.telemetry import router
            app.include_router(router, prefix="/v1")
        elif router_name == "lab":
            from inference.app.routers.lab import router
            app.include_router(router, prefix="/v1")
        elif router_name == "openai":
            from inference.app.routers.openai import router
            app.include_router(router, prefix="/v1")
        
        _router_loaded[router_name] = True
        return True
    except Exception as e:
        print(f"Failed to load {router_name} router: {e}")
        return False

# Lazy loading endpoints
@app.get("/v1/workspace")
async def workspace():
    # Return instant response first
    from inference.app.workspace_minimal import get_instant_status
    return get_instant_status()

@app.get("/v1/workspace/full")
async def workspace_full():
    """Full workspace with lazy loading - only when explicitly requested."""
    load_router_if_needed("workspace")
    from inference.app.routers.workspace import workspace_overview
    return workspace_overview()

@app.get("/v1/models")
async def models():
    load_router_if_needed("metadata")
    from inference.app.routers.metadata import get_models
    return {"models": get_models()}

@app.get("/v1/datasets")
async def datasets():
    load_router_if_needed("datasets")
    from inference.app.routers.datasets import get_dataset_dashboard
    return get_dataset_dashboard()

# Exception handler
try:
    from inference.app.services.openai_service import OpenAIError
    
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
except ImportError:
    pass
