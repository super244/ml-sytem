import asyncio
import json
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from ai_factory.config import settings
from ai_factory.database import init_db
from ai_factory.api.websocket import manager
from ai_factory.api.v1 import jobs, cluster, datasets, automl, agents, models, inference, feedback
from ai_factory.simulation import start_simulation


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    if settings.simulation_mode:
        await start_simulation()
    yield


app = FastAPI(title="AI-Factory Backend", version="1.0.0", lifespan=lifespan)

replit_domain = os.environ.get("REPLIT_DEV_DOMAIN", "")
allowed_origins = [
    "http://localhost:5000",
    "http://0.0.0.0:5000",
]
if replit_domain:
    allowed_origins.append(f"https://{replit_domain}")
    allowed_origins.append("https://*.replit.dev")

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(jobs.router, prefix="/api/v1")
app.include_router(cluster.router, prefix="/api/v1")
app.include_router(datasets.router, prefix="/api/v1")
app.include_router(automl.router, prefix="/api/v1")
app.include_router(agents.router, prefix="/api/v1")
app.include_router(models.router, prefix="/api/v1")
app.include_router(inference.router, prefix="/api/v1")
app.include_router(feedback.router, prefix="/api/v1")


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.websocket("/ws/telemetry")
async def websocket_telemetry(ws: WebSocket):
    await manager.connect(ws)
    try:
        while True:
            data = await ws.receive_text()
            try:
                msg = json.loads(data)
                if msg.get("type") == "subscribe" and "topic" in msg:
                    manager.subscriptions[msg["topic"]].add(ws)
            except (json.JSONDecodeError, KeyError):
                pass
    except WebSocketDisconnect:
        manager.disconnect(ws)
    except Exception:
        manager.disconnect(ws)
