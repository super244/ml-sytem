# AI-FACTORY v2.0

## Overview
A full-stack AI operations dashboard (Mission Control) for monitoring and managing ML training jobs, models, inference, datasets, AutoML, agents, and GPU clusters. The frontend is React+TypeScript; the backend is Python/FastAPI running in simulation mode.

## Tech Stack

### Frontend
- **Framework**: React 18 + TypeScript
- **Build Tool**: Vite (with @vitejs/plugin-react-swc)
- **Routing**: React Router v6
- **UI Components**: Radix UI primitives + shadcn/ui
- **Styling**: Tailwind CSS
- **State/Data**: TanStack React Query
- **Forms**: React Hook Form + Zod
- **Charts**: Recharts
- **Animations**: Framer Motion

### Backend
- **Framework**: Python 3.11 + FastAPI
- **Database**: PostgreSQL (via SQLAlchemy 2.0 async + asyncpg)
- **WebSocket**: FastAPI native WebSocket
- **ASGI Server**: Uvicorn
- **Validation**: Pydantic v2

## Project Structure

### Frontend
- `src/pages/` — Route-level page components (Dashboard, Monitoring, ModelRegistry, InferenceChat, DatasetStudio, AutoMLExplorer, AgentMonitor, ClusterPage, Solve, SettingsPage)
- `src/components/` — Shared UI components
- `src/hooks/` — Custom React hooks
- `src/lib/` — Utility functions
- `src/data/` — Static/mock data

### Backend (`ai_factory/`)
- `ai_factory/main.py` — FastAPI app factory, lifespan, router mounting
- `ai_factory/config.py` — Pydantic Settings for env vars
- `ai_factory/database.py` — SQLAlchemy async engine, session factory
- `ai_factory/models/` — SQLAlchemy ORM models (TrainingJob, Dataset, ModelCheckpoint, ClusterNode, AgentRecord, AutoMLSearch, LineageNode, etc.)
- `ai_factory/schemas/` — Pydantic v2 request/response/websocket schemas
- `ai_factory/services/` — Business logic layer (JobService, ClusterService, DatasetService, ModelService, AutoMLService, AgentService, InferenceService)
- `ai_factory/api/v1/` — FastAPI routers for all API endpoints
- `ai_factory/api/websocket.py` — WebSocket ConnectionManager
- `ai_factory/simulation.py` — Simulation engine: database seeding + asyncio background tasks for telemetry

## API Endpoints
- `GET /health` — Health check
- `GET/POST /api/v1/jobs` — Training job listing and creation
- `GET /api/v1/jobs/{id}` — Job detail with loss history
- `POST /api/v1/jobs/{id}/stop` — Stop a running job
- `GET /api/v1/cluster/nodes` — Cluster node statuses with GPU metrics
- `GET /api/v1/datasets` — Dataset listing
- `GET /api/v1/datasets/{id}/samples` — Dataset samples (paginated)
- `GET /api/v1/automl/searches` — AutoML search results
- `GET /api/v1/automl/searches/{id}` — AutoML search detail with runs
- `GET /api/v1/agents` — Agent statuses
- `GET /api/v1/agents/decisions` — Agent decision logs
- `GET /api/v1/models` — Model registry
- `GET /api/v1/models/{id}/lineage` — Model lineage graph
- `POST /api/v1/inference/completions` — Inference completion (simulated)
- `POST /api/v1/feedback/flag` — Feedback flagging
- `WS /ws/telemetry` — WebSocket telemetry stream

## WebSocket Message Types
- `gpu_telemetry` — Every 2s per node
- `job_update` — Every 5s per running job
- `job_complete` — Emitted when a job finishes all steps
- `job_failed` — Emitted on rare random failure (0.5% chance per tick)
- `cluster_update` — Every 10s
- `agent_decision` — Periodic agent decisions
- `log_line` — Every 3s
- `automl_update` — Every 8s

## Database Migrations (Alembic)
- Config: `alembic.ini` (uses DATABASE_URL env var at runtime)
- Scripts: `ai_factory/alembic/` (env.py imports all ORM models for autogenerate)
- Versions: `ai_factory/alembic/versions/`
- Generate new migration: `python -m alembic revision --autogenerate -m "description"`
- Run migrations: `python -m alembic upgrade head`
- On first deploy the initial migration is already stamped as head

## Running the App
- **Backend**: `python -m uvicorn ai_factory.main:app --host 0.0.0.0 --port 8000` (port 8000)
- **Frontend**: `npm run dev` (port 5000, proxies /api and /ws to port 8000)
- **Build**: `npm run build`

## Replit Configuration
- Workflow "Start FastAPI backend" runs uvicorn on port 8000 (console)
- Workflow "Start application" runs `npm run dev` on port 5000 (webview)
- Workflow "Project" runs both in parallel
- Vite proxy configured to forward `/api`, `/ws`, `/health` to FastAPI backend
- PostgreSQL database stores all entities, seeded on startup with simulation data
- Simulation mode runs asyncio background tasks producing realistic telemetry streams
