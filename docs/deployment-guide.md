# Deployment Guide

AI-Factory ships with lightweight deployment scaffolding for a demo or pre-production style environment.

## Included Assets

- `Dockerfile.api`
- `Dockerfile.frontend`
- `docker-compose.yml`
- `.env.example`

## Recommended Local Stack

1. Mount `artifacts/` so packaged model assets persist across restarts.
2. Set `NEXT_PUBLIC_API_BASE_URL` for the frontend.
3. Point the API at the intended model registry and artifact directories.
4. Keep telemetry and cache directories on writable storage.
5. Keep `artifacts/control_plane/` on durable writable storage so SQLite task state, heartbeats, and events survive restarts.

## Compose Flow

```bash
docker compose up --build
```

## Production Notes

- Model weights are not bundled in the repo.
- The API is single-operator by default, but OpenAI-compatible routes can be protected with `OPENAI_API_KEYS`.
- The control plane is SQLite local-first and broker-ready: durable enough for single-node production-style operation, with interfaces shaped for later Redis/Postgres adapters.
- Multiple worker processes are future-ready, but this pass only ships local and SSH/cloud executors.
