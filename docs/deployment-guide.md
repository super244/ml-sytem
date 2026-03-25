# Deployment Guide

Atlas Math Lab ships with lightweight deployment scaffolding for a demo or pre-production style environment.

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

## Compose Flow

```bash
docker compose up --build
```

## Production Notes

- Model weights are not bundled in the repo.
- The API is currently single-operator and unauthenticated.
- Background job orchestration is intentionally out of scope for this MVP+, but the artifact and manifest conventions are ready for later queue-based orchestration.
