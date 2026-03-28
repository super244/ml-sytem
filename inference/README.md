# Inference Layer

The inference layer is a product and research backend for math-specialist serving. It exposes lazy-loaded base and fine-tuned models, structured generation controls, verification-aware reranking, metadata dashboards, caching, and telemetry.

## Core Modules

- `app/main.py`: FastAPI assembly and route registration.
- `app/routers/`: health, metadata, and generation endpoints.
- `app/routers/orchestration.py`: additive control-plane routes for runs, tasks, events, retry/cancel, and summary views.
- `app/services/generation_service.py`: end-to-end request orchestration.
- `app/services/metadata_service.py`: models, datasets, prompts, benchmarks, and runs metadata.
- `app/model_loader.py`: YAML-backed model registry plus lazy loading.
- `app/prompts.py`: prompt preset loading and solver-mode composition.
- `app/generation.py`: candidate sampling, extraction, verification, and response shaping.
- `app/cache.py`: local request/response caching hooks.
- `app/telemetry.py`: JSONL telemetry recording.
- `configs/model_registry.yaml`: runtime model definitions.
- `configs/prompt_presets.yaml`: prompt library and example prompts.

## Supported Features

- prompt presets and solver modes
- compare-two-models mode
- batch generation
- self-consistency and candidate reranking
- reasoning visibility toggles
- structured JSON output
- answer verification endpoint
- OpenAI-compatible chat completions with optional API-key auth and in-memory rate limiting
- request usage tracking and streaming responses
- dataset, benchmark, run, and status metadata

## Primary Routes

- `GET /v1/health`
- `GET /v1/status`
- `GET /v1/models`
- `GET /v1/models/{model_id}`
- `GET /v1/datasets`
- `GET /v1/prompts`
- `GET /v1/benchmarks`
- `GET /v1/runs`
- `GET /v1/orchestration/runs`
- `GET /v1/orchestration/runs/{run_id}`
- `GET /v1/orchestration/runs/{run_id}/tasks`
- `GET /v1/orchestration/runs/{run_id}/events`
- `GET /v1/orchestration/summary`
- `POST /v1/orchestration/runs/{run_id}/cancel`
- `POST /v1/orchestration/tasks/{task_id}/retry`
- `POST /v1/generate`
- `POST /v1/generate/batch`
- `POST /v1/compare`
- `POST /v1/verify`
- `POST /v1/chat/completions`
- `GET /v1/usage`

## OpenAI-Compatible Settings

- `OPENAI_API_KEYS`: optional comma-separated bearer tokens for `/v1/chat/completions` and `/v1/usage`
- `OPENAI_RATE_LIMIT_REQUESTS_PER_MINUTE`: optional in-memory limit per API key or client IP
- `OPENAI_RATE_LIMIT_WINDOW_SECONDS`: rate-limit window, default `60`

## Example Command

```bash
uvicorn inference.app.main:app --host 0.0.0.0 --port 8000 --reload
```
