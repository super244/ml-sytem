# Inference Layer

The inference layer is a product and research backend for math-specialist serving. It exposes lazy-loaded base and fine-tuned models, structured generation controls, verification-aware reranking, metadata dashboards, caching, and telemetry.

## Core Modules

- `app/main.py`: FastAPI assembly and route registration.
- `app/routers/`: health, metadata, and generation endpoints.
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
- dataset, benchmark, run, and status metadata

## Primary Routes

- `GET /v1/health`
- `GET /v1/status`
- `GET /v1/models`
- `GET /v1/datasets`
- `GET /v1/prompts`
- `GET /v1/benchmarks`
- `GET /v1/runs`
- `POST /v1/generate`
- `POST /v1/generate/batch`
- `POST /v1/compare`
- `POST /v1/verify`

## Example Command

```bash
uvicorn inference.app.main:app --host 0.0.0.0 --port 8000 --reload
```
