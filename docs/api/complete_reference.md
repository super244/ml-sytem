# AI-Factory API Complete Reference

## Overview
AI-Factory exposes a unified FastAPI surface for lifecycle orchestration, datasets, training/evaluation control, and inference.

## Authentication
The local development server does not require auth by default. Production deployments should enforce gateway auth.

## Key Endpoints

### Health
`GET /v1/health`

### Mission Control
`GET /v1/status`

### Instances
`GET /v1/instances`
`POST /v1/instances`
`GET /v1/instances/{instance_id}`
`POST /v1/instances/{instance_id}/retry`
`POST /v1/instances/{instance_id}/cancel`

### Orchestration
`GET /v1/orchestration/runs`
`GET /v1/orchestration/runs/{run_id}`
`GET /v1/orchestration/runs/{run_id}/tasks`
`GET /v1/orchestration/runs/{run_id}/events`
`POST /v1/orchestration/runs/{run_id}/cancel`
`POST /v1/orchestration/tasks/{task_id}/retry`
`GET /v1/orchestration/summary`

### Data/Lab
`GET /v1/datasets`
`GET /v1/lab/overview`
`GET /v1/lab/provenance`

### Inference (OpenAI-compatible)
`POST /v1/chat/completions`
`POST /v1/completions`

## Example Request
```bash
curl -sS http://127.0.0.1:8000/v1/health | jq
```

## Example Response
```json
{
  "status": "ok",
  "service": "ai-factory-inference",
  "version": "0.3.0"
}
```

