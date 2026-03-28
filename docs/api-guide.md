# API Guide

## Metadata Endpoints

- `GET /v1/health`
- `GET /v1/status`
- `GET /v1/models`
- `GET /v1/models/{model_id}`
- `GET /v1/prompts`
- `GET /v1/datasets`
- `GET /v1/benchmarks`
- `GET /v1/runs`

## Generation Endpoints

- `POST /v1/generate`
- `POST /v1/generate/batch`
- `POST /v1/compare`
- `POST /v1/verify`

## OpenAI-Compatible Endpoints

- `POST /v1/chat/completions`
- `GET /v1/usage`

`GET /v1/models` now includes both the legacy `models` key and an OpenAI-style `data` list so existing clients keep working while OpenAI clients can discover models.

## Example Generate Payload

```json
{
  "question": "Evaluate \\int_0^1 x e^{x^2} dx.",
  "model_variant": "finetuned",
  "prompt_preset": "atlas_rigorous",
  "solver_mode": "rigorous",
  "num_samples": 3,
  "use_calculator": true,
  "show_reasoning": true,
  "output_format": "text",
  "use_cache": true
}
```

## Example Compare Payload

```json
{
  "question": "Find the maximum of x(1-x)^2 on [0,1].",
  "primary_model": "calculus_specialist",
  "secondary_model": "base",
  "prompt_preset": "atlas_exam",
  "num_samples": 3
}
```
