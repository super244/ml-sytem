# Inference System

The inference layer is designed to support both product interaction and benchmark-serving.

## Core Concepts

- YAML-backed model registry
- lazy loading for base and specialist variants
- prompt preset library
- solver modes
- self-consistency sampling
- answer extraction
- arithmetic and symbolic verification hooks
- reranking and comparison mode
- cache and telemetry hooks

## Model Registry

`inference/configs/model_registry.yaml` defines base, fine-tuned, and specialist entries. Metadata from the registry feeds both the API and the frontend.

## Prompt Presets

`inference/configs/prompt_presets.yaml` defines reusable prompt styles such as:

- `atlas_rigorous`
- `atlas_exam`
- `atlas_verifier`
- `atlas_calculus`

## Endpoint Families

- health and status
- models, prompts, datasets, benchmarks, runs
- generate, batch-generate, compare, verify

Versioned `/v1/*` routes are the primary interface.

## Response Shape

Responses can include:

- final answer
- full answer text
- reasoning steps
- candidate list
- selected score
- verification metadata
- comparison block
- latency
- cache status
- structured JSON payload
