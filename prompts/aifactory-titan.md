# AI-Factory Titan Prompt

Upgrade the Titan engine into a serious local runtime that moves toward a `llama.cpp`-style architecture while staying compatible with AI-Factory's Python, API, and dashboard layers.

## Primary Goal

Make `ai_factory_titan/` feel like the beginning of a real inference core: structured runtime metadata, quantization-aware memory layouts, scheduler/runtime primitives, and optional native acceleration paths.

## Read This First

- `prompts/shared-repo-context.md`
- `ai_factory_titan/Cargo.toml`
- `ai_factory_titan/src/lib.rs`
- `ai_factory_titan/src/backend.rs`
- `ai_factory_titan/src/detect.rs`
- `ai_factory_titan/src/scheduler.rs`
- `ai_factory_titan/src/quantization.rs`
- `ai_factory_titan/src/bin/titan-status.rs`
- `ai_factory/titan.py`
- `inference/app/routers/titan.py`
- `frontend/lib/titan-schema.ts`
- `tests/test_titan_api.py`
- `tests/test_titan_probe.py`

## What "Like llama.cpp" Means Here

- Strong local-runtime introspection
- Explicit quantization formats and memory-layout contracts
- Predictable CPU fallback with vectorized hot paths
- Native backends that can grow into Metal/CUDA/C++ kernel acceleration
- Small, composable, testable primitives rather than opaque magic

## Immediate Priorities

- Add runtime descriptors for tensor layout, scheduler capacity, cache strategy, and acceleration capabilities
- Add a C or C++ bridge for performance-sensitive CPU kernels with a Rust fallback
- Make status and probe output better reflect what Titan can actually do
- Keep the Python/Titan status contract coherent for web and API consumers

## Definition Of Done

- Titan exposes clearer runtime capabilities than a stub probe.
- There is a real native acceleration foothold, even if limited in scope.
- Tests cover the new runtime/quantization/kernel behavior.
- Follow-on engine work becomes easier because the internal structure is clearer.
