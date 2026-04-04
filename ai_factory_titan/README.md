# AI-Factory Titan Engine

Titan is the systems-facing Rust core for AI-Factory. It now exposes hardware/runtime introspection, scheduler primitives, quantization metadata, a minimal GGUF/KV/sampler runtime surface, and optional C++ CPU kernel acceleration.

## Current Focus

Titan is moving toward a `llama.cpp`-style local runtime in stages:

- hardware detection and backend selection
- scheduler/runtime metadata for local execution
- quantization-aware tensor layout contracts
- GGUF header probing, paged KV cache primitives, and sampler scaffolding
- pure-Rust CPU kernels with optional C++ acceleration
- Python bridge hooks for future runtime integration

## Module Map

- `src/backend.rs`: backend identity and runtime mode metadata
- `src/detect.rs`: local hardware probing for Metal, CUDA, and CPU fallback
- `src/engine.rs`: Titan engine descriptor and acceleration capability map
- `src/runtime.rs`: runtime selection and env-driven runtime descriptor
- `src/tensor.rs`: tensor-shape and quantized-layout contracts
- `src/quantization.rs`: Arrow schema and quantization layout helpers
- `src/gguf.rs`: minimal GGUF header parsing
- `src/kv_cache.rs`: paged KV cache primitives
- `src/sampler.rs`: deterministic sampler stack primitives
- `src/cpu_kernels.rs`: CPU fallback math kernels
- `src/cpp.rs`: optional bridge to C++ kernels behind the `cpp` feature
- `src/scheduler.rs`: bounded Tokio scheduler primitive
- `src/telemetry.rs`: telemetry frame contract
- `src/python.rs`: PyO3 bridge stub for Python integration

## Building

```bash
cd ai_factory_titan
cargo build --release
```

To include the optional C++ kernel path:

```bash
cd ai_factory_titan
cargo build --release --features cpp
```

## Testing

```bash
cd ai_factory_titan
cargo test
cargo test --features cpp
```

## Near-Term Direction

Titan still does not replace the Python Transformers inference stack. The current MVP is enough to report richer runtime metadata through Python/API/frontend and to start growing a real local engine around GGUF ingestion, KV-cache management, decode loops, and backend-specific kernels.
