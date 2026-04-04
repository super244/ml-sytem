# AI-Factory Titan Prompt

Upgrade the **Titan Engine** into a serious, production-grade local runtime. Your objective is to move the architecture toward a `llama.cpp`-style foundation while guaranteeing seamless integration with AI-Factory's Python ecosystem, API layer, and dashboard interfaces.

## Primary Goal

Evolve `ai_factory_titan/` from a basic probe into a legitimate inference core. You must implement structured runtime metadata, explicit quantization-aware memory layouts, robust scheduler primitives, and lay the groundwork for optional native acceleration paths (SIMD, Metal, CUDA).

## Read This First (Mandatory Ingestion)

- `prompts/shared-repo-context.md`
- `ai_factory_titan/Cargo.toml`
- `ai_factory_titan/src/lib.rs` (The FFI / Python bridge boundary)
- `ai_factory_titan/src/backend.rs` (Device dispatch layer)
- `ai_factory_titan/src/detect.rs` (Hardware telemetry)
- `ai_factory_titan/src/scheduler.rs` (Batching and request scheduling)
- `ai_factory_titan/src/quantization.rs` (Format parsing: GGUF, AWQ, EXL2)
- `ai_factory_titan/src/bin/titan-status.rs` (CLI diagnostic tool)
- `ai_factory/titan.py` (Python bindings wrapper)
- `inference/app/routers/titan.py` (API exposure)
- `frontend/lib/titan-schema.ts` (Frontend contract)

## What "Like llama.cpp" Means Here (Strict Constraints)

1. **Strong Local-Runtime Introspection**: The engine must report exact hardware capabilities (AVX2, AVX512, NEON, Metal, CUDA), current VRAM usage, and active execution mode.
2. **Explicit Memory Layouts**: Zero-copy parsing via `mmap`. Strict separation between parameter storage (Q4_0, Q8_0, FP16) and KV cache blocks.
3. **Predictable CPU Fallbacks**: Every operation must have a slow-but-correct scalar fallback, alongside vectorized hot paths.
4. **C/C++ Kernel Bridge**: Establish an architecture that allows raw C/C++ or assembly kernels to be invoked seamlessly via Rust FFI without sacrificing memory safety.
5. **Small, Composable Primitives**: No opaque magic blocks. Tensor math, tokenization, and graph execution must be testable in isolation.

## Immediate Priorities & Execution Plan

1. **Runtime Descriptors**: Expand the engine descriptor structs to report explicit tensor layouts, maximum scheduler batch capacity, KV cache pagination strategies, and compiled acceleration backends.
2. **FFI Hardening**: Ensure the Python bridge (`ctypes` or `PyO3`) does not leak memory and gracefully propagates Rust `Result` types as Python exceptions.
3. **Telemetry Fidelity**: Upgrade `titan-status.rs` to output rich JSON metrics that the web dashboard can digest in real-time.
4. **Contract Coherence**: Verify that changes in Rust structs are immediately mirrored in `titan-schema.ts` and `inference/app/schemas.py`.

## Definition Of Done

- Titan correctly exposes rich runtime capabilities via API, completely replacing the old stub probe.
- At least one primitive native acceleration foothold (e.g., a basic vectorized dot product) is integrated and benchmarked.
- 100% test coverage on new quantization formats and kernel fallback logic.
- Follow-on C++ acceleration can be injected easily due to the newly established abstraction layers.