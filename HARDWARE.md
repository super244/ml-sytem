# HARDWARE

This file is generated from the local Titan hardware probe.

## Titan Summary

- Silicon: Apple M5 Max
- Platform: Darwin
- Backend: metal
- Mode: Metal-Direct
- GPU: Apple M5 Max
- GPU vendor: Apple
- GPU count: 1
- Unified memory: 64.0 GB
- Estimated bandwidth: 614 GB/s
- CPU threads: 18
- CPU fallback threads: 18
- Preferred backend: metal
- Remote execution: no
- Cloud provider: n/a
- Silent mode: enabled
- GPU cap: 90%

## Capability Map

- Zero-copy path: yes
- Metal/MPS path: yes
- CUDA path: no
- CUDA compute capability: n/a
- CUDA memory: n/a GB
- CUDA driver: n/a
- MLX path: yes
- PyO3 bridge: no

## Runtime Contracts

- Runtime selection: python-fallback
- Runtime env var: AI_FACTORY_TITAN_RUNTIME
- Runtime source: rust-binary
- GGUF support: no
- KV cache: yes
- Sampler stack: argmax, temperature, top_k, top_p, repetition_penalty
- Scheduler runtime: crossbeam+tokio
- Queue policy: three-band-priority
- UI frame budget: 120 Hz
- Max inflight tasks: 192
- Priority bands: 3
- Quantization: q4_0, q4_k, q8_0, f16
- Quantized layout: blocked-row-major
- Layout block size: 32
- Layout bytes/block: 24

## Engine

- Architecture: llm-runtime
- Decode model: llama.cpp-inspired
- Max context: 131072
- Max batch tokens: 8192
- Cache strategy: paged_kv
- Scheduler queue depth: 256
- C++ kernels: yes

## Rust Core

- Crate root: /Users/luca/Projects/ai-factory/ai_factory_titan
- Cargo manifest: /Users/luca/Projects/ai-factory/ai_factory_titan/Cargo.toml
- Toolchain available: yes
- PyO3 bridge stub: /Users/luca/Projects/ai-factory/ai_factory_titan/src/python.rs
- Build script: /Users/luca/Projects/ai-factory/ai_factory_titan/build.rs
- GGUF module: /Users/luca/Projects/ai-factory/ai_factory_titan/src/gguf.rs
- KV cache module: /Users/luca/Projects/ai-factory/ai_factory_titan/src/kv_cache.rs
- Runtime module: /Users/luca/Projects/ai-factory/ai_factory_titan/src/runtime.rs
- Sampler module: /Users/luca/Projects/ai-factory/ai_factory_titan/src/sampler.rs
- Status binary: /Users/luca/Projects/ai-factory/ai_factory_titan/target/debug/titan-status

## Raw Probe

```json
{
  "silicon": "Apple M5 Max",
  "platform": "Darwin",
  "backend": "metal",
  "mode": "Metal-Direct",
  "unified_memory_gb": 64.0,
  "bandwidth_gbps": 614,
  "gpu_name": "Apple M5 Max",
  "gpu_vendor": "Apple",
  "gpu_count": 1,
  "cpu_threads": 18,
  "cpu_fallback_threads": 18,
  "zero_copy_supported": true,
  "supports_metal": true,
  "supports_cuda": false,
  "supports_mlx": true,
  "supports_pyo3_bridge": false,
  "remote_execution": false,
  "cloud_provider": null,
  "cuda_compute_capability": null,
  "cuda_memory_gb": null,
  "cuda_driver_version": null,
  "silent_mode": true,
  "gpu_cap_pct": 90,
  "preferred_training_backend": "metal",
  "runtime_source": "rust-binary",
  "scheduler": {
    "runtime": "crossbeam+tokio",
    "queue_policy": "three-band-priority",
    "ui_frame_budget_hz": 120,
    "max_inflight_tasks": 192,
    "priority_bands": 3
  },
  "quantization": {
    "formats": [
      "q4_0",
      "q4_k",
      "q8_0",
      "f16"
    ],
    "memory_layout": "blocked-row-major",
    "default_layout": {
      "format": "q4_k",
      "block_size": 32,
      "bytes_per_block": 24,
      "alignment_bytes": 64,
      "storage": "blocked-row-major"
    }
  },
  "telemetry": {
    "bridge": "pyo3",
    "target_latency_ms": 1,
    "metrics": [
      "thermals",
      "memory_pressure",
      "flops",
      "queue_depth"
    ]
  },
  "rust_core": {
    "crate_root": "/Users/luca/Projects/ai-factory/ai_factory_titan",
    "cargo_toml": "/Users/luca/Projects/ai-factory/ai_factory_titan/Cargo.toml",
    "toolchain_available": true,
    "python_bridge_stub": "/Users/luca/Projects/ai-factory/ai_factory_titan/src/python.rs",
    "build_script": "/Users/luca/Projects/ai-factory/ai_factory_titan/build.rs",
    "build_rs": "/Users/luca/Projects/ai-factory/ai_factory_titan/build.rs",
    "cpp_bridge": "/Users/luca/Projects/ai-factory/ai_factory_titan/src/cpp.rs",
    "gguf_module": "/Users/luca/Projects/ai-factory/ai_factory_titan/src/gguf.rs",
    "kv_cache_module": "/Users/luca/Projects/ai-factory/ai_factory_titan/src/kv_cache.rs",
    "runtime_module": "/Users/luca/Projects/ai-factory/ai_factory_titan/src/runtime.rs",
    "sampler_module": "/Users/luca/Projects/ai-factory/ai_factory_titan/src/sampler.rs",
    "features": [
      "async-runtime",
      "cpp",
      "cuda",
      "cuda-graphs",
      "cuda-secure",
      "default",
      "distributed",
      "metal",
      "metal-cpp",
      "nccl-compat",
      "python",
      "rdma",
      "supercharged",
      "ultimate"
    ],
    "cpp_feature_available": true,
    "status_binary": "/Users/luca/Projects/ai-factory/ai_factory_titan/target/debug/titan-status",
    "status_binary_available": true
  },
  "runtime": {
    "selected": "python-fallback",
    "env_var": "AI_FACTORY_TITAN_RUNTIME",
    "runtime_flag": "AI_FACTORY_TITAN_RUNTIME",
    "runtime_enabled": false,
    "status_source": "rust-binary",
    "status_binary_available": true,
    "gguf_support": false,
    "kv_cache": {
      "enabled": true,
      "strategy": "paged-kv",
      "capacity_tokens": 2048,
      "page_size_tokens": 32,
      "stored_tokens": 0,
      "vram_usage_mb": 512
    },
    "sampler": {
      "stack": [
        "argmax",
        "temperature",
        "top_k",
        "top_p",
        "min_p",
        "repetition_penalty",
        "typical_p"
      ],
      "min_p": 0.0,
      "repetition_penalty": 1.100000023841858,
      "seed": null,
      "temperature": 0.800000011920929,
      "top_k": 40,
      "top_p": 0.949999988079071,
      "typical_p": 0.0
    },
    "sampler_stack": [
      "argmax",
      "temperature",
      "top_k",
      "top_p",
      "repetition_penalty"
    ]
  },
  "engine": {
    "architecture": "llm-runtime",
    "decode_model": "llama.cpp-inspired",
    "runtime_mode": "python-fallback",
    "runtime_ready": false,
    "runtime_reason": "Titan exposes runtime metadata today, but live generation remains Python-backed.",
    "max_context_tokens": 131072,
    "max_batch_tokens": 8192,
    "cache_strategy": "paged_kv",
    "scheduler_queue_depth": 256,
    "runtime_env": "AI_FACTORY_TITAN_RUNTIME",
    "supports_gguf": false,
    "supports_kv_cache": true,
    "supports_sampler_stack": true,
    "gguf_support": true,
    "kv_cache": true,
    "sampler_stack": [
      "argmax",
      "temperature",
      "top_k",
      "top_p",
      "min_p",
      "repetition_penalty",
      "typical_p"
    ],
    "supported_quantizations": [
      "q4_0",
      "q4_k",
      "q5_k",
      "q6_k",
      "q8_0",
      "f16",
      "b_f16"
    ],
    "default_layout": {
      "format": "q4_k",
      "block_size": 32,
      "bytes_per_block": 24,
      "alignment_bytes": 64,
      "storage": "blocked-row-major"
    },
    "acceleration": {
      "rust_fallback": true,
      "cpp_kernels": true,
      "metal_backend": true,
      "cuda_backend": true,
      "bf16_compute": true,
      "cpu_adamw": true,
      "fp8_compute": true,
      "fused_norm_silu": true,
      "paged_kv_cache": true,
      "priority_scheduler": true,
      "rayon_parallel": true
    },
    "max_batch_sequences": 64,
    "priority_bands": 3,
    "version": "0.5.0"
  }
}
```
