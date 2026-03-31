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
- PyO3 bridge: yes

## Runtime Contracts

- Scheduler runtime: tokio
- Queue policy: bounded-priority
- UI frame budget: 120 Hz
- Quantization: 4bit, 8bit
- Quantized layout: arrow-columnar

## Rust Core

- Crate root: /Users/luca/Projects/ai-factory/ai_factory_titan
- Cargo manifest: /Users/luca/Projects/ai-factory/ai_factory_titan/Cargo.toml
- Toolchain available: yes
- PyO3 bridge stub: /Users/luca/Projects/ai-factory/ai_factory_titan/src/python.rs

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
  "supports_pyo3_bridge": true,
  "remote_execution": false,
  "cloud_provider": null,
  "cuda_compute_capability": null,
  "cuda_memory_gb": null,
  "cuda_driver_version": null,
  "silent_mode": true,
  "gpu_cap_pct": 90,
  "preferred_training_backend": "metal",
  "scheduler": {
    "runtime": "tokio",
    "queue_policy": "bounded-priority",
    "ui_frame_budget_hz": 120
  },
  "quantization": {
    "formats": [
      "4bit",
      "8bit"
    ],
    "memory_layout": "arrow-columnar"
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
    "python_bridge_stub": "/Users/luca/Projects/ai-factory/ai_factory_titan/src/python.rs"
  }
}
```
