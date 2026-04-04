from __future__ import annotations

from ai_factory import titan


def test_detect_titan_status_prefers_remote_cuda_when_configured(monkeypatch) -> None:
    monkeypatch.setenv("AI_FACTORY_TITAN_REMOTE_EXECUTION", "1")
    monkeypatch.setenv("AI_FACTORY_TITAN_REMOTE_GPU_NAME", "NVIDIA H100")
    monkeypatch.setenv("AI_FACTORY_TITAN_REMOTE_GPU_COUNT", "4")
    monkeypatch.setenv("AI_FACTORY_TITAN_REMOTE_GPU_MEMORY_GB", "80")
    monkeypatch.setenv("AI_FACTORY_TITAN_REMOTE_GPU_COMPUTE_CAP", "9.0")
    monkeypatch.setenv("AI_FACTORY_TITAN_REMOTE_CUDA_DRIVER", "555.10")
    monkeypatch.setenv("AI_FACTORY_TITAN_CLOUD_PROVIDER", "runpod")
    monkeypatch.setenv("AI_FACTORY_TITAN_FORCE_BACKEND", "cuda")
    monkeypatch.setattr(titan, "_detect_apple_gpu", lambda: (None, None))
    monkeypatch.setattr(
        titan,
        "_detect_nvidia_gpu",
        lambda: {
            "name": None,
            "count": 0,
            "memory_gb": None,
            "compute_capability": None,
            "driver_version": None,
        },
    )
    monkeypatch.setattr(
        titan,
        "_load_rust_status",
        lambda repo_root: {
            "scheduler": {
                "runtime": "tokio",
                "queue_policy": "bounded-priority",
                "ui_budget_hz": 120,
                "max_inflight_tasks": 64,
                "priority_bands": 3,
            },
            "engine": {
                "architecture": "llm-runtime",
                "decode_model": "llama.cpp-inspired",
            },
            "runtime": {
                "selected": "rust",
                "status_source": "rust-binary",
                "gguf_support": True,
                "kv_cache": True,
                "sampler_stack": ["argmax", "top_k"],
            },
            "quantization": {
                "formats": ["q4_k", "q8_0"],
                "layout": {"storage": "blocked-row-major"},
            },
        },
    )

    status = titan.detect_titan_status("/tmp")

    assert status["backend"] == "cuda"
    assert status["mode"] == "CUDA-Hopper"
    assert status["gpu_name"] == "NVIDIA H100"
    assert status["gpu_count"] == 4
    assert status["cuda_memory_gb"] == 80.0
    assert status["cuda_compute_capability"] == "9.0"
    assert status["cloud_provider"] == "runpod"
    assert status["remote_execution"] is True
    assert status["runtime_source"] == "rust-binary"
    assert status["scheduler"]["max_inflight_tasks"] == 64
    assert status["quantization"]["memory_layout"] == "blocked-row-major"
    assert status["runtime"]["selected"] == "rust"
    assert status["runtime"]["gguf_support"] is True
    assert status["engine"]["runtime_mode"] == "rust-primary"
    assert status["engine"]["runtime_ready"] is True
    assert status["engine"]["supports_gguf"] is True
    assert status["engine"]["supports_kv_cache"] is True
    assert status["engine"]["supports_sampler_stack"] is True
