from __future__ import annotations

from ai_factory import titan


def test_detect_titan_status_prefers_remote_cuda_when_configured(monkeypatch):
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

    status = titan.detect_titan_status("/tmp")

    assert status["backend"] == "cuda"
    assert status["mode"] == "CUDA-Hopper"
    assert status["gpu_name"] == "NVIDIA H100"
    assert status["gpu_count"] == 4
    assert status["cuda_memory_gb"] == 80.0
    assert status["cuda_compute_capability"] == "9.0"
    assert status["cloud_provider"] == "runpod"
    assert status["remote_execution"] is True
