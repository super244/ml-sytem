from __future__ import annotations

from types import SimpleNamespace

import pytest

import training.train as training_entrypoint
from training.src.config import ExperimentConfig
from training.src.hardware import TrainingHardware
from training.src.modeling import build_quantization_config
from training.src.preflight import build_training_preflight_report


def _minimal_config(*, use_4bit: bool = True, use_8bit: bool = False) -> ExperimentConfig:
    return ExperimentConfig.model_validate(
        {
            "run_name": "unit-run",
            "seed": 7,
            "profile_name": "unit-profile",
            "config_path": "training/configs/profiles/baseline_qlora.yaml",
            "model": {
                "base_model_name": "Qwen/Qwen2.5-Math-1.5B-Instruct",
                "use_4bit": use_4bit,
                "use_8bit": use_8bit,
            },
            "data": {},
            "training": {},
            "adapter": {},
        }
    )


def test_resolve_precision_flags_keep_mps_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    config = _minimal_config(use_4bit=False)

    monkeypatch.setattr(training_entrypoint.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        training_entrypoint.torch,
        "backends",
        SimpleNamespace(mps=SimpleNamespace(is_available=lambda: True)),
    )

    flags = training_entrypoint._resolve_precision_flags(config)

    assert flags == {"bf16": False, "fp16": False, "use_cpu": False}


def test_build_quantization_config_falls_back_without_linux_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    config = _minimal_config(use_4bit=True)

    monkeypatch.setattr(
        "training.src.modeling.detect_training_hardware",
        lambda: TrainingHardware(
            backend="mps",
            system="Darwin",
            machine="arm64",
            cuda_available=False,
            cuda_device_count=0,
            mps_available=True,
            cpu_threads=8,
            bitsandbytes_supported=False,
        ),
    )

    assert build_quantization_config(config) is None


def test_preflight_warns_for_quantized_runtime_without_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("training.src.preflight.torch.cuda.device_count", lambda: 0)
    monkeypatch.setattr(
        "training.src.preflight.detect_training_hardware",
        lambda: TrainingHardware(
            backend="mps",
            system="Darwin",
            machine="arm64",
            cuda_available=False,
            cuda_device_count=0,
            mps_available=True,
            cpu_threads=8,
            bitsandbytes_supported=False,
        ),
    )

    payload = build_training_preflight_report("training/configs/profiles/failure_aware.yaml")
    checks = {check["id"]: check for check in payload["checks"]}

    assert checks["quantization-runtime"]["status"] == "warn"
    assert "fall back to non-quantized loading" in checks["quantization-runtime"]["detail"]
