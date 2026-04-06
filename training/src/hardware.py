from __future__ import annotations

import platform
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class TrainingHardware:
    backend: str
    system: str
    machine: str
    cuda_available: bool
    cuda_device_count: int
    mps_available: bool
    cpu_threads: int
    bitsandbytes_supported: bool


def detect_training_hardware() -> TrainingHardware:
    system = platform.system()
    machine = platform.machine().lower()
    cuda_available = torch.cuda.is_available()
    cuda_device_count = torch.cuda.device_count() if cuda_available else 0
    mps_backend = getattr(torch.backends, "mps", None)
    mps_available = bool(mps_backend and mps_backend.is_available())
    backend = "cuda" if cuda_available else "mps" if mps_available else "cpu"
    return TrainingHardware(
        backend=backend,
        system=system,
        machine=machine,
        cuda_available=cuda_available,
        cuda_device_count=cuda_device_count,
        mps_available=mps_available,
        cpu_threads=max(1, torch.get_num_threads()),
        bitsandbytes_supported=cuda_available and system == "Linux" and machine not in {"aarch64", "arm64"},
    )
