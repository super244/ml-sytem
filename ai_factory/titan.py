from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

_APPLE_BANDWIDTH_GBPS = {
    "m1 max": 400,
    "m2 max": 400,
    "m3 max": 400,
    "m4 max": 546,
    "m5 max": 614,
}


@dataclass(frozen=True)
class TitanStatus:
    silicon: str
    platform: str
    backend: str
    mode: str
    unified_memory_gb: float | None
    bandwidth_gbps: int | None
    gpu_name: str | None
    gpu_vendor: str | None
    gpu_count: int
    cpu_threads: int
    cpu_fallback_threads: int
    zero_copy_supported: bool
    supports_metal: bool
    supports_cuda: bool
    supports_mlx: bool
    supports_pyo3_bridge: bool
    remote_execution: bool
    cloud_provider: str | None
    cuda_compute_capability: str | None
    cuda_memory_gb: float | None
    cuda_driver_version: str | None
    silent_mode: bool
    gpu_cap_pct: int
    preferred_training_backend: str
    scheduler: dict[str, Any]
    quantization: dict[str, Any]
    telemetry: dict[str, Any]
    rust_core: dict[str, Any]


def _run_command(args: list[str]) -> str | None:
    try:
        completed = subprocess.run(
            args,
            capture_output=True,
            check=False,
            text=True,
            timeout=2,
        )
    except (FileNotFoundError, subprocess.SubprocessError, TimeoutError):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip() or None


def _detect_apple_gpu() -> tuple[str | None, float | None]:
    chip_name = _run_command(["sysctl", "-n", "machdep.cpu.brand_string"])
    mem_bytes = _run_command(["sysctl", "-n", "hw.memsize"])
    unified_memory_gb = None
    if mem_bytes and mem_bytes.isdigit():
        unified_memory_gb = round(int(mem_bytes) / (1024**3), 1)
    return chip_name, unified_memory_gb


def _detect_nvidia_gpu() -> dict[str, Any]:
    output = _run_command(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,compute_cap,driver_version",
            "--format=csv,noheader",
        ]
    )
    if not output:
        return {
            "name": None,
            "count": 0,
            "memory_gb": None,
            "compute_capability": None,
            "driver_version": None,
        }

    records = [line.strip() for line in output.splitlines() if line.strip()]
    first = [part.strip() for part in records[0].split(",")] if records else []
    memory_gb = None
    if len(first) > 1:
        memory_text = first[1].split()[0]
        try:
            memory_gb = round(float(memory_text) / 1024, 1)
        except ValueError:
            memory_gb = None
    return {
        "name": first[0] if first else None,
        "count": len(records),
        "memory_gb": memory_gb,
        "compute_capability": first[2] if len(first) > 2 else None,
        "driver_version": first[3] if len(first) > 3 else None,
    }


def _env_flag(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() not in {"0", "false", "no"}


def _preferred_cuda_mode(compute_capability: str | None) -> str:
    if not compute_capability:
        return "CUDA-Direct"
    try:
        major = int(compute_capability.split(".", 1)[0])
    except ValueError:
        return "CUDA-Direct"
    if major >= 9:
        return "CUDA-Hopper"
    if major >= 7:
        return "CUDA-Turing+"
    return "CUDA-Direct"


def _bandwidth_for_silicon(silicon: str | None) -> int | None:
    if not silicon:
        return None
    normalized = silicon.lower()
    for key, value in _APPLE_BANDWIDTH_GBPS.items():
        if key in normalized:
            return value
    return None


def _rust_core_manifest(repo_root: Path) -> dict[str, Any]:
    crate_root = repo_root / "ai_factory_titan"
    return {
        "crate_root": str(crate_root),
        "cargo_toml": str(crate_root / "Cargo.toml"),
        "toolchain_available": shutil.which("cargo") is not None,
        "python_bridge_stub": str(crate_root / "src" / "python.rs"),
    }


def detect_titan_status(repo_root: str | Path | None = None) -> dict[str, Any]:
    resolved_root = Path(repo_root or Path(__file__).resolve().parents[1]).resolve()
    system = platform.system()
    silicon = platform.processor() or platform.machine() or platform.platform()
    gpu_name: str | None = None
    gpu_vendor: str | None = None
    unified_memory_gb: float | None = None
    supports_metal = False
    supports_cuda = False
    supports_mlx = False
    gpu_count = 0
    cloud_provider = os.getenv("AI_FACTORY_TITAN_CLOUD_PROVIDER") or None
    remote_execution = _env_flag("AI_FACTORY_TITAN_REMOTE_EXECUTION")
    cuda_compute_capability: str | None = None
    cuda_memory_gb: float | None = None
    cuda_driver_version: str | None = None

    if system == "Darwin":
        apple_gpu, unified_memory_gb = _detect_apple_gpu()
        if apple_gpu:
            silicon = apple_gpu
            gpu_name = apple_gpu
            gpu_vendor = "Apple"
            gpu_count = 1
            supports_metal = True
            supports_mlx = True

    nvidia_gpu = _detect_nvidia_gpu()
    if nvidia_gpu["count"] > 0:
        supports_cuda = True
        cuda_compute_capability = nvidia_gpu["compute_capability"]
        cuda_memory_gb = nvidia_gpu["memory_gb"]
        cuda_driver_version = nvidia_gpu["driver_version"]
        if not supports_metal:
            gpu_name = nvidia_gpu["name"]
            gpu_vendor = "NVIDIA"
            gpu_count = int(nvidia_gpu["count"])

    remote_gpu_name = os.getenv("AI_FACTORY_TITAN_REMOTE_GPU_NAME") or None
    remote_gpu_count = os.getenv("AI_FACTORY_TITAN_REMOTE_GPU_COUNT")
    if remote_execution and remote_gpu_name and not supports_cuda:
        supports_cuda = True
        gpu_name = remote_gpu_name
        gpu_vendor = "NVIDIA"
        try:
            gpu_count = max(1, int(remote_gpu_count or "1"))
        except ValueError:
            gpu_count = 1
        cuda_compute_capability = os.getenv("AI_FACTORY_TITAN_REMOTE_GPU_COMPUTE_CAP") or None
        try:
            cuda_memory_gb = float(os.getenv("AI_FACTORY_TITAN_REMOTE_GPU_MEMORY_GB", "").strip())
        except ValueError:
            cuda_memory_gb = None
        cuda_driver_version = os.getenv("AI_FACTORY_TITAN_REMOTE_CUDA_DRIVER") or None

    force_backend = (os.getenv("AI_FACTORY_TITAN_FORCE_BACKEND") or "").strip().lower()

    backend = "cpu-fallback"
    mode = "CPU-Fallback"
    if force_backend == "cuda" and supports_cuda:
        backend = "cuda"
        mode = _preferred_cuda_mode(cuda_compute_capability)
    elif force_backend == "metal" and supports_metal:
        backend = "metal"
        mode = "Metal-Direct"
    elif supports_metal:
        backend = "metal"
        mode = "Metal-Direct"
    elif supports_cuda:
        backend = "cuda"
        mode = _preferred_cuda_mode(cuda_compute_capability)

    silent_mode = os.getenv("AI_FACTORY_TITAN_SILENT_MODE", "1" if supports_metal else "0").lower() not in {
        "0",
        "false",
        "no",
    }
    gpu_cap_pct = 90 if silent_mode else 100
    cpu_threads = os.cpu_count() or 1
    cpu_fallback_threads = min(cpu_threads, 64)

    status = TitanStatus(
        silicon=silicon,
        platform=system,
        backend=backend,
        mode=mode,
        unified_memory_gb=unified_memory_gb,
        bandwidth_gbps=_bandwidth_for_silicon(silicon),
        gpu_name=gpu_name,
        gpu_vendor=gpu_vendor,
        gpu_count=gpu_count,
        cpu_threads=cpu_threads,
        cpu_fallback_threads=cpu_fallback_threads,
        zero_copy_supported=backend == "metal",
        supports_metal=supports_metal,
        supports_cuda=supports_cuda,
        supports_mlx=supports_mlx,
        supports_pyo3_bridge=True,
        remote_execution=remote_execution,
        cloud_provider=cloud_provider,
        cuda_compute_capability=cuda_compute_capability,
        cuda_memory_gb=cuda_memory_gb,
        cuda_driver_version=cuda_driver_version,
        silent_mode=silent_mode,
        gpu_cap_pct=gpu_cap_pct,
        preferred_training_backend=backend,
        scheduler={
            "runtime": "tokio",
            "queue_policy": "bounded-priority",
            "ui_frame_budget_hz": 120,
        },
        quantization={
            "formats": ["4bit", "8bit"],
            "memory_layout": "arrow-columnar",
        },
        telemetry={
            "bridge": "pyo3",
            "target_latency_ms": 1,
            "metrics": ["thermals", "memory_pressure", "flops", "queue_depth"],
        },
        rust_core=_rust_core_manifest(resolved_root),
    )
    return asdict(status)


def build_hardware_markdown(status: dict[str, Any]) -> str:
    lines = [
        "# HARDWARE",
        "",
        "This file is generated from the local Titan hardware probe.",
        "",
        "## Titan Summary",
        "",
        f"- Silicon: {status['silicon']}",
        f"- Platform: {status['platform']}",
        f"- Backend: {status['backend']}",
        f"- Mode: {status['mode']}",
        f"- GPU: {status.get('gpu_name') or 'none detected'}",
        f"- GPU vendor: {status.get('gpu_vendor') or 'n/a'}",
        f"- GPU count: {status['gpu_count']}",
        f"- Unified memory: {status.get('unified_memory_gb') or 'n/a'} GB",
        f"- Estimated bandwidth: {status.get('bandwidth_gbps') or 'n/a'} GB/s",
        f"- CPU threads: {status['cpu_threads']}",
        f"- CPU fallback threads: {status['cpu_fallback_threads']}",
        f"- Preferred backend: {status['preferred_training_backend']}",
        f"- Remote execution: {'yes' if status['remote_execution'] else 'no'}",
        f"- Cloud provider: {status.get('cloud_provider') or 'n/a'}",
        f"- Silent mode: {'enabled' if status['silent_mode'] else 'disabled'}",
        f"- GPU cap: {status['gpu_cap_pct']}%",
        "",
        "## Capability Map",
        "",
        f"- Zero-copy path: {'yes' if status['zero_copy_supported'] else 'no'}",
        f"- Metal/MPS path: {'yes' if status['supports_metal'] else 'no'}",
        f"- CUDA path: {'yes' if status['supports_cuda'] else 'no'}",
        f"- CUDA compute capability: {status.get('cuda_compute_capability') or 'n/a'}",
        f"- CUDA memory: {status.get('cuda_memory_gb') or 'n/a'} GB",
        f"- CUDA driver: {status.get('cuda_driver_version') or 'n/a'}",
        f"- MLX path: {'yes' if status['supports_mlx'] else 'no'}",
        f"- PyO3 bridge: {'yes' if status['supports_pyo3_bridge'] else 'no'}",
        "",
        "## Runtime Contracts",
        "",
        f"- Scheduler runtime: {status['scheduler']['runtime']}",
        f"- Queue policy: {status['scheduler']['queue_policy']}",
        f"- UI frame budget: {status['scheduler']['ui_frame_budget_hz']} Hz",
        f"- Quantization: {', '.join(status['quantization']['formats'])}",
        f"- Quantized layout: {status['quantization']['memory_layout']}",
        "",
        "## Rust Core",
        "",
        f"- Crate root: {status['rust_core']['crate_root']}",
        f"- Cargo manifest: {status['rust_core']['cargo_toml']}",
        f"- Toolchain available: {'yes' if status['rust_core']['toolchain_available'] else 'no'}",
        f"- PyO3 bridge stub: {status['rust_core']['python_bridge_stub']}",
        "",
        "## Raw Probe",
        "",
        "```json",
        json.dumps(status, indent=2),
        "```",
        "",
    ]
    return "\n".join(lines)


def write_hardware_markdown(
    path: str | Path = "HARDWARE.md",
    *,
    repo_root: str | Path | None = None,
) -> Path:
    resolved_root = Path(repo_root or Path(__file__).resolve().parents[1]).resolve()
    output_path = Path(path)
    if not output_path.is_absolute():
        output_path = resolved_root / output_path
    status = detect_titan_status(resolved_root)
    output_path.write_text(build_hardware_markdown(status))
    return output_path
