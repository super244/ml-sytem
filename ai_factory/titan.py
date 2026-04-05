from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
from dataclasses import asdict, dataclass
from importlib.util import find_spec
from pathlib import Path
from typing import Any, cast

import tomllib

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
    runtime_source: str
    scheduler: dict[str, Any]
    quantization: dict[str, Any]
    telemetry: dict[str, Any]
    rust_core: dict[str, Any]
    runtime: dict[str, Any]
    engine: dict[str, Any]


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return cast(dict[str, Any], value)
    return {}


def _run_command(args: list[str]) -> str | None:
    if not args:
        return None
    executable = shutil.which(args[0])
    if executable is None:
        return None
    safe_args = [executable, *args[1:]]
    try:
        completed = subprocess.run(  # nosec B603 - fixed hardware probe commands with absolute executable path
            safe_args,
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


def _run_json_command(args: list[str]) -> dict[str, Any] | None:
    output = _run_command(args)
    if not output:
        return None
    try:
        payload = json.loads(output)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


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


def _default_layout() -> dict[str, Any]:
    return {
        "format": "q4_k",
        "block_size": 32,
        "bytes_per_block": 24,
        "alignment_bytes": 64,
        "storage": "blocked-row-major",
    }


def _load_cargo_features(crate_root: Path) -> list[str]:
    cargo_toml = crate_root / "Cargo.toml"
    if not cargo_toml.exists():
        return []
    try:
        parsed = tomllib.loads(cargo_toml.read_text())
    except (tomllib.TOMLDecodeError, OSError):
        return []
    features = parsed.get("features", {})
    if not isinstance(features, dict):
        return []
    return sorted(str(name) for name in features.keys())


def _find_titan_status_binary(repo_root: Path) -> Path | None:
    crate_root = repo_root / "ai_factory_titan"
    candidates = [
        crate_root / "target" / profile / binary_name
        for profile in ("debug", "release")
        for binary_name in ("titan-status", "titan-status.exe")
    ]
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def _rust_core_manifest(repo_root: Path) -> dict[str, Any]:
    crate_root = repo_root / "ai_factory_titan"
    status_binary = _find_titan_status_binary(repo_root)
    features = _load_cargo_features(crate_root)
    return {
        "crate_root": str(crate_root),
        "cargo_toml": str(crate_root / "Cargo.toml"),
        "toolchain_available": shutil.which("cargo") is not None,
        "python_bridge_stub": str(crate_root / "src" / "python.rs"),
        "build_script": str(crate_root / "build.rs"),
        "build_rs": str(crate_root / "build.rs"),
        "cpp_bridge": str(crate_root / "src" / "cpp.rs"),
        "gguf_module": str(crate_root / "src" / "gguf.rs"),
        "kv_cache_module": str(crate_root / "src" / "kv_cache.rs"),
        "runtime_module": str(crate_root / "src" / "runtime.rs"),
        "sampler_module": str(crate_root / "src" / "sampler.rs"),
        "features": features,
        "cpp_feature_available": "cpp" in features,
        "status_binary": str(status_binary) if status_binary else None,
        "status_binary_available": status_binary is not None,
    }


def _detect_pyo3_bridge_support() -> bool:
    if _env_flag("AI_FACTORY_TITAN_ENABLE_PYO3_BRIDGE"):
        return True
    try:
        return find_spec("ai_factory_titan_py") is not None
    except (ImportError, ValueError):
        return False


def _source_engine_manifest(repo_root: Path) -> dict[str, Any]:
    crate_root = repo_root / "ai_factory_titan" / "src"
    features = set(_load_cargo_features(repo_root / "ai_factory_titan"))
    runtime_selected = (os.getenv("AI_FACTORY_TITAN_RUNTIME") or "python").strip().lower() or "python"
    runtime_mode = (
        "rust-primary"
        if runtime_selected in {"rust", "rust-primary"}
        else "rust-canary"
        if runtime_selected == "rust-canary"
        else "python-fallback"
    )
    return {
        "architecture": "llm-runtime",
        "decode_model": "llama.cpp-inspired",
        "runtime_mode": runtime_mode,
        "runtime_ready": False,
        "runtime_reason": "Titan exposes runtime metadata today, but live generation remains Python-backed.",
        "max_context_tokens": 8192,
        "max_batch_tokens": 2048,
        "cache_strategy": "paged_kv",
        "scheduler_queue_depth": 64,
        "runtime_env": "AI_FACTORY_TITAN_RUNTIME",
        "supports_gguf": (crate_root / "gguf.rs").exists(),
        "supports_kv_cache": (crate_root / "kv_cache.rs").exists(),
        "supports_sampler_stack": (crate_root / "sampler.rs").exists(),
        "gguf_support": (crate_root / "gguf.rs").exists(),
        "kv_cache": (crate_root / "kv_cache.rs").exists(),
        "sampler_stack": ["argmax", "temperature", "top_k", "top_p", "repetition_penalty"],
        "supported_quantizations": ["q4_0", "q4_k", "q8_0", "f16"],
        "default_layout": _default_layout(),
        "acceleration": {
            "rust_fallback": True,
            "cpp_kernels": "cpp" in features,
            "metal_backend": "metal" in features,
            "cuda_backend": "cuda" in features,
        },
    }


def _load_rust_status(repo_root: Path) -> dict[str, Any] | None:
    if _env_flag("AI_FACTORY_TITAN_DISABLE_STATUS_BINARY"):
        return None
    binary = _find_titan_status_binary(repo_root)
    if binary is None:
        return None
    return _run_json_command([str(binary)])


def _merge_dict(base: dict[str, Any], overlay: dict[str, Any] | None) -> dict[str, Any]:
    if not overlay:
        return base
    merged = dict(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


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

    rust_core = _rust_core_manifest(resolved_root)
    rust_status = _load_rust_status(resolved_root)
    source_engine = _source_engine_manifest(resolved_root)
    rust_engine = rust_status.get("engine") if rust_status else None
    engine = _merge_dict(source_engine, rust_engine)

    scheduler = {
        "runtime": "tokio",
        "queue_policy": "bounded-priority",
        "ui_frame_budget_hz": 120,
        "max_inflight_tasks": 64,
        "priority_bands": 3,
    }
    if rust_status and isinstance(rust_status.get("scheduler"), dict):
        rust_scheduler = dict(rust_status["scheduler"])
        scheduler = _merge_dict(
            scheduler,
            {
                "runtime": rust_scheduler.get("runtime"),
                "queue_policy": rust_scheduler.get("queue_policy"),
                "ui_frame_budget_hz": rust_scheduler.get("ui_budget_hz", rust_scheduler.get("ui_frame_budget_hz")),
                "max_inflight_tasks": rust_scheduler.get("max_inflight_tasks"),
                "priority_bands": rust_scheduler.get("priority_bands"),
            },
        )

    quantization = {
        "formats": ["4bit", "8bit", "q4_k", "q8_0", "f16"],
        "memory_layout": _default_layout()["storage"],
        "default_layout": _default_layout(),
    }
    if rust_status and isinstance(rust_status.get("quantization"), dict):
        quantization = _merge_dict(
            quantization,
            {
                "formats": rust_status["quantization"].get("formats"),
                "default_layout": rust_status["quantization"].get("layout"),
                "memory_layout": (rust_status["quantization"].get("layout") or {}).get("storage"),
            },
        )

    telemetry = {
        "bridge": "pyo3",
        "target_latency_ms": 1,
        "metrics": ["thermals", "memory_pressure", "flops", "queue_depth"],
    }

    runtime_selected = (os.getenv("AI_FACTORY_TITAN_RUNTIME") or "python").strip().lower() or "python"
    runtime = {
        "selected": runtime_selected,
        "env_var": "AI_FACTORY_TITAN_RUNTIME",
        "runtime_flag": "AI_FACTORY_TITAN_RUNTIME",
        "runtime_enabled": False,
        "status_source": "rust-binary" if rust_status else "python-probe",
        "status_binary_available": rust_core["status_binary_available"],
        "gguf_support": bool(engine.get("supports_gguf", engine.get("gguf_support"))),
        "kv_cache": {
            "enabled": bool(engine.get("supports_kv_cache", engine.get("kv_cache"))),
            "strategy": engine.get("cache_strategy") or "paged_kv",
        },
        "sampler": {
            "stack": list(engine.get("sampler_stack") or []),
        },
        "sampler_stack": list(engine.get("sampler_stack") or []),
    }
    if rust_status and isinstance(rust_status.get("runtime"), dict):
        runtime = _merge_dict(runtime, rust_status["runtime"])
    engine = _merge_dict(
        engine,
        {
            "runtime_mode": (
                "rust-primary"
                if runtime.get("selected") in {"rust", "rust-primary"}
                else "rust-canary"
                if runtime.get("selected") == "rust-canary"
                else "python-fallback"
            ),
            "runtime_ready": bool(runtime.get("runtime_enabled")),
            "runtime_reason": str(runtime.get("reason") or engine.get("runtime_reason") or ""),
            "supports_gguf": bool(runtime.get("gguf_support")),
            "supports_kv_cache": bool(runtime.get("kv_cache")),
            "supports_sampler_stack": bool(runtime.get("sampler_stack")),
        },
    )

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
        supports_pyo3_bridge=_detect_pyo3_bridge_support(),
        remote_execution=remote_execution,
        cloud_provider=cloud_provider,
        cuda_compute_capability=cuda_compute_capability,
        cuda_memory_gb=cuda_memory_gb,
        cuda_driver_version=cuda_driver_version,
        silent_mode=silent_mode,
        gpu_cap_pct=gpu_cap_pct,
        preferred_training_backend=backend,
        runtime_source=runtime.get("status_source", "python-probe"),
        scheduler=scheduler,
        quantization=quantization,
        telemetry=telemetry,
        rust_core=rust_core,
        runtime=runtime,
        engine=engine,
    )
    return asdict(status)


def build_hardware_markdown(status: dict[str, Any]) -> str:
    runtime = status.get("runtime") or {}
    engine = status.get("engine") or {}
    default_layout = (status.get("quantization") or {}).get("default_layout") or {}
    kv_cache = runtime.get("kv_cache")
    kv_cache_enabled = kv_cache if isinstance(kv_cache, bool) else bool((kv_cache or {}).get("enabled", kv_cache))
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
        f"- Runtime selection: {runtime.get('selected') or 'python'}",
        f"- Runtime env var: {runtime.get('env_var') or 'AI_FACTORY_TITAN_RUNTIME'}",
        f"- Runtime source: {runtime.get('status_source') or 'python-probe'}",
        f"- GGUF support: {'yes' if runtime.get('gguf_support') else 'no'}",
        f"- KV cache: {'yes' if kv_cache_enabled else 'no'}",
        f"- Sampler stack: {', '.join(runtime.get('sampler_stack') or ['argmax'])}",
        f"- Scheduler runtime: {status['scheduler']['runtime']}",
        f"- Queue policy: {status['scheduler']['queue_policy']}",
        f"- UI frame budget: {status['scheduler']['ui_frame_budget_hz']} Hz",
        f"- Max inflight tasks: {status['scheduler'].get('max_inflight_tasks') or 'n/a'}",
        f"- Priority bands: {status['scheduler'].get('priority_bands') or 'n/a'}",
        f"- Quantization: {', '.join(status['quantization']['formats'])}",
        f"- Quantized layout: {status['quantization']['memory_layout']}",
        f"- Layout block size: {default_layout.get('block_size') or 'n/a'}",
        f"- Layout bytes/block: {default_layout.get('bytes_per_block') or 'n/a'}",
        "",
        "## Engine",
        "",
        f"- Architecture: {engine.get('architecture') or 'n/a'}",
        f"- Decode model: {engine.get('decode_model') or 'n/a'}",
        f"- Max context: {engine.get('max_context_tokens') or 'n/a'}",
        f"- Max batch tokens: {engine.get('max_batch_tokens') or 'n/a'}",
        f"- Cache strategy: {engine.get('cache_strategy') or 'n/a'}",
        f"- Scheduler queue depth: {engine.get('scheduler_queue_depth') or 'n/a'}",
        f"- C++ kernels: {'yes' if (engine.get('acceleration') or {}).get('cpp_kernels') else 'no'}",
        "",
        "## Rust Core",
        "",
        f"- Crate root: {status['rust_core']['crate_root']}",
        f"- Cargo manifest: {status['rust_core']['cargo_toml']}",
        f"- Toolchain available: {'yes' if status['rust_core']['toolchain_available'] else 'no'}",
        f"- PyO3 bridge stub: {status['rust_core']['python_bridge_stub']}",
        f"- Build script: {status['rust_core'].get('build_rs') or 'n/a'}",
        f"- GGUF module: {status['rust_core'].get('gguf_module') or 'n/a'}",
        f"- KV cache module: {status['rust_core'].get('kv_cache_module') or 'n/a'}",
        f"- Runtime module: {status['rust_core'].get('runtime_module') or 'n/a'}",
        f"- Sampler module: {status['rust_core'].get('sampler_module') or 'n/a'}",
        f"- Status binary: {status['rust_core'].get('status_binary') or 'n/a'}",
        "",
        "## Raw Probe",
        "",
        "```json",
        json.dumps(status, indent=2),
        "```",
        "",
    ]
    return "\n".join(lines)


def titan_diagnostics(repo_root: str | Path | None = None) -> dict[str, Any]:
    resolved_root = Path(repo_root or Path(__file__).resolve().parents[1]).resolve()
    status = detect_titan_status(resolved_root)
    rust_status = _load_rust_status(resolved_root)
    runtime = _as_dict(status.get("runtime"))
    engine = _as_dict(status.get("engine"))
    return {
        "status": status,
        "rust_status": rust_status,
        "runtime": {
            "selected": runtime.get("selected"),
            "status_source": runtime.get("status_source"),
            "status_binary_available": runtime.get("status_binary_available"),
            "gguf_support": runtime.get("gguf_support"),
            "kv_cache": runtime.get("kv_cache"),
            "sampler_stack": runtime.get("sampler_stack"),
            "canary_generation_requested": runtime.get("selected") in {"rust", "rust-primary", "rust-canary"},
            "canary_generation_enabled": _env_flag("AI_FACTORY_TITAN_ENABLE_CANARY_GENERATION"),
        },
        "engine": {
            "decode_model": engine.get("decode_model"),
            "cache_strategy": engine.get("cache_strategy"),
            "runtime_mode": engine.get("runtime_mode"),
            "runtime_ready": engine.get("runtime_ready"),
            "supports_gguf": engine.get("supports_gguf", engine.get("gguf_support")),
            "supports_kv_cache": engine.get("supports_kv_cache", engine.get("kv_cache")),
            "sampler_stack": engine.get("sampler_stack"),
        },
    }


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
