from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import torch
from transformers.utils import is_flash_attn_2_available

from ai_factory.core.model_scales import get_model_scale_spec
from training.src.config import load_experiment_config, resolve_path_reference
from training.src.scaling import resolve_scratch_architecture


def _status_rank(value: str) -> int:
    return {"ok": 0, "warn": 1, "error": 2}.get(value, 2)


def _make_check(
    check_id: str,
    label: str,
    status: str,
    detail: str,
    *,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "id": check_id,
        "label": label,
        "status": status,
        "detail": detail,
        "metadata": metadata or {},
    }


def _looks_like_local_reference(value: str) -> bool:
    path = Path(value).expanduser()
    return path.is_absolute() or value.startswith(("./", "../", "~/", "artifacts/", "data/", "models/"))


def _is_writable_directory(path: Path) -> bool:
    probe_dir = path if path.exists() else path.parent
    try:
        probe_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        return False
    probe_file = probe_dir / ".ai_factory_write_probe"
    try:
        probe_file.write_text("ok")
        probe_file.unlink()
        return True
    except OSError:
        return False


def _nearest_existing_path(path: Path) -> Path:
    current = path
    while not current.exists() and current != current.parent:
        current = current.parent
    return current


def _estimate_min_free_disk_gb(config) -> float:
    parameter_size_b: float | None = config.model.parameter_size_b
    if parameter_size_b is None:
        try:
            scale_spec = get_model_scale_spec(config.model.scale or config.model.target_parameters)
        except ValueError:
            scale_spec = None
        parameter_size_b = scale_spec.parameter_size_b if scale_spec is not None else None

    method = config.adapter.method.lower()
    if config.model.initialization.lower() == "scratch" or method in {"full", "sft"}:
        if parameter_size_b is None:
            return 20.0
        return max(20.0, round(parameter_size_b * max(config.training.save_total_limit, 2) * 2.0, 1))
    if parameter_size_b is None:
        return 8.0
    return max(8.0, round(parameter_size_b * 4.0, 1))


def build_training_preflight_report(config_path: str) -> dict[str, Any]:
    config = load_experiment_config(config_path)
    checks: list[dict[str, Any]] = []

    artifacts_dir = (
        resolve_path_reference(config.training.artifacts_dir, config.config_path)
        or Path(config.training.artifacts_dir).resolve()
    )
    disk_usage = shutil.disk_usage(_nearest_existing_path(artifacts_dir))
    free_gb = round(disk_usage.free / (1024**3), 1)
    threshold_gb = _estimate_min_free_disk_gb(config)
    disk_status = "ok" if free_gb >= threshold_gb else "warn"
    if free_gb < 5.0:
        disk_status = "error"

    checks.append(
        _make_check(
            "artifacts-dir",
            "Artifacts directory",
            "ok" if _is_writable_directory(artifacts_dir) else "error",
            f"Artifacts will be written under {artifacts_dir}.",
            metadata={"artifacts_dir": str(artifacts_dir)},
        )
    )
    checks.append(
        _make_check(
            "disk-space",
            "Free disk space",
            disk_status,
            f"{free_gb} GiB free at {artifacts_dir}; estimated minimum headroom is {threshold_gb} GiB.",
            metadata={"free_gb": free_gb, "recommended_min_gb": threshold_gb},
        )
    )

    train_file = resolve_path_reference(config.data.train_file, config.config_path)
    train_size_mb = round(train_file.stat().st_size / (1024 * 1024), 1) if train_file and train_file.exists() else 0.0
    checks.append(
        _make_check(
            "training-data",
            "Training data",
            "ok" if train_file and train_file.exists() else "error",
            f"Training input resolves to {train_file} ({train_size_mb} MiB)."
            if train_file
            else "Training data path is missing.",
            metadata={"train_file": str(train_file) if train_file else None, "size_mb": train_size_mb},
        )
    )

    tokenizer_path = resolve_path_reference(config.model.tokenizer_path, config.config_path)
    tokenizer_files = ("tokenizer.json", "tokenizer_config.json")
    tokenizer_ready = bool(
        tokenizer_path
        and tokenizer_path.exists()
        and all((tokenizer_path / required).exists() for required in tokenizer_files)
    )
    if config.model.initialization.lower() == "scratch":
        tokenizer_status = "ok" if tokenizer_ready else "error"
        tokenizer_detail = (
            f"Scratch training will use the local tokenizer at {tokenizer_path}."
            if tokenizer_ready
            else (
                f"Scratch training requires a local tokenizer at {tokenizer_path}. "
                f"Run `python training/scripts/train_tokenizer.py --config {config.config_path} "
                f"--output-dir {config.model.tokenizer_path}` first."
            )
        )
    else:
        tokenizer_status = "ok" if tokenizer_ready else "warn"
        tokenizer_detail = (
            f"Tokenizer resolves to the local artifact at {tokenizer_path}."
            if tokenizer_ready
            else f"Tokenizer will fall back to {config.model.tokenizer_name or config.model.base_model_name}."
        )
    checks.append(
        _make_check(
            "tokenizer",
            "Tokenizer source",
            tokenizer_status,
            tokenizer_detail,
            metadata={"tokenizer_path": str(tokenizer_path) if tokenizer_path else None, "ready": tokenizer_ready},
        )
    )

    if config.model.initialization.lower() == "scratch":
        architecture, estimate = resolve_scratch_architecture(
            model_type=config.model.model_type or "",
            architecture_overrides=config.model.architecture,
            target_parameters=config.model.target_parameters,
        )
        checks.append(
            _make_check(
                "model-source",
                "Model source",
                "ok",
                (
                    f"Scratch {config.model.model_type} model resolved to hidden={architecture.get('hidden_size')} "
                    f"layers={architecture.get('num_hidden_layers')} "
                    f"estimated_parameters={estimate or 'n/a'}."
                ),
                metadata={"estimated_parameters": estimate, "architecture": architecture},
            )
        )
    else:
        base_model_ref = config.model.base_model_name
        local_base_model = _looks_like_local_reference(base_model_ref)
        if local_base_model:
            base_model_path = resolve_path_reference(base_model_ref, config.config_path)
            base_model_ready = bool(base_model_path and base_model_path.exists())
            checks.append(
                _make_check(
                    "model-source",
                    "Model source",
                    "ok" if base_model_ready else "error",
                    f"Base model resolves to {base_model_path}."
                    if base_model_ready
                    else f"Configured local base model was not found: {base_model_ref}.",
                    metadata={"base_model_path": str(base_model_path) if base_model_path else None},
                )
            )
        else:
            checks.append(
                _make_check(
                    "model-source",
                    "Model source",
                    "warn",
                    f"Base model will be pulled from {base_model_ref}; confirm network/auth availability before launch.",
                    metadata={"base_model_ref": base_model_ref},
                )
            )

    cuda_available = torch.cuda.is_available()
    device_count = torch.cuda.device_count() if cuda_available else 0
    gpu_names = [torch.cuda.get_device_name(index) for index in range(device_count)] if cuda_available else []
    gpu_status = "ok" if cuda_available else "warn"
    if config.model.initialization.lower() == "scratch" and not cuda_available:
        gpu_status = "error"
    elif config.adapter.method.lower() in {"full", "sft"} and not cuda_available:
        gpu_status = "error"
    checks.append(
        _make_check(
            "hardware",
            "Hardware",
            gpu_status,
            f"CUDA available={cuda_available}, device_count={device_count}, devices={gpu_names or ['cpu-only']}.",
            metadata={"cuda_available": cuda_available, "device_count": device_count, "gpu_names": gpu_names},
        )
    )

    if config.model.use_flash_attention and not is_flash_attn_2_available():
        checks.append(
            _make_check(
                "flash-attention",
                "FlashAttention2",
                "warn",
                "FlashAttention2 is requested by the config but is unavailable in the current environment.",
            )
        )
    else:
        checks.append(
            _make_check(
                "flash-attention",
                "FlashAttention2",
                "ok",
                "Attention backend is aligned with the current environment.",
            )
        )

    status = max((check["status"] for check in checks), key=_status_rank, default="ok")
    summary = {
        "ok": sum(1 for check in checks if check["status"] == "ok"),
        "warn": sum(1 for check in checks if check["status"] == "warn"),
        "error": sum(1 for check in checks if check["status"] == "error"),
    }
    recommended_commands = [
        f"python -m training.train --config {config_path} --dry-run",
        f"python -m training.train --config {config_path}",
    ]
    if config.model.initialization.lower() == "scratch" and not tokenizer_ready:
        recommended_commands.insert(
            0,
            f"python training/scripts/train_tokenizer.py --config {config.config_path} "
            f"--output-dir {config.model.tokenizer_path}",
        )

    return {
        "status": status,
        "config_path": config.config_path,
        "profile_name": config.profile_name,
        "run_name": config.run_name,
        "checks": checks,
        "summary": summary,
        "recommended_commands": recommended_commands,
    }
