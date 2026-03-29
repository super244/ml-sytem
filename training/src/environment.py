from __future__ import annotations

import importlib.metadata
import os
import platform
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from ai_factory.core.artifacts import ArtifactLayout, current_git_sha, detect_run_env
from ai_factory.core.hashing import sha256_file
from ai_factory.core.io import write_json
from training.src.config import ExperimentConfig

IMPORTANT_ENV_VARS = (
    "AI_FACTORY_REPO_ROOT",
    "ARTIFACTS_DIR",
    "CUDA_VISIBLE_DEVICES",
    "HF_HOME",
    "HF_TOKEN",
    "WANDB_MODE",
    "WANDB_PROJECT",
    "MLFLOW_TRACKING_URI",
)


def _package_versions(*package_names: str) -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in package_names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _installed_packages() -> dict[str, str]:
    packages: dict[str, str] = {}
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name")
        if not name:
            continue
        packages[name] = distribution.version
    return dict(sorted(packages.items()))


def _file_snapshot(path_like: str | None) -> dict[str, Any] | None:
    if not path_like:
        return None
    path = Path(path_like)
    if not path.exists():
        return {"path": str(path), "exists": False}
    return {
        "path": str(path.resolve()),
        "exists": True,
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def collect_environment_snapshot(
    config: ExperimentConfig,
    layout: ArtifactLayout,
) -> tuple[Path, dict[str, Any]]:
    repo_root = Path(__file__).resolve().parents[2]
    runtime_env = detect_run_env()
    snapshot = {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "run_id": layout.run_id,
        "run_name": config.run_name,
        "profile_name": config.profile_name,
        "seed": config.seed,
        "git_sha": current_git_sha(repo_root),
        "python": {
            "executable": sys.executable,
            "version": sys.version,
            "version_info": list(sys.version_info[:5]),
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "platform": platform.platform(),
            "runtime": asdict(runtime_env),
        },
        "torch": {
            "version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count(),
            "cuda_version": torch.version.cuda,
            "mps_available": bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()),
        },
        "libraries": _package_versions(
            "accelerate",
            "datasets",
            "mlflow",
            "peft",
            "torch",
            "transformers",
            "trl",
            "wandb",
        ),
        "runtime_config": asdict(config.runtime),
        "files": {
            "config": _file_snapshot(config.config_path),
            "train_file": _file_snapshot(config.data.train_file),
            "eval_file": _file_snapshot(config.data.eval_file),
            "test_file": _file_snapshot(config.data.test_file),
            "pack_manifest": _file_snapshot(config.data.pack_manifest),
        },
        "environment_variables": {
            key: ("***redacted***" if "TOKEN" in key or "KEY" in key else value)
            for key, value in (
                (name, os.getenv(name))
                for name in IMPORTANT_ENV_VARS
            )
            if value is not None
        },
    }
    if config.tracking.capture_installed_packages:
        snapshot["installed_packages"] = _installed_packages()
    output_path = layout.manifests_dir / "environment_snapshot.json"
    write_json(output_path, snapshot)
    return output_path, snapshot
