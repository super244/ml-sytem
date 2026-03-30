from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_CHECKPOINT_PATTERN = re.compile(r"^checkpoint-(\d+)$")


@dataclass
class CheckpointInfo:
    path: Path
    step: int
    modified_at: float
    size_bytes: int
    has_optimizer: bool
    has_scheduler: bool
    has_adapter: bool
    trainer_state: dict[str, Any] | None = None


def _read_trainer_state(ckpt_path: Path) -> dict[str, Any] | None:
    state_file = ckpt_path / "trainer_state.json"
    if not state_file.exists():
        return None
    try:
        return json.loads(state_file.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def inspect_checkpoint(ckpt_path: Path) -> CheckpointInfo | None:
    match = _CHECKPOINT_PATTERN.match(ckpt_path.name)
    if not match or not ckpt_path.is_dir():
        return None
    step = int(match.group(1))
    stat = ckpt_path.stat()
    total_size = sum(f.stat().st_size for f in ckpt_path.rglob("*") if f.is_file())
    return CheckpointInfo(
        path=ckpt_path,
        step=step,
        modified_at=stat.st_mtime,
        size_bytes=total_size,
        has_optimizer=any(ckpt_path.glob("optimizer*")),
        has_scheduler=any(ckpt_path.glob("scheduler*")),
        has_adapter=(ckpt_path / "adapter_model.safetensors").exists() or (ckpt_path / "adapter_model.bin").exists(),
        trainer_state=_read_trainer_state(ckpt_path),
    )


def list_checkpoints(checkpoints_dir: str | Path) -> list[CheckpointInfo]:
    base = Path(checkpoints_dir)
    if not base.exists():
        return []
    results: list[CheckpointInfo] = []
    for path in sorted(base.iterdir()):
        info = inspect_checkpoint(path)
        if info is not None:
            results.append(info)
    results.sort(key=lambda c: c.step)
    return results


def find_latest_checkpoint(checkpoints_dir: str | Path) -> Path | None:
    checkpoints = list_checkpoints(checkpoints_dir)
    if not checkpoints:
        return None
    return max(checkpoints, key=lambda c: (c.step, c.modified_at)).path


def validate_checkpoint(ckpt_path: str | Path) -> list[str]:
    path = Path(ckpt_path)
    errors: list[str] = []
    if not path.exists():
        errors.append(f"Checkpoint path does not exist: {path}")
        return errors
    if not path.is_dir():
        errors.append(f"Checkpoint path is not a directory: {path}")
        return errors
    match = _CHECKPOINT_PATTERN.match(path.name)
    if not match:
        errors.append(f"Checkpoint directory name does not match pattern checkpoint-N: {path.name}")
    has_model = any(path.glob("*.safetensors")) or any(path.glob("*.bin")) or any(path.glob("pytorch_model*"))
    if not has_model:
        errors.append("No model weights found (safetensors/bin/pytorch_model)")
    state = _read_trainer_state(path)
    if state is None:
        errors.append("trainer_state.json missing or unreadable")
    return errors


def resolve_resume_checkpoint(
    checkpoint_dir: str | Path,
    explicit_checkpoint: str | None = None,
    resume_from_latest: bool = False,
) -> tuple[str | None, dict[str, Any]]:
    report: dict[str, Any] = {
        "checkpoint_dir": str(checkpoint_dir),
        "explicit_checkpoint": explicit_checkpoint,
        "resume_from_latest": resume_from_latest,
        "resolved": None,
        "validation_errors": [],
        "available_checkpoints": [],
    }
    available = list_checkpoints(checkpoint_dir)
    report["available_checkpoints"] = [
        {"step": c.step, "path": str(c.path), "size_mb": round(c.size_bytes / (1024 * 1024), 1)} for c in available
    ]
    if explicit_checkpoint:
        errors = validate_checkpoint(explicit_checkpoint)
        report["validation_errors"] = errors
        if not errors:
            report["resolved"] = explicit_checkpoint
        return (explicit_checkpoint if not errors else None), report
    if not resume_from_latest:
        return None, report
    latest = find_latest_checkpoint(checkpoint_dir)
    if latest is not None:
        errors = validate_checkpoint(latest)
        report["validation_errors"] = errors
        if not errors:
            report["resolved"] = str(latest)
            return str(latest), report
    return None, report
