from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import os
from pathlib import Path
import subprocess
from typing import Any

from ai_factory.core.hashing import sha256_file, sha256_text
from ai_factory.core.io import load_json, read_jsonl, write_json, write_jsonl, write_markdown


@dataclass(frozen=True)
class RunEnv:
    python: str | None
    platform: str | None


@dataclass(frozen=True)
class ArtifactLayout:
    run_id: str
    root: Path
    logs_dir: Path
    reports_dir: Path
    metrics_dir: Path
    checkpoints_dir: Path
    manifests_dir: Path
    latest_pointer: Path


def current_git_sha(repo_root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    return result.stdout.strip() or None


def detect_run_env() -> RunEnv:
    return RunEnv(python=os.getenv("PYTHON_VERSION"), platform=os.getenv("RUNNER_OS"))


def prepare_run_layout(base_dir: str | Path, run_name: str, explicit_run_id: str | None = None) -> ArtifactLayout:
    base_path = Path(base_dir)
    run_id = explicit_run_id or f"{run_name}-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
    root = base_path / "runs" / run_id
    layout = ArtifactLayout(
        run_id=run_id,
        root=root,
        logs_dir=root / "logs",
        reports_dir=root / "reports",
        metrics_dir=root / "metrics",
        checkpoints_dir=root / "checkpoints",
        manifests_dir=root / "manifests",
        latest_pointer=base_path / "runs" / "LATEST_RUN.json",
    )
    for path in (
        layout.root,
        layout.logs_dir,
        layout.reports_dir,
        layout.metrics_dir,
        layout.checkpoints_dir,
        layout.manifests_dir,
    ):
        path.mkdir(parents=True, exist_ok=True)
    return layout


def ensure_latest_pointer(pointer_path: str | Path, target_dir: str | Path, metadata: dict[str, Any] | None = None) -> None:
    payload = {
        "target_dir": str(Path(target_dir)),
        **(metadata or {}),
    }
    write_json(pointer_path, payload)
    pointer = Path(pointer_path)
    symlink_path = pointer.parent / "latest"
    try:
        if symlink_path.is_symlink() or symlink_path.exists():
            symlink_path.unlink()
        symlink_path.symlink_to(Path(target_dir).resolve())
    except Exception:
        pass


__all__ = [
    "ArtifactLayout",
    "RunEnv",
    "current_git_sha",
    "detect_run_env",
    "ensure_latest_pointer",
    "load_json",
    "prepare_run_layout",
    "read_jsonl",
    "sha256_file",
    "sha256_text",
    "write_json",
    "write_jsonl",
    "write_markdown",
]
