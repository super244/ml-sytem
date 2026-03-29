from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PlatformSettings:
    repo_root: str
    artifacts_dir: str
    control_plane_dir: str
    control_db_path: str
    worker_concurrency: int
    heartbeat_interval_s: int
    stale_after_s: int
    telemetry_sink: str
    local_execution_backend: str
    cloud_execution_backend: str
    plugin_modules: tuple[str, ...]


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw in {None, ""}:
        return default
    try:
        return int(raw) if raw is not None else 0
    except ValueError:
        return default


def _split_env(name: str) -> tuple[str, ...]:
    raw = os.getenv(name, "")
    return tuple(part.strip() for part in raw.split(",") if part.strip())


def get_platform_settings(
    *,
    repo_root: str | Path | None = None,
    artifacts_dir: str | Path | None = None,
) -> PlatformSettings:
    resolved_repo_root = Path(repo_root or os.getenv("AI_FACTORY_REPO_ROOT") or Path.cwd()).resolve()
    resolved_artifacts_dir = Path(
        artifacts_dir or os.getenv("ARTIFACTS_DIR") or resolved_repo_root / "artifacts"
    ).resolve()
    control_plane_dir = Path(
        os.getenv("AI_FACTORY_CONTROL_PLANE_DIR", str(resolved_artifacts_dir / "control_plane"))
    ).resolve()
    control_db_path = Path(
        os.getenv("AI_FACTORY_CONTROL_DB_PATH", str(control_plane_dir / "control_plane.db"))
    ).resolve()
    telemetry_sink = os.getenv(
        "AI_FACTORY_CONTROL_TELEMETRY_PATH",
        str(control_plane_dir / "events.jsonl"),
    )
    return PlatformSettings(
        repo_root=str(resolved_repo_root),
        artifacts_dir=str(resolved_artifacts_dir),
        control_plane_dir=str(control_plane_dir),
        control_db_path=str(control_db_path),
        worker_concurrency=max(_int_env("AI_FACTORY_WORKER_CONCURRENCY", 4), 1),
        heartbeat_interval_s=max(_int_env("AI_FACTORY_HEARTBEAT_INTERVAL_S", 5), 1),
        stale_after_s=max(_int_env("AI_FACTORY_TASK_STALE_AFTER_S", 60), 5),
        telemetry_sink=telemetry_sink,
        local_execution_backend=os.getenv("AI_FACTORY_LOCAL_EXECUTION_BACKEND", "local"),
        cloud_execution_backend=os.getenv("AI_FACTORY_CLOUD_EXECUTION_BACKEND", "ssh"),
        plugin_modules=_split_env("AI_FACTORY_PLUGIN_MODULES"),
    )
