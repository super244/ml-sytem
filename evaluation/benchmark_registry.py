from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def _find_project_root(start: Path) -> Path | None:
    for candidate in [start, *start.parents]:
        if (candidate / "pyproject.toml").exists():
            return candidate
    return None


def _resolve_relative_path(base_path: Path, path: str) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    direct = (base_path.parent / candidate).resolve()
    if direct.exists():
        return direct

    project_root = _find_project_root(base_path.parent.resolve())
    if project_root is not None:
        rooted = (project_root / candidate).resolve()
        if rooted.exists():
            return rooted

    cwd_relative = (Path.cwd() / candidate).resolve()
    if cwd_relative.exists():
        return cwd_relative
    return direct


def load_benchmark_registry(path: str | Path) -> list[dict[str, Any]]:
    registry_path = Path(path).expanduser()
    if not registry_path.exists():
        raise FileNotFoundError(f"Benchmark registry not found: {registry_path}")
    payload = yaml.safe_load(registry_path.read_text()) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Benchmark registry must be a mapping: {registry_path}")
    benchmarks = payload.get("benchmarks", [])
    if not isinstance(benchmarks, list):
        raise ValueError(f"Benchmark registry benchmarks entry must be a list: {registry_path}")
    return benchmarks


def resolve_benchmark_file(
    registry_path: str | Path,
    benchmark_id: str | None = None,
    benchmark_file: str | None = None,
) -> tuple[str, dict[str, Any]]:
    registry_path_obj = Path(registry_path).expanduser()
    if benchmark_file:
        resolved = _resolve_relative_path(registry_path_obj, benchmark_file)
        if not resolved.exists():
            raise FileNotFoundError(f"Benchmark file not found: {benchmark_file}")
        return benchmark_file, {"id": benchmark_id or Path(benchmark_file).stem, "path": benchmark_file}
    registry = load_benchmark_registry(registry_path_obj)
    if benchmark_id is None:
        raise ValueError("Either benchmark_id or benchmark_file must be provided.")
    for entry in registry:
        if entry["id"] == benchmark_id:
            resolved = _resolve_relative_path(registry_path_obj, entry["path"])
            if not resolved.exists():
                raise FileNotFoundError(f"Benchmark file not found for {benchmark_id}: {entry['path']}")
            return entry["path"], entry
    raise KeyError(f"Unknown benchmark id: {benchmark_id}")
