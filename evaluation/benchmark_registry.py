from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_benchmark_registry(path: str | Path) -> list[dict[str, Any]]:
    payload = yaml.safe_load(Path(path).read_text()) or {}
    return payload.get("benchmarks", [])


def resolve_benchmark_file(
    registry_path: str | Path,
    benchmark_id: str | None = None,
    benchmark_file: str | None = None,
) -> tuple[str, dict[str, Any]]:
    if benchmark_file:
        return benchmark_file, {"id": benchmark_id or Path(benchmark_file).stem, "path": benchmark_file}
    registry = load_benchmark_registry(registry_path)
    if benchmark_id is None:
        raise ValueError("Either benchmark_id or benchmark_file must be provided.")
    for entry in registry:
        if entry["id"] == benchmark_id:
            return entry["path"], entry
    raise KeyError(f"Unknown benchmark id: {benchmark_id}")
