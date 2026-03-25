from __future__ import annotations

from pathlib import Path
from typing import Any

from ai_factory.core.io import load_json


def load_run_summary(run_dir: str | Path) -> dict[str, Any]:
    base = Path(run_dir)
    manifest = load_json(base / "manifests" / "run_manifest.json", default={})
    metrics = load_json(base / "metrics" / "metrics.json", default={})
    return {
        "run_dir": str(base),
        "run_name": manifest.get("run_name"),
        "profile_name": manifest.get("profile_name"),
        "metrics": metrics,
        "metadata": manifest.get("metadata", {}),
    }


def compare_runs(left_run_dir: str | Path, right_run_dir: str | Path) -> dict[str, Any]:
    left = load_run_summary(left_run_dir)
    right = load_run_summary(right_run_dir)
    keys = sorted(set(left["metrics"]) | set(right["metrics"]))
    delta = {
        key: (right["metrics"].get(key, 0.0) or 0.0) - (left["metrics"].get(key, 0.0) or 0.0)
        for key in keys
        if isinstance(left["metrics"].get(key, 0.0), (int, float)) and isinstance(right["metrics"].get(key, 0.0), (int, float))
    }
    return {
        "left": left,
        "right": right,
        "delta": delta,
    }
