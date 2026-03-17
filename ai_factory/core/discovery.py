from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def list_training_runs(artifacts_dir: str = "artifacts") -> list[dict[str, Any]]:
    base_dir = Path(artifacts_dir) / "runs"
    if not base_dir.exists():
        return []
    runs: list[dict[str, Any]] = []
    for child in sorted(base_dir.iterdir()):
        if not child.is_dir():
            continue
        manifest_path = child / "manifests" / "run_manifest.json"
        if not manifest_path.exists():
            continue
        run_payload = json.loads(manifest_path.read_text())
        metrics_path = child / "metrics" / "metrics.json"
        model_report_path = child / "metrics" / "model_report.json"
        dataset_report_path = child / "metrics" / "dataset_report.json"
        runs.append(
            {
                "run_id": run_payload.get("run_id"),
                "run_name": run_payload.get("run_name"),
                "profile_name": run_payload.get("profile_name"),
                "base_model": run_payload.get("base_model"),
                "output_dir": str(child),
                "metrics": json.loads(metrics_path.read_text()) if metrics_path.exists() else {},
                "model_report": json.loads(model_report_path.read_text()) if model_report_path.exists() else {},
                "dataset_report": json.loads(dataset_report_path.read_text()) if dataset_report_path.exists() else {},
            }
        )
    return runs


def load_benchmark_registry(path: str | Path) -> list[dict[str, Any]]:
    registry_path = Path(path)
    if not registry_path.exists():
        return []
    try:
        import yaml

        payload = yaml.safe_load(registry_path.read_text()) or {}
        return payload.get("benchmarks", [])
    except Exception:
        return []
