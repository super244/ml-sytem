from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_RUN_ID_TIMESTAMP_RE = re.compile(r"(?P<date>\d{8})-(?P<time>\d{6})")


def _load_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def _parse_created_at(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        normalized = str(value).replace("Z", "+00:00")
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _parse_run_id_timestamp(value: Any) -> datetime | None:
    if not value:
        return None
    match = _RUN_ID_TIMESTAMP_RE.search(str(value))
    if match is None:
        return None
    try:
        timestamp = f"{match.group('date')}-{match.group('time')}"
        return datetime.strptime(timestamp, "%Y%m%d-%H%M%S").replace(tzinfo=UTC)
    except ValueError:
        return None


def _run_recency_key(run: dict[str, Any]) -> tuple[int, datetime, str]:
    created_at = _parse_created_at(run.get("created_at"))
    if created_at is not None:
        return (3, created_at, str(run.get("run_id") or run.get("output_dir") or ""))

    run_id_timestamp = _parse_run_id_timestamp(run.get("run_id"))
    if run_id_timestamp is not None:
        return (2, run_id_timestamp, str(run.get("run_id") or ""))

    output_dir = Path(str(run.get("output_dir") or "."))
    try:
        modified_at = datetime.fromtimestamp(output_dir.stat().st_mtime, tz=UTC)
    except OSError:
        modified_at = datetime.fromtimestamp(0, tz=UTC)
    return (1, modified_at, str(output_dir))


def list_training_runs(artifacts_dir: str | Path = "artifacts") -> list[dict[str, Any]]:
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
        try:
            run_payload = json.loads(manifest_path.read_text())
        except json.JSONDecodeError:
            continue
        if not isinstance(run_payload, dict):
            continue
        metrics_path = child / "metrics" / "metrics.json"
        model_report_path = child / "metrics" / "model_report.json"
        dataset_report_path = child / "metrics" / "dataset_report.json"
        runs.append(
            {
                "run_id": run_payload.get("run_id"),
                "run_name": run_payload.get("run_name"),
                "profile_name": run_payload.get("profile_name"),
                "created_at": run_payload.get("created_at"),
                "base_model": run_payload.get("base_model"),
                "output_dir": str(child),
                "metrics": _load_json_if_exists(metrics_path),
                "model_report": _load_json_if_exists(model_report_path),
                "dataset_report": _load_json_if_exists(dataset_report_path),
            }
        )
    return runs


def latest_training_run(runs: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not runs:
        return None
    return max(runs, key=_run_recency_key)


def load_benchmark_registry(path: str | Path) -> list[dict[str, Any]]:
    registry_path = Path(path)
    if not registry_path.exists():
        return []
    try:
        import yaml

        payload = yaml.safe_load(registry_path.read_text()) or {}
        benchmarks = payload.get("benchmarks", [])
        return benchmarks if isinstance(benchmarks, list) else []
    except Exception:
        return []
