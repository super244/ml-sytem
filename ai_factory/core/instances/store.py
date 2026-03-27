from __future__ import annotations

from pathlib import Path
from typing import Any

from ai_factory.core.io import load_json, read_jsonl, write_json
from ai_factory.core.instances.models import InstanceManifest, MetricPoint
from ai_factory.core.monitoring.events import InstanceEvent


class FileInstanceStore:
    def __init__(self, artifacts_dir: str | Path = "artifacts"):
        self.artifacts_dir = Path(artifacts_dir)
        self.instances_dir = self.artifacts_dir / "instances"
        self.instances_dir.mkdir(parents=True, exist_ok=True)

    def instance_dir(self, instance_id: str) -> Path:
        return self.instances_dir / instance_id

    def manifest_path(self, instance_id: str) -> Path:
        return self.instance_dir(instance_id) / "instance.json"

    def config_snapshot_path(self, instance_id: str) -> Path:
        return self.instance_dir(instance_id) / "config_snapshot.json"

    def events_path(self, instance_id: str) -> Path:
        return self.instance_dir(instance_id) / "events.jsonl"

    def stdout_path(self, instance_id: str) -> Path:
        return self.instance_dir(instance_id) / "logs" / "stdout.log"

    def stderr_path(self, instance_id: str) -> Path:
        return self.instance_dir(instance_id) / "logs" / "stderr.log"

    def current_metrics_path(self, instance_id: str) -> Path:
        return self.instance_dir(instance_id) / "metrics" / "current.json"

    def timeseries_metrics_path(self, instance_id: str) -> Path:
        return self.instance_dir(instance_id) / "metrics" / "timeseries.jsonl"

    def decision_path(self, instance_id: str) -> Path:
        return self.instance_dir(instance_id) / "reports" / "decision.json"

    def _ensure_layout(self, instance_id: str) -> None:
        for path in (
            self.instance_dir(instance_id),
            self.instance_dir(instance_id) / "logs",
            self.instance_dir(instance_id) / "metrics",
            self.instance_dir(instance_id) / "reports",
        ):
            path.mkdir(parents=True, exist_ok=True)

    def create(self, manifest: InstanceManifest, config_snapshot: dict[str, Any]) -> InstanceManifest:
        self._ensure_layout(manifest.id)
        manifest.config_snapshot_path = str(self.config_snapshot_path(manifest.id))
        self.save(manifest)
        write_json(self.config_snapshot_path(manifest.id), config_snapshot)
        self.append_event(
            manifest.id,
            InstanceEvent(
                type="instance.created",
                message=f"Created {manifest.type} instance.",
                payload={"status": manifest.status, "environment": manifest.environment.kind},
            ),
        )
        return manifest

    def save(self, manifest: InstanceManifest) -> InstanceManifest:
        self._ensure_layout(manifest.id)
        manifest.touch()
        write_json(self.manifest_path(manifest.id), manifest.model_dump(mode="json"))
        return manifest

    def load(self, instance_id: str) -> InstanceManifest:
        payload = load_json(self.manifest_path(instance_id))
        if not payload:
            raise FileNotFoundError(f"Unknown instance: {instance_id}")
        return InstanceManifest.model_validate(payload)

    def list_instances(self) -> list[InstanceManifest]:
        manifests: list[InstanceManifest] = []
        for child in sorted(self.instances_dir.iterdir()) if self.instances_dir.exists() else []:
            manifest_path = child / "instance.json"
            if not manifest_path.exists():
                continue
            manifests.append(InstanceManifest.model_validate(load_json(manifest_path)))
        return manifests

    def load_config_snapshot(self, instance_id: str) -> dict[str, Any]:
        return load_json(self.config_snapshot_path(instance_id), default={}) or {}

    def append_event(self, instance_id: str, event: InstanceEvent) -> None:
        path = self.events_path(instance_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as handle:
            handle.write(event.model_dump_json() + "\n")

    def read_events(self, instance_id: str) -> list[dict[str, Any]]:
        return read_jsonl(self.events_path(instance_id))

    def append_metric_points(self, instance_id: str, points: list[MetricPoint]) -> None:
        if not points:
            return
        path = self.timeseries_metrics_path(instance_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as handle:
            for point in points:
                handle.write(point.model_dump_json() + "\n")

    def read_metric_points(self, instance_id: str) -> list[dict[str, Any]]:
        return read_jsonl(self.timeseries_metrics_path(instance_id))

    def write_current_metrics(self, instance_id: str, metrics: dict[str, Any]) -> None:
        write_json(self.current_metrics_path(instance_id), metrics)

    def read_current_metrics(self, instance_id: str) -> dict[str, Any]:
        return load_json(self.current_metrics_path(instance_id), default={}) or {}

    def read_logs(self, instance_id: str) -> dict[str, str]:
        stdout_path = self.stdout_path(instance_id)
        stderr_path = self.stderr_path(instance_id)
        return {
            "stdout": stdout_path.read_text() if stdout_path.exists() else "",
            "stderr": stderr_path.read_text() if stderr_path.exists() else "",
        }

    def write_decision_report(self, instance_id: str, payload: dict[str, Any]) -> None:
        write_json(self.decision_path(instance_id), payload)
