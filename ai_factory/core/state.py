from __future__ import annotations

from typing import Any

from ai_factory.core.instances.models import InstanceManifest
from ai_factory.core.monitoring.collectors import collect_metrics_for_instance, collect_progress_for_instance


class LifecycleStateManager:
    def __init__(self, store, orchestration):
        self.store = store
        self.orchestration = orchestration

    def _snapshot(self, instance_id: str) -> dict[str, Any]:
        return self.store.load_config_snapshot(instance_id)

    def _live_metrics(self, manifest: InstanceManifest) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
        snapshot = self._snapshot(manifest.id)
        if not snapshot:
            return self.store.read_current_metrics(manifest.id), self.store.read_metric_points(manifest.id), {}
        summary, points, refs = collect_metrics_for_instance(manifest, snapshot, collect_gpu=False)
        stored_summary = dict(self.store.read_current_metrics(manifest.id))
        stored_summary.update({key: value for key, value in summary.items() if value is not None})
        if points:
            point_payload = [item.model_dump(mode="json") for item in points]
        else:
            point_payload = self.store.read_metric_points(manifest.id)
        return stored_summary, point_payload, refs

    def project_manifest(self, manifest: InstanceManifest) -> InstanceManifest:
        projected = self.orchestration.project_manifest(manifest)
        snapshot = self._snapshot(projected.id)
        if not snapshot:
            return projected

        progress = collect_progress_for_instance(projected, snapshot)
        if progress is not None and (projected.status in {"pending", "running"} or projected.progress is None):
            projected.progress = progress

        summary, _, refs = self._live_metrics(projected)
        if summary:
            projected.metrics_summary = summary
        if refs:
            projected.artifact_refs.update(refs)
        return projected

    def get_metrics(self, instance_id: str) -> dict[str, Any]:
        manifest = self.project_manifest(self.store.load(instance_id))
        summary, points, refs = self._live_metrics(manifest)
        if refs:
            manifest.artifact_refs.update(refs)
        return {"summary": summary or manifest.metrics_summary, "points": points}
