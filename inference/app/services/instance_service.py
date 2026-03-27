from __future__ import annotations

from pathlib import Path

from fastapi import HTTPException

from ai_factory.core.config.loader import save_cloud_profile
from ai_factory.core.instances.manager import InstanceManager
from ai_factory.core.instances.models import InstanceManifest
from ai_factory.core.instances.store import FileInstanceStore
from inference.app.config import AppSettings
from inference.app.schemas import (
    InstanceCreateRequest,
    InstanceDeployRequest,
    InstanceDetail,
    InstanceEvaluateRequest,
    InstanceLogsResponse,
    InstanceMetricsResponse,
)


class InstanceService:
    def __init__(self, settings: AppSettings):
        self.settings = settings
        self.store = FileInstanceStore(settings.artifacts_dir)
        self.manager = InstanceManager(self.store)

    def _ensure_config_path(self, config_path: str | None) -> str:
        if not config_path:
            raise HTTPException(status_code=400, detail="config_path is required")
        path = Path(config_path)
        if not path.exists():
            raise HTTPException(status_code=404, detail=f"Config not found: {config_path}")
        return str(path)

    def _manifest_to_detail(self, manifest: InstanceManifest) -> InstanceDetail:
        snapshot = self.store.load_config_snapshot(manifest.id)
        logs = self.get_logs(manifest.id)
        metrics = self.get_metrics(manifest.id)
        detail = InstanceDetail.model_validate(manifest.model_dump(mode="json"))
        detail.config_snapshot = snapshot
        detail.logs = logs
        detail.metrics = metrics
        detail.children = self.manager.list_instances(parent_instance_id=manifest.id)
        detail.events = self.store.read_events(manifest.id)
        return detail

    def list_instances(
        self,
        *,
        instance_type: str | None = None,
        status: str | None = None,
        parent_instance_id: str | None = None,
    ) -> list[InstanceManifest]:
        return self.manager.list_instances(
            instance_type=instance_type,
            status=status,
            parent_instance_id=parent_instance_id,
        )

    def get_instance(self, instance_id: str) -> InstanceDetail:
        try:
            manifest = self.manager.get_instance(instance_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return self._manifest_to_detail(manifest)

    def create_instance(self, request: InstanceCreateRequest) -> InstanceDetail:
        config_path = self._ensure_config_path(request.config_path)
        if request.environment and request.environment.kind == "cloud" and request.environment.profile_name:
            save_cloud_profile(request.environment.profile_name, request.environment)
        manifest = self.manager.create_instance(
            config_path,
            start=request.start,
            environment_override=request.environment,
            parent_instance_id=request.parent_instance_id,
        )
        return self.get_instance(manifest.id)

    def evaluate_instance(self, instance_id: str, request: InstanceEvaluateRequest | None = None) -> InstanceDetail:
        request = request or InstanceEvaluateRequest()
        if request.config_path is not None:
            config_path = self._ensure_config_path(request.config_path)
        else:
            config_path = "configs/eval.yaml"
        manifest = self.manager.create_evaluation_instance(
            instance_id,
            config_path=config_path,
            start=request.start,
        )
        return self.get_instance(manifest.id)

    def deploy_instance(self, instance_id: str, request: InstanceDeployRequest) -> InstanceDetail:
        config_path = self._ensure_config_path(request.config_path or "configs/deploy.yaml")
        manifest = self.manager.create_deployment_instance(
            instance_id,
            target=request.target,
            config_path=config_path,
            start=request.start,
        )
        return self.get_instance(manifest.id)

    def get_logs(self, instance_id: str) -> InstanceLogsResponse:
        try:
            self.manager.get_instance(instance_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        logs = self.store.read_logs(instance_id)
        return InstanceLogsResponse(
            stdout=logs.get("stdout", ""),
            stderr=logs.get("stderr", ""),
            stdout_path=str(self.store.stdout_path(instance_id)),
            stderr_path=str(self.store.stderr_path(instance_id)),
        )

    def get_metrics(self, instance_id: str) -> InstanceMetricsResponse:
        try:
            self.manager.get_instance(instance_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        metrics = self.manager.get_metrics(instance_id)
        return InstanceMetricsResponse(
            summary=metrics.get("summary", {}),
            points=metrics.get("points", []),
        )
