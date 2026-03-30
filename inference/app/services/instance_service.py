from __future__ import annotations

from pathlib import Path

from fastapi import HTTPException

from ai_factory.core.config.loader import save_cloud_profile
from ai_factory.core.control.service import FactoryControlService
from ai_factory.core.platform.container import build_platform_container
from inference.app.config import AppSettings
from inference.app.schemas import (
    FoundationOverviewResponse,
    InstanceActionRequest,
    InstanceCreateRequest,
    InstanceDeployRequest,
    InstanceDetail,
    InstanceEvaluateRequest,
    InstanceInferenceRequest,
    InstanceLogsResponse,
    InstanceMetricsResponse,
    InstanceStreamResponse,
    OrchestrationEventListResponse,
    OrchestrationRunDetail,
    OrchestrationRunListResponse,
    OrchestrationSummaryResponse,
    OrchestrationTaskListResponse,
)


class InstanceService:
    def __init__(
        self,
        settings: AppSettings,
        *,
        control_service: FactoryControlService | None = None,
    ):
        self.settings = settings
        self.repo_root = Path(settings.repo_root).resolve()
        self.control = (
            control_service
            or build_platform_container(
                repo_root=self.repo_root,
                artifacts_dir=settings.artifacts_dir,
            ).control_service
        )
        self.store = self.control.store
        self.manager = self.control.manager

    def _ensure_config_path(self, config_path: str | None) -> str:
        if not config_path:
            raise HTTPException(status_code=400, detail="config_path is required")
        path = Path(config_path)
        resolved = path.resolve() if path.is_absolute() else (self.repo_root / path).resolve()
        if resolved.exists():
            return str(resolved)
        raise HTTPException(status_code=404, detail=f"Config not found: {config_path}")

    def _detail(self, instance_id: str) -> InstanceDetail:
        return InstanceDetail.model_validate(self.control.get_instance_detail(instance_id).model_dump(mode="json"))

    def list_instances(
        self,
        *,
        instance_type: str | None = None,
        status: str | None = None,
        parent_instance_id: str | None = None,
    ):
        return self.control.list_instances(
            instance_type=instance_type,
            status=status,
            parent_instance_id=parent_instance_id,
        )

    def get_instance(self, instance_id: str) -> InstanceDetail:
        try:
            self.control.get_instance(instance_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return self._detail(instance_id)

    def create_instance(self, request: InstanceCreateRequest) -> InstanceDetail:
        config_path = self._ensure_config_path(request.config_path)
        if request.environment and request.environment.kind == "cloud" and request.environment.profile_name:
            save_cloud_profile(request.environment.profile_name, request.environment)
        manifest = self.control.create_instance(
            config_path,
            start=request.start,
            environment_override=request.environment,
            parent_instance_id=request.parent_instance_id,
            name_override=request.name,
            user_level_override=request.user_level,
            lifecycle_override=request.lifecycle,
            subsystem_updates=request.subsystem_overrides or None,
            metadata_updates=request.metadata or None,
        )
        return self._detail(manifest.id)

    def evaluate_instance(self, instance_id: str, request: InstanceEvaluateRequest | None = None) -> InstanceDetail:
        request = request or InstanceEvaluateRequest()
        config_path = self._ensure_config_path(request.config_path or "configs/eval.yaml")
        manifest = self.control.create_evaluation_instance(
            instance_id,
            config_path=config_path,
            start=request.start,
        )
        return self._detail(manifest.id)

    def deploy_instance(self, instance_id: str, request: InstanceDeployRequest) -> InstanceDetail:
        config_path = self._ensure_config_path(request.config_path or "configs/deploy.yaml")
        manifest = self.control.create_deployment_instance(
            instance_id,
            target=request.target,
            config_path=config_path,
            start=request.start,
        )
        return self._detail(manifest.id)

    def start_inference_instance(
        self,
        instance_id: str,
        request: InstanceInferenceRequest | None = None,
    ) -> InstanceDetail:
        request = request or InstanceInferenceRequest()
        config_path = self._ensure_config_path(request.config_path or "configs/inference.yaml")
        manifest = self.control.create_inference_instance(
            instance_id,
            config_path=config_path,
            start=request.start,
        )
        return self._detail(manifest.id)

    def run_instance_action(self, instance_id: str, request: InstanceActionRequest) -> InstanceDetail:
        try:
            config_path = self._ensure_config_path(request.config_path) if request.config_path else None
            manifest = self.control.execute_action(
                instance_id,
                action=request.action,
                config_path=config_path,
                deployment_target=request.deployment_target,
                start=request.start,
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return self._detail(manifest.id)

    def get_logs(self, instance_id: str) -> InstanceLogsResponse:
        try:
            self.manager.get_instance(instance_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return InstanceLogsResponse.model_validate(self.control.get_logs(instance_id).model_dump(mode="json"))

    def get_metrics(self, instance_id: str) -> InstanceMetricsResponse:
        try:
            self.manager.get_instance(instance_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return InstanceMetricsResponse.model_validate(self.control.get_metrics(instance_id).model_dump(mode="json"))

    def get_live_snapshot(self, instance_id: str) -> InstanceStreamResponse:
        try:
            snapshot = self.control.get_live_instance_snapshot(instance_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return InstanceStreamResponse.model_validate(snapshot.model_dump(mode="json"))

    def list_orchestration_runs(self) -> OrchestrationRunListResponse:
        return OrchestrationRunListResponse(runs=self.control.list_orchestration_runs())

    def get_orchestration_run(self, run_id: str) -> OrchestrationRunDetail:
        try:
            payload = self.control.get_orchestration_run(run_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return OrchestrationRunDetail.model_validate(payload)

    def list_orchestration_tasks(self, run_id: str) -> OrchestrationTaskListResponse:
        try:
            self.control.get_orchestration_run(run_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return OrchestrationTaskListResponse(tasks=self.control.list_tasks(run_id))

    def list_orchestration_events(
        self,
        run_id: str,
        *,
        limit: int | None = None,
    ) -> OrchestrationEventListResponse:
        try:
            self.control.get_orchestration_run(run_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return OrchestrationEventListResponse(events=self.control.list_orchestration_events(run_id, limit=limit))

    def cancel_orchestration_run(self, run_id: str) -> OrchestrationRunDetail:
        try:
            self.control.cancel_instance(run_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return self.get_orchestration_run(run_id)

    def retry_orchestration_task(self, task_id: str) -> OrchestrationRunDetail:
        try:
            task = self.manager.orchestration.control_plane.get_task(task_id)
            if task is None:
                task = self.manager.orchestration.control_plane.get_task_by_legacy_instance(task_id)
            if task is None:
                raise FileNotFoundError(f"Unknown orchestration task: {task_id}")
            retry_target = task.legacy_instance_id
            if retry_target is None:
                run = self.manager.orchestration.control_plane.get_run(task.run_id)
                if run is not None and run.legacy_instance_id:
                    retry_target = run.legacy_instance_id
            if retry_target is None:
                raise FileNotFoundError(f"Unknown orchestration task: {task_id}")
            self.control.retry_instance(retry_target)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return self.get_orchestration_run(task.run_id)

    def get_orchestration_summary(self) -> OrchestrationSummaryResponse:
        return OrchestrationSummaryResponse(summary=self.control.monitoring_summary())

    def get_foundation_overview(self) -> FoundationOverviewResponse:
        return FoundationOverviewResponse.model_validate(self.control.describe_foundation().model_dump(mode="json"))
