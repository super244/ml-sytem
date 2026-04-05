from __future__ import annotations

from pathlib import Path
from typing import Any

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
    OrchestrationRecoveryResponse,
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
        try:
            resolved = path.resolve() if path.is_absolute() else (self.repo_root / path).resolve()
        except OSError as exc:
            raise HTTPException(status_code=400, detail=f"Invalid config path: {config_path}") from exc
        try:
            resolved.relative_to(self.repo_root)
        except ValueError as exc:
            raise HTTPException(
                status_code=403,
                detail=f"Config path must stay within repo root: {self.repo_root}",
            ) from exc
        if resolved.is_dir():
            raise HTTPException(status_code=400, detail=f"Config path must point to a file: {config_path}")
        if resolved.is_file():
            return str(resolved)
        raise HTTPException(status_code=404, detail=f"Config not found: {config_path}")

    def _detail(self, instance_id: str) -> InstanceDetail:
        manifest = self.control.get_instance(instance_id)
        run_target = manifest.orchestration_run_id or manifest.id
        config_snapshot = {}
        try:
            config_snapshot = self.store.load_config_snapshot(instance_id)
        except FileNotFoundError:
            config_snapshot = {}
        payload = {
            **manifest.model_dump(mode="json"),
            "config_snapshot": config_snapshot,
            "logs": self.control.get_logs(instance_id).model_dump(mode="json"),
            "metrics": self.control.get_metrics(instance_id).model_dump(mode="json"),
            "children": [item.model_dump(mode="json") for item in self.control.get_children(instance_id)],
            "events": self.store.read_events(instance_id),
            "available_actions": self.manager.get_available_actions(instance_id),
            "tasks": self.control.list_tasks(run_target),
            "orchestration_events": self.control.list_orchestration_events(run_target, limit=200),
            "orchestration_summary": self.control.monitoring_summary(),
        }
        return InstanceDetail.model_validate(payload)

    def _run_task_summary(self, run_id: str) -> dict[str, Any]:
        tasks = self.control.list_tasks(run_id)
        status_counts: dict[str, int] = {}
        for task in tasks:
            status = task["status"] if isinstance(task, dict) else task.status
            status_counts[status] = status_counts.get(status, 0) + 1
        return {
            "task_count": len(tasks),
            "task_status_counts": status_counts,
            "ready_tasks": status_counts.get("ready", 0) + status_counts.get("queued", 0),
            "running_tasks": status_counts.get("running", 0),
            "blocked_tasks": status_counts.get("blocked", 0),
            "retry_waiting_tasks": status_counts.get("retry_waiting", 0),
            "failed_tasks": status_counts.get("failed", 0) + status_counts.get("dead_lettered", 0),
        }

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
        try:
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
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return self._detail(manifest.id)

    def evaluate_instance(self, instance_id: str, request: InstanceEvaluateRequest | None = None) -> InstanceDetail:
        request = request or InstanceEvaluateRequest()
        config_path = self._ensure_config_path(request.config_path or "configs/eval.yaml")
        try:
            manifest = self.control.create_evaluation_instance(
                instance_id,
                config_path=config_path,
                start=request.start,
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return self._detail(manifest.id)

    def deploy_instance(self, instance_id: str, request: InstanceDeployRequest) -> InstanceDetail:
        config_path = self._ensure_config_path(request.config_path or "configs/deploy.yaml")
        try:
            manifest = self.control.create_deployment_instance(
                instance_id,
                target=request.target,
                config_path=config_path,
                start=request.start,
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return self._detail(manifest.id)

    def start_inference_instance(
        self,
        instance_id: str,
        request: InstanceInferenceRequest | None = None,
    ) -> InstanceDetail:
        request = request or InstanceInferenceRequest()
        config_path = self._ensure_config_path(request.config_path or "configs/inference.yaml")
        try:
            manifest = self.control.create_inference_instance(
                instance_id,
                config_path=config_path,
                start=request.start,
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
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
            manifest = self.control.get_instance(instance_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        run_target = manifest.orchestration_run_id or manifest.id
        snapshot = {
            "instance": manifest.model_dump(mode="json"),
            "logs": self.control.get_logs(instance_id).model_dump(mode="json"),
            "metrics": self.control.get_metrics(instance_id).model_dump(mode="json"),
            "events": self.store.read_events(instance_id),
            "tasks": self.control.list_tasks(run_target),
            "available_actions": self.manager.get_available_actions(instance_id),
            "orchestration_summary": self.control.monitoring_summary(),
        }
        return InstanceStreamResponse.model_validate(snapshot)

    def list_orchestration_runs(
        self,
        *,
        status: str | None = None,
        limit: int | None = None,
    ) -> OrchestrationRunListResponse:
        runs = self.control.list_orchestration_runs()
        if status is not None:
            runs = [run for run in runs if (run["status"] if isinstance(run, dict) else run.status) == status]
        if limit is not None:
            runs = runs[:limit]
        return OrchestrationRunListResponse(runs=runs)

    def get_orchestration_run(self, run_id: str) -> OrchestrationRunDetail:
        try:
            payload = self.control.get_orchestration_run(run_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        run_payload = payload.get("run")
        resolved_run_id = run_payload["id"] if isinstance(run_payload, dict) else run_payload.id
        payload["summary"] = {**payload.get("summary", {}), **self._run_task_summary(resolved_run_id)}
        return OrchestrationRunDetail.model_validate(payload)

    def list_orchestration_tasks(
        self,
        run_id: str,
        *,
        status: str | None = None,
        task_type: str | None = None,
        agent_type: str | None = None,
        limit: int | None = None,
    ) -> OrchestrationTaskListResponse:
        try:
            self.control.get_orchestration_run(run_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        tasks = self.control.list_tasks(run_id)
        if status is not None:
            tasks = [task for task in tasks if (task["status"] if isinstance(task, dict) else task.status) == status]
        if task_type is not None:
            tasks = [task for task in tasks if (task["task_type"] if isinstance(task, dict) else task.task_type) == task_type]
        if agent_type is not None:
            tasks = [task for task in tasks if (task["agent_type"] if isinstance(task, dict) else task.agent_type) == agent_type]
        if limit is not None:
            tasks = tasks[:limit]
        return OrchestrationTaskListResponse(tasks=tasks)

    def list_orchestration_events(
        self,
        run_id: str,
        *,
        limit: int | None = None,
        event_type: str | None = None,
        level: str | None = None,
    ) -> OrchestrationEventListResponse:
        try:
            self.control.get_orchestration_run(run_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        events = self.control.list_orchestration_events(run_id, limit=limit)
        if event_type is not None:
            events = [
                event
                for event in events
                if (event["event_type"] if isinstance(event, dict) else event.event_type) == event_type
            ]
        if level is not None:
            events = [event for event in events if (event["level"] if isinstance(event, dict) else event.level) == level]
        return OrchestrationEventListResponse(events=events)

    def cancel_orchestration_run(self, run_id: str) -> OrchestrationRunDetail:
        try:
            self.control.cancel_instance(run_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return self.get_orchestration_run(run_id)

    def recover_stalled_orchestration_tasks(self) -> OrchestrationRecoveryResponse:
        recovered = self.manager.orchestration.recover_stalled_tasks()
        recovered_ids = [task.id for task in recovered]
        return OrchestrationRecoveryResponse(
            recovered_task_ids=recovered_ids,
            recovered_count=len(recovered_ids),
            summary=self.control.monitoring_summary(),
        )

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
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return self.get_orchestration_run(task.run_id)

    def get_orchestration_summary(self) -> OrchestrationSummaryResponse:
        return OrchestrationSummaryResponse(summary=self.control.monitoring_summary())

    def get_foundation_overview(self) -> FoundationOverviewResponse:
        return FoundationOverviewResponse.model_validate(self.control.describe_foundation().model_dump(mode="json"))
