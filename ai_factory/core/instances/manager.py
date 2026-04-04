from __future__ import annotations

import asyncio
import uuid
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, cast

import yaml

from ai_factory.core.config.loader import (
    build_orchestration_config,
    load_cloud_profile,
    load_orchestration_config,
    resolve_path_from_config,
)
from ai_factory.core.config.schema import OrchestrationConfig
from ai_factory.core.decisions.rules import build_feedback_recommendations, decide_next_step
from ai_factory.core.execution.base import CommandSpec
from ai_factory.core.execution.commands import UnsupportedInstanceTypeError, build_command
from ai_factory.core.execution.local import LocalExecutor
from ai_factory.core.execution.ssh import SshExecutor
from ai_factory.core.instances.models import (
    DecisionAction,
    DecisionResult,
    DeploymentTarget,
    EnvironmentSpec,
    ExecutionHandle,
    FeedbackRecommendation,
    InstanceError,
    InstanceManifest,
    InstanceStatus,
    InstanceType,
    LifecycleProfile,
    LifecycleStage,
    ProgressSnapshot,
    UserLevel,
    utc_now_iso,
)
from ai_factory.core.instances.queries import InstanceQueryService
from ai_factory.core.instances.store import FileInstanceStore
from ai_factory.core.instances.utils import (
    _deep_merge,
    _SafeTemplateDict,
    _source_artifact_ref,
    _stage_for_instance_type,
)
from ai_factory.core.io import write_json
from ai_factory.core.monitoring.collectors import collect_metrics_for_instance
from ai_factory.core.monitoring.events import InstanceEvent
from ai_factory.core.orchestration.service import OrchestrationService
from ai_factory.core.orchestration.sqlite import SqliteControlPlane
from ai_factory.core.platform.settings import PlatformSettings, get_platform_settings
from ai_factory.core.plugins.registry import PluginRegistry, build_default_plugin_registry
from ai_factory.core.state import LifecycleStateManager


class InstanceManager:
    def __init__(
        self,
        store: FileInstanceStore,
        *,
        orchestration: OrchestrationService | None = None,
        platform_settings: PlatformSettings | None = None,
        plugin_registry: PluginRegistry | None = None,
    ) -> None:
        self.store = store
        self.queries = InstanceQueryService(store)
        self.platform_settings = platform_settings or get_platform_settings(artifacts_dir=store.artifacts_dir)
        self.plugin_registry = plugin_registry or build_default_plugin_registry(self.platform_settings.plugin_modules)
        self.orchestration = orchestration or OrchestrationService(
            control_plane=SqliteControlPlane(self.platform_settings.control_db_path),
            settings=self.platform_settings,
        )
        self.state = LifecycleStateManager(self.store, self.orchestration)

    def _make_instance_id(self, instance_type: InstanceType) -> str:
        return (
            f"{instance_type}-"
            f"{utc_now_iso().replace(':', '').replace('+00:00', 'z').replace('-', '').replace('.', '')}-"
            f"{uuid.uuid4().hex[:8]}"
        )

    def _resolve_environment(self, environment: EnvironmentSpec) -> EnvironmentSpec:
        if environment.kind != "cloud" or environment.host:
            return environment
        if not environment.profile_name:
            return environment
        profile = load_cloud_profile(environment.profile_name)
        if profile is None:
            raise FileNotFoundError(f"Unknown cloud profile: {environment.profile_name}")
        payload = profile.model_dump(mode="json")
        for key, value in environment.model_dump(mode="json").items():
            if value not in (None, {}):
                payload[key] = value
        return EnvironmentSpec.model_validate(payload)

    def _persist_snapshot(self, instance_id: str, snapshot: dict[str, Any]) -> None:
        write_json(self.store.config_snapshot_path(instance_id), snapshot)

    def _project_manifest(self, manifest: InstanceManifest) -> InstanceManifest:
        return self.state.project_manifest(manifest)

    def _checkpoint_hint_from_refs(self, refs: dict[str, Any]) -> str | None:
        run_dir = refs.get("run_dir")
        if not run_dir:
            return None
        checkpoints_dir = Path(str(run_dir)) / "checkpoints"
        if not checkpoints_dir.exists():
            return None
        candidates = [path for path in checkpoints_dir.iterdir() if path.is_dir()]
        if not candidates:
            return None
        candidates.sort(key=lambda path: path.stat().st_mtime)
        return str(candidates[-1])

    def _resume_command_if_available(
        self,
        manifest: InstanceManifest,
        config: OrchestrationConfig,
        command: CommandSpec,
    ) -> CommandSpec:
        if manifest.type not in {"train", "finetune"}:
            return command
        if not config.resilience.enable_checkpoint_resume:
            return command
        if "--resume-from-checkpoint" in command.argv:
            return command
        checkpoint_hint = self.orchestration.latest_checkpoint(manifest.id)
        if checkpoint_hint:
            command.argv = [*command.argv, "--resume-from-checkpoint", checkpoint_hint]
        return command

    def _legacy_instance_id(self, target: str) -> str:
        try:
            self.store.load(target)
            return target
        except FileNotFoundError:
            run = self.orchestration.control_plane.get_run(
                target
            ) or self.orchestration.control_plane.get_run_by_legacy_instance(target)
            if run is None or not run.legacy_instance_id:
                raise FileNotFoundError(f"Unknown instance or orchestration run: {target}") from None
            return run.legacy_instance_id

    def _progress(
        self,
        *,
        stage: str,
        message: str,
        percent: float | None = None,
        completed_steps: int | None = None,
        total_steps: int | None = None,
        metrics: dict[str, Any] | None = None,
    ) -> ProgressSnapshot:
        return ProgressSnapshot(
            stage=stage,
            status_message=message,
            percent=percent,
            completed_steps=completed_steps,
            total_steps=total_steps,
            metrics=metrics or {},
        )

    def _base_metadata(
        self,
        config: OrchestrationConfig,
        *,
        metadata_updates: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        metadata = deepcopy(config.metadata)
        metadata.setdefault("guardrail_notes", [])
        metadata["remote_access"] = config.remote_access.model_dump(mode="json")
        metadata["sub_agents"] = config.sub_agents.model_dump(mode="json")
        metadata["publish_hooks"] = [hook.model_dump(mode="json") for hook in config.publish_hooks]
        if metadata_updates:
            metadata = _deep_merge(metadata, metadata_updates)
        metadata.setdefault("automation_depth", 0)
        return metadata

    def _resolved_lifecycle(self, config: OrchestrationConfig) -> LifecycleProfile:
        payload = config.lifecycle.model_dump(mode="json")
        if not payload.get("stage"):
            payload["stage"] = _stage_for_instance_type(config.instance.type)
        return LifecycleProfile.model_validate(payload)

    def _apply_config_overrides(
        self,
        config: OrchestrationConfig,
        *,
        name_override: str | None = None,
        user_level_override: UserLevel | None = None,
        lifecycle_override: LifecycleProfile | None = None,
        subsystem_updates: dict[str, Any] | None = None,
        execution_updates: dict[str, Any] | None = None,
        metadata_updates: dict[str, Any] | None = None,
    ) -> OrchestrationConfig:
        merged = config.model_dump(mode="json")
        if name_override:
            merged.setdefault("instance", {})["name"] = name_override
        if user_level_override:
            merged.setdefault("experience", {})["level"] = user_level_override
        if lifecycle_override is not None:
            merged["lifecycle"] = lifecycle_override.model_dump(mode="json")
        if subsystem_updates:
            merged["subsystem"] = _deep_merge(merged.get("subsystem", {}), subsystem_updates)
        if execution_updates:
            merged["execution"] = _deep_merge(merged.get("execution", {}), execution_updates)
        if metadata_updates:
            merged["metadata"] = _deep_merge(merged.get("metadata", {}), metadata_updates)
        return build_orchestration_config(merged, config_path=config.config_path)

    def _inherit_lifecycle(
        self,
        source: InstanceManifest,
        *,
        stage: LifecycleStage,
        evaluation_suite: dict[str, Any] | None = None,
    ) -> LifecycleProfile:
        payload = source.lifecycle.model_dump(mode="json")
        payload["stage"] = stage
        if evaluation_suite:
            payload["evaluation_suite"] = evaluation_suite
        return LifecycleProfile.model_validate(payload)

    def _base_manifest(
        self,
        config: OrchestrationConfig,
        *,
        config_path: str | None,
        parent_instance_id: str | None = None,
        metadata_updates: dict[str, Any] | None = None,
    ) -> InstanceManifest:
        instance_id = self._make_instance_id(config.instance.type)
        environment = self._resolve_environment(config.instance.environment)
        return InstanceManifest(
            id=instance_id,
            type=config.instance.type,
            status="pending",
            environment=environment,
            user_level=config.experience.level,
            orchestration_mode=config.orchestration_mode,
            lifecycle=self._resolved_lifecycle(config),
            name=config.instance.name or instance_id,
            parent_instance_id=parent_instance_id or config.instance.parent_instance_id,
            config_path=config_path or config.config_path,
            config_snapshot_path=str(self.store.config_snapshot_path(instance_id)),
            artifact_refs={"instance_dir": str(self.store.instance_dir(instance_id))},
            progress=self._progress(stage="created", message="Instance created and waiting to start.", percent=0.0),
            metadata=self._base_metadata(config, metadata_updates=metadata_updates),
        )

    def _child_context(
        self,
        manifest: InstanceManifest,
        *,
        summary: dict[str, Any] | None = None,
        refs: dict[str, Any] | None = None,
        decision: DecisionResult | None = None,
    ) -> dict[str, Any]:
        context: dict[str, Any] = {
            "instance_id": manifest.id,
            "instance_name": manifest.name,
            "instance_type": manifest.type,
            "source_instance_id": manifest.id,
            "source_artifact": _source_artifact_ref(manifest) or "",
            "decision_action": decision.action if decision else "",
            "decision_rule": decision.rule if decision else "",
        }
        for payload in (refs or {}, summary or {}):
            for key, value in payload.items():
                if value is None:
                    continue
                if isinstance(value, (str, int, float, bool)):
                    context[key] = value
        return context

    def _render_templates(self, value: Any, context: dict[str, Any]) -> Any:
        if isinstance(value, str):
            return value.format_map(_SafeTemplateDict(context))
        if isinstance(value, list):
            return [self._render_templates(item, context) for item in value]
        if isinstance(value, dict):
            return {key: self._render_templates(item, context) for key, item in value.items()}
        return value

    def _apply_snapshot_overrides(
        self,
        instance_id: str,
        *,
        context: dict[str, Any] | None = None,
        subsystem_updates: dict[str, Any] | None = None,
        metadata_updates: dict[str, Any] | None = None,
    ) -> None:
        snapshot = self.store.load_config_snapshot(instance_id)
        snapshot.setdefault("subsystem", {})
        if context:
            snapshot["subsystem"] = self._render_templates(snapshot["subsystem"], context)
        if subsystem_updates:
            updates = self._render_templates(subsystem_updates, context or {})
            snapshot["subsystem"] = _deep_merge(snapshot["subsystem"], updates)
        if metadata_updates:
            snapshot["metadata"] = _deep_merge(snapshot.get("metadata", {}), metadata_updates)
        self._persist_snapshot(instance_id, snapshot)

    def _apply_manifest_metadata(self, instance_id: str, metadata_updates: dict[str, Any] | None) -> None:
        if not metadata_updates:
            return
        manifest = self.store.load(instance_id)
        manifest.metadata = _deep_merge(manifest.metadata, metadata_updates)
        self.store.save(manifest)

    def _child_metadata(
        self, parent: InstanceManifest, *, reason: str, extra: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        metadata = {
            "automation_reason": reason,
            "automation_parent_instance_id": parent.id,
            "automation_depth": int(parent.metadata.get("automation_depth", 0)) + 1,
        }
        if extra:
            metadata = _deep_merge(metadata, extra)
        return metadata

    def _can_spawn_children(self, manifest: InstanceManifest, config: OrchestrationConfig) -> bool:
        current_children = len(self.queries.get_children(manifest.id))
        if current_children >= config.pipeline.max_auto_children:
            return False
        if not config.sub_agents.allow_nested and manifest.parent_instance_id:
            return False
        return int(manifest.metadata.get("automation_depth", 0)) < config.pipeline.max_auto_cycles

    def _workload_enabled(
        self,
        config: OrchestrationConfig,
        workload: Literal["preprocess", "metrics", "evaluation", "finetune", "publish"],
    ) -> bool:
        return bool(config.sub_agents.enabled and workload in config.sub_agents.workloads)

    def create_instance(
        self,
        config_path: str,
        *,
        start: bool = True,
        environment_override: EnvironmentSpec | None = None,
        parent_instance_id: str | None = None,
        name_override: str | None = None,
        user_level_override: UserLevel | None = None,
        lifecycle_override: LifecycleProfile | None = None,
        subsystem_updates: dict[str, Any] | None = None,
        execution_updates: dict[str, Any] | None = None,
        metadata_updates: dict[str, Any] | None = None,
    ) -> InstanceManifest:
        config = load_orchestration_config(config_path)
        config = self._apply_config_overrides(
            config,
            name_override=name_override,
            user_level_override=user_level_override,
            lifecycle_override=lifecycle_override,
            subsystem_updates=subsystem_updates,
            execution_updates=execution_updates,
            metadata_updates=metadata_updates,
        )
        if environment_override is not None:
            config.instance.environment = environment_override

        # Create the base instance using the original manager logic
        manifest = self._base_manifest(
            config,
            config_path=config_path,
            parent_instance_id=parent_instance_id,
            metadata_updates=metadata_updates,
        )

        # Save using store
        self.store.create(manifest, config.model_dump(mode="json"))

        self.orchestration.ensure_run_for_instance(manifest, config.model_dump(mode="json"))
        manifest = self._project_manifest(manifest)
        self.store.save(manifest)

        if (
            self._workload_enabled(config, "preprocess")
            and manifest.type in {"train", "finetune"}
            and self._can_spawn_children(manifest, config)
        ):
            self.create_instance(
                resolve_path_from_config(manifest.config_path, config.pipeline.default_prepare_config)
                or config.pipeline.default_prepare_config,
                start=True,
                environment_override=manifest.environment,
                parent_instance_id=manifest.id,
                metadata_updates=self._child_metadata(manifest, reason="preprocess_sub_agent"),
            )

        return self.start_instance(manifest.id) if start else self.get_instance(manifest.id)

    def start_instance(self, instance_id: str) -> InstanceManifest:
        manifest = self.store.load(instance_id)
        snapshot = self.store.load_config_snapshot(instance_id)
        config = build_orchestration_config(snapshot, config_path=manifest.config_path)
        self.orchestration.ensure_run_for_instance(manifest, snapshot)
        self.orchestration.recover_stalled_tasks()
        try:
            command = build_command(config, manifest, plugin_registry=self.plugin_registry)
            command = self._resume_command_if_available(manifest, config, command)
        except UnsupportedInstanceTypeError as exc:
            return self.mark_failed(
                instance_id,
                code="unsupported_instance_type",
                message=str(exc),
                details={"instance_type": manifest.type},
            )
        except Exception as exc:
            return self.mark_failed(
                instance_id,
                code="command_resolution_failed",
                message=str(exc),
                details={"instance_type": manifest.type},
            )

        use_ssh = config.execution.backend == "ssh" or manifest.environment.kind == "cloud"
        executor = SshExecutor() if use_ssh else LocalExecutor()
        run, task, attempt = self.orchestration.begin_attempt(
            legacy_instance_id=instance_id,
            stdout_path=str(self.store.stdout_path(instance_id)),
            stderr_path=str(self.store.stderr_path(instance_id)),
            lease_owner=executor.backend_name,
            metadata={"argv": command.argv},
        )
        manifest.execution = ExecutionHandle(
            backend=executor.backend_name,
            stdout_path=str(self.store.stdout_path(instance_id)),
            stderr_path=str(self.store.stderr_path(instance_id)),
            metadata={
                "orchestration_run_id": run.id,
                "orchestration_task_id": task.id,
                "attempt_id": attempt.id,
                "argv": command.argv,
            },
        )
        manifest.progress = self._progress(stage="queued", message="Execution requested.", percent=0.0)
        self.store.save(self._project_manifest(manifest))
        handle = executor.start(
            manifest,
            command,
            artifacts_dir=self.store.artifacts_dir,
            stdout_path=str(self.store.stdout_path(instance_id)),
            stderr_path=str(self.store.stderr_path(instance_id)),
        )
        refreshed = self.store.load(instance_id)
        if refreshed.execution is None:
            refreshed.execution = handle
        else:
            refreshed.execution.pid = handle.pid
            refreshed.execution.metadata.update(handle.metadata)
            refreshed.execution.metadata.setdefault("attempt_id", attempt.id)
            refreshed.execution.metadata.setdefault("orchestration_run_id", run.id)
            refreshed.execution.metadata.setdefault("orchestration_task_id", task.id)
        self.store.save(self._project_manifest(refreshed))
        self.store.append_event(
            instance_id,
            InstanceEvent(
                type="instance.start_requested",
                message=f"Queued {manifest.type} instance for {handle.backend} execution.",
                payload={"backend": handle.backend, "pid": handle.pid, "attempt_id": attempt.id},
            ),
        )
        return self._project_manifest(refreshed)

    def mark_running(self, instance_id: str) -> InstanceManifest:
        manifest = self.store.load(instance_id)
        manifest.status = "running"
        execution = manifest.execution or ExecutionHandle(
            backend="ssh" if manifest.environment.kind == "cloud" else "local",
            stdout_path=str(self.store.stdout_path(instance_id)),
            stderr_path=str(self.store.stderr_path(instance_id)),
        )
        execution.started_at = utc_now_iso()
        manifest.execution = execution
        manifest.progress = self._progress(stage="running", message=f"{manifest.type} instance is running.")
        attempt_id = (execution.metadata or {}).get("attempt_id")
        if isinstance(attempt_id, str):
            try:
                self.orchestration.heartbeat(attempt_id)
            except FileNotFoundError:
                pass
        self.store.save(self._project_manifest(manifest))
        self.store.append_event(
            instance_id,
            InstanceEvent(
                type="instance.running",
                message=f"{manifest.type} instance is running.",
                payload={"backend": execution.backend},
            ),
        )
        return self._project_manifest(manifest)

    def _retry_failed_instance(self, manifest: InstanceManifest, config: OrchestrationConfig) -> bool:
        retry_limit = max(int(config.execution.retry_limit or 0), int(config.sub_agents.retry_limit or 0))
        if retry_limit <= 0:
            return False
        task_summary = manifest.task_summary or {}
        current_attempt = int(task_summary.get("current_attempt") or 0)
        if current_attempt >= retry_limit:
            return False
        retry_task = self.orchestration.retry_task(manifest.id)
        available_at = datetime.fromisoformat(retry_task.available_at.replace("Z", "+00:00"))
        if available_at <= datetime.now(timezone.utc):
            self.start_instance(manifest.id)
            return True
        return False

    def mark_failed(
        self,
        instance_id: str,
        *,
        code: str,
        message: str,
        details: dict[str, Any],
    ) -> InstanceManifest:
        manifest = self.store.load(instance_id)
        snapshot = self.store.load_config_snapshot(instance_id)
        config = build_orchestration_config(snapshot, config_path=manifest.config_path)
        manifest.status = "failed"
        manifest.error = InstanceError(code=code, message=message, details=details)
        manifest.progress = self._progress(stage="failed", message=message, percent=100.0)
        if manifest.execution is not None:
            manifest.execution.ended_at = utc_now_iso()
            manifest.execution.exit_code = (
                manifest.execution.exit_code if manifest.execution.exit_code is not None else 1
            )
        self.store.save(self._project_manifest(manifest))
        self.store.append_event(
            instance_id,
            InstanceEvent(type="instance.failed", message=message, payload={"code": code, **details}),
        )
        self._retry_failed_instance(manifest, config)
        return self._project_manifest(manifest)

    def _schedule_report_instance(
        self,
        parent: InstanceManifest,
        config: OrchestrationConfig,
        *,
        context: dict[str, Any],
        reason: str,
    ) -> None:
        if not self._can_spawn_children(parent, config):
            return
        report_config = resolve_path_from_config(parent.config_path, config.pipeline.default_report_config)
        if not report_config:
            return
        child = self.create_instance(
            report_config,
            start=False,
            environment_override=parent.environment,
            parent_instance_id=parent.id,
            metadata_updates=self._child_metadata(parent, reason=reason),
        )
        self._apply_snapshot_overrides(child.id, context=context)
        self.start_instance(child.id)

    def _schedule_publish_hooks(
        self,
        parent: InstanceManifest,
        config: OrchestrationConfig,
        *,
        when: Literal["manual", "on_success", "after_evaluation"],
        context: dict[str, Any],
    ) -> bool:
        if not self._can_spawn_children(parent, config):
            return False
        scheduled = False
        for hook in config.publish_hooks:
            if not hook.enabled or hook.when != when:
                continue
            deploy_config = resolve_path_from_config(
                parent.config_path,
                hook.config_path or config.pipeline.default_deploy_config,
            )
            if not deploy_config:
                continue
            target: DeploymentTarget = "custom_api" if hook.target == "api" else hook.target
            child = self.create_deployment_instance(
                parent.id,
                target=target,
                config_path=deploy_config,
                start=False,
                metadata_updates=self._child_metadata(parent, reason=f"publish_hook:{target}"),
            )
            self._apply_snapshot_overrides(
                child.id,
                context=context,
                subsystem_updates={"provider_options": hook.provider_options},
            )
            self.start_instance(child.id)
            scheduled = True
        return scheduled

    def _queue_recommendations(
        self,
        manifest: InstanceManifest,
        config: OrchestrationConfig,
        *,
        context: dict[str, Any],
    ) -> bool:
        if not self._can_spawn_children(manifest, config):
            return False
        scheduled = False
        for recommendation in manifest.recommendations[: config.feedback_loop.max_recommendations]:
            should_queue = (recommendation.action == "finetune" and config.feedback_loop.auto_queue_finetune) or (
                recommendation.action == "retrain" and config.feedback_loop.auto_queue_retrain
            )
            if not should_queue or not recommendation.target_instance_type or not recommendation.config_path:
                continue
            if recommendation.target_instance_type == "deploy":
                child = self.create_deployment_instance(
                    manifest.id,
                    target=recommendation.deployment_target or "huggingface",
                    config_path=resolve_path_from_config(manifest.config_path, recommendation.config_path)
                    or recommendation.config_path,
                    start=False,
                    metadata_updates=self._child_metadata(
                        manifest,
                        reason=f"recommendation:{recommendation.action}",
                    ),
                )
                self._apply_snapshot_overrides(child.id, context=context)
                self.start_instance(child.id)
                scheduled = True
                continue
            if recommendation.target_instance_type == "evaluate":
                child = self.create_evaluation_instance(
                    manifest.id,
                    config_path=resolve_path_from_config(manifest.config_path, recommendation.config_path)
                    or recommendation.config_path,
                    start=False,
                    metadata_updates=self._child_metadata(
                        manifest,
                        reason=f"recommendation:{recommendation.action}",
                    ),
                )
                self._apply_snapshot_overrides(child.id, context=context)
                self.start_instance(child.id)
                scheduled = True
                continue
            child = self.create_instance(
                resolve_path_from_config(manifest.config_path, recommendation.config_path)
                or recommendation.config_path,
                start=False,
                environment_override=manifest.environment,
                parent_instance_id=manifest.id,
                metadata_updates=self._child_metadata(
                    manifest,
                    reason=f"recommendation:{recommendation.action}",
                ),
            )
            subsystem_updates = {"command_override": recommendation.command} if recommendation.command else None
            self._apply_snapshot_overrides(child.id, context=context, subsystem_updates=subsystem_updates)
            self.start_instance(child.id)
            scheduled = True
        return scheduled

    def _post_success_automation(
        self,
        manifest: InstanceManifest,
        config: OrchestrationConfig,
        *,
        summary: dict[str, Any],
        refs: dict[str, Any],
    ) -> None:
        context = self._child_context(manifest, summary=summary, refs=refs, decision=manifest.decision)
        scheduled = False

        if manifest.type in {"train", "finetune"}:
            if self._workload_enabled(config, "evaluation") or config.feedback_loop.queue_follow_up_evaluation:
                child = self.create_evaluation_instance(
                    manifest.id,
                    config_path=resolve_path_from_config(manifest.config_path, config.pipeline.default_eval_config)
                    or config.pipeline.default_eval_config,
                    start=False,
                    metadata_updates=self._child_metadata(manifest, reason="evaluation_follow_up"),
                )
                self._apply_snapshot_overrides(child.id, context=context)
                self.start_instance(child.id)
                scheduled = True
            if self._schedule_publish_hooks(manifest, config, when="on_success", context=context):
                scheduled = True

        if manifest.type == "evaluate":
            if self._workload_enabled(config, "metrics") or config.feedback_loop.suggest_failure_analysis:
                self._schedule_report_instance(manifest, config, context=context, reason="failure_analysis")
                scheduled = True
            if manifest.decision and manifest.decision.action == "deploy":
                if self._schedule_publish_hooks(manifest, config, when="after_evaluation", context=context):
                    scheduled = True
            if self._queue_recommendations(manifest, config, context=context):
                scheduled = True
            if (
                (config.pipeline.auto_continue or config.decision_policy.auto_continue)
                and manifest.decision
                and not scheduled
            ):
                self._auto_continue(manifest, config, manifest.decision.action)

    def finalize_instance(
        self,
        instance_id: str,
        exit_code: int,
        *,
        runtime_metadata: dict[str, Any] | None = None,
    ) -> InstanceManifest:
        manifest = self.store.load(instance_id)
        snapshot = self.store.load_config_snapshot(instance_id)
        config = build_orchestration_config(snapshot, config_path=manifest.config_path)
        if manifest.execution is None:
            manifest.execution = ExecutionHandle(
                backend="ssh" if manifest.environment.kind == "cloud" else "local",
                stdout_path=str(self.store.stdout_path(instance_id)),
                stderr_path=str(self.store.stderr_path(instance_id)),
            )
        manifest.execution.ended_at = utc_now_iso()
        manifest.execution.exit_code = exit_code
        if runtime_metadata:
            manifest.execution.metadata.update(runtime_metadata)
        manifest.status = "completed" if exit_code == 0 else "failed"
        if exit_code != 0 and manifest.error is None:
            manifest.error = InstanceError(
                code="execution_failed",
                message=f"Instance exited with code {exit_code}",
                details={"exit_code": exit_code},
            )

        summary, points, refs = collect_metrics_for_instance(
            manifest,
            snapshot,
            collect_gpu=config.monitoring.collect_gpu,
        )
        manifest.metrics_summary = summary
        manifest.artifact_refs.update(refs)
        manifest.progress = self._progress(
            stage="completed" if manifest.status == "completed" else "failed",
            message=f"{manifest.type} instance {manifest.status}.",
            percent=100.0,
            completed_steps=summary.get("latest_step"),
            total_steps=summary.get("total_steps"),
            metrics={key: value for key, value in summary.items() if isinstance(value, (int, float, bool))},
        )
        self.store.write_current_metrics(instance_id, summary)
        if config.monitoring.write_timeseries:
            self.store.append_metric_points(instance_id, points)

        if manifest.type == "evaluate" and manifest.status == "completed":
            decision = decide_next_step(summary, config.decision_policy)
            manifest.decision = decision
            manifest.recommendations = build_feedback_recommendations(
                summary,
                config.decision_policy,
                default_prepare_config=resolve_path_from_config(
                    manifest.config_path, config.pipeline.default_prepare_config
                )
                or config.pipeline.default_prepare_config,
                default_train_config=resolve_path_from_config(
                    manifest.config_path, config.pipeline.default_train_config
                )
                or config.pipeline.default_train_config,
                default_finetune_config=resolve_path_from_config(
                    manifest.config_path,
                    config.pipeline.default_finetune_config,
                )
                or config.pipeline.default_finetune_config,
                default_eval_config=resolve_path_from_config(manifest.config_path, config.pipeline.default_eval_config)
                or config.pipeline.default_eval_config,
                default_inference_config=resolve_path_from_config(
                    manifest.config_path,
                    config.pipeline.default_inference_config,
                )
                or config.pipeline.default_inference_config,
                default_deploy_config=resolve_path_from_config(
                    manifest.config_path,
                    config.pipeline.default_deploy_config,
                )
                or config.pipeline.default_deploy_config,
                default_report_config=resolve_path_from_config(
                    manifest.config_path,
                    config.pipeline.default_report_config,
                )
                or config.pipeline.default_report_config,
                improvement_floor=config.feedback_loop.improvement_floor,
                suggest_failure_analysis=config.feedback_loop.suggest_failure_analysis,
            )[: config.feedback_loop.max_recommendations]
            self.store.write_decision_report(instance_id, decision.model_dump(mode="json"))
            self.store.write_recommendations_report(
                instance_id,
                [item.model_dump(mode="json") for item in manifest.recommendations],
            )
        elif manifest.type in {"train", "finetune"} and manifest.status == "completed":
            if self._workload_enabled(config, "evaluation") or config.feedback_loop.queue_follow_up_evaluation:
                manifest.recommendations = [
                    FeedbackRecommendation(
                        action="evaluate",
                        reason=(
                            "A follow-up evaluation is recommended so training metrics "
                            "can feed the next orchestration step."
                        ),
                        priority=2,
                        target_instance_type="evaluate",
                        config_path=resolve_path_from_config(manifest.config_path, config.pipeline.default_eval_config)
                        or config.pipeline.default_eval_config,
                        metadata={"source": manifest.type},
                    )
                ]
                self.store.write_recommendations_report(
                    instance_id,
                    [item.model_dump(mode="json") for item in manifest.recommendations],
                )

        checkpoint_hint = self._checkpoint_hint_from_refs(refs)
        attempt_id = ((manifest.execution or ExecutionHandle(backend="local")).metadata or {}).get("attempt_id")
        if isinstance(attempt_id, str):
            self.orchestration.finalize_attempt(
                legacy_instance_id=instance_id,
                attempt_id=attempt_id,
                exit_code=exit_code,
                summary=summary,
                metrics={key: value for key, value in summary.items() if isinstance(value, (int, float, bool))},
                artifacts=refs,
                recommendations=[item.model_dump(mode="json") for item in manifest.recommendations],
                checkpoint_hint=checkpoint_hint,
                error_code=manifest.error.code if manifest.error else None,
                error_message=manifest.error.message if manifest.error else None,
            )
        self.store.save(self._project_manifest(manifest))
        self.store.append_event(
            instance_id,
            InstanceEvent(
                type=f"instance.{manifest.status}",
                message=f"{manifest.type} instance {manifest.status}.",
                payload={"exit_code": exit_code, "metrics_summary": summary},
            ),
        )

        if manifest.status == "completed":
            self._post_success_automation(manifest, config, summary=summary, refs=refs)
        else:
            self._retry_failed_instance(manifest, config)
        return self._project_manifest(manifest)

    def _auto_continue(
        self,
        manifest: InstanceManifest,
        config: OrchestrationConfig,
        action: DecisionAction,
    ) -> None:
        if not self._can_spawn_children(manifest, config):
            return
        if action == "deploy":
            self.create_deployment_instance(
                manifest.parent_instance_id or manifest.id,
                target="huggingface",
                config_path=resolve_path_from_config(manifest.config_path, config.pipeline.default_deploy_config)
                or config.pipeline.default_deploy_config,
                start=True,
                metadata_updates=self._child_metadata(manifest, reason="auto_continue_deploy"),
            )
        elif action == "finetune":
            self.create_instance(
                resolve_path_from_config(manifest.config_path, config.pipeline.default_finetune_config)
                or config.pipeline.default_finetune_config,
                start=True,
                environment_override=manifest.environment,
                parent_instance_id=manifest.id,
                metadata_updates=self._child_metadata(manifest, reason="auto_continue_finetune"),
            )
        elif action == "retrain":
            self.create_instance(
                resolve_path_from_config(manifest.config_path, config.pipeline.default_train_config)
                or config.pipeline.default_train_config,
                start=True,
                environment_override=manifest.environment,
                parent_instance_id=manifest.id,
                metadata_updates=self._child_metadata(manifest, reason="auto_continue_retrain"),
            )
        elif action == "open_inference":
            self.create_inference_instance(
                manifest.parent_instance_id or manifest.id,
                config_path=resolve_path_from_config(manifest.config_path, config.pipeline.default_inference_config)
                or config.pipeline.default_inference_config,
                start=True,
                metadata_updates=self._child_metadata(manifest, reason="auto_continue_inference"),
            )
        elif action == "re_evaluate":
            self.create_evaluation_instance(
                manifest.parent_instance_id or manifest.id,
                config_path=resolve_path_from_config(manifest.config_path, config.pipeline.default_eval_config)
                or config.pipeline.default_eval_config,
                start=True,
                metadata_updates=self._child_metadata(manifest, reason="auto_continue_re_evaluate"),
            )

    def _write_generated_model_registry(
        self,
        manifest: InstanceManifest,
        source: InstanceManifest,
        *,
        registry_path: str,
        description: str,
        tags: list[str],
    ) -> tuple[str, str]:
        source_artifact = _source_artifact_ref(source)
        if not source_artifact:
            raise ValueError("source instance does not expose a publishable artifact yet")
        registry_payload = yaml.safe_load(Path(registry_path).read_text()) or {}
        generated_name = f"instance_{source.id.replace('-', '_')}"
        registry_payload.setdefault("models", [])
        registry_payload["models"] = [item for item in registry_payload["models"] if item.get("name") != generated_name]
        registry_payload["models"].append(
            {
                "name": generated_name,
                "label": source.name,
                "base_model": source.artifact_refs.get("base_model", "Qwen/Qwen2.5-Math-1.5B-Instruct"),
                "adapter_path": source_artifact,
                "load_in_4bit": True,
                "load_in_8bit": False,
                "dtype": "bfloat16",
                "description": description,
                "tags": tags,
            }
        )
        generated_registry_path = self.store.instance_dir(manifest.id) / "generated_model_registry.yaml"
        generated_registry_path.write_text(yaml.safe_dump(registry_payload, sort_keys=False))
        return generated_name, str(generated_registry_path)

    def _write_lifecycle(self, instance_id: str, lifecycle: LifecycleProfile) -> None:
        manifest = self.store.load(instance_id)
        manifest.lifecycle = lifecycle
        self.store.save(manifest)

    def _build_eval_snapshot(self, manifest: InstanceManifest, source: InstanceManifest) -> dict[str, Any]:
        snapshot = self.store.load_config_snapshot(manifest.id)
        source_artifact = _source_artifact_ref(source)
        if not source_artifact:
            snapshot["subsystem"]["source_instance_id"] = source.id
            snapshot.setdefault("metadata", {})
            snapshot["metadata"]["source_instance_id"] = source.id
            return snapshot

        eval_config_path = snapshot.get("resolved_subsystem_config_path") or snapshot["subsystem"].get("config_ref")
        if not eval_config_path:
            snapshot["subsystem"]["source_instance_id"] = source.id
            return snapshot

        eval_payload = yaml.safe_load(Path(eval_config_path).read_text()) or {}
        registry_path = Path(
            eval_payload.get("models", {}).get("registry_path", "inference/configs/model_registry.yaml")
        )
        generated_name, generated_registry_path = self._write_generated_model_registry(
            manifest,
            source,
            registry_path=str(registry_path),
            description=f"Generated evaluation model for {source.id}",
            tags=["instance", "evaluation", source.type],
        )
        generated_eval_path = self.store.instance_dir(manifest.id) / "generated_eval_config.yaml"

        eval_payload.setdefault("models", {})
        eval_payload["models"]["registry_path"] = generated_registry_path
        eval_payload["models"]["primary_model"] = generated_name
        eval_payload["models"]["primary_label"] = source.name
        generated_eval_path.write_text(yaml.safe_dump(eval_payload, sort_keys=False))

        snapshot["subsystem"]["source_instance_id"] = source.id
        snapshot["subsystem"]["source_artifact_ref"] = source_artifact
        snapshot["resolved_subsystem_config_path"] = str(generated_eval_path)
        snapshot["resolved_subsystem_config"] = eval_payload
        snapshot.setdefault("metadata", {})
        snapshot["metadata"]["source_instance_id"] = source.id
        return snapshot

    def _build_inference_snapshot(self, manifest: InstanceManifest, source: InstanceManifest) -> dict[str, Any]:
        snapshot = self.store.load_config_snapshot(manifest.id)
        source_artifact = _source_artifact_ref(source)
        if not source_artifact:
            snapshot["subsystem"]["source_instance_id"] = source.id
            snapshot.setdefault("metadata", {})
            snapshot["metadata"]["source_instance_id"] = source.id
            return snapshot

        execution = snapshot.setdefault("execution", {})
        env = execution.setdefault("env", {})
        registry_path = env.get("MODEL_REGISTRY_PATH") or "inference/configs/model_registry.yaml"
        generated_name, generated_registry_path = self._write_generated_model_registry(
            manifest,
            source,
            registry_path=registry_path,
            description=f"Generated inference model for {source.id}",
            tags=["instance", "inference", source.type],
        )
        env["MODEL_REGISTRY_PATH"] = generated_registry_path
        snapshot["subsystem"]["model_variant"] = generated_name
        snapshot["subsystem"]["source_instance_id"] = source.id
        snapshot["subsystem"]["source_artifact_ref"] = source_artifact
        snapshot.setdefault("metadata", {})
        snapshot["metadata"]["source_instance_id"] = source.id
        return snapshot

    def create_evaluation_instance(
        self,
        source_instance_id: str,
        *,
        config_path: str = "configs/eval.yaml",
        start: bool = True,
        metadata_updates: dict[str, Any] | None = None,
    ) -> InstanceManifest:
        source = self.store.load(source_instance_id)
        manifest = self.create_instance(
            config_path,
            start=False,
            environment_override=source.environment,
            parent_instance_id=source.id,
            metadata_updates=metadata_updates,
        )
        snapshot = self._build_eval_snapshot(manifest, source)
        self._persist_snapshot(manifest.id, snapshot)
        if _source_artifact_ref(source):
            manifest.artifact_refs["source_artifact"] = _source_artifact_ref(source)
            manifest.artifact_refs["source_instance_id"] = source.id
            self.store.save(manifest)
        eval_payload = snapshot.get("resolved_subsystem_config") or {}
        raw_models_payload = eval_payload.get("models")
        models_payload: dict[str, Any] = raw_models_payload if isinstance(raw_models_payload, dict) else {}
        self._write_lifecycle(
            manifest.id,
            self._inherit_lifecycle(
                source,
                stage="evaluate",
                evaluation_suite={
                    "id": Path(str(snapshot.get("resolved_subsystem_config_path") or config_path)).stem,
                    "label": source.name,
                    "benchmark_config": str(snapshot.get("resolved_subsystem_config_path") or config_path),
                    "compare_to_models": [str(primary_model)]
                    if isinstance(primary_model := models_payload.get("primary_model"), str)
                    else [],
                },
            ),
        )
        return self.start_instance(manifest.id) if start else self.get_instance(manifest.id)

    def create_deployment_instance(
        self,
        source_instance_id: str,
        *,
        target: DeploymentTarget,
        config_path: str = "configs/deploy.yaml",
        start: bool = True,
        metadata_updates: dict[str, Any] | None = None,
    ) -> InstanceManifest:
        source = self.store.load(source_instance_id)
        manifest = self.create_instance(
            config_path,
            start=False,
            environment_override=source.environment,
            parent_instance_id=source.id,
            metadata_updates=metadata_updates,
        )
        snapshot = self.store.load_config_snapshot(manifest.id)
        snapshot["subsystem"]["provider"] = target
        snapshot["subsystem"]["source_instance_id"] = source.id
        snapshot["subsystem"]["source_artifact_ref"] = _source_artifact_ref(source)
        snapshot.setdefault("metadata", {})
        snapshot["metadata"]["source_instance_id"] = source.id
        self._persist_snapshot(manifest.id, snapshot)
        if _source_artifact_ref(source):
            manifest.artifact_refs["source_artifact"] = _source_artifact_ref(source)
            manifest.artifact_refs["source_instance_id"] = source.id
            self.store.save(manifest)
        lifecycle = self._inherit_lifecycle(source, stage="publish")
        if target not in lifecycle.deployment_targets:
            lifecycle.deployment_targets = [*lifecycle.deployment_targets, target]
        self._write_lifecycle(manifest.id, lifecycle)
        return self.start_instance(manifest.id) if start else self.get_instance(manifest.id)

    def create_inference_instance(
        self,
        source_instance_id: str,
        *,
        config_path: str = "configs/inference.yaml",
        start: bool = True,
        metadata_updates: dict[str, Any] | None = None,
    ) -> InstanceManifest:
        source = self.store.load(source_instance_id)
        manifest = self.create_instance(
            config_path,
            start=False,
            environment_override=source.environment,
            parent_instance_id=source.id,
            metadata_updates=metadata_updates,
        )
        snapshot = self._build_inference_snapshot(manifest, source)
        self._persist_snapshot(manifest.id, snapshot)
        if _source_artifact_ref(source):
            manifest.artifact_refs["source_artifact"] = _source_artifact_ref(source)
            manifest.artifact_refs["source_instance_id"] = source.id
            self.store.save(manifest)
        self._write_lifecycle(manifest.id, self._inherit_lifecycle(source, stage="infer"))
        return self.start_instance(manifest.id) if start else manifest

    def _default_config_path_for_action(
        self,
        source: InstanceManifest,
        action: str,
        config: OrchestrationConfig,
    ) -> str | None:
        mapping = {
            "prepare": config.pipeline.default_prepare_config,
            "train": config.pipeline.default_train_config,
            "retrain": config.pipeline.default_train_config,
            "evaluate": config.pipeline.default_eval_config,
            "re_evaluate": config.pipeline.default_eval_config,
            "finetune": config.pipeline.default_finetune_config,
            "open_inference": config.pipeline.default_inference_config,
            "deploy": config.pipeline.default_deploy_config,
            "report": config.pipeline.default_report_config,
        }
        template = mapping.get(action)
        if template is None:
            return None
        return resolve_path_from_config(source.config_path, template) or template

    def _default_deployment_target(self, manifest: InstanceManifest, config: OrchestrationConfig) -> DeploymentTarget:
        for recommendation in manifest.recommendations:
            if recommendation.action == "deploy" and recommendation.deployment_target:
                return recommendation.deployment_target
        if manifest.lifecycle.deployment_targets:
            return manifest.lifecycle.deployment_targets[0]
        for hook in config.publish_hooks:
            if hook.target:
                return hook.target
        return "ollama"

    def execute_action(
        self,
        instance_id: str,
        *,
        action: str,
        config_path: str | None = None,
        deployment_target: DeploymentTarget | None = None,
        start: bool = True,
    ) -> InstanceManifest:
        source = self.get_instance(instance_id)
        snapshot = self.store.load_config_snapshot(source.id)
        config = build_orchestration_config(snapshot, config_path=source.config_path)
        context = self._child_context(
            source,
            summary=source.metrics_summary,
            refs=source.artifact_refs,
            decision=source.decision,
        )
        metadata = self._child_metadata(source, reason=f"manual_action:{action}")

        if action in {"evaluate", "re_evaluate"}:
            child = self.create_evaluation_instance(
                source.id,
                config_path=config_path
                or self._default_config_path_for_action(source, action, config)
                or "configs/eval.yaml",
                start=False,
                metadata_updates=metadata,
            )
        elif action == "open_inference":
            child = self.create_inference_instance(
                source.id,
                config_path=config_path
                or self._default_config_path_for_action(source, action, config)
                or "configs/inference.yaml",
                start=False,
                metadata_updates=metadata,
            )
        elif action == "deploy":
            child = self.create_deployment_instance(
                source.id,
                target=deployment_target or self._default_deployment_target(source, config),
                config_path=config_path
                or self._default_config_path_for_action(source, action, config)
                or "configs/deploy.yaml",
                start=False,
                metadata_updates=metadata,
            )
        else:
            resolved_config_path = config_path or self._default_config_path_for_action(source, action, config)
            if resolved_config_path is None:
                raise ValueError(f"Unsupported action: {action}")
            child = self.create_instance(
                resolved_config_path,
                start=False,
                environment_override=source.environment,
                parent_instance_id=source.id,
                metadata_updates=metadata,
            )

        self._apply_snapshot_overrides(child.id, context=context)
        return self.start_instance(child.id) if start else self.get_instance(child.id)

    def get_instance(self, instance_id: str) -> InstanceManifest:
        return self._project_manifest(self.store.load(instance_id))

    def list_instances(
        self,
        *,
        instance_type: InstanceType | None = None,
        status: InstanceStatus | None = None,
        parent_instance_id: str | None = None,
    ) -> list[InstanceManifest]:
        return [
            self._project_manifest(item)
            for item in self.queries.list_instances(
                instance_type=instance_type,
                status=status,
                parent_instance_id=parent_instance_id,
            )
        ]

    def get_children(self, instance_id: str) -> list[InstanceManifest]:
        return [self._project_manifest(item) for item in self.queries.get_children(instance_id)]

    def get_logs(self, instance_id: str) -> dict[str, str]:
        return self.store.read_logs(instance_id)

    def get_metrics(self, instance_id: str) -> dict[str, Any]:
        return self.state.get_metrics(instance_id)

    def get_available_actions(self, instance_id: str) -> list[dict[str, Any]]:
        manifest = self.get_instance(instance_id)
        if manifest.status != "completed":
            return []
        snapshot = self.store.load_config_snapshot(instance_id)
        config = build_orchestration_config(snapshot, config_path=manifest.config_path)
        actions: list[dict[str, Any]] = []
        seen: set[tuple[str, str | None, str | None, str | None]] = set()

        def add_action(
            action: str,
            label: str,
            description: str,
            *,
            target_instance_type: InstanceType | None = None,
            config_path: str | None = None,
            deployment_target: DeploymentTarget | None = None,
        ) -> None:
            key = (action, target_instance_type, config_path, deployment_target)
            if key in seen:
                return
            seen.add(key)
            actions.append(
                {
                    "action": action,
                    "label": label,
                    "description": description,
                    "target_instance_type": target_instance_type,
                    "config_path": config_path,
                    "deployment_target": deployment_target,
                }
            )

        eval_config_path = (
            resolve_path_from_config(manifest.config_path, config.pipeline.default_eval_config)
            or config.pipeline.default_eval_config
        )
        inference_config_path = (
            resolve_path_from_config(manifest.config_path, config.pipeline.default_inference_config)
            or config.pipeline.default_inference_config
        )
        deploy_config_path = (
            resolve_path_from_config(manifest.config_path, config.pipeline.default_deploy_config)
            or config.pipeline.default_deploy_config
        )

        if manifest.type in {"train", "finetune"}:
            add_action(
                "evaluate",
                "Evaluate",
                "Run the managed benchmark suite and attach the results to this lifecycle branch.",
                target_instance_type="evaluate",
                config_path=eval_config_path,
            )
            if _source_artifact_ref(manifest):
                add_action(
                    "open_inference",
                    "Start Inference",
                    "Launch an inference sandbox backed by this trained artifact.",
                    target_instance_type="inference",
                    config_path=inference_config_path,
                )

        if manifest.type in {"train", "finetune", "evaluate"} and _source_artifact_ref(manifest):
            add_action(
                "open_inference",
                "Start Inference",
                "Launch an inference sandbox backed by this lifecycle branch.",
                target_instance_type="inference",
                config_path=inference_config_path,
            )

        if manifest.type == "evaluate":
            targets: list[DeploymentTarget] = []
            targets.extend(manifest.lifecycle.deployment_targets)
            targets.extend(
                item.deployment_target for item in manifest.recommendations if item.deployment_target is not None
            )
            targets.extend(hook.target for hook in config.publish_hooks if hook.target)
            deduped_targets: list[DeploymentTarget] = []
            for target in targets:
                if target not in deduped_targets:
                    deduped_targets.append(target)
            if not deduped_targets and manifest.decision and manifest.decision.action == "deploy":
                deduped_targets.append("ollama")
            for target in deduped_targets:
                add_action(
                    "deploy",
                    f"Prepare Publish: {target.replace('_', ' ').title()}",
                    "Create a managed publish/deploy child instance for this evaluated artifact.",
                    target_instance_type="deploy",
                    config_path=deploy_config_path,
                    deployment_target=target,
                )
            if manifest.decision and manifest.decision.action == "re_evaluate":
                add_action(
                    "re_evaluate",
                    "Re-run Evaluation",
                    "Refresh the benchmark results before deciding on the next stage.",
                    target_instance_type="evaluate",
                    config_path=eval_config_path,
                )
            for recommendation in manifest.recommendations:
                if recommendation.action == "deploy" and recommendation.deployment_target:
                    add_action(
                        "deploy",
                        f"Prepare Publish: {recommendation.deployment_target.replace('_', ' ').title()}",
                        recommendation.reason,
                        target_instance_type="deploy",
                        config_path=recommendation.config_path or deploy_config_path,
                        deployment_target=recommendation.deployment_target,
                    )
                elif recommendation.action == "open_inference":
                    add_action(
                        "open_inference",
                        "Start Inference",
                        recommendation.reason,
                        target_instance_type="inference",
                        config_path=recommendation.config_path or inference_config_path,
                    )
                elif recommendation.action in {"finetune", "retrain", "report", "evaluate", "re_evaluate"}:
                    label_map = {
                        "finetune": "Start Finetune Iteration",
                        "retrain": "Start Retraining Branch",
                        "report": "Analyze Failures",
                        "evaluate": "Evaluate",
                        "re_evaluate": "Re-run Evaluation",
                    }
                    add_action(
                        recommendation.action,
                        label_map[recommendation.action],
                        recommendation.reason,
                        target_instance_type=recommendation.target_instance_type,
                        config_path=recommendation.config_path,
                    )

        return actions

    def list_tasks(self, legacy_or_run_id: str | None = None) -> list[dict[str, Any]]:
        return [item.model_dump(mode="json") for item in self.orchestration.list_tasks(legacy_or_run_id)]

    def list_orchestration_runs(self) -> list[dict[str, Any]]:
        return [item.model_dump(mode="json") for item in self.orchestration.list_runs()]

    def get_orchestration_run(self, legacy_or_run_id: str) -> dict[str, Any]:
        run = self.orchestration.control_plane.get_run(
            legacy_or_run_id
        ) or self.orchestration.control_plane.get_run_by_legacy_instance(legacy_or_run_id)
        if run is None:
            raise FileNotFoundError(f"Unknown orchestration run: {legacy_or_run_id}")
        tasks = self.list_tasks(run.id)
        events = self.list_orchestration_events(run.id)
        return {
            "run": run.model_dump(mode="json"),
            "tasks": tasks,
            "events": events,
            "summary": self.orchestration.monitoring_summary(),
        }

    def list_orchestration_events(self, legacy_or_run_id: str, *, limit: int | None = None) -> list[dict[str, Any]]:
        return [item.model_dump(mode="json") for item in self.orchestration.list_events(legacy_or_run_id, limit=limit)]

    def cancel_instance(self, instance_id: str) -> InstanceManifest:
        legacy_instance_id = self._legacy_instance_id(instance_id)
        self.orchestration.cancel_run(legacy_instance_id)
        manifest = self.store.load(legacy_instance_id)
        manifest.status = "failed"
        manifest.progress = self._progress(stage="cancelled", message="Instance was cancelled.")
        self.store.save(self._project_manifest(manifest))
        return self._project_manifest(manifest)

    def retry_instance(self, instance_id: str) -> InstanceManifest:
        legacy_instance_id = self._legacy_instance_id(instance_id)
        self.orchestration.retry_task(legacy_instance_id)
        return self.start_instance(legacy_instance_id)

    def watch_instance(self, instance_id: str, *, timeout_s: float = 30.0) -> dict[str, Any]:
        legacy_instance_id = self._legacy_instance_id(instance_id)
        return asyncio.run(self.orchestration.watch_run(legacy_instance_id, timeout_s=timeout_s))

    def dispatch_ready_tasks(self) -> list[str]:
        started: list[str] = []
        for task in self.orchestration.ready_tasks():
            legacy_instance_id = task.legacy_instance_id
            if not legacy_instance_id:
                continue
            self.start_instance(legacy_instance_id)
            started.append(legacy_instance_id)
        return started

    def monitoring_summary(self) -> dict[str, Any]:
        return self.orchestration.monitoring_summary()

    def heartbeat_instance_attempt(self, instance_id: str, attempt_id: str) -> None:
        self.orchestration.heartbeat(attempt_id)
        manifest = self.store.load(instance_id)
        self.store.save(self._project_manifest(manifest))
