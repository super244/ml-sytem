from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import uuid
from typing import Any

import yaml

from ai_factory.core.config.loader import (
    apply_environment_override,
    build_orchestration_config,
    load_cloud_profile,
    load_orchestration_config,
)
from ai_factory.core.decisions.rules import decide_next_step
from ai_factory.core.execution.commands import UnsupportedInstanceTypeError, build_command
from ai_factory.core.execution.local import LocalExecutor
from ai_factory.core.execution.ssh import SshExecutor
from ai_factory.core.instances.models import (
    EnvironmentSpec,
    ExecutionHandle,
    InstanceError,
    InstanceManifest,
    InstanceStatus,
    utc_now_iso,
)
from ai_factory.core.instances.queries import InstanceQueryService
from ai_factory.core.instances.store import FileInstanceStore
from ai_factory.core.io import write_json
from ai_factory.core.monitoring.collectors import collect_metrics_for_instance
from ai_factory.core.monitoring.events import InstanceEvent


class InstanceManager:
    def __init__(self, store: FileInstanceStore):
        self.store = store
        self.queries = InstanceQueryService(store)

    def _make_instance_id(self, instance_type: str) -> str:
        return f"{instance_type}-{utc_now_iso().replace(':', '').replace('+00:00', 'z').replace('-', '').replace('.', '')}-{uuid.uuid4().hex[:8]}"

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

    def _base_manifest(
        self,
        config,
        *,
        config_path: str | None,
        parent_instance_id: str | None = None,
    ) -> InstanceManifest:
        instance_id = self._make_instance_id(config.instance.type)
        environment = self._resolve_environment(config.instance.environment)
        return InstanceManifest(
            id=instance_id,
            type=config.instance.type,
            status="pending",
            environment=environment,
            name=config.instance.name or instance_id,
            parent_instance_id=parent_instance_id or config.instance.parent_instance_id,
            config_path=config_path or config.config_path,
            config_snapshot_path=str(self.store.config_snapshot_path(instance_id)),
            artifact_refs={"instance_dir": str(self.store.instance_dir(instance_id))},
        )

    def create_instance(
        self,
        config_path: str,
        *,
        start: bool = True,
        environment_override: EnvironmentSpec | None = None,
        parent_instance_id: str | None = None,
    ) -> InstanceManifest:
        config = load_orchestration_config(config_path)
        config = apply_environment_override(config, environment_override)
        manifest = self._base_manifest(config, config_path=config_path, parent_instance_id=parent_instance_id)
        self.store.create(manifest, config.model_dump(mode="json"))
        return self.start_instance(manifest.id) if start else manifest

    def start_instance(self, instance_id: str) -> InstanceManifest:
        manifest = self.store.load(instance_id)
        snapshot = self.store.load_config_snapshot(instance_id)
        config = build_orchestration_config(snapshot, config_path=manifest.config_path)
        try:
            command = build_command(config, manifest)
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

        executor = SshExecutor() if manifest.environment.kind == "cloud" else LocalExecutor()
        manifest.execution = ExecutionHandle(
            backend=executor.backend_name,
            stdout_path=str(self.store.stdout_path(instance_id)),
            stderr_path=str(self.store.stderr_path(instance_id)),
        )
        self.store.save(manifest)
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
        self.store.save(refreshed)
        self.store.append_event(
            instance_id,
            InstanceEvent(
                type="instance.start_requested",
                message=f"Queued {manifest.type} instance for {handle.backend} execution.",
                payload={"backend": handle.backend, "pid": handle.pid},
            ),
        )
        return manifest

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
        self.store.save(manifest)
        self.store.append_event(
            instance_id,
            InstanceEvent(
                type="instance.running",
                message=f"{manifest.type} instance is running.",
                payload={"backend": execution.backend},
            ),
        )
        return manifest

    def mark_failed(
        self,
        instance_id: str,
        *,
        code: str,
        message: str,
        details: dict[str, Any],
    ) -> InstanceManifest:
        manifest = self.store.load(instance_id)
        manifest.status = "failed"
        manifest.error = InstanceError(code=code, message=message, details=details)
        if manifest.execution is not None:
            manifest.execution.ended_at = utc_now_iso()
            manifest.execution.exit_code = manifest.execution.exit_code if manifest.execution.exit_code is not None else 1
        self.store.save(manifest)
        self.store.append_event(
            instance_id,
            InstanceEvent(type="instance.failed", message=message, payload={"code": code, **details}),
        )
        return manifest

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
        self.store.write_current_metrics(instance_id, summary)
        if config.monitoring.write_timeseries:
            self.store.append_metric_points(instance_id, points)

        if manifest.type == "evaluate" and manifest.status == "completed":
            decision = decide_next_step(summary, config.decision_policy)
            manifest.decision = decision
            self.store.write_decision_report(instance_id, decision.model_dump(mode="json"))
            if config.pipeline.auto_continue or config.decision_policy.auto_continue:
                self._auto_continue(manifest, config, decision.action)

        self.store.save(manifest)
        self.store.append_event(
            instance_id,
            InstanceEvent(
                type=f"instance.{manifest.status}",
                message=f"{manifest.type} instance {manifest.status}.",
                payload={"exit_code": exit_code, "metrics_summary": summary},
            ),
        )
        return manifest

    def _auto_continue(self, manifest: InstanceManifest, config, action: str) -> None:
        if action == "deploy":
            self.create_deployment_instance(
                manifest.parent_instance_id or manifest.id,
                target="huggingface",
                config_path=config.pipeline.default_deploy_config,
                start=True,
            )
        elif action == "finetune":
            self.create_instance(
                config.pipeline.default_finetune_config,
                start=True,
                parent_instance_id=manifest.id,
            )

    def _build_eval_snapshot(self, manifest: InstanceManifest, source: InstanceManifest) -> dict[str, Any]:
        snapshot = self.store.load_config_snapshot(manifest.id)
        source_artifact = (
            (source.artifact_refs.get("published") or {}).get("final_adapter")
            or (source.artifact_refs.get("published") or {}).get("merged_model")
            or source.artifact_refs.get("source_artifact")
        )
        if not source_artifact:
            snapshot["subsystem"]["source_instance_id"] = source.id
            snapshot["metadata"]["source_instance_id"] = source.id
            return snapshot

        eval_config_path = snapshot.get("resolved_subsystem_config_path") or snapshot["subsystem"].get("config_ref")
        if not eval_config_path:
            snapshot["subsystem"]["source_instance_id"] = source.id
            return snapshot

        eval_payload = yaml.safe_load(Path(eval_config_path).read_text()) or {}
        registry_path = Path(eval_payload.get("models", {}).get("registry_path", "inference/configs/model_registry.yaml"))
        registry_payload = yaml.safe_load(registry_path.read_text()) or {}
        generated_name = f"instance_{source.id.replace('-', '_')}"
        registry_payload.setdefault("models", [])
        registry_payload["models"] = [
            item for item in registry_payload["models"] if item.get("name") != generated_name
        ]
        registry_payload["models"].append(
            {
                "name": generated_name,
                "label": source.name,
                "base_model": source.artifact_refs.get("base_model", "Qwen/Qwen2.5-Math-1.5B-Instruct"),
                "adapter_path": source_artifact,
                "load_in_4bit": True,
                "load_in_8bit": False,
                "dtype": "bfloat16",
                "description": f"Generated instance model for {source.id}",
                "tags": ["instance", source.type],
            }
        )
        generated_registry_path = self.store.instance_dir(manifest.id) / "generated_model_registry.yaml"
        generated_eval_path = self.store.instance_dir(manifest.id) / "generated_eval_config.yaml"
        generated_registry_path.write_text(yaml.safe_dump(registry_payload, sort_keys=False))

        eval_payload.setdefault("models", {})
        eval_payload["models"]["registry_path"] = str(generated_registry_path)
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

    def create_evaluation_instance(
        self,
        source_instance_id: str,
        *,
        config_path: str = "configs/eval.yaml",
        start: bool = True,
    ) -> InstanceManifest:
        source = self.store.load(source_instance_id)
        manifest = self.create_instance(
            config_path,
            start=False,
            environment_override=source.environment,
            parent_instance_id=source.id,
        )
        snapshot = self._build_eval_snapshot(manifest, source)
        self._persist_snapshot(manifest.id, snapshot)
        return self.start_instance(manifest.id) if start else manifest

    def create_deployment_instance(
        self,
        source_instance_id: str,
        *,
        target: str,
        config_path: str = "configs/deploy.yaml",
        start: bool = True,
    ) -> InstanceManifest:
        source = self.store.load(source_instance_id)
        manifest = self.create_instance(
            config_path,
            start=False,
            environment_override=EnvironmentSpec(kind="local"),
            parent_instance_id=source.id,
        )
        snapshot = self.store.load_config_snapshot(manifest.id)
        snapshot["subsystem"]["provider"] = target
        snapshot["subsystem"]["source_instance_id"] = source.id
        snapshot["subsystem"]["source_artifact_ref"] = (
            (source.artifact_refs.get("published") or {}).get("final_adapter")
            or (source.artifact_refs.get("published") or {}).get("merged_model")
            or source.artifact_refs.get("source_artifact")
        )
        snapshot.setdefault("metadata", {})
        snapshot["metadata"]["source_instance_id"] = source.id
        self._persist_snapshot(manifest.id, snapshot)
        return self.start_instance(manifest.id) if start else manifest

    def get_instance(self, instance_id: str) -> InstanceManifest:
        return self.store.load(instance_id)

    def list_instances(self) -> list[InstanceManifest]:
        return self.queries.list_instances()

    def get_logs(self, instance_id: str) -> dict[str, str]:
        return self.store.read_logs(instance_id)

    def get_metrics(self, instance_id: str) -> dict[str, Any]:
        return {
            "summary": self.store.read_current_metrics(instance_id),
            "points": self.store.read_metric_points(instance_id),
        }
