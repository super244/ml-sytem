from __future__ import annotations

import asyncio
import hashlib
import secrets
import uuid
from collections import Counter
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any, Literal, cast

if TYPE_CHECKING:
    from ai_factory.core.orchestration.sqlite import SqliteControlPlane

from ai_factory.core.instances.models import InstanceManifest
from ai_factory.core.orchestration.agents import AgentRegistry
from ai_factory.core.orchestration.models import (
    AgentType,
    CircuitState,
    OrchestrationEvent,
    OrchestrationRun,
    OrchestrationTask,
    ResourceClass,
    RetryPolicy,
    TaskAttempt,
    TaskDependency,
    TaskInputEnvelope,
    TaskLease,
    TaskOutputEnvelope,
    TaskType,
    utc_now_iso,
)
from ai_factory.core.platform.settings import PlatformSettings
from ai_factory.titan import detect_titan_status

DEFAULT_CIRCUIT_FAILURE_THRESHOLD = 3
DEFAULT_CIRCUIT_REOPEN_AFTER_S = 60


def _now() -> datetime:
    return datetime.now(UTC)


def _iso_plus(seconds: float) -> str:
    return (_now() + timedelta(seconds=seconds)).isoformat()


def _parse_iso(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _stable_hash(parts: list[str]) -> str:
    body = "|".join(parts)
    return hashlib.sha256(body.encode("utf-8")).hexdigest()[:12]


EventLevel = Literal["debug", "info", "warning", "error"]


class OrchestrationService:
    def __init__(self, control_plane: SqliteControlPlane, settings: PlatformSettings) -> None:
        self.control_plane = control_plane
        self.settings = settings
        self.registry = AgentRegistry()
        self._titan_status: dict[str, Any] | None = None
        self._titan_status_cached_at: datetime | None = None

    def _get_titan_status(self) -> dict[str, Any]:
        """Get cached Titan status with TTL of 60 seconds."""
        now = _now()
        if (
            self._titan_status is None
            or self._titan_status_cached_at is None
            or (now - self._titan_status_cached_at).total_seconds() > 60
        ):
            try:
                self._titan_status = detect_titan_status()
                self._titan_status_cached_at = now
            except Exception:
                self._titan_status = {}
                self._titan_status_cached_at = now
        return self._titan_status

    def get_hardware_capabilities(self) -> dict[str, Any]:
        """Get hardware capabilities for scheduling decisions."""
        titan = self._get_titan_status()
        return {
            "backend": titan.get("backend", "unknown"),
            "gpu_count": titan.get("gpu_count", 0),
            "gpu_name": titan.get("gpu_name"),
            "supports_cuda": titan.get("supports_cuda", False),
            "supports_metal": titan.get("supports_metal", False),
            "cpu_threads": titan.get("cpu_threads", 1),
            "unified_memory_gb": titan.get("unified_memory_gb"),
        }

    def _primary_ids(self, legacy_instance_id: str) -> tuple[str, str]:
        suffix = _stable_hash([legacy_instance_id])
        return f"run-{suffix}", f"task-{suffix}"

    def _event_id(self) -> str:
        return f"evt-{uuid.uuid4().hex[:12]}"

    def _attempt_id(self) -> str:
        return f"att-{uuid.uuid4().hex[:12]}"

    def _agent_for_manifest(self, manifest: InstanceManifest) -> AgentType:
        mapping: dict[str, AgentType] = {
            "prepare": "data_processing",
            "train": "training_orchestration",
            "finetune": "training_orchestration",
            "evaluate": "evaluation_benchmarking",
            "report": "monitoring_telemetry",
            "inference": "monitoring_telemetry",
            "deploy": "deployment",
        }
        return mapping.get(manifest.type, "monitoring_telemetry")

    def _resource_for_manifest(self, manifest: InstanceManifest) -> ResourceClass:
        if manifest.type in {"train", "finetune"}:
            return "gpu"
        if manifest.type in {"prepare", "report"}:
            return "io"
        if manifest.type == "deploy":
            return "network"
        return "cpu"

    def _retry_policy_for_manifest(self, manifest: InstanceManifest, snapshot: dict[str, Any]) -> RetryPolicy:
        execution = snapshot.get("execution") or {}
        sub_agents = snapshot.get("sub_agents") or {}
        resilience = snapshot.get("resilience") or {}
        retry_limit = max(
            int(execution.get("retry_limit") or 0),
            int(sub_agents.get("retry_limit") or 0),
            1,
        )
        base_delay_s = int(resilience.get("base_delay_s") or 5)
        max_delay_s = int(resilience.get("max_delay_s") or 300)
        return RetryPolicy(
            max_attempts=max(retry_limit, 1),
            base_delay_s=base_delay_s,
            max_delay_s=max_delay_s,
            multiplier=float(resilience.get("multiplier") or 2.0),
            jitter_s=float(resilience.get("jitter_s") or 0.25),
        )

    def _task_lineage_context(
        self,
        task: OrchestrationTask,
        run: OrchestrationRun | None,
        attempt: TaskAttempt | None = None,
    ) -> dict[str, Any]:
        lineage: dict[str, Any] = {
            "run_id": task.run_id,
            "legacy_instance_id": task.legacy_instance_id,
            "task_id": task.id,
            "task_type": task.task_type,
            "agent_type": task.agent_type,
            "parent_task_id": task.parent_task_id,
        }
        if run is not None:
            lineage.update(
                {
                    "run_status": run.status,
                    "root_run_id": run.root_run_id,
                    "parent_run_id": run.parent_run_id,
                    "run_legacy_instance_id": run.legacy_instance_id,
                }
            )
        if attempt is not None:
            lineage.update(
                {
                    "attempt_id": attempt.id,
                    "attempt_sequence": attempt.sequence,
                    "attempt_status": attempt.status,
                }
            )
        return lineage

    def _circuit_failure_threshold(self, agent_type: AgentType) -> int:
        descriptor = self.registry.get(agent_type)
        return max(descriptor.retry_policy.max_attempts, DEFAULT_CIRCUIT_FAILURE_THRESHOLD)

    def _circuit_reopen_after_s(self, agent_type: AgentType) -> int:
        descriptor = self.registry.get(agent_type)
        return max(descriptor.retry_policy.max_delay_s, DEFAULT_CIRCUIT_REOPEN_AFTER_S)

    def _workload_to_task_type(self, workload: str) -> TaskType | None:
        mapping: dict[str, TaskType] = {
            "preprocess": "prepare",
            "metrics": "monitor",
            "evaluation": "evaluate",
            "finetune": "finetune",
            "publish": "deploy",
        }
        return mapping.get(workload)

    def _seed_sub_agent_tasks(
        self,
        run: OrchestrationRun,
        primary_task: OrchestrationTask,
        manifest: InstanceManifest,
        snapshot: dict[str, Any],
    ) -> None:
        raw_sub_agents = snapshot.get("sub_agents") or {}
        if not raw_sub_agents.get("enabled"):
            return

        workloads = list(raw_sub_agents.get("workloads") or [])
        if not workloads:
            return

        max_parallelism = max(int(raw_sub_agents.get("max_parallelism") or 1), 1)
        created_by_workload: dict[str, OrchestrationTask] = {}

        for index, workload in enumerate(workloads):
            task_type = self._workload_to_task_type(str(workload))
            if task_type is None:
                continue
            if task_type == cast(TaskType, manifest.type):
                # Avoid spawning duplicate sub-agents for the exact primary task type.
                continue

            dependencies: list[str] = [primary_task.id]
            if task_type == "prepare":
                dependencies = []
            elif task_type == "deploy" and "evaluation" in created_by_workload:
                dependencies = [created_by_workload["evaluation"].id]

            created_task = self.create_task(
                run_id=run.id,
                task_type=task_type,
                dependencies=dependencies,
                parent_task_id=primary_task.id,
                idempotency_key=f"{run.id}:sub-agent:{workload}",
                input_payload={
                    "source_instance_id": manifest.id,
                    "workload": workload,
                    "parent_task_id": primary_task.id,
                    "config_path": manifest.config_path,
                },
                metadata={
                    "spawned_by": "sub_agents",
                    "workload": workload,
                    "parallelism_cap": max_parallelism,
                    "sub_agent_slot": (index % max_parallelism) + 1,
                },
            )
            created_by_workload[str(workload)] = created_task

        preprocess_task = created_by_workload.get("preprocess")
        if preprocess_task is not None:
            self.control_plane.create_dependency(
                TaskDependency(task_id=primary_task.id, depends_on_task_id=preprocess_task.id)
            )
            primary_task.status = "blocked"
            primary_task.updated_at = utc_now_iso()
            self.control_plane.upsert_task(primary_task)
            self.append_event(
                run.id,
                primary_task.id,
                None,
                event_type="task.blocked",
                message="Primary task is waiting for preprocess sub-agent completion.",
                agent_type=primary_task.agent_type,
                payload={"depends_on": preprocess_task.id},
            )
        deploy_task = created_by_workload.get("publish")
        evaluation_task = created_by_workload.get("evaluation")
        if deploy_task is not None and evaluation_task is not None:
            self.control_plane.create_dependency(
                TaskDependency(task_id=deploy_task.id, depends_on_task_id=evaluation_task.id)
            )
            deploy_task.updated_at = utc_now_iso()
            self.control_plane.upsert_task(deploy_task)

    def ensure_run_for_instance(
        self,
        manifest: InstanceManifest,
        snapshot: dict[str, Any],
    ) -> tuple[OrchestrationRun, OrchestrationTask]:
        existing_run = self.control_plane.get_run_by_legacy_instance(manifest.id)
        if existing_run:
            task = self.control_plane.get_task_by_legacy_instance(manifest.id)
            if task is None:
                raise FileNotFoundError(f"Primary orchestration task missing for {manifest.id}")
            return existing_run, task

        run_id, task_id = self._primary_ids(manifest.id)
        metadata = {
            "instance_type": manifest.type,
            "environment": manifest.environment.model_dump(mode="json"),
            "parent_instance_id": manifest.parent_instance_id,
            "config_path": manifest.config_path,
            "lineage": {
                "legacy_instance_id": manifest.id,
                "root_run_id": run_id,
                "parent_run_id": None,
            },
        }
        run = OrchestrationRun(
            id=run_id,
            legacy_instance_id=manifest.id,
            name=manifest.name,
            status="queued",
            parent_run_id=None,
            root_run_id=run_id,
            idempotency_key=(snapshot.get("metadata") or {}).get("idempotency_key"),
            metadata=metadata,
        )
        task = OrchestrationTask(
            id=task_id,
            run_id=run.id,
            legacy_instance_id=manifest.id,
            task_type=cast(TaskType, manifest.type),
            agent_type=self._agent_for_manifest(manifest),
            status="queued",
            resource_class=self._resource_for_manifest(manifest),
            retry_policy=self._retry_policy_for_manifest(manifest, snapshot),
            input=TaskInputEnvelope(
                task_type=cast(TaskType, manifest.type),
                legacy_instance_id=manifest.id,
                config_path=manifest.config_path,
                payload={
                    "manifest_name": manifest.name,
                    "snapshot_ref": manifest.config_snapshot_path,
                },
                environment=manifest.environment.model_dump(mode="json"),
                labels=list((snapshot.get("subsystem") or {}).get("labels") or []),
                idempotency_key=(snapshot.get("metadata") or {}).get("idempotency_key"),
                resource_class=self._resource_for_manifest(manifest),
            ),
            metadata={
                "agent_capabilities": [
                    capability.model_dump(mode="json") for capability in self.registry.list_capabilities()
                ]
            },
        )
        self.control_plane.upsert_run(run)
        self.control_plane.upsert_task(task)
        self.append_event(
            run.id,
            task.id,
            None,
            event_type="run.created",
            message=f"Created orchestration run for {manifest.type} instance.",
            payload={"legacy_instance_id": manifest.id, "agent_type": task.agent_type},
        )
        if manifest.parent_instance_id:
            parent_run = self.control_plane.get_run_by_legacy_instance(manifest.parent_instance_id)
            if parent_run:
                run.parent_run_id = parent_run.id
                run.root_run_id = parent_run.root_run_id or parent_run.id
                self.control_plane.upsert_run(run)
                self.append_event(
                    run.id,
                    task.id,
                    None,
                    event_type="run.lineage.attached",
                    message="Attached run to parent orchestration lineage.",
                    payload={
                        "parent_run_id": parent_run.id,
                        "root_run_id": parent_run.root_run_id or parent_run.id,
                        "legacy_parent_instance_id": manifest.parent_instance_id,
                    },
                )
        self._seed_sub_agent_tasks(run, task, manifest, snapshot)
        return run, task

    def create_task(
        self,
        *,
        run_id: str,
        task_type: TaskType,
        input_payload: dict[str, Any] | None = None,
        dependencies: list[str] | None = None,
        parent_task_id: str | None = None,
        legacy_instance_id: str | None = None,
        idempotency_key: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> OrchestrationTask:
        descriptor = self.registry.agent_for_task_type(task_type)
        if idempotency_key:
            for task in self.control_plane.list_tasks(run_id=run_id):
                if task.input.idempotency_key == idempotency_key:
                    return task
        task_id = f"task-{uuid.uuid4().hex[:12]}"
        status: Literal["blocked", "ready"] = "blocked" if dependencies else "ready"
        task = OrchestrationTask(
            id=task_id,
            run_id=run_id,
            legacy_instance_id=legacy_instance_id,
            parent_task_id=parent_task_id,
            task_type=task_type,
            agent_type=descriptor.agent_type,
            status=status,
            resource_class=descriptor.capability().resource_classes[0],
            retry_policy=descriptor.retry_policy,
            input=TaskInputEnvelope(
                task_type=task_type,
                legacy_instance_id=legacy_instance_id,
                payload=input_payload or {},
                idempotency_key=idempotency_key,
                resource_class=descriptor.capability().resource_classes[0],
            ),
            metadata=metadata or {},
        )
        self.control_plane.upsert_task(task)
        for dependency_id in dependencies or []:
            self.control_plane.create_dependency(TaskDependency(task_id=task.id, depends_on_task_id=dependency_id))
        self.append_event(
            run_id,
            task.id,
            None,
            event_type="task.created",
            message=f"Queued {task.task_type} task for {descriptor.label}.",
            agent_type=descriptor.agent_type,
            payload={
                "dependencies": dependencies or [],
                "parent_task_id": parent_task_id,
                "lineage": self._task_lineage_context(task, self.control_plane.get_run(run_id)),
                "registry_candidate": self.registry.describe_task_type(task_type),
            },
        )
        return task

    def _resolve_task(self, target: str) -> OrchestrationTask | None:
        return self.control_plane.get_task(target) or self.control_plane.get_task_by_legacy_instance(target)

    def _candidate_ready_tasks(
        self,
        run_id: str | None = None,
        *,
        legacy_only: bool = False,
    ) -> list[OrchestrationTask]:
        tasks = self.control_plane.list_tasks(run_id=run_id)
        completed = {task.id for task in tasks if task.status == "completed"}
        ready: list[OrchestrationTask] = []
        for task in tasks:
            if legacy_only and task.legacy_instance_id is None:
                continue
            if task.status not in {"ready", "queued", "retry_waiting", "blocked"}:
                continue
            dependencies = self.control_plane.list_dependencies(task.id)
            if any(dep.depends_on_task_id not in completed for dep in dependencies):
                continue
            if self.control_plane.get_lease(task.id) is not None:
                continue
            if self.is_circuit_open(task.agent_type):
                continue
            if _parse_iso(task.available_at) > _now():
                continue
            if task.status != "ready":
                task.status = "ready"
                task.updated_at = utc_now_iso()
                self.control_plane.upsert_task(task)
            ready.append(task)
        ready.sort(key=lambda item: (item.priority, item.available_at, item.created_at))
        return ready

    def _dispatchable_ready_tasks(
        self,
        *,
        run_id: str | None = None,
        limit: int | None = None,
        legacy_only: bool = False,
    ) -> list[OrchestrationTask]:
        ready = self._candidate_ready_tasks(run_id=run_id, legacy_only=legacy_only)
        running_tasks = self.control_plane.list_tasks(status="running")
        remaining_global_slots = max(self.settings.worker_concurrency - len(running_tasks), 0)
        running_by_agent = Counter(task.agent_type for task in running_tasks)
        dispatchable: list[OrchestrationTask] = []

        for task in ready:
            descriptor = self.registry.get(task.agent_type)
            if running_by_agent[task.agent_type] >= descriptor.max_concurrency:
                continue
            if remaining_global_slots <= 0:
                break
            dispatchable.append(task)
            running_by_agent[task.agent_type] += 1
            remaining_global_slots -= 1
            if limit is not None and len(dispatchable) >= limit:
                break
        return dispatchable

    def _task_counts_toward_run(self, task: OrchestrationTask) -> bool:
        if task.legacy_instance_id is not None:
            return True
        if task.current_attempt > 0:
            return True
        return task.status not in {"queued", "ready", "blocked"}

    def _sync_run_status(self, run_id: str) -> OrchestrationRun:
        run = self.control_plane.get_run(run_id)
        if run is None:
            raise FileNotFoundError(f"Unknown orchestration run: {run_id}")

        tasks = self.control_plane.list_tasks(run_id=run_id)
        relevant_tasks = [task for task in tasks if self._task_counts_toward_run(task)] or tasks

        if not relevant_tasks:
            run.status = "queued"
        elif all(task.status == "cancelled" for task in relevant_tasks):
            run.status = "cancelled"
        elif any(task.status in {"failed", "dead_lettered"} for task in relevant_tasks):
            run.status = "failed"
        elif any(task.status in {"running", "retry_waiting"} for task in relevant_tasks):
            run.status = "running"
        elif all(task.status == "completed" for task in relevant_tasks):
            run.status = "completed"
        elif any(task.status in {"ready", "blocked", "queued"} for task in relevant_tasks):
            run.status = "queued"
        run.updated_at = utc_now_iso()
        self.control_plane.upsert_run(run)
        return run

    def _require_active_attempt(self, task: OrchestrationTask, attempt: TaskAttempt) -> TaskLease:
        if attempt.task_id != task.id:
            raise ValueError(f"Attempt {attempt.id} does not belong to task {task.id}")
        if task.status != "running":
            raise ValueError(f"Task {task.id} is not running")
        if task.current_attempt != attempt.sequence:
            raise ValueError(f"Attempt {attempt.id} is no longer active for task {task.id}")
        lease = self.control_plane.get_lease(task.id)
        if lease is None:
            raise ValueError(f"Task {task.id} does not have an active lease")
        if lease.attempt_id != attempt.id:
            raise ValueError(f"Lease for task {task.id} belongs to a different attempt")
        return lease

    def task_readiness(self, task_id: str) -> dict[str, Any]:
        task = self.control_plane.get_task(task_id)
        if task is None:
            raise FileNotFoundError(f"Unknown orchestration task: {task_id}")

        blockers: list[str] = []
        dependency_ids: list[str] = []
        for dependency in self.control_plane.list_dependencies(task.id):
            dependency_task = self.control_plane.get_task(dependency.depends_on_task_id)
            if dependency_task is None or dependency_task.status != "completed":
                dependency_ids.append(dependency.depends_on_task_id)

        if dependency_ids:
            blockers.append("waiting_on_dependencies")
        if task.status not in {"queued", "ready", "retry_waiting", "blocked"}:
            blockers.append(f"status:{task.status}")
        if self.control_plane.get_lease(task.id) is not None:
            blockers.append("active_lease")
        if self.is_circuit_open(task.agent_type):
            blockers.append(f"circuit_open:{task.agent_type}")
        if _parse_iso(task.available_at) > _now():
            blockers.append("waiting_for_backoff")

        return {
            "task_id": task.id,
            "ready": len(blockers) == 0,
            "blockers": blockers,
            "dependency_ids": dependency_ids,
            "available_at": task.available_at,
        }

    def describe_task(self, task_id: str) -> dict[str, Any]:
        task = self.control_plane.get_task(task_id)
        if task is None:
            raise FileNotFoundError(f"Unknown orchestration task: {task_id}")
        dependencies = self.control_plane.list_dependencies(task.id)
        dependents = [item for item in self.control_plane.list_dependencies() if item.depends_on_task_id == task.id]
        lease = self.control_plane.get_lease(task.id)
        run = self.control_plane.get_run(task.run_id)
        return {
            "task": task.model_dump(mode="json"),
            "run": run.model_dump(mode="json") if run else None,
            "attempts": [item.model_dump(mode="json") for item in self.control_plane.list_attempts(task.id)],
            "dependencies": [item.model_dump(mode="json") for item in dependencies],
            "dependents": [item.model_dump(mode="json") for item in dependents],
            "lease": lease.model_dump(mode="json") if lease else None,
            "readiness": self.task_readiness(task.id),
        }

    def ready_tasks(self, run_id: str | None = None) -> list[OrchestrationTask]:
        return self._candidate_ready_tasks(run_id=run_id)

    def begin_attempt(
        self,
        *,
        legacy_instance_id: str,
        stdout_path: str | None = None,
        stderr_path: str | None = None,
        lease_owner: str = "local-runner",
        metadata: dict[str, Any] | None = None,
    ) -> tuple[OrchestrationRun, OrchestrationTask, TaskAttempt]:
        task = self.control_plane.get_task_by_legacy_instance(legacy_instance_id)
        if task is None:
            raise FileNotFoundError(f"No orchestration task found for {legacy_instance_id}")
        return self.begin_task_attempt(
            task_id=task.id,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            lease_owner=lease_owner,
            metadata=metadata,
        )

    def begin_task_attempt(
        self,
        *,
        task_id: str,
        stdout_path: str | None = None,
        stderr_path: str | None = None,
        lease_owner: str = "local-runner",
        metadata: dict[str, Any] | None = None,
    ) -> tuple[OrchestrationRun, OrchestrationTask, TaskAttempt]:
        task = self.control_plane.get_task(task_id)
        if task is None:
            raise FileNotFoundError(f"Unknown orchestration task: {task_id}")
        readiness = self.task_readiness(task.id)
        if not readiness["ready"]:
            blocker_text = ", ".join(str(item) for item in readiness["blockers"])
            raise ValueError(f"Task {task.id} is not ready to start: {blocker_text}")
        run = self.control_plane.get_run(task.run_id)
        if run is None:
            raise FileNotFoundError(f"Unknown orchestration run: {task.run_id}")
        task.current_attempt += 1
        task.status = "running"
        task.started_at = task.started_at or utc_now_iso()
        task.updated_at = utc_now_iso()
        attempt = TaskAttempt(
            id=self._attempt_id(),
            task_id=task.id,
            sequence=task.current_attempt,
            status="running",
            lease_owner=lease_owner,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            metadata=metadata or {},
        )
        self.control_plane.upsert_task(task)
        self.control_plane.upsert_attempt(attempt)
        self.control_plane.write_lease(
            task_id=task.id,
            attempt_id=attempt.id,
            lease_owner=lease_owner,
            acquired_at=attempt.started_at,
            heartbeat_at=attempt.heartbeat_at,
            expires_at=_iso_plus(self.settings.stale_after_s),
        )
        self.append_event(
            run.id,
            task.id,
            attempt.id,
            event_type="task.running",
            message=f"{task.task_type} task is running.",
            agent_type=task.agent_type,
            payload={"attempt": attempt.sequence, "lease_owner": lease_owner},
        )
        run = self._sync_run_status(task.run_id)
        return run, task, attempt

    def heartbeat(self, attempt_id: str) -> TaskAttempt:
        attempt = self.control_plane.get_attempt(attempt_id)
        if attempt is None:
            raise FileNotFoundError(f"Unknown attempt: {attempt_id}")
        task = self.control_plane.get_task(attempt.task_id)
        if task is None:
            raise FileNotFoundError(f"Unknown orchestration task: {attempt.task_id}")
        self._require_active_attempt(task, attempt)
        attempt.heartbeat_at = utc_now_iso()
        self.control_plane.upsert_attempt(attempt)
        self.control_plane.write_lease(
            task_id=attempt.task_id,
            attempt_id=attempt.id,
            lease_owner=attempt.lease_owner or "runner",
            acquired_at=attempt.started_at,
            heartbeat_at=attempt.heartbeat_at,
            expires_at=_iso_plus(self.settings.stale_after_s),
        )
        task = self.control_plane.get_task(attempt.task_id)
        if task is not None:
            run = self.control_plane.get_run(task.run_id)
            if run is not None:
                self.append_event(
                    run.id,
                    task.id,
                    attempt.id,
                    event_type="task.heartbeat",
                    message="Task heartbeat received.",
                    agent_type=task.agent_type,
                    payload={"lease_owner": attempt.lease_owner},
                )
        return attempt

    def finalize_attempt(
        self,
        *,
        legacy_instance_id: str,
        attempt_id: str,
        exit_code: int,
        summary: dict[str, Any] | None = None,
        metrics: dict[str, Any] | None = None,
        artifacts: dict[str, Any] | None = None,
        recommendations: list[dict[str, Any]] | None = None,
        checkpoint_hint: str | None = None,
        error_code: str | None = None,
        error_message: str | None = None,
    ) -> tuple[OrchestrationRun, OrchestrationTask, TaskAttempt]:
        task = self.control_plane.get_task_by_legacy_instance(legacy_instance_id)
        if task is None:
            raise FileNotFoundError(f"Unknown orchestration state for {legacy_instance_id}")
        return self.finalize_task_attempt(
            task_id=task.id,
            attempt_id=attempt_id,
            exit_code=exit_code,
            summary=summary,
            metrics=metrics,
            artifacts=artifacts,
            recommendations=recommendations,
            checkpoint_hint=checkpoint_hint,
            error_code=error_code,
            error_message=error_message,
        )

    def finalize_task_attempt(
        self,
        *,
        task_id: str,
        attempt_id: str,
        exit_code: int,
        summary: dict[str, Any] | None = None,
        metrics: dict[str, Any] | None = None,
        artifacts: dict[str, Any] | None = None,
        recommendations: list[dict[str, Any]] | None = None,
        checkpoint_hint: str | None = None,
        error_code: str | None = None,
        error_message: str | None = None,
    ) -> tuple[OrchestrationRun, OrchestrationTask, TaskAttempt]:
        task = self.control_plane.get_task(task_id)
        attempt = self.control_plane.get_attempt(attempt_id)
        if task is None or attempt is None:
            raise FileNotFoundError(f"Unknown orchestration task or attempt: {task_id}")
        run = self.control_plane.get_run(task.run_id)
        if run is None:
            raise FileNotFoundError(f"Unknown orchestration run: {task.run_id}")
        if attempt.finished_at is not None:
            if task.current_attempt != attempt.sequence:
                raise ValueError(f"Attempt {attempt_id} is no longer active for task {task_id}")
            return run, task, attempt

        self._require_active_attempt(task, attempt)

        attempt.status = "completed" if exit_code == 0 else "failed"
        attempt.finished_at = utc_now_iso()
        attempt.heartbeat_at = attempt.finished_at
        attempt.exit_code = exit_code
        attempt.checkpoint_hint = checkpoint_hint
        attempt.error_code = error_code
        attempt.error_message = error_message
        self.control_plane.upsert_attempt(attempt)
        self.control_plane.drop_lease(task.id)

        task.output = TaskOutputEnvelope(
            summary=summary or {},
            metrics=metrics or {},
            artifacts=artifacts or {},
            checkpoint_hint=checkpoint_hint,
            recommendations=recommendations or [],
            status_updates=[],
        )
        task.checkpoint_hint = checkpoint_hint
        task.updated_at = utc_now_iso()
        if exit_code == 0:
            task.status = "completed"
            task.finished_at = task.updated_at
            task.last_error_code = None
            task.last_error_message = None
            self.record_agent_success(task.agent_type)
        else:
            task.last_error_code = error_code or "execution_failed"
            task.last_error_message = error_message or f"Task exited with code {exit_code}"
            retry_allowed = task.current_attempt < task.retry_policy.max_attempts
            if retry_allowed:
                delay_s = self.compute_retry_delay(task.retry_policy, task.current_attempt)
                task.status = "retry_waiting"
                task.available_at = _iso_plus(delay_s)
            else:
                task.status = "dead_lettered"
                task.finished_at = task.updated_at
            self.record_agent_failure(task.agent_type, task.last_error_message)
        self.control_plane.upsert_task(task)
        self.append_event(
            run.id,
            task.id,
            attempt.id,
            event_type="task.completed" if exit_code == 0 else "task.failed",
            level="info" if exit_code == 0 else "error",
            message=(
                f"{task.task_type} task completed successfully." if exit_code == 0 else f"{task.task_type} task failed."
            ),
            agent_type=task.agent_type,
            payload={
                "exit_code": exit_code,
                "retry_allowed": task.status == "retry_waiting",
                "checkpoint_hint": checkpoint_hint,
                "next_available_at": task.available_at if task.status == "retry_waiting" else None,
                "current_attempt": task.current_attempt,
                "retry_policy": task.retry_policy.model_dump(mode="json"),
                "lineage": self._task_lineage_context(task, run, attempt),
            },
        )
        run = self._sync_run_status(task.run_id)
        return run, task, attempt

    def cancel_run(self, legacy_or_run_id: str) -> OrchestrationRun:
        run = self.control_plane.get_run(legacy_or_run_id) or self.control_plane.get_run_by_legacy_instance(
            legacy_or_run_id
        )
        if run is None:
            raise FileNotFoundError(f"Unknown orchestration run: {legacy_or_run_id}")
        run.status = "cancelled"
        run.updated_at = utc_now_iso()
        self.control_plane.upsert_run(run)
        for task in self.control_plane.list_tasks(run_id=run.id):
            if task.status in {"completed", "failed", "dead_lettered"}:
                continue
            active_attempt = self.control_plane.get_active_attempt_for_task(task.id)
            if active_attempt is not None:
                active_attempt.status = "cancelled"
                active_attempt.finished_at = utc_now_iso()
                active_attempt.heartbeat_at = active_attempt.finished_at
                active_attempt.error_code = "cancelled"
                active_attempt.error_message = "Run cancelled."
                self.control_plane.upsert_attempt(active_attempt)
            self.control_plane.drop_lease(task.id)
            task.status = "cancelled"
            task.finished_at = utc_now_iso()
            task.updated_at = task.finished_at
            self.control_plane.upsert_task(task)
        self.append_event(
            run.id,
            None,
            None,
            event_type="run.cancelled",
            level="warning",
            message="Cancelled orchestration run.",
        )
        return self._sync_run_status(run.id)

    def retry_task(self, task_or_legacy_id: str) -> OrchestrationTask:
        task = self._resolve_task(task_or_legacy_id)
        if task is None:
            raise FileNotFoundError(f"Unknown orchestration task for {task_or_legacy_id}")
        if task.status == "running":
            raise ValueError(f"Cannot retry running task: {task.id}")
        task.status = "ready"
        task.available_at = utc_now_iso()
        task.finished_at = None
        task.updated_at = utc_now_iso()
        task.last_error_code = None
        task.last_error_message = None
        self.control_plane.upsert_task(task)
        run = self._sync_run_status(task.run_id)
        self.append_event(
            run.id,
            task.id,
            None,
            event_type="task.retry_requested",
            level="warning",
            message="Manual retry requested.",
            agent_type=task.agent_type,
            payload={
                "current_attempt": task.current_attempt,
                "available_at": task.available_at,
                "lineage": self._task_lineage_context(task, run),
            },
        )
        return task

    def recover_stalled_tasks(self) -> list[OrchestrationTask]:
        stale_before = utc_now_iso()
        recovered: list[OrchestrationTask] = []
        for lease in self.control_plane.list_stale_leases(stale_before=stale_before):
            task = self.control_plane.get_task(str(lease["task_id"]))
            attempt = self.control_plane.get_attempt(str(lease["attempt_id"]))
            if task is None or attempt is None:
                self.control_plane.drop_lease(str(lease["task_id"]))
                continue
            if task.status in {"completed", "failed", "cancelled", "dead_lettered"}:
                self.control_plane.drop_lease(task.id)
                continue
            if attempt.status != "running":
                self.control_plane.drop_lease(task.id)
                continue
            attempt.status = "failed"
            attempt.finished_at = utc_now_iso()
            attempt.error_code = "lease_expired"
            attempt.error_message = "Task heartbeat expired."
            self.control_plane.upsert_attempt(attempt)
            self.control_plane.drop_lease(task.id)
            retry_allowed = task.current_attempt < task.retry_policy.max_attempts
            if retry_allowed:
                task.status = "retry_waiting"
                task.available_at = _iso_plus(self.compute_retry_delay(task.retry_policy, task.current_attempt))
            else:
                task.status = "dead_lettered"
                task.finished_at = utc_now_iso()
            task.last_error_code = "lease_expired"
            task.last_error_message = "Task heartbeat expired."
            task.updated_at = utc_now_iso()
            self.control_plane.upsert_task(task)
            recovered.append(task)
            self.record_agent_failure(task.agent_type, task.last_error_message)
            run = self._sync_run_status(task.run_id)
            retry_delay_s = self.compute_retry_delay(task.retry_policy, task.current_attempt) if retry_allowed else None
            self.append_event(
                run.id,
                task.id,
                attempt.id,
                event_type="task.recovered",
                level="warning",
                message="Recovered stalled task after heartbeat expiry.",
                agent_type=task.agent_type,
                payload={
                    "retry_allowed": retry_allowed,
                    "lease_owner": attempt.lease_owner,
                    "expires_at": lease["expires_at"],
                    "retry_delay_s": retry_delay_s,
                    "lineage": self._task_lineage_context(task, run, attempt),
                },
            )
        return recovered

    def compute_retry_delay(self, policy: RetryPolicy, current_attempt: int) -> int:
        raw = policy.base_delay_s * (policy.multiplier ** max(current_attempt - 1, 0))
        bounded = min(int(raw), policy.max_delay_s)
        jitter = 0.0
        if policy.jitter_s > 0:
            jitter_ms = max(int(policy.jitter_s * 1000), 0)
            jitter = (secrets.randbelow(jitter_ms + 1) / 1000.0) if jitter_ms > 0 else 0.0
        return max(min(int(round(bounded + jitter)), policy.max_delay_s), 0)

    def record_agent_failure(self, agent_type: AgentType, message: str) -> CircuitState:
        circuit = self.control_plane.get_circuit(agent_type) or CircuitState(agent_type=agent_type)
        circuit.failure_count += 1
        threshold = self._circuit_failure_threshold(agent_type)
        cooldown_s = self._circuit_reopen_after_s(agent_type)
        if circuit.failure_count >= threshold:
            if circuit.status != "open":
                circuit.opened_at = utc_now_iso()
            circuit.status = "open"
            circuit.reopen_after = _iso_plus(cooldown_s)
        circuit.last_error = message
        circuit.updated_at = utc_now_iso()
        return self.control_plane.upsert_circuit(circuit)

    def record_agent_success(self, agent_type: AgentType) -> CircuitState:
        circuit = self.control_plane.get_circuit(agent_type) or CircuitState(agent_type=agent_type)
        circuit.status = "closed"
        circuit.failure_count = 0
        circuit.opened_at = None
        circuit.reopen_after = None
        circuit.last_error = None
        circuit.updated_at = utc_now_iso()
        return self.control_plane.upsert_circuit(circuit)

    def is_circuit_open(self, agent_type: str) -> bool:
        circuit = self.control_plane.get_circuit(agent_type)
        if circuit is None:
            return False
        if circuit.status != "open":
            return False
        if circuit.reopen_after:
            reopen_after = datetime.fromisoformat(circuit.reopen_after.replace("Z", "+00:00"))
            if reopen_after <= _now():
                circuit.status = "half_open"
                circuit.updated_at = utc_now_iso()
                self.control_plane.upsert_circuit(circuit)
                return False
        return True

    def latest_checkpoint(self, legacy_instance_id: str) -> str | None:
        task = self.control_plane.get_task_by_legacy_instance(legacy_instance_id)
        if task is None:
            return None
        if task.checkpoint_hint:
            return task.checkpoint_hint
        attempts = self.control_plane.list_attempts(task.id)
        for attempt in reversed(attempts):
            if attempt.checkpoint_hint:
                return attempt.checkpoint_hint
        return None

    def list_runs(self) -> list[OrchestrationRun]:
        return self.control_plane.list_runs()

    def list_tasks(self, legacy_or_run_id: str | None = None) -> list[OrchestrationTask]:
        if legacy_or_run_id is None:
            return self.control_plane.list_tasks()
        run = self.control_plane.get_run(legacy_or_run_id) or self.control_plane.get_run_by_legacy_instance(
            legacy_or_run_id
        )
        if run is not None:
            return self.control_plane.list_tasks(run_id=run.id)
        task = self.control_plane.get_task(legacy_or_run_id) or self.control_plane.get_task_by_legacy_instance(
            legacy_or_run_id
        )
        return [task] if task else []

    def list_events(self, legacy_or_run_id: str, *, limit: int | None = None) -> list[OrchestrationEvent]:
        run = self.control_plane.get_run(legacy_or_run_id) or self.control_plane.get_run_by_legacy_instance(
            legacy_or_run_id
        )
        if run is None:
            return []
        return self.control_plane.list_events(run_id=run.id, limit=limit)

    def summarize_run(self, legacy_or_run_id: str) -> dict[str, Any]:
        run = self.control_plane.get_run(legacy_or_run_id) or self.control_plane.get_run_by_legacy_instance(
            legacy_or_run_id
        )
        if run is None:
            raise FileNotFoundError(f"Unknown orchestration run: {legacy_or_run_id}")
        tasks = self.control_plane.list_tasks(run_id=run.id)
        task_status_counts = dict(Counter(task.status for task in tasks))
        task_type_counts = dict(Counter(task.task_type for task in tasks))
        ready_tasks = self.ready_tasks(run.id)
        dispatchable_tasks = self._dispatchable_ready_tasks(run_id=run.id)
        run_leases = [
            lease
            for lease in self.control_plane.list_leases()
            if (task := self.control_plane.get_task(lease.task_id)) is not None and task.run_id == run.id
        ]
        attempts = [attempt for task in tasks for attempt in self.control_plane.list_attempts(task.id)]
        events = self.control_plane.list_events(run_id=run.id)
        return {
            "run_id": run.id,
            "status": run.status,
            "lineage": {
                "legacy_instance_id": run.legacy_instance_id,
                "root_run_id": run.root_run_id,
                "parent_run_id": run.parent_run_id,
            },
            "tasks": len(tasks),
            "task_status_counts": task_status_counts,
            "task_type_counts": task_type_counts,
            "ready_tasks": len(ready_tasks),
            "dispatchable_tasks": len(dispatchable_tasks),
            "blocked_tasks": task_status_counts.get("blocked", 0),
            "retry_waiting_tasks": task_status_counts.get("retry_waiting", 0),
            "dead_lettered_tasks": task_status_counts.get("dead_lettered", 0),
            "active_leases": len(run_leases),
            "last_event_at": events[-1].created_at if events else None,
            "last_event_type": events[-1].event_type if events else None,
            "last_heartbeat_at": attempts[-1].heartbeat_at if attempts else None,
            "open_circuits": sorted({task.agent_type for task in tasks if self.is_circuit_open(task.agent_type)}),
        }

    def monitoring_summary(self) -> dict[str, Any]:
        runs = self.control_plane.list_runs()
        tasks = self.control_plane.list_tasks()
        run_status_counts: dict[str, int] = {}
        status_counts: dict[str, int] = {}
        task_type_counts: dict[str, int] = {}
        resource_class_counts: dict[str, int] = {}
        for task in tasks:
            status_counts[task.status] = status_counts.get(task.status, 0) + 1
            task_type_counts[task.task_type] = task_type_counts.get(task.task_type, 0) + 1
            resource_class_counts[task.resource_class] = resource_class_counts.get(task.resource_class, 0) + 1
        for run in runs:
            run_status_counts[run.status] = run_status_counts.get(run.status, 0) + 1
        ready_tasks = self.ready_tasks()
        dispatchable_tasks = self._dispatchable_ready_tasks()
        ready_by_resource: dict[str, int] = {}
        running_by_agent = dict(Counter(task.agent_type for task in tasks if task.status == "running"))
        dispatchable_by_agent = dict(Counter(task.agent_type for task in dispatchable_tasks))
        for task in ready_tasks:
            ready_by_resource[task.resource_class] = ready_by_resource.get(task.resource_class, 0) + 1
        circuits = {
            circuit.agent_type: circuit.model_dump(mode="json") for circuit in self.control_plane.list_circuits()
        }
        return {
            "runs": len(runs),
            "tasks": len(tasks),
            "active_runs": run_status_counts.get("running", 0),
            "run_status_counts": run_status_counts,
            "task_status_counts": status_counts,
            "task_type_counts": task_type_counts,
            "resource_class_counts": resource_class_counts,
            "ready_tasks": len(ready_tasks),
            "dispatchable_tasks": len(dispatchable_tasks),
            "ready_by_resource": ready_by_resource,
            "running_by_agent": running_by_agent,
            "dispatchable_by_agent": dispatchable_by_agent,
            "blocked_tasks": status_counts.get("blocked", 0),
            "retry_waiting_tasks": status_counts.get("retry_waiting", 0),
            "stale_leases": len(self.control_plane.list_stale_leases(stale_before=utc_now_iso())),
            "circuits": circuits,
            "open_circuits": sorted(
                agent_type for agent_type, circuit in circuits.items() if circuit["status"] == "open"
            ),
        }

    def projection_for_instance(self, legacy_instance_id: str) -> dict[str, Any]:
        run = self.control_plane.get_run_by_legacy_instance(legacy_instance_id)
        task = self.control_plane.get_task_by_legacy_instance(legacy_instance_id)
        if run is None or task is None:
            return {
                "orchestration_run_id": None,
                "task_summary": {},
                "last_heartbeat_at": None,
                "active_agents": [],
            }
        attempts = self.control_plane.list_attempts(task.id)
        last_heartbeat_at = attempts[-1].heartbeat_at if attempts else None
        latest_attempt = attempts[-1] if attempts else None
        active_agents = [task.agent_type] if task.status == "running" else []
        task_summary = {
            "status": task.status,
            "task_type": task.task_type,
            "attempts": len(attempts),
            "resource_class": task.resource_class,
            "current_attempt": task.current_attempt,
            "parent_task_id": task.parent_task_id,
            "checkpoint_hint": task.checkpoint_hint,
            "last_error_code": task.last_error_code,
            "last_error_message": task.last_error_message,
            "latest_attempt_status": latest_attempt.status if latest_attempt else None,
            "lineage": self._task_lineage_context(task, run, latest_attempt),
        }
        return {
            "orchestration_run_id": run.id,
            "task_summary": task_summary,
            "last_heartbeat_at": last_heartbeat_at,
            "active_agents": active_agents,
        }

    def project_manifest(self, manifest: InstanceManifest) -> InstanceManifest:
        projection = self.projection_for_instance(manifest.id)
        manifest.orchestration_run_id = projection["orchestration_run_id"]
        manifest.task_summary = projection["task_summary"]
        manifest.last_heartbeat_at = projection["last_heartbeat_at"]
        manifest.active_agents = projection["active_agents"]
        return manifest

    def append_event(
        self,
        run_id: str,
        task_id: str | None,
        attempt_id: str | None,
        *,
        event_type: str,
        message: str,
        payload: dict[str, Any] | None = None,
        level: EventLevel = "info",
        agent_type: AgentType | None = None,
    ) -> OrchestrationEvent:
        event = OrchestrationEvent(
            id=self._event_id(),
            run_id=run_id,
            task_id=task_id,
            attempt_id=attempt_id,
            event_type=event_type,
            level=level,
            agent_type=agent_type,
            message=message,
            payload=payload or {},
        )
        return self.control_plane.append_event(event)

    async def dispatch_ready_tasks(self, *, limit: int | None = None) -> list[OrchestrationTask]:
        return self._dispatchable_ready_tasks(limit=limit)

    async def dispatch_ready_legacy_tasks(self, *, limit: int | None = None) -> list[OrchestrationTask]:
        return self._dispatchable_ready_tasks(limit=limit, legacy_only=True)

    async def watch_run(
        self,
        legacy_or_run_id: str,
        *,
        poll_interval_s: float = 0.5,
        timeout_s: float = 30.0,
    ) -> dict[str, Any]:
        started = _now()
        initial_count = 0
        while (_now() - started).total_seconds() < timeout_s:
            run = self.control_plane.get_run(legacy_or_run_id) or self.control_plane.get_run_by_legacy_instance(
                legacy_or_run_id
            )
            if run is None:
                return {"run": None, "events": []}
            events = self.control_plane.list_events(run_id=run.id)
            if run.status in {"completed", "failed", "cancelled"} and len(events) >= initial_count:
                return {"run": run.model_dump(mode="json"), "events": [item.model_dump(mode="json") for item in events]}
            initial_count = len(events)
            await asyncio.sleep(poll_interval_s)
        run = self.control_plane.get_run(legacy_or_run_id) or self.control_plane.get_run_by_legacy_instance(
            legacy_or_run_id
        )
        events = self.control_plane.list_events(run_id=run.id) if run else []
        return {
            "run": run.model_dump(mode="json") if run else None,
            "events": [item.model_dump(mode="json") for item in events],
        }
