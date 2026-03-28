from __future__ import annotations

import asyncio
import hashlib
import random
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from ai_factory.core.instances.models import InstanceManifest
from ai_factory.core.orchestration.agents import AgentRegistry
from ai_factory.core.orchestration.models import (
    CircuitState,
    OrchestrationEvent,
    OrchestrationRun,
    OrchestrationTask,
    RetryPolicy,
    TaskAttempt,
    TaskDependency,
    TaskInputEnvelope,
    TaskOutputEnvelope,
    utc_now_iso,
)
from ai_factory.core.platform.settings import PlatformSettings


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso_plus(seconds: float) -> str:
    return (_now() + timedelta(seconds=seconds)).isoformat()


def _stable_hash(parts: list[str]) -> str:
    body = "|".join(parts)
    return hashlib.sha1(body.encode("utf-8")).hexdigest()[:12]


class OrchestrationService:
    def __init__(self, control_plane, settings: PlatformSettings):
        self.control_plane = control_plane
        self.settings = settings
        self.registry = AgentRegistry()

    def _primary_ids(self, legacy_instance_id: str) -> tuple[str, str]:
        suffix = _stable_hash([legacy_instance_id])
        return f"run-{suffix}", f"task-{suffix}"

    def _event_id(self) -> str:
        return f"evt-{uuid.uuid4().hex[:12]}"

    def _attempt_id(self) -> str:
        return f"att-{uuid.uuid4().hex[:12]}"

    def _agent_for_manifest(self, manifest: InstanceManifest) -> str:
        mapping = {
            "prepare": "data_processing",
            "train": "training_orchestration",
            "finetune": "training_orchestration",
            "evaluate": "evaluation_benchmarking",
            "report": "monitoring_telemetry",
            "inference": "monitoring_telemetry",
            "deploy": "deployment",
        }
        return mapping.get(manifest.type, "monitoring_telemetry")

    def _resource_for_manifest(self, manifest: InstanceManifest) -> str:
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
        retry_limit = max(
            int(execution.get("retry_limit") or 0),
            int(sub_agents.get("retry_limit") or 0),
            1,
        )
        base_delay_s = int((snapshot.get("resilience") or {}).get("base_delay_s") or 5)
        max_delay_s = int((snapshot.get("resilience") or {}).get("max_delay_s") or 300)
        return RetryPolicy(
            max_attempts=max(retry_limit, 1),
            base_delay_s=base_delay_s,
            max_delay_s=max_delay_s,
        )

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
            task_type=manifest.type,
            agent_type=self._agent_for_manifest(manifest),
            status="queued",
            resource_class=self._resource_for_manifest(manifest),
            retry_policy=self._retry_policy_for_manifest(manifest, snapshot),
            input=TaskInputEnvelope(
                task_type=manifest.type,
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
                    capability.model_dump(mode="json")
                    for capability in self.registry.list_capabilities()
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
                    payload={"parent_run_id": parent_run.id},
                )
        return run, task

    def create_task(
        self,
        *,
        run_id: str,
        task_type: str,
        input_payload: dict[str, Any] | None = None,
        dependencies: list[str] | None = None,
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
        status = "blocked" if dependencies else "ready"
        task = OrchestrationTask(
            id=task_id,
            run_id=run_id,
            legacy_instance_id=legacy_instance_id,
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
            self.control_plane.create_dependency(
                TaskDependency(task_id=task.id, depends_on_task_id=dependency_id)
            )
        self.append_event(
            run_id,
            task.id,
            None,
            event_type="task.created",
            message=f"Queued {task.task_type} task for {descriptor.label}.",
            agent_type=descriptor.agent_type,
            payload={"dependencies": dependencies or []},
        )
        return task

    def ready_tasks(self, run_id: str | None = None) -> list[OrchestrationTask]:
        tasks = self.control_plane.list_tasks(run_id=run_id)
        completed = {task.id for task in tasks if task.status == "completed"}
        ready: list[OrchestrationTask] = []
        for task in tasks:
            if task.status not in {"ready", "queued", "retry_waiting", "blocked"}:
                continue
            dependencies = self.control_plane.list_dependencies(task.id)
            if any(dep.depends_on_task_id not in completed for dep in dependencies):
                continue
            if self.is_circuit_open(task.agent_type):
                continue
            if datetime.fromisoformat(task.available_at.replace("Z", "+00:00")) > _now():
                continue
            if task.status != "ready":
                task.status = "ready"
                task.updated_at = utc_now_iso()
                self.control_plane.upsert_task(task)
            ready.append(task)
        ready.sort(key=lambda item: (item.priority, item.available_at, item.created_at))
        return ready

    def begin_attempt(
        self,
        *,
        legacy_instance_id: str,
        stdout_path: str | None = None,
        stderr_path: str | None = None,
        lease_owner: str = "local-runner",
        metadata: dict[str, Any] | None = None,
    ) -> tuple[OrchestrationRun, OrchestrationTask, TaskAttempt]:
        run = self.control_plane.get_run_by_legacy_instance(legacy_instance_id)
        task = self.control_plane.get_task_by_legacy_instance(legacy_instance_id)
        if run is None or task is None:
            raise FileNotFoundError(f"No orchestration task found for {legacy_instance_id}")
        task.current_attempt += 1
        task.status = "running"
        task.started_at = task.started_at or utc_now_iso()
        task.updated_at = utc_now_iso()
        run.status = "running"
        run.updated_at = utc_now_iso()
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
        self.control_plane.upsert_run(run)
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
        return run, task, attempt

    def heartbeat(self, attempt_id: str) -> TaskAttempt:
        attempt = self.control_plane.get_attempt(attempt_id)
        if attempt is None:
            raise FileNotFoundError(f"Unknown attempt: {attempt_id}")
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
        run = self.control_plane.get_run_by_legacy_instance(legacy_instance_id)
        task = self.control_plane.get_task_by_legacy_instance(legacy_instance_id)
        attempt = self.control_plane.get_attempt(attempt_id)
        if run is None or task is None or attempt is None:
            raise FileNotFoundError(f"Unknown orchestration state for {legacy_instance_id}")

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
            run.status = "completed"
            self.record_agent_success(task.agent_type)
        else:
            task.last_error_code = error_code or "execution_failed"
            task.last_error_message = error_message or f"Task exited with code {exit_code}"
            retry_allowed = task.current_attempt < task.retry_policy.max_attempts
            if retry_allowed:
                delay_s = self.compute_retry_delay(task.retry_policy, task.current_attempt)
                task.status = "retry_waiting"
                task.available_at = _iso_plus(delay_s)
                run.status = "running"
            else:
                task.status = "dead_lettered"
                task.finished_at = task.updated_at
                run.status = "failed"
            self.record_agent_failure(task.agent_type, task.last_error_message)
        run.updated_at = utc_now_iso()
        self.control_plane.upsert_task(task)
        self.control_plane.upsert_run(run)
        self.append_event(
            run.id,
            task.id,
            attempt.id,
            event_type="task.completed" if exit_code == 0 else "task.failed",
            level="info" if exit_code == 0 else "error",
            message=(
                f"{task.task_type} task completed successfully."
                if exit_code == 0
                else f"{task.task_type} task failed."
            ),
            agent_type=task.agent_type,
            payload={
                "exit_code": exit_code,
                "retry_allowed": task.status == "retry_waiting",
                "checkpoint_hint": checkpoint_hint,
            },
        )
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
        return run

    def retry_task(self, legacy_instance_id: str) -> OrchestrationTask:
        run = self.control_plane.get_run_by_legacy_instance(legacy_instance_id)
        task = self.control_plane.get_task_by_legacy_instance(legacy_instance_id)
        if run is None or task is None:
            raise FileNotFoundError(f"Unknown orchestration task for {legacy_instance_id}")
        task.status = "ready"
        task.available_at = utc_now_iso()
        task.updated_at = utc_now_iso()
        task.last_error_code = None
        task.last_error_message = None
        run.status = "queued"
        run.updated_at = utc_now_iso()
        self.control_plane.upsert_task(task)
        self.control_plane.upsert_run(run)
        self.append_event(
            run.id,
            task.id,
            None,
            event_type="task.retry_requested",
            level="warning",
            message="Manual retry requested.",
            agent_type=task.agent_type,
        )
        return task

    def recover_stalled_tasks(self) -> list[OrchestrationTask]:
        stale_before = utc_now_iso()
        recovered: list[OrchestrationTask] = []
        for lease in self.control_plane.list_stale_leases(stale_before=stale_before):
            task = self.control_plane.get_task(str(lease["task_id"]))
            attempt = self.control_plane.get_attempt(str(lease["attempt_id"]))
            if task is None or attempt is None:
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
            run = self.control_plane.get_run(task.run_id)
            if run is not None:
                run.status = "running" if retry_allowed else "failed"
                run.updated_at = utc_now_iso()
                self.control_plane.upsert_run(run)
                self.append_event(
                    run.id,
                    task.id,
                    attempt.id,
                    event_type="task.recovered",
                    level="warning",
                    message="Recovered stalled task after heartbeat expiry.",
                    agent_type=task.agent_type,
                    payload={"retry_allowed": retry_allowed},
                )
        return recovered

    def compute_retry_delay(self, policy: RetryPolicy, current_attempt: int) -> int:
        raw = policy.base_delay_s * (policy.multiplier ** max(current_attempt - 1, 0))
        bounded = min(int(raw), policy.max_delay_s)
        jitter = random.uniform(0, policy.jitter_s) if policy.jitter_s > 0 else 0.0
        return max(int(round(bounded + jitter)), 0)

    def record_agent_failure(self, agent_type: str, message: str) -> CircuitState:
        circuit = self.control_plane.get_circuit(agent_type) or CircuitState(agent_type=agent_type)
        circuit.failure_count += 1
        if circuit.failure_count >= 3:
            circuit.status = "open"
            circuit.opened_at = utc_now_iso()
            circuit.reopen_after = _iso_plus(60)
        circuit.last_error = message
        circuit.updated_at = utc_now_iso()
        return self.control_plane.upsert_circuit(circuit)

    def record_agent_success(self, agent_type: str) -> CircuitState:
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

    def monitoring_summary(self) -> dict[str, Any]:
        runs = self.control_plane.list_runs()
        tasks = self.control_plane.list_tasks()
        status_counts: dict[str, int] = {}
        for task in tasks:
            status_counts[task.status] = status_counts.get(task.status, 0) + 1
        return {
            "runs": len(runs),
            "tasks": len(tasks),
            "task_status_counts": status_counts,
            "open_circuits": [
                circuit.agent_type
                for circuit in (
                    self.control_plane.get_circuit("data_processing"),
                    self.control_plane.get_circuit("training_orchestration"),
                    self.control_plane.get_circuit("evaluation_benchmarking"),
                    self.control_plane.get_circuit("monitoring_telemetry"),
                    self.control_plane.get_circuit("optimization_feedback"),
                    self.control_plane.get_circuit("deployment"),
                )
                if circuit is not None and circuit.status == "open"
            ],
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
        active_agents = [task.agent_type] if task.status == "running" else []
        task_summary = {
            "status": task.status,
            "task_type": task.task_type,
            "attempts": len(attempts),
            "resource_class": task.resource_class,
            "current_attempt": task.current_attempt,
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
        level: str = "info",
        agent_type: str | None = None,
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
        ready = self.ready_tasks()
        if limit is not None:
            ready = ready[:limit]
        return ready

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
        return {"run": run.model_dump(mode="json") if run else None, "events": [item.model_dump(mode="json") for item in events]}
