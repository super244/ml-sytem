from __future__ import annotations

import uuid
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from ai_factory.platform.monitoring import hardware
from inference.app import workspace as workspace_module
from inference.app.config import AppSettings
from inference.app.schemas import InstanceActionRequest, InstanceCreateRequest

ActionKind = Literal["launch_training", "run_action", "advisory"]
ActionStatus = Literal["planned", "executed", "blocked", "failed", "skipped"]
LoopStatus = Literal["planned", "executed", "partial", "blocked", "failed"]


class AutonomousLoopAction(BaseModel):
    id: str
    kind: ActionKind
    title: str
    detail: str
    priority: int
    executable: bool = False
    status: ActionStatus = "planned"
    source_instance_id: str | None = None
    source_instance_name: str | None = None
    action: str | None = None
    config_path: str | None = None
    deployment_target: str | None = None
    surface: str = "dashboard"
    href: str = "/dashboard"
    command: str | None = None
    created_instance_id: str | None = None
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class AutonomousLoopRun(BaseModel):
    id: str
    created_at: str
    status: LoopStatus
    dry_run: bool = False
    start_instances: bool = False
    blockers: list[str] = Field(default_factory=list)
    summary: dict[str, Any] = Field(default_factory=dict)
    actions: list[AutonomousLoopAction] = Field(default_factory=list)


class AutonomousLoopSnapshot(BaseModel):
    generated_at: str
    ready: bool
    blockers: list[str] = Field(default_factory=list)
    summary: dict[str, Any] = Field(default_factory=dict)
    actions: list[AutonomousLoopAction] = Field(default_factory=list)
    recent_loops: list[AutonomousLoopRun] = Field(default_factory=list)
    latest_loop: AutonomousLoopRun | None = None


class AutonomousLoopService:
    def __init__(self, settings: AppSettings, *, instance_service: Any):
        self.settings = settings
        self.instance_service = instance_service
        self.repo_root = Path(settings.repo_root).resolve()
        self.loops_file = self.repo_root / "data" / "autonomous" / "loops.jsonl"

    def _now_iso(self) -> str:
        return datetime.now(UTC).isoformat()

    def _load_loop_runs(self) -> list[AutonomousLoopRun]:
        if not self.loops_file.exists():
            return []
        runs: list[AutonomousLoopRun] = []
        for line in self.loops_file.read_text().splitlines():
            if not line.strip():
                continue
            try:
                runs.append(AutonomousLoopRun.model_validate_json(line))
            except Exception:
                continue
        runs.sort(key=lambda item: item.created_at, reverse=True)
        return runs

    def _append_loop_run(self, run: AutonomousLoopRun) -> None:
        self.loops_file.parent.mkdir(parents=True, exist_ok=True)
        with self.loops_file.open("a", encoding="utf-8") as handle:
            handle.write(run.model_dump_json() + "\n")

    def _instance_rows(self) -> list[dict[str, Any]]:
        rows = self.instance_service.list_instances()
        payload: list[dict[str, Any]] = []
        for item in rows:
            if hasattr(item, "model_dump"):
                payload.append(item.model_dump(mode="json"))
            else:
                payload.append(dict(item))
        return payload

    def _orchestration_summary(self) -> dict[str, Any]:
        summary = self.instance_service.get_orchestration_summary()
        raw = getattr(summary, "summary", summary)
        return raw if isinstance(raw, dict) else {}

    def _candidate_actions(self) -> tuple[list[AutonomousLoopAction], dict[str, Any], list[str]]:
        workspace = workspace_module.build_workspace_overview(self.repo_root)
        workspace_summary = workspace.get("summary", {})
        ready_checks = int(workspace_summary.get("ready_checks", 0) or 0)
        total_checks = int(workspace_summary.get("total_checks", 0) or 0)
        workspace_ready = total_checks > 0 and ready_checks == total_checks

        instances = self._instance_rows()
        running_instances = [item for item in instances if item.get("status") == "running"]
        failed_instances = [item for item in instances if item.get("status") == "failed"]
        completed_instances = [item for item in instances if item.get("status") == "completed"]
        completed_by_type = Counter(str(item.get("type", "unknown")) for item in completed_instances)

        orchestration_summary = self._orchestration_summary()
        open_circuits = (
            orchestration_summary.get("open_circuits", []) if isinstance(orchestration_summary, dict) else []
        )
        telemetry_dir = self.repo_root / "data" / "telemetry" / "flagged.jsonl"
        telemetry_backlog = 0
        if telemetry_dir.exists():
            telemetry_backlog = sum(1 for line in telemetry_dir.read_text().splitlines() if line.strip())
        cluster_nodes = hardware.get_cluster_nodes()
        idle_nodes = [node for node in cluster_nodes if node.get("status") == "idle"]

        blockers: list[str] = []
        if total_checks and not workspace_ready:
            blockers.append(f"workspace readiness is incomplete ({ready_checks}/{total_checks})")
        if failed_instances:
            blockers.append(f"{len(failed_instances)} failed managed instance(s) need attention")
        if open_circuits:
            blockers.append(f"{len(open_circuits)} orchestration circuit(s) are open")

        actions: list[AutonomousLoopAction] = []
        seen: set[tuple[str | None, str | None, str | None, str | None]] = set()

        def add_action(action: AutonomousLoopAction) -> None:
            key = (action.source_instance_id, action.action, action.config_path, action.deployment_target)
            if key in seen:
                return
            seen.add(key)
            actions.append(action)

        if workspace_ready and not instances:
            add_action(
                AutonomousLoopAction(
                    id=f"auto-{uuid.uuid4().hex[:10]}",
                    kind="launch_training",
                    title="Seed the lab with a baseline training branch",
                    detail=(
                        "The workspace is ready and there are no managed instances yet. "
                        "Queue a baseline training branch to start the autonomous lifecycle."
                    ),
                    priority=100,
                    executable=True,
                    config_path="configs/train.yaml",
                    surface="training",
                    href="/dashboard/training",
                    metadata={"reason": "bootstrap"},
                )
            )

        for item in completed_instances:
            instance_id = item.get("id")
            if not isinstance(instance_id, str):
                continue
            detail = self.instance_service.get_instance(instance_id)
            available_actions = getattr(detail, "available_actions", [])
            name = getattr(detail, "name", item.get("name", instance_id))
            instance_type = str(getattr(detail, "type", item.get("type", "unknown")))
            base_priority = 80 if instance_type == "evaluate" else 70
            for available_action in available_actions:
                target_action = available_action.action
                if target_action not in {
                    "evaluate",
                    "re_evaluate",
                    "finetune",
                    "retrain",
                    "deploy",
                    "report",
                    "open_inference",
                }:
                    continue
                priority_boost = {
                    "deploy": 18,
                    "finetune": 16,
                    "retrain": 14,
                    "evaluate": 15,
                    "re_evaluate": 13,
                    "report": 10,
                    "open_inference": 8,
                }.get(target_action, 0)
                add_action(
                    AutonomousLoopAction(
                        id=f"auto-{uuid.uuid4().hex[:10]}",
                        kind="run_action",
                        title=available_action.label,
                        detail=available_action.description,
                        priority=base_priority + priority_boost,
                        executable=True,
                        source_instance_id=instance_id,
                        source_instance_name=str(name),
                        action=target_action,
                        config_path=available_action.config_path,
                        deployment_target=available_action.deployment_target,
                        surface="runs",
                        href=f"/runs/{instance_id}",
                        metadata={"source_type": instance_type},
                    )
                )

        if telemetry_backlog:
            add_action(
                AutonomousLoopAction(
                    id=f"auto-{uuid.uuid4().hex[:10]}",
                    kind="advisory",
                    title="Review flagged telemetry and promote new training rows",
                    detail=(
                        f"{telemetry_backlog} flagged interaction(s) are waiting in the backlog. "
                        "Triaging them now will keep the self-improvement loop grounded in real failures."
                    ),
                    priority=64,
                    executable=False,
                    surface="datasets",
                    href="/dashboard/datasets",
                    metadata={"telemetry_backlog": telemetry_backlog},
                )
            )

        if idle_nodes:
            add_action(
                AutonomousLoopAction(
                    id=f"auto-{uuid.uuid4().hex[:10]}",
                    kind="advisory",
                    title="Use idle cluster capacity for search or evaluation",
                    detail=(
                        f"{len(idle_nodes)} cluster node(s) are idle right now. "
                        "This is a good window for AutoML search, evaluation replays, or a fresh training branch."
                    ),
                    priority=52,
                    executable=False,
                    surface="cluster",
                    href="/dashboard/cluster",
                    metadata={"idle_nodes": len(idle_nodes)},
                )
            )

        actions.sort(key=lambda item: (-item.priority, item.title, item.id))
        summary = {
            "workspace_ready": workspace_ready,
            "ready_checks": ready_checks,
            "total_checks": total_checks,
            "instances": len(instances),
            "completed_instances": len(completed_instances),
            "running_instances": len(running_instances),
            "failed_instances": len(failed_instances),
            "completed_evaluations": completed_by_type.get("evaluate", 0),
            "completed_training_branches": completed_by_type.get("train", 0) + completed_by_type.get("finetune", 0),
            "telemetry_backlog": telemetry_backlog,
            "idle_nodes": len(idle_nodes),
            "open_circuits": len(open_circuits),
            "total_actions": len(actions),
            "executable_actions": sum(1 for item in actions if item.executable),
            "advisory_actions": sum(1 for item in actions if not item.executable),
        }
        return actions, summary, blockers

    def snapshot(self, *, max_actions: int = 10) -> AutonomousLoopSnapshot:
        actions, summary, blockers = self._candidate_actions()
        recent_loops = self._load_loop_runs()[:8]
        ready = bool(summary.get("workspace_ready")) and not blockers
        return AutonomousLoopSnapshot(
            generated_at=self._now_iso(),
            ready=ready,
            blockers=blockers,
            summary=summary,
            actions=actions[:max_actions],
            recent_loops=recent_loops,
            latest_loop=recent_loops[0] if recent_loops else None,
        )

    def plan(self, *, max_actions: int = 6) -> AutonomousLoopRun:
        snapshot = self.snapshot(max_actions=max_actions)
        status: LoopStatus = "planned" if snapshot.ready and snapshot.summary.get("total_actions") else "blocked"
        run = AutonomousLoopRun(
            id=f"loop-{uuid.uuid4().hex[:10]}",
            created_at=self._now_iso(),
            status=status,
            blockers=snapshot.blockers,
            summary=snapshot.summary,
            actions=snapshot.actions,
        )
        self._append_loop_run(run)
        return run

    def execute(
        self,
        *,
        max_actions: int = 2,
        dry_run: bool = False,
        start_instances: bool = False,
    ) -> AutonomousLoopRun:
        snapshot = self.snapshot(max_actions=max_actions * 3)
        executable_actions = [item.model_copy(deep=True) for item in snapshot.actions if item.executable][:max_actions]
        if not executable_actions:
            run = AutonomousLoopRun(
                id=f"loop-{uuid.uuid4().hex[:10]}",
                created_at=self._now_iso(),
                status="blocked",
                dry_run=dry_run,
                start_instances=start_instances,
                blockers=snapshot.blockers or ["no executable autonomous actions are available"],
                summary=snapshot.summary,
                actions=[],
            )
            self._append_loop_run(run)
            return run

        results: list[AutonomousLoopAction] = []
        created_instance_ids: list[str] = []
        failures = 0

        for action in executable_actions:
            if dry_run:
                action.status = "planned"
                results.append(action)
                continue
            try:
                if action.kind == "launch_training":
                    detail = self.instance_service.create_instance(
                        InstanceCreateRequest(
                            config_path=action.config_path or "configs/train.yaml",
                            start=start_instances,
                            user_level="hobbyist",
                            metadata={
                                "source": "autonomous_loop",
                                "reason": action.metadata.get("reason", "autonomous_queue"),
                            },
                        )
                    )
                elif action.kind == "run_action":
                    if not action.source_instance_id or not action.action:
                        raise ValueError("autonomous action is missing source_instance_id or action")
                    detail = self.instance_service.run_instance_action(
                        action.source_instance_id,
                        InstanceActionRequest(
                            action=action.action,
                            config_path=action.config_path,
                            deployment_target=action.deployment_target,
                            start=start_instances,
                        ),
                    )
                else:
                    action.status = "skipped"
                    results.append(action)
                    continue

                action.status = "executed"
                action.created_instance_id = detail.id
                created_instance_ids.append(detail.id)
            except Exception as exc:
                failures += 1
                action.status = "failed"
                action.error = str(exc)
            results.append(action)

        status: LoopStatus
        if failures == len(results):
            status = "failed"
        elif failures:
            status = "partial"
        else:
            status = "executed"

        summary = dict(snapshot.summary)
        summary.update(
            {
                "requested_actions": len(executable_actions),
                "executed_actions": sum(1 for item in results if item.status == "executed"),
                "failed_actions": failures,
                "created_instances": len(created_instance_ids),
                "created_instance_ids": created_instance_ids,
            }
        )
        run = AutonomousLoopRun(
            id=f"loop-{uuid.uuid4().hex[:10]}",
            created_at=self._now_iso(),
            status=status,
            dry_run=dry_run,
            start_instances=start_instances,
            blockers=snapshot.blockers,
            summary=summary,
            actions=results,
        )
        self._append_loop_run(run)
        return run
