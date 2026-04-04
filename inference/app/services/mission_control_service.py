from __future__ import annotations

import json
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from ai_factory.platform.monitoring import hardware
from ai_factory.titan import detect_titan_status
from inference.app import workspace as workspace_module
from inference.app.config import AppSettings
from inference.app.services.autonomous_lab import AutonomousLabService


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except Exception:
            continue
        if isinstance(payload, dict):
            records.append(payload)
    return records


def _timestamp(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        if isinstance(value, str):
            try:
                return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
            except ValueError:
                return 0.0
        return 0.0


def _status_counts(items: list[dict[str, Any]], field_name: str = "status") -> dict[str, int]:
    counts = Counter(str(item.get(field_name, "unknown")) for item in items)
    return dict(counts)


def _recent(items: list[dict[str, Any]], field_name: str, limit: int = 12) -> list[dict[str, Any]]:
    return sorted(items, key=lambda item: _timestamp(item.get(field_name)), reverse=True)[:limit]


def _slugify(value: str) -> str:
    cleaned = []
    for char in value.lower():
        cleaned.append(char if char.isalnum() else "-")
    return "".join(cleaned).strip("-") or "item"


_AGENT_ROLE_HINTS: dict[str, tuple[str, ...]] = {
    "data_processing": ("data", "curat", "dataset", "prepare", "preprocess", "pack", "telemetry"),
    "training_orchestration": ("train", "trainer", "lora", "finetune", "runner", "orchestrat"),
    "evaluation_benchmarking": ("eval", "benchmark", "red-team", "judge", "verifier", "score"),
    "monitoring_telemetry": ("monitor", "telemetry", "health", "observ", "heartbeat"),
    "optimization_feedback": ("optimiz", "search", "sweep", "feedback", "tune", "automl"),
    "deployment": ("deploy", "publish", "registry", "release", "serve", "ship"),
}


class AutonomyStageSnapshot(BaseModel):
    id: str
    title: str
    status: Literal["blocked", "active", "attention", "ready", "idle"]
    headline: str
    detail: str
    href: str
    metric_label: str | None = None
    metric_value: str | None = None
    counts: dict[str, int] = Field(default_factory=dict)


class AutonomyAgentCoverage(BaseModel):
    agent_type: str
    label: str
    status: Literal["blocked", "active", "attention", "ready", "idle"]
    queued_tasks: int = 0
    running_tasks: int = 0
    active_swarm_agents: int = 0
    open_circuit: bool = False
    resource_classes: list[str] = Field(default_factory=list)
    recommended_action: str


class AutonomyLineageAlert(BaseModel):
    id: str
    severity: Literal["critical", "warning", "opportunity", "info"]
    title: str
    detail: str
    href: str
    instance_id: str | None = None


class AutonomyNextAction(BaseModel):
    id: str
    title: str
    detail: str
    href: str
    category: Literal["stabilize", "dispatch", "optimize", "lineage"]
    blocking: bool = False
    command: str | None = None


class AutonomyCapacitySnapshot(BaseModel):
    status: Literal["blocked", "active", "ready", "idle"]
    idle_nodes: int = 0
    busy_nodes: int = 0
    offline_nodes: int = 0
    active_gpu_tasks: int = 0
    active_cpu_tasks: int = 0
    schedulable_trials: int = 0
    suggested_parallelism: int = 0
    bottleneck: str
    execution_modes: dict[str, int] = Field(default_factory=dict)


class AutonomyOverview(BaseModel):
    status: Literal["blocked", "degraded", "active", "ready", "idle"]
    mode: Literal["manual", "assisted", "autonomous"]
    summary: str
    open_circuits: int = 0
    telemetry_backlog: int = 0
    active_runs: int = 0
    running_sweeps: int = 0
    stalled_runs: int = 0
    stages: list[AutonomyStageSnapshot] = Field(default_factory=list)
    agent_coverage: list[AutonomyAgentCoverage] = Field(default_factory=list)
    capacity: AutonomyCapacitySnapshot
    lineage_alerts: list[AutonomyLineageAlert] = Field(default_factory=list)
    next_actions: list[AutonomyNextAction] = Field(default_factory=list)


class MissionControlSnapshot(BaseModel):
    generated_at: str
    repo_root: str
    workspace: dict[str, Any] = Field(default_factory=dict)
    orchestration: dict[str, Any] = Field(default_factory=dict)
    watchlist: dict[str, Any] = Field(default_factory=dict)
    control_plane: dict[str, Any] = Field(default_factory=dict)
    autonomous: dict[str, Any] = Field(default_factory=dict)
    agents: dict[str, Any] = Field(default_factory=dict)
    automl: dict[str, Any] = Field(default_factory=dict)
    cluster: dict[str, Any] = Field(default_factory=dict)
    lineage: dict[str, Any] = Field(default_factory=dict)
    telemetry: dict[str, Any] = Field(default_factory=dict)
    titan: dict[str, Any] = Field(default_factory=dict)
    autonomy: AutonomyOverview
    criticality: dict[str, Any] = Field(default_factory=dict)
    recommendations: list[dict[str, Any]] = Field(default_factory=list)
    summary: dict[str, Any] = Field(default_factory=dict)


class MissionControlRecommendation(BaseModel):
    id: str
    severity: Literal["critical", "warning", "opportunity", "info"]
    title: str
    detail: str
    surface: str
    href: str
    metric_label: str | None = None
    metric_value: str | None = None
    command: str | None = None


def _severity_rank(value: str) -> int:
    order = {
        "critical": 0,
        "warning": 1,
        "opportunity": 2,
        "info": 3,
    }
    return order.get(value, len(order))


class MissionControlService:
    def __init__(self, settings: AppSettings, *, instance_service: Any):
        self.settings = settings
        self.instance_service = instance_service
        self.repo_root = Path(settings.repo_root).resolve()

    def _resolve_path(self, raw_path: str | Path) -> Path:
        path = Path(raw_path).expanduser()
        return path if path.is_absolute() else (self.repo_root / path)

    def _agents(self) -> dict[str, Any]:
        agents_path = self.repo_root / "data" / "agents" / "registry.jsonl"
        now = time.time()
        agents = _load_jsonl(agents_path)
        for agent in agents:
            agent["uptime_s"] = max(0, int(now - _timestamp(agent.get("created_at"))))
        return {
            "path": str(agents_path),
            "count": len(agents),
            "status_counts": _status_counts(agents),
            "swarm": agents,
        }

    def _automl(self) -> dict[str, Any]:
        sweeps_path = self.repo_root / "data" / "automl" / "sweeps.jsonl"
        sweeps = _load_jsonl(sweeps_path)
        sweeps = sorted(sweeps, key=lambda item: _timestamp(item.get("created_at")), reverse=True)
        return {
            "path": str(sweeps_path),
            "count": len(sweeps),
            "status_counts": _status_counts(sweeps),
            "latest": sweeps[0] if sweeps else None,
            "sweeps": sweeps[:10],
        }

    def _telemetry(self) -> dict[str, Any]:
        flagged_path = self.repo_root / "data" / "telemetry" / "flagged.jsonl"
        requests_path = self._resolve_path(self.settings.telemetry_path)
        flagged = _load_jsonl(flagged_path)
        requests = _load_jsonl(requests_path)
        model_counts = Counter(
            str(record.get("model_variant", "unknown")) for record in requests if record.get("model_variant")
        )
        latest_flag = max(flagged, key=lambda item: _timestamp(item.get("timestamp")), default=None)
        return {
            "flagged": {
                "path": str(flagged_path),
                "count": len(flagged),
                "recent": _recent(flagged, "timestamp", limit=8),
                "latest": latest_flag,
            },
            "requests": {
                "path": str(requests_path),
                "count": len(requests),
                "by_model": dict(model_counts),
                "recent": _recent(requests, "timestamp", limit=8),
            },
        }

    def _task_payloads(self) -> list[dict[str, Any]]:
        manager = getattr(self.instance_service, "manager", None)
        orchestration = getattr(manager, "orchestration", None)
        if orchestration is None:
            return []
        return [task.model_dump(mode="json") for task in orchestration.list_tasks()]

    def _agent_capabilities(self) -> list[dict[str, Any]]:
        manager = getattr(self.instance_service, "manager", None)
        orchestration = getattr(manager, "orchestration", None)
        if orchestration is None:
            return []
        return [cap.model_dump(mode="json") for cap in orchestration.registry.list_capabilities()]

    def _build_context(self) -> dict[str, Any]:
        workspace = workspace_module.build_workspace_overview(self.repo_root)
        workspace_summary = workspace.get("summary", {})
        instances = [item.model_dump(mode="json") for item in self.instance_service.list_instances()]
        runs_payload = self.instance_service.list_orchestration_runs()
        runs = [
            run.model_dump(mode="json") if hasattr(run, "model_dump") else dict(run)
            for run in getattr(runs_payload, "runs", runs_payload)
        ]
        orchestration_summary = self.instance_service.get_orchestration_summary()
        orchestration_summary_payload = getattr(orchestration_summary, "summary", orchestration_summary)
        autonomous = AutonomousLabService(self.settings, instance_service=self.instance_service).snapshot().model_dump(
            mode="json"
        )
        ready_actions = list(autonomous.get("ready_actions", []))
        autonomous.setdefault(
            "summary",
            {
                "total_actions": len(ready_actions),
                "executable_actions": len(ready_actions),
                "advisory_actions": 0,
                "telemetry_backlog": int((autonomous.get("loop_health") or {}).get("flagged_backlog", 0) or 0),
                "idle_nodes": int((autonomous.get("cluster") or {}).get("idle_nodes", 0) or 0),
            },
        )
        autonomous.setdefault("ready", bool(ready_actions))
        autonomous.setdefault("blockers", [])
        autonomous.setdefault(
            "actions",
            [
                {
                    "id": action.get("id"),
                    "kind": "advisory",
                    "title": action.get("title"),
                    "detail": action.get("detail"),
                    "priority": max(len(ready_actions) - index, 1),
                    "executable": action.get("status") == "planned",
                    "status": "planned",
                    "source_instance_id": action.get("source_instance_id"),
                    "source_instance_name": None,
                    "action": action.get("kind"),
                    "config_path": action.get("config_path"),
                    "deployment_target": None,
                    "surface": "autonomous",
                    "href": "/dashboard/autonomous",
                    "command": None,
                    "created_instance_id": action.get("instance_id"),
                    "error": None,
                    "metadata": action.get("metadata") or {},
                }
                for index, action in enumerate(ready_actions)
            ],
        )
        cluster_nodes = list(autonomous.get("cluster", {}).get("nodes", [])) or hardware.get_cluster_nodes()
        agents = self._agents()
        automl = self._automl()
        telemetry = self._telemetry()
        titan = detect_titan_status(self.repo_root)
        tasks = self._task_payloads()
        capabilities = self._agent_capabilities()
        return {
            "workspace": workspace,
            "workspace_summary": workspace_summary,
            "instances": instances,
            "runs": runs,
            "orchestration_summary": orchestration_summary_payload if isinstance(orchestration_summary_payload, dict) else {},
            "autonomous": autonomous,
            "cluster_nodes": cluster_nodes,
            "agents": agents,
            "automl": automl,
            "telemetry": telemetry,
            "titan": titan,
            "tasks": tasks,
            "capabilities": capabilities,
        }

    def _recommendations(
        self,
        *,
        workspace_summary: dict[str, Any],
        running_instances: list[dict[str, Any]],
        failed_instances: list[dict[str, Any]],
        autonomous: dict[str, Any],
        agents: dict[str, Any],
        automl: dict[str, Any],
        telemetry: dict[str, Any],
        cluster_nodes: list[dict[str, Any]],
        open_circuits: list[Any],
        instances: list[dict[str, Any]],
    ) -> list[MissionControlRecommendation]:
        recommendations: list[MissionControlRecommendation] = []
        ready_checks = int(workspace_summary.get("ready_checks", 0) or 0)
        total_checks = int(workspace_summary.get("total_checks", 0) or 0)
        failed_checks = max(total_checks - ready_checks, 0)
        telemetry_flags = int(telemetry["flagged"]["count"])
        active_agents = int(agents["status_counts"].get("active", 0))
        running_sweeps = int(automl["status_counts"].get("running", 0))
        ready_autonomous_actions = len(autonomous.get("ready_actions", []))
        lineage_gap_count = int((autonomous.get("lineage") or {}).get("gap_count", 0) or 0)
        idle_nodes = [node for node in cluster_nodes if node.get("status") == "idle"]

        if failed_checks:
            recommendations.append(
                MissionControlRecommendation(
                    id="workspace-readiness",
                    severity="critical",
                    title="Finish workspace readiness before scaling the loop",
                    detail=(
                        f"{failed_checks} readiness checks are still unresolved. "
                        "Fix the local control surfaces before launching more automation."
                    ),
                    surface="workspace",
                    href="/workspace",
                    metric_label="ready",
                    metric_value=f"{ready_checks}/{total_checks}",
                )
            )

        if failed_instances:
            latest_failed = failed_instances[0]
            recommendations.append(
                MissionControlRecommendation(
                    id="failed-instances",
                    severity="critical",
                    title="Investigate failed managed instances",
                    detail=(
                        f"{len(failed_instances)} managed instance(s) are failed. "
                        f"Start with '{latest_failed.get('name', latest_failed.get('id', 'latest run'))}'."
                    ),
                    surface="monitoring",
                    href="/dashboard/monitoring",
                    metric_label="failed",
                    metric_value=str(len(failed_instances)),
                )
            )

        if open_circuits:
            recommendations.append(
                MissionControlRecommendation(
                    id="open-circuits",
                    severity="warning",
                    title="Resolve orchestration circuits before dispatching more work",
                    detail=(
                        f"{len(open_circuits)} orchestration circuit(s) are open. "
                        "Review stalled flows and clear the blockers before expanding the queue."
                    ),
                    surface="monitoring",
                    href="/dashboard/monitoring",
                    metric_label="circuits",
                    metric_value=str(len(open_circuits)),
                )
            )

        if telemetry_flags:
            recommendations.append(
                MissionControlRecommendation(
                    id="telemetry-backlog",
                    severity="warning",
                    title="Promote flagged telemetry into the dataset loop",
                    detail=(
                        f"{telemetry_flags} flagged interaction(s) are waiting in backlog. "
                        "Review them now to keep the failure-replay pipeline current."
                    ),
                    surface="datasets",
                    href="/dashboard/datasets",
                    metric_label="flagged",
                    metric_value=str(telemetry_flags),
                )
            )

        if telemetry_flags and active_agents == 0:
            recommendations.append(
                MissionControlRecommendation(
                    id="wake-agents",
                    severity="warning",
                    title="Wake or deploy agents to cover the backlog",
                    detail=(
                        "The lab has work queued but no active agents. "
                        "Bring a curator or evaluator online before the backlog grows."
                    ),
                    surface="agents",
                    href="/dashboard/agents",
                    metric_label="active agents",
                    metric_value=str(active_agents),
                )
            )

        if ready_autonomous_actions and autonomous.get("ready"):
            recommendations.append(
                MissionControlRecommendation(
                    id="autonomous-wave",
                    severity="opportunity",
                    title="Launch the next autonomous wave",
                    detail=(
                        f"{ready_autonomous_actions} autonomous action(s) are ready to dispatch. "
                        "Queue the next managed actions to keep the lifecycle moving automatically."
                    ),
                    surface="dashboard",
                    href="/dashboard",
                    metric_label="ready actions",
                    metric_value=str(ready_autonomous_actions),
                )
            )

        if lineage_gap_count:
            recommendations.append(
                MissionControlRecommendation(
                    id="lineage-gaps",
                    severity="warning",
                    title="Repair lineage coverage before promoting more work",
                    detail=(
                        f"{lineage_gap_count} completed managed instance(s) are missing lineage records. "
                        "Register provenance so downstream decisions stay auditable."
                    ),
                    surface="autonomous",
                    href="/dashboard/autonomous",
                    metric_label="lineage gaps",
                    metric_value=str(lineage_gap_count),
                )
            )

        if idle_nodes and running_sweeps == 0 and total_checks > 0 and ready_checks == total_checks:
            recommendations.append(
                MissionControlRecommendation(
                    id="idle-cluster",
                    severity="opportunity",
                    title="Use idle cluster capacity for a new sweep",
                    detail=(
                        f"{len(idle_nodes)} cluster node(s) are idle and no sweep is running. "
                        "This is a clean window to launch an AutoML search."
                    ),
                    surface="automl",
                    href="/dashboard/automl",
                    metric_label="idle nodes",
                    metric_value=str(len(idle_nodes)),
                )
            )

        if not instances and total_checks > 0 and ready_checks == total_checks:
            recommendations.append(
                MissionControlRecommendation(
                    id="first-run",
                    severity="info",
                    title="Launch the first managed training branch",
                    detail=(
                        "The workspace is ready but there are no managed instances yet. "
                        "Kick off a baseline run to seed the rest of the lifecycle."
                    ),
                    surface="training",
                    href="/dashboard/training",
                    metric_label="instances",
                    metric_value="0",
                    command="python3 -m training.train --config training/configs/profiles/baseline_qlora.yaml --dry-run",
                )
            )

        if running_instances and active_agents == 0:
            recommendations.append(
                MissionControlRecommendation(
                    id="uncovered-runs",
                    severity="opportunity",
                    title="Attach agents to active runs",
                    detail=(
                        f"{len(running_instances)} run(s) are in flight without any active agents. "
                        "Add monitor or evaluator coverage so the loop can react automatically."
                    ),
                    surface="agents",
                    href="/dashboard/agents",
                    metric_label="running",
                    metric_value=str(len(running_instances)),
                )
            )

        recommendations.sort(key=lambda item: (_severity_rank(item.severity), item.id))
        return recommendations[:6]

    def _swarm_agent_count(self, swarm: list[dict[str, Any]], agent_type: str) -> int:
        hints = _AGENT_ROLE_HINTS.get(agent_type, ())
        if not hints:
            return 0
        count = 0
        for agent in swarm:
            if agent.get("status") not in {"active", "sleeping"}:
                continue
            haystack = f"{agent.get('name', '')} {agent.get('role', '')}".lower()
            if any(hint in haystack for hint in hints):
                count += 1
        return count

    def _lineage_alerts(
        self,
        *,
        instances: list[dict[str, Any]],
        telemetry_backlog: int,
    ) -> list[AutonomyLineageAlert]:
        children_by_parent: dict[str, list[dict[str, Any]]] = {}
        for instance in instances:
            parent_id = instance.get("parent_instance_id")
            if isinstance(parent_id, str) and parent_id:
                children_by_parent.setdefault(parent_id, []).append(instance)

        alerts: list[AutonomyLineageAlert] = []
        for instance in instances:
            instance_id = str(instance.get("id", ""))
            children = children_by_parent.get(instance_id, [])
            child_types = {str(child.get("type", "")) for child in children}
            lifecycle = instance.get("lifecycle") or {}

            if instance.get("status") == "completed" and instance.get("type") in {"train", "finetune"} and "evaluate" not in child_types:
                alerts.append(
                    AutonomyLineageAlert(
                        id=f"lineage-eval-{instance_id}",
                        severity="warning",
                        title="Completed model lacks downstream evaluation",
                        detail=(
                            f"{instance.get('name', instance_id)} finished without an evaluation child run. "
                            "The loop cannot safely promote it without benchmark evidence."
                        ),
                        href=f"/runs/{instance_id}",
                        instance_id=instance_id,
                    )
                )

            decision = instance.get("decision") or {}
            if (
                instance.get("status") == "completed"
                and instance.get("type") == "evaluate"
                and decision.get("action") in {"deploy", "open_inference"}
                and not ({"deploy", "inference"} & child_types)
            ):
                alerts.append(
                    AutonomyLineageAlert(
                        id=f"lineage-promote-{instance_id}",
                        severity="opportunity",
                        title="Evaluation winner has not been promoted",
                        detail=(
                            f"{instance.get('name', instance_id)} recommends {decision.get('action')} but no child deployment or inference run exists yet."
                        ),
                        href=f"/runs/{instance_id}",
                        instance_id=instance_id,
                    )
                )

            if (
                instance.get("status") == "completed"
                and lifecycle.get("origin") == "existing_model"
                and instance.get("type") in {"train", "finetune", "evaluate", "deploy"}
                and not lifecycle.get("source_model")
            ):
                alerts.append(
                    AutonomyLineageAlert(
                        id=f"lineage-source-{instance_id}",
                        severity="warning",
                        title="Lifecycle lineage is missing source model metadata",
                        detail=(
                            f"{instance.get('name', instance_id)} completed without a `source_model` in lifecycle metadata. "
                            "That weakens provenance across the loop."
                        ),
                        href=f"/runs/{instance_id}",
                        instance_id=instance_id,
                    )
                )

        if telemetry_backlog and not any(instance.get("type") == "prepare" for instance in instances):
            alerts.append(
                AutonomyLineageAlert(
                    id="lineage-feedback-gap",
                    severity="opportunity",
                    title="Telemetry failures are not flowing back into data preparation",
                    detail=(
                        f"{telemetry_backlog} flagged prompt failure(s) are waiting, but there is no prepare-stage run in the lineage graph."
                    ),
                    href="/dashboard/datasets",
                )
            )

        alerts.sort(key=lambda item: (_severity_rank(item.severity), item.id))
        return alerts[:6]

    def autonomy_overview(
        self,
        context: dict[str, Any] | None = None,
        recommendations: list[MissionControlRecommendation] | None = None,
    ) -> AutonomyOverview:
        context = context or self._build_context()
        workspace_summary = context["workspace_summary"]
        instances = context["instances"]
        tasks = context["tasks"]
        capabilities = context["capabilities"]
        cluster_nodes = context["cluster_nodes"]
        agents = context["agents"]
        automl = context["automl"]
        telemetry = context["telemetry"]
        orchestration_summary = context["orchestration_summary"]

        ready_checks = int(workspace_summary.get("ready_checks", 0) or 0)
        total_checks = int(workspace_summary.get("total_checks", 0) or 0)
        failed_checks = max(total_checks - ready_checks, 0)
        open_circuits = list(orchestration_summary.get("open_circuits", []))
        telemetry_backlog = int(telemetry["flagged"]["count"])
        running_instances = [item for item in instances if item.get("status") == "running"]
        failed_instances = [item for item in instances if item.get("status") == "failed"]
        stalled_runs = len(failed_instances) + int(orchestration_summary.get("task_status_counts", {}).get("retry_waiting", 0))
        running_sweeps = int(automl["status_counts"].get("running", 0))
        active_runs = int(orchestration_summary.get("active_runs", 0))

        if failed_checks or open_circuits or failed_instances:
            status: Literal["blocked", "degraded", "active", "ready", "idle"] = "blocked"
        elif running_instances or running_sweeps or active_runs:
            status = "active"
        elif total_checks and ready_checks == total_checks:
            status = "ready"
        elif instances or agents["count"] or automl["count"]:
            status = "degraded"
        else:
            status = "idle"

        if any((item.get("metadata") or {}).get("sub_agents", {}).get("enabled") for item in instances):
            mode: Literal["manual", "assisted", "autonomous"] = "autonomous"
        elif agents["count"] or automl["count"]:
            mode = "assisted"
        else:
            mode = "manual"

        children_by_parent: dict[str, list[dict[str, Any]]] = {}
        for instance in instances:
            parent_id = instance.get("parent_instance_id")
            if isinstance(parent_id, str) and parent_id:
                children_by_parent.setdefault(parent_id, []).append(instance)

        completed_models = [
            item for item in instances if item.get("status") == "completed" and item.get("type") in {"train", "finetune"}
        ]
        completed_without_eval = 0
        deploy_ready = 0
        for instance in instances:
            children = children_by_parent.get(str(instance.get("id", "")), [])
            child_types = {str(child.get("type", "")) for child in children}
            if instance.get("status") == "completed" and instance.get("type") in {"train", "finetune"} and "evaluate" not in child_types:
                completed_without_eval += 1
            if (
                instance.get("status") == "completed"
                and instance.get("type") == "evaluate"
                and (instance.get("decision") or {}).get("action") in {"deploy", "open_inference"}
                and not ({"deploy", "inference"} & child_types)
            ):
                deploy_ready += 1

        prepare_running = sum(1 for item in instances if item.get("type") == "prepare" and item.get("status") == "running")
        train_running = sum(
            1 for item in instances if item.get("type") in {"train", "finetune"} and item.get("status") == "running"
        )
        eval_running = sum(
            1 for item in instances if item.get("type") in {"evaluate", "report"} and item.get("status") == "running"
        )
        deploy_running = sum(
            1 for item in instances if item.get("type") in {"deploy", "inference"} and item.get("status") == "running"
        )

        stages = [
            AutonomyStageSnapshot(
                id="datasets",
                title="Dataset Assembly",
                status=(
                    "blocked"
                    if failed_checks
                    else "active"
                    if prepare_running
                    else "attention"
                    if telemetry_backlog
                    else "ready"
                    if int(workspace_summary.get("datasets", 0) or 0) > 0
                    else "idle"
                ),
                headline=(
                    "Workspace blockers are preventing safe data refreshes"
                    if failed_checks
                    else f"{prepare_running} prepare run(s) are curating data"
                    if prepare_running
                    else f"{telemetry_backlog} flagged interaction(s) are waiting for replay"
                    if telemetry_backlog
                    else "Dataset inventory is ready for the next loop"
                    if int(workspace_summary.get("datasets", 0) or 0) > 0
                    else "No dataset inventory detected yet"
                ),
                detail=(
                    "The V2 loop should continuously convert flagged prompts and raw sources into curated packs with lineage."
                ),
                href="/dashboard/datasets",
                metric_label="datasets",
                metric_value=str(int(workspace_summary.get("datasets", 0) or 0)),
                counts={
                    "datasets": int(workspace_summary.get("datasets", 0) or 0),
                    "flagged": telemetry_backlog,
                    "running_prepare": prepare_running,
                },
            ),
            AutonomyStageSnapshot(
                id="training",
                title="Training + Search",
                status=(
                    "active"
                    if train_running or running_sweeps
                    else "attention"
                    if status == "ready" and not completed_models
                    else "ready"
                    if completed_models
                    else "idle"
                ),
                headline=(
                    f"{train_running} managed training branch(es) and {running_sweeps} sweep(s) are active"
                    if train_running or running_sweeps
                    else "No trained branches are available to branch from yet"
                    if status == "ready" and not completed_models
                    else f"{len(completed_models)} trained branch(es) are available for evaluation or finetune"
                    if completed_models
                    else "Training has not started yet"
                ),
                detail="The loop should keep cluster capacity warm by branching promising winners into deeper finetunes.",
                href="/dashboard/automl",
                metric_label="running sweeps",
                metric_value=str(running_sweeps),
                counts={
                    "running_train": train_running,
                    "running_sweeps": running_sweeps,
                    "completed_models": len(completed_models),
                },
            ),
            AutonomyStageSnapshot(
                id="evaluation",
                title="Evaluation + Feedback",
                status=(
                    "active"
                    if eval_running
                    else "attention"
                    if completed_without_eval
                    else "ready"
                    if any(item.get("type") == "evaluate" and item.get("status") == "completed" for item in instances)
                    else "idle"
                ),
                headline=(
                    f"{eval_running} evaluation job(s) are scoring current candidates"
                    if eval_running
                    else f"{completed_without_eval} completed model(s) still need evaluation coverage"
                    if completed_without_eval
                    else "Evaluation history exists for current lineage"
                    if any(item.get("type") == "evaluate" and item.get("status") == "completed" for item in instances)
                    else "No benchmark evidence has been recorded yet"
                ),
                detail="Evaluators should prune weak branches, score hard failure clusters, and feed follow-up recommendations.",
                href="/dashboard/evaluate",
                metric_label="unevaluated",
                metric_value=str(completed_without_eval),
                counts={
                    "running_eval": eval_running,
                    "unevaluated_models": completed_without_eval,
                },
            ),
            AutonomyStageSnapshot(
                id="deployment",
                title="Inference + Deployment",
                status=(
                    "active"
                    if deploy_running
                    else "attention"
                    if deploy_ready
                    else "ready"
                    if any(item.get("type") == "deploy" and item.get("status") == "completed" for item in instances)
                    else "idle"
                ),
                headline=(
                    f"{deploy_running} deployment or inference job(s) are in flight"
                    if deploy_running
                    else f"{deploy_ready} evaluated branch(es) are ready for promotion"
                    if deploy_ready
                    else "Deployment history is attached to the lineage graph"
                    if any(item.get("type") == "deploy" and item.get("status") == "completed" for item in instances)
                    else "No promoted deployment branches yet"
                ),
                detail="Inference telemetry should close the loop by generating the next wave of curated training examples.",
                href="/dashboard/deploy",
                metric_label="promotion backlog",
                metric_value=str(deploy_ready),
                counts={
                    "running_deploy": deploy_running,
                    "promotion_backlog": deploy_ready,
                },
            ),
        ]

        queued_statuses = {"queued", "ready", "retry_waiting", "blocked"}
        resource_counts = Counter(str(task.get("resource_class", "control")) for task in tasks if task.get("status") == "running")
        execution_modes = Counter(str((instance.get("environment") or {}).get("kind", "local")) for instance in running_instances)
        idle_nodes = sum(1 for node in cluster_nodes if node.get("status") == "idle")
        busy_nodes = sum(1 for node in cluster_nodes if node.get("status") in {"online", "busy"})
        offline_nodes = sum(1 for node in cluster_nodes if node.get("status") == "offline")
        suggested_parallelism = max(min(idle_nodes or 1, 5), 1) if status != "blocked" else 0
        capacity = AutonomyCapacitySnapshot(
            status=(
                "blocked"
                if failed_checks or open_circuits
                else "active"
                if active_runs or running_sweeps
                else "ready"
                if idle_nodes
                else "idle"
            ),
            idle_nodes=idle_nodes,
            busy_nodes=busy_nodes,
            offline_nodes=offline_nodes,
            active_gpu_tasks=int(resource_counts.get("gpu", 0)),
            active_cpu_tasks=int(resource_counts.get("cpu", 0) + resource_counts.get("io", 0) + resource_counts.get("control", 0)),
            schedulable_trials=max(idle_nodes, 0),
            suggested_parallelism=suggested_parallelism,
            bottleneck=(
                "workspace readiness"
                if failed_checks
                else "open orchestration circuits"
                if open_circuits
                else "cluster capacity"
                if not idle_nodes and (running_sweeps or active_runs)
                else "evaluation coverage"
                if completed_without_eval
                else "none"
            ),
            execution_modes=dict(execution_modes),
        )

        agent_coverage: list[AutonomyAgentCoverage] = []
        for capability in capabilities:
            agent_type = str(capability.get("agent_type", "monitoring_telemetry"))
            label = agent_type.replace("_", " ").title()
            queued_tasks = sum(
                1 for task in tasks if task.get("agent_type") == agent_type and task.get("status") in queued_statuses
            )
            running_tasks = sum(
                1 for task in tasks if task.get("agent_type") == agent_type and task.get("status") == "running"
            )
            active_swarm_agents = self._swarm_agent_count(agents["swarm"], agent_type)
            open_circuit = agent_type in open_circuits
            coverage_status: Literal["blocked", "active", "attention", "ready", "idle"]
            if open_circuit:
                coverage_status = "blocked"
            elif running_tasks or active_swarm_agents:
                coverage_status = "active"
            elif queued_tasks:
                coverage_status = "attention"
            elif agents["count"]:
                coverage_status = "ready"
            else:
                coverage_status = "idle"
            agent_coverage.append(
                AutonomyAgentCoverage(
                    agent_type=agent_type,
                    label=label,
                    status=coverage_status,
                    queued_tasks=queued_tasks,
                    running_tasks=running_tasks,
                    active_swarm_agents=active_swarm_agents,
                    open_circuit=open_circuit,
                    resource_classes=[str(item) for item in capability.get("resource_classes", [])],
                    recommended_action=(
                        "Clear the circuit and retry blocked work."
                        if open_circuit
                        else "Attach a swarm agent or reduce queue depth for this role."
                        if queued_tasks and not active_swarm_agents
                        else "Coverage is keeping pace with current demand."
                    ),
                )
            )

        agent_coverage.sort(key=lambda item: (item.status not in {"blocked", "attention"}, item.label))
        lineage_alerts = self._lineage_alerts(instances=instances, telemetry_backlog=telemetry_backlog)

        recommendations = recommendations or self._recommendations(
            workspace_summary=workspace_summary,
            running_instances=running_instances,
            failed_instances=failed_instances,
            autonomous=context.get("autonomous", {}),
            agents=agents,
            automl=automl,
            telemetry=telemetry,
            cluster_nodes=cluster_nodes,
            open_circuits=open_circuits,
            instances=instances,
        )
        next_actions: list[AutonomyNextAction] = [
            AutonomyNextAction(
                id=f"rec-{item.id}",
                title=item.title,
                detail=item.detail,
                href=item.href,
                category=(
                    "stabilize"
                    if item.severity in {"critical", "warning"}
                    else "optimize"
                    if item.surface in {"automl", "training"}
                    else "dispatch"
                ),
                blocking=item.severity in {"critical", "warning"},
                command=item.command,
            )
            for item in recommendations
        ]
        for alert in lineage_alerts[:2]:
            next_actions.append(
                AutonomyNextAction(
                    id=f"alert-{alert.id}",
                    title=alert.title,
                    detail=alert.detail,
                    href=alert.href,
                    category="lineage",
                    blocking=alert.severity in {"critical", "warning"},
                )
            )
        next_actions.sort(key=lambda item: (not item.blocking, item.category, item.id))

        summary = (
            "Autonomous loop is blocked by readiness, failed runs, or open circuits."
            if status == "blocked"
            else "Autonomous loop is actively dispatching work across training, evaluation, or deployment."
            if status == "active"
            else "Autonomous loop is ready to expand into a new sweep or branch."
            if status == "ready"
            else "Autonomous loop has partial coverage but needs more connected surfaces."
            if status == "degraded"
            else "Autonomous loop is idle and waiting for the first managed branch."
        )

        return AutonomyOverview(
            status=status,
            mode=mode,
            summary=summary,
            open_circuits=len(open_circuits),
            telemetry_backlog=telemetry_backlog,
            active_runs=active_runs,
            running_sweeps=running_sweeps,
            stalled_runs=stalled_runs,
            stages=stages,
            agent_coverage=agent_coverage,
            capacity=capacity,
            lineage_alerts=lineage_alerts,
            next_actions=next_actions[:8],
        )

    def snapshot(self) -> MissionControlSnapshot:
        context = self._build_context()
        workspace = context["workspace"]
        workspace_summary = context["workspace_summary"]
        instances = context["instances"]
        runs = context["runs"]
        orchestration_summary_payload = context["orchestration_summary"]
        cluster_nodes = context["cluster_nodes"]
        agents = context["agents"]
        automl = context["automl"]
        telemetry = context["telemetry"]
        titan = context["titan"]
        autonomous = context["autonomous"]

        recent_instances = _recent(instances, "updated_at", limit=12)
        recent_runs = _recent(runs, "updated_at", limit=12)
        running_instances = [item for item in instances if item.get("status") == "running"]
        failed_instances = [item for item in instances if item.get("status") == "failed"]
        cluster_status_counts = _status_counts(cluster_nodes)
        open_circuits = list(orchestration_summary_payload.get("open_circuits", []))
        ready_autonomous_actions = int((autonomous.get("summary") or {}).get("executable_actions", 0) or 0)
        recommendations = self._recommendations(
            workspace_summary=workspace_summary,
            running_instances=running_instances,
            failed_instances=failed_instances,
            autonomous=autonomous,
            agents=agents,
            automl=automl,
            telemetry=telemetry,
            cluster_nodes=cluster_nodes,
            open_circuits=open_circuits,
            instances=instances,
        )
        autonomy = self.autonomy_overview(context=context, recommendations=recommendations)
        criticality_counts = Counter(item.severity for item in recommendations)
        highest_level = recommendations[0].severity if recommendations else "info"
        summary = {
            "workspace_ready_checks": workspace_summary.get("ready_checks", 0),
            "workspace_total_checks": workspace_summary.get("total_checks", 0),
            "instances": len(instances),
            "running_instances": len(running_instances),
            "failed_instances": len(failed_instances),
            "orchestration_runs": len(runs),
            "autonomous_actions": (autonomous.get("summary") or {}).get("total_actions", ready_autonomous_actions),
            "autonomous_executable_actions": ready_autonomous_actions,
            "autonomous_ready": bool(autonomous.get("ready")),
            "agents": agents["count"],
            "automl_sweeps": automl["count"],
            "cluster_nodes": len(cluster_nodes),
            "telemetry_flags": telemetry["flagged"]["count"],
            "telemetry_requests": telemetry["requests"]["count"],
            "autonomous_campaigns": 0,
            "ready_autonomous_actions": ready_autonomous_actions,
            "lineage_records": 0,
            "lineage_gaps": 0,
            "active_agents": agents["status_counts"].get("active", 0),
            "running_sweeps": automl["status_counts"].get("running", 0),
            "telemetry_backlog": telemetry["flagged"]["count"],
            "ready_checks": workspace_summary.get("ready_checks", 0),
            "total_checks": workspace_summary.get("total_checks", 0),
            "datasets": workspace_summary.get("datasets", 0),
            "training_profiles": workspace_summary.get("training_profiles", 0),
            "open_circuits": len(open_circuits),
            "titan_backend": titan.get("backend"),
            "titan_mode": titan.get("mode"),
            "autonomous_blockers": len(autonomous.get("blockers", [])),
            "autonomy_status": autonomy.status,
            "autonomy_mode": autonomy.mode,
            "stalled_runs": autonomy.stalled_runs,
        }
        return MissionControlSnapshot(
            generated_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            repo_root=str(self.repo_root),
            workspace=workspace,
            orchestration=orchestration_summary_payload,
            watchlist={
                "instances": recent_instances,
                "running_instances": _recent(running_instances, "updated_at", limit=12),
                "failed_instances": _recent(failed_instances, "updated_at", limit=12),
                "agents": agents["swarm"][:8],
                "sweeps": automl["sweeps"][:8],
                "cluster_nodes": cluster_nodes,
                "telemetry": telemetry["flagged"]["recent"],
            },
            control_plane={
                "instances": instances,
                "runs": recent_runs,
                "orchestration_summary": orchestration_summary_payload,
            },
            autonomous=autonomous,
            agents=agents,
            automl=automl,
            cluster={
                "nodes": cluster_nodes,
                "status_counts": cluster_status_counts,
            },
            lineage={},
            telemetry=telemetry,
            titan=titan,
            autonomy=autonomy,
            criticality={
                "level": highest_level,
                "counts": dict(criticality_counts),
            },
            recommendations=[item.model_dump(mode="json") for item in recommendations],
            summary=summary,
        )
