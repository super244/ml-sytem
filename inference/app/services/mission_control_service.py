from __future__ import annotations

import json
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from ai_factory.platform.monitoring import hardware
from inference.app import workspace as workspace_module
from inference.app.config import AppSettings


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


class MissionControlSnapshot(BaseModel):
    generated_at: str
    repo_root: str
    workspace: dict[str, Any] = Field(default_factory=dict)
    orchestration: dict[str, Any] = Field(default_factory=dict)
    watchlist: dict[str, Any] = Field(default_factory=dict)
    control_plane: dict[str, Any] = Field(default_factory=dict)
    agents: dict[str, Any] = Field(default_factory=dict)
    automl: dict[str, Any] = Field(default_factory=dict)
    cluster: dict[str, Any] = Field(default_factory=dict)
    telemetry: dict[str, Any] = Field(default_factory=dict)
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

    def _recommendations(
        self,
        *,
        workspace_summary: dict[str, Any],
        running_instances: list[dict[str, Any]],
        failed_instances: list[dict[str, Any]],
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

    def snapshot(self) -> MissionControlSnapshot:
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
        cluster_nodes = hardware.get_cluster_nodes()
        agents = self._agents()
        automl = self._automl()
        telemetry = self._telemetry()
        recent_instances = _recent(instances, "updated_at", limit=12)
        recent_runs = _recent(runs, "updated_at", limit=12)
        running_instances = [item for item in instances if item.get("status") == "running"]
        failed_instances = [item for item in instances if item.get("status") == "failed"]
        cluster_status_counts = _status_counts(cluster_nodes)
        open_circuits = (
            orchestration_summary_payload.get("open_circuits", [])
            if isinstance(orchestration_summary_payload, dict)
            else []
        )
        recommendations = self._recommendations(
            workspace_summary=workspace_summary,
            running_instances=running_instances,
            failed_instances=failed_instances,
            agents=agents,
            automl=automl,
            telemetry=telemetry,
            cluster_nodes=cluster_nodes,
            open_circuits=open_circuits,
            instances=instances,
        )
        criticality_counts = Counter(item.severity for item in recommendations)
        highest_level = recommendations[0].severity if recommendations else "info"
        summary = {
            "workspace_ready_checks": workspace_summary.get("ready_checks", 0),
            "workspace_total_checks": workspace_summary.get("total_checks", 0),
            "instances": len(instances),
            "running_instances": len(running_instances),
            "failed_instances": len(failed_instances),
            "orchestration_runs": len(runs),
            "agents": agents["count"],
            "automl_sweeps": automl["count"],
            "cluster_nodes": len(cluster_nodes),
            "telemetry_flags": telemetry["flagged"]["count"],
            "telemetry_requests": telemetry["requests"]["count"],
            # Backward-compatible aliases used by the current dashboard.
            "active_agents": agents["status_counts"].get("active", 0),
            "running_sweeps": automl["status_counts"].get("running", 0),
            "telemetry_backlog": telemetry["flagged"]["count"],
            "ready_checks": workspace_summary.get("ready_checks", 0),
            "total_checks": workspace_summary.get("total_checks", 0),
            "datasets": workspace_summary.get("datasets", 0),
            "training_profiles": workspace_summary.get("training_profiles", 0),
            "open_circuits": len(open_circuits),
        }
        return MissionControlSnapshot(
            generated_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            repo_root=str(self.repo_root),
            workspace=workspace,
            orchestration=orchestration_summary_payload if isinstance(orchestration_summary_payload, dict) else {},
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
            agents=agents,
            automl=automl,
            cluster={
                "nodes": cluster_nodes,
                "status_counts": cluster_status_counts,
            },
            telemetry=telemetry,
            criticality={
                "level": highest_level,
                "counts": dict(criticality_counts),
            },
            recommendations=[item.model_dump(mode="json") for item in recommendations],
            summary=summary,
        )
