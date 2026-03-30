from __future__ import annotations

import json
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

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
    summary: dict[str, Any] = Field(default_factory=dict)


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
            summary=summary,
        )
