from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, Field

from ai_factory.core.lineage.models import LineageRecord
from ai_factory.core.lineage.registry import LineageRegistry
from ai_factory.platform.monitoring import hardware
from inference.app.config import AppSettings

if TYPE_CHECKING:
    from inference.app.services.instance_service import InstanceService


ActionKind = Literal["prepare", "train", "finetune", "evaluate", "inference", "deploy", "lineage"]
ActionStatus = Literal["planned", "started", "blocked", "completed", "failed", "skipped"]
CampaignStatus = Literal["planned", "running", "completed", "degraded"]


class AutonomousAction(BaseModel):
    id: str
    kind: ActionKind
    title: str
    detail: str
    status: ActionStatus = "planned"
    config_path: str | None = None
    source_instance_id: str | None = None
    instance_id: str | None = None
    depends_on: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class AutonomousCampaign(BaseModel):
    campaign_id: str
    experiment_name: str
    goal: str
    status: CampaignStatus = "planned"
    created_at: str
    updated_at: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    plan: list[AutonomousAction] = Field(default_factory=list)
    execution: list[dict[str, Any]] = Field(default_factory=list)
    summary: dict[str, Any] = Field(default_factory=dict)


class AutonomousExperimentRequest(BaseModel):
    experiment_name: str = Field(..., description="The name of the autonomous experiment")
    goal: str = Field(..., description="The primary goal or objective of the experiment")
    parameters: dict[str, str | int | float | bool] | None = Field(
        default=None,
        description="Optional execution parameters for the campaign.",
    )
    auto_start: bool = Field(
        default=True,
        description="Whether the service should immediately translate ready plan steps into managed instances.",
    )
    max_actions: int = Field(default=3, ge=1, le=8, description="Maximum ready steps to launch immediately.")


class AutonomousLoopSnapshot(BaseModel):
    status: Literal["available"] = "available"
    write_enabled: bool = True
    path: str
    count: int = 0
    status_counts: dict[str, int] = Field(default_factory=dict)
    active_campaigns: int = 0
    campaigns: list[AutonomousCampaign] = Field(default_factory=list)
    ready_actions: list[AutonomousAction] = Field(default_factory=list)
    loop_health: dict[str, Any] = Field(default_factory=dict)
    lineage: dict[str, Any] = Field(default_factory=dict)
    cluster: dict[str, Any] = Field(default_factory=dict)


class AutonomousExperimentResponse(BaseModel):
    experiment_id: str = Field(..., description="The unique identifier for the campaign")
    status: str = Field(..., description="The current status of the campaign")
    message: str = Field(..., description="Status message regarding the campaign")
    campaign: AutonomousCampaign


class AutonomousLabService:
    def __init__(self, settings: AppSettings, *, instance_service: InstanceService) -> None:
        self.settings = settings
        self.instance_service = instance_service
        self.repo_root = Path(settings.repo_root).resolve()
        self.campaigns_file = self.repo_root / "data" / "autonomous" / "campaigns.jsonl"
        self.sweeps_file = self.repo_root / "data" / "automl" / "sweeps.jsonl"
        self.flagged_file = self.repo_root / "data" / "telemetry" / "flagged.jsonl"
        self.lineage_registry = LineageRegistry(Path(settings.artifacts_dir) / "lineage")

    def _utc_now(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _campaign_id(self, name: str, goal: str) -> str:
        body = f"{name}|{goal}|{time.time():.6f}"
        return f"camp-{hashlib.sha256(body.encode('utf-8')).hexdigest()[:12]}"

    def _resolve_repo_path(self, value: str) -> str:
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = self.repo_root / path
        return str(path.resolve())

    def _load_jsonl(self, path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            return []
        rows: list[dict[str, Any]] = []
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                rows.append(payload)
        return rows

    def _load_campaigns(self) -> list[AutonomousCampaign]:
        campaigns = [AutonomousCampaign.model_validate(row) for row in self._load_jsonl(self.campaigns_file)]
        campaigns.sort(key=lambda item: item.updated_at, reverse=True)
        return campaigns

    def _save_campaign(self, campaign: AutonomousCampaign) -> None:
        campaigns = {item.campaign_id: item for item in self._load_campaigns()}
        campaigns[campaign.campaign_id] = campaign
        self.campaigns_file.parent.mkdir(parents=True, exist_ok=True)
        with self.campaigns_file.open("w", encoding="utf-8") as handle:
            for item in campaigns.values():
                handle.write(json.dumps(item.model_dump(mode="json")) + "\n")

    def _status_counts(self, values: list[dict[str, Any]] | list[AutonomousCampaign], field: str = "status") -> dict[str, int]:
        counts: dict[str, int] = {}
        for item in values:
            if isinstance(item, BaseModel):
                status = getattr(item, field, "unknown")
            else:
                status = item.get(field, "unknown")
            key = str(status or "unknown")
            counts[key] = counts.get(key, 0) + 1
        return counts

    def _instances(self) -> list[dict[str, Any]]:
        records = []
        for item in self.instance_service.list_instances():
            records.append(item.model_dump(mode="json") if hasattr(item, "model_dump") else dict(item))
        records.sort(key=lambda row: str(row.get("updated_at", "")), reverse=True)
        return records

    def _lineage_summary(self, instances: list[dict[str, Any]]) -> dict[str, Any]:
        records = self.lineage_registry.list_lineage(limit=200)
        covered_instance_ids = {
            str(record.metadata.get("instance_id"))
            for record in records
            if record.metadata.get("instance_id") is not None
        }
        model_instances = [
            item
            for item in instances
            if item.get("status") == "completed" and item.get("type") in {"train", "finetune", "evaluate", "deploy", "inference"}
        ]
        gaps = [item for item in model_instances if item.get("id") not in covered_instance_ids]
        return {
            "records": len(records),
            "roots": sum(1 for record in records if record.parent_id is None),
            "covered_instances": len(covered_instance_ids),
            "gap_count": len(gaps),
            "gaps": [
                {
                    "instance_id": item.get("id"),
                    "name": item.get("name"),
                    "type": item.get("type"),
                    "updated_at": item.get("updated_at"),
                }
                for item in gaps[:12]
            ],
            "latest": [record.model_dump(mode="json") for record in records[:8]],
        }

    def _cluster_summary(self, instances: list[dict[str, Any]]) -> dict[str, Any]:
        nodes = hardware.get_cluster_nodes()
        idle_nodes = [node for node in nodes if node.get("status") == "idle" or int(node.get("activeJobs", 0) or 0) == 0]
        running_gpu = [
            item
            for item in instances
            if item.get("status") == "running" and item.get("type") in {"train", "finetune"}
        ]
        queued_eval = [
            item
            for item in instances
            if item.get("status") in {"pending", "running"} and item.get("type") == "evaluate"
        ]
        placements = []
        for node in nodes:
            node_type = str(node.get("type", "")).lower()
            preferred_workload = "gpu-heavy" if "nvidia" in node_type or "mps" in node_type or "gpu" in node_type else "cpu-telemetry"
            placements.append(
                {
                    "node_id": node.get("id"),
                    "node_name": node.get("name"),
                    "status": node.get("status"),
                    "preferred_workload": preferred_workload,
                    "active_jobs": int(node.get("activeJobs", 0) or 0),
                    "usage": int(node.get("usage", 0) or 0),
                }
            )
        return {
            "nodes": nodes,
            "idle_nodes": len(idle_nodes),
            "placements": placements,
            "pressure": {
                "running_gpu_workloads": len(running_gpu),
                "evaluation_backlog": len(queued_eval),
                "saturation": "high" if running_gpu and not idle_nodes else "balanced",
            },
        }

    def _latest(self, instances: list[dict[str, Any]], *, kind: str, status: str = "completed") -> dict[str, Any] | None:
        for item in instances:
            if item.get("type") == kind and item.get("status") == status:
                return item
        return None

    def _has_child(self, instances: list[dict[str, Any]], source_instance_id: str, child_type: str) -> bool:
        return any(
            item.get("parent_instance_id") == source_instance_id and item.get("type") == child_type
            for item in instances
        )

    def _plan_actions(
        self,
        *,
        goal: str,
        parameters: dict[str, Any],
        instances: list[dict[str, Any]],
        orchestration_summary: dict[str, Any],
        lineage_summary: dict[str, Any],
        cluster_summary: dict[str, Any],
    ) -> list[AutonomousAction]:
        flagged = len(self._load_jsonl(self.flagged_file))
        sweeps = self._load_jsonl(self.sweeps_file)
        running_sweeps = sum(1 for item in sweeps if item.get("status") == "running")
        running_instances = [item for item in instances if item.get("status") == "running"]
        completed_model = self._latest(instances, kind="finetune") or self._latest(instances, kind="train")
        latest_eval = self._latest(instances, kind="evaluate")
        actions: list[AutonomousAction] = []

        if flagged > 0:
            actions.append(
                AutonomousAction(
                    id="prepare-telemetry-replay",
                    kind="prepare",
                    title="Refresh the dataset pack from flagged telemetry",
                    detail=(
                        f"{flagged} flagged interaction(s) are waiting to be promoted into the curation loop. "
                        "Run the managed prepare pipeline before the next finetune wave."
                    ),
                    config_path=self._resolve_repo_path("configs/prepare.yaml"),
                    metadata={"flagged_records": flagged, "goal": goal},
                )
            )

        if completed_model and not self._has_child(instances, str(completed_model.get("id")), "evaluate"):
            actions.append(
                AutonomousAction(
                    id=f"evaluate-{completed_model['id']}",
                    kind="evaluate",
                    title="Benchmark the latest model branch",
                    detail=(
                        f"{completed_model.get('name', completed_model['id'])} finished without a managed evaluation child. "
                        "Create benchmark coverage before promoting it deeper into the loop."
                    ),
                    config_path=self._resolve_repo_path("configs/eval.yaml"),
                    source_instance_id=str(completed_model.get("id")),
                    metadata={"source_type": completed_model.get("type")},
                )
            )

        if latest_eval and not self._has_child(instances, str(latest_eval.get("id")), "inference"):
            actions.append(
                AutonomousAction(
                    id=f"inference-{latest_eval['id']}",
                    kind="inference",
                    title="Open an inference branch for the strongest evaluated artifact",
                    detail=(
                        f"{latest_eval.get('name', latest_eval['id'])} is evaluation-complete. "
                        "Spin up an inference branch so telemetry can start feeding the next dataset cycle."
                    ),
                    config_path=self._resolve_repo_path("configs/inference.yaml"),
                    source_instance_id=str(latest_eval.get("id")),
                )
            )

        decision = latest_eval.get("decision") if latest_eval else None
        if latest_eval and isinstance(decision, dict) and decision.get("action") == "deploy":
            actions.append(
                AutonomousAction(
                    id=f"deploy-{latest_eval['id']}",
                    kind="deploy",
                    title="Publish the latest evaluation winner",
                    detail=(
                        f"{latest_eval.get('name', latest_eval['id'])} meets the deployment gate according to the decision engine."
                    ),
                    config_path=self._resolve_repo_path("configs/deploy.yaml"),
                    source_instance_id=str(latest_eval.get("id")),
                    depends_on=[f"inference-{latest_eval['id']}"],
                    metadata={"deployment_target": "huggingface"},
                )
            )

        if not running_instances and running_sweeps == 0:
            if completed_model and flagged > 0:
                actions.append(
                    AutonomousAction(
                        id=f"finetune-{completed_model['id']}",
                        kind="finetune",
                        title="Launch the next finetune branch",
                        detail=(
                            "The telemetry backlog and the completed parent model are both ready. "
                            "Start a fresh finetune branch to continue the autonomous loop."
                        ),
                        config_path=self._resolve_repo_path("configs/finetune.yaml"),
                        source_instance_id=str(completed_model.get("id")),
                        depends_on=["prepare-telemetry-replay"] if flagged > 0 else [],
                        metadata={"goal": goal, "idle_nodes": cluster_summary.get("idle_nodes", 0)},
                    )
                )
            elif completed_model is None:
                actions.append(
                    AutonomousAction(
                        id="train-foundation",
                        kind="train",
                        title="Seed the loop with a baseline training branch",
                        detail="No completed train or finetune branch exists yet, so the lab needs a baseline source model.",
                        config_path=self._resolve_repo_path("configs/train.yaml"),
                        metadata={"goal": goal},
                    )
                )

        if lineage_summary.get("gap_count", 0) > 0:
            actions.append(
                AutonomousAction(
                    id="lineage-reconcile",
                    kind="lineage",
                    title="Reconcile lineage coverage for completed runs",
                    detail=(
                        f"{lineage_summary['gap_count']} completed managed instance(s) are missing a lineage record. "
                        "Materialize provenance now so the lab can reason over parentage and outcomes."
                    ),
                    metadata={"instance_ids": [item["instance_id"] for item in lineage_summary.get("gaps", [])]},
                )
            )

        ready = orchestration_summary.get("ready_tasks", 0)
        if ready and cluster_summary.get("idle_nodes", 0) == 0:
            actions.append(
                AutonomousAction(
                    id="dispatch-capacity-warning",
                    kind="lineage",
                    title="Hold for capacity before dispatching more work",
                    detail=(
                        f"{ready} orchestration task(s) are ready, but no idle cluster nodes are visible. "
                        "Let the current wave clear before starting another branch."
                    ),
                    status="blocked",
                    metadata={"ready_tasks": ready},
                )
            )

        return actions

    def _config_path(self, value: str) -> str:
        path = Path(value)
        return str(path if path.is_absolute() else (self.repo_root / path).resolve())

    def _stable_hash(self, payload: Any) -> str:
        encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _record_lineage(self, instance_id: str) -> dict[str, Any]:
        manifest = self.instance_service.control.get_instance(instance_id)
        snapshot = self.instance_service.store.load_config_snapshot(instance_id)
        parent_id = manifest.parent_instance_id
        record_id = self._stable_hash({"instance_id": manifest.id, "config": snapshot, "type": manifest.type})[:16]
        parent_record_id = None
        if parent_id:
            parent_record_id = self._stable_hash({"instance_id": parent_id})[:16]
        progress_metrics = manifest.progress.metrics if manifest.progress else {}
        dataset_fingerprint = self._stable_hash(snapshot.get("subsystem", snapshot or {}))[:16]
        record = LineageRecord(
            id=record_id,
            parent_id=parent_record_id,
            base_model=(manifest.lifecycle.source_model or manifest.name or manifest.id),
            dataset_hash=dataset_fingerprint,
            job_id=manifest.id,
            training_config=snapshot.get("subsystem", {}),
            snapshot_config=snapshot,
            metrics={
                key: float(value)
                for key, value in progress_metrics.items()
                if isinstance(value, (int, float))
            },
            metadata={
                "instance_id": manifest.id,
                "instance_type": manifest.type,
                "orchestration_run_id": manifest.orchestration_run_id,
                "config_path": manifest.config_path,
            },
            tags=[manifest.type, manifest.lifecycle.stage],
        )
        self.instance_service.control.record_lineage(record)
        return {"instance_id": manifest.id, "record_id": record.id}

    def _execute_action(self, campaign: AutonomousCampaign, action: AutonomousAction) -> dict[str, Any]:
        metadata = {
            "source": "autonomous_lab",
            "campaign_id": campaign.campaign_id,
            "goal": campaign.goal,
            "action_id": action.id,
        }
        if action.kind in {"prepare", "train"}:
            manifest = self.instance_service.control.create_instance(
                self._config_path(action.config_path or "configs/train.yaml"),
                start=True,
                metadata_updates={**metadata, **action.metadata},
            )
            return {"instance_id": manifest.id, "instance_type": manifest.type}
        if action.kind == "finetune":
            manifest = self.instance_service.control.create_instance(
                self._config_path(action.config_path or "configs/finetune.yaml"),
                start=True,
                parent_instance_id=action.source_instance_id,
                metadata_updates={**metadata, **action.metadata},
            )
            return {"instance_id": manifest.id, "instance_type": manifest.type, "source_instance_id": action.source_instance_id}
        if action.kind in {"evaluate", "inference", "deploy"}:
            action_name = {
                "evaluate": "evaluate",
                "inference": "open_inference",
                "deploy": "deploy",
            }[action.kind]
            manifest = self.instance_service.control.execute_action(
                action.source_instance_id or "",
                action=action_name,
                config_path=self._config_path(action.config_path or f"configs/{action.kind}.yaml"),
                deployment_target=action.metadata.get("deployment_target"),
                start=True,
            )
            return {"instance_id": manifest.id, "instance_type": manifest.type, "source_instance_id": action.source_instance_id}
        if action.kind == "lineage":
            records = [self._record_lineage(instance_id) for instance_id in action.metadata.get("instance_ids", []) if instance_id]
            return {"records": records, "record_count": len(records)}
        raise ValueError(f"Unsupported autonomous action kind: {action.kind}")

    def snapshot(self) -> AutonomousLoopSnapshot:
        campaigns = self._load_campaigns()
        instances = self._instances()
        orchestration_response = self.instance_service.get_orchestration_summary()
        orchestration_summary = getattr(orchestration_response, "summary", orchestration_response)
        lineage_summary = self._lineage_summary(instances)
        cluster_summary = self._cluster_summary(instances)
        planner_actions = self._plan_actions(
            goal="Maintain an always-on autonomous research loop.",
            parameters={},
            instances=instances,
            orchestration_summary=orchestration_summary,
            lineage_summary=lineage_summary,
            cluster_summary=cluster_summary,
        )
        active_campaigns = sum(1 for item in campaigns if item.status in {"planned", "running"})
        return AutonomousLoopSnapshot(
            path=str(self.campaigns_file),
            count=len(campaigns),
            status_counts=self._status_counts(campaigns),
            active_campaigns=active_campaigns,
            campaigns=campaigns[:12],
            ready_actions=[action for action in planner_actions if action.status == "planned"][:10],
            loop_health={
                "flagged_backlog": len(self._load_jsonl(self.flagged_file)),
                "running_instances": sum(1 for item in instances if item.get("status") == "running"),
                "open_circuits": len(orchestration_summary.get("open_circuits", [])),
                "ready_tasks": int(orchestration_summary.get("ready_tasks", 0) or 0),
                "running_sweeps": sum(1 for item in self._load_jsonl(self.sweeps_file) if item.get("status") == "running"),
            },
            lineage=lineage_summary,
            cluster=cluster_summary,
        )

    def create_campaign(self, request: AutonomousExperimentRequest) -> AutonomousCampaign:
        instances = self._instances()
        orchestration_response = self.instance_service.get_orchestration_summary()
        orchestration_summary = getattr(orchestration_response, "summary", orchestration_response)
        lineage_summary = self._lineage_summary(instances)
        cluster_summary = self._cluster_summary(instances)
        parameters = dict(request.parameters or {})
        plan = self._plan_actions(
            goal=request.goal,
            parameters=parameters,
            instances=instances,
            orchestration_summary=orchestration_summary,
            lineage_summary=lineage_summary,
            cluster_summary=cluster_summary,
        )
        campaign = AutonomousCampaign(
            campaign_id=self._campaign_id(request.experiment_name, request.goal),
            experiment_name=request.experiment_name,
            goal=request.goal,
            status="planned",
            created_at=self._utc_now(),
            updated_at=self._utc_now(),
            parameters={**parameters, "auto_start": request.auto_start, "max_actions": request.max_actions},
            plan=plan,
            summary={
                "ready_actions": sum(1 for action in plan if action.status == "planned"),
                "blocked_actions": sum(1 for action in plan if action.status == "blocked"),
                "lineage_gap_count": lineage_summary.get("gap_count", 0),
                "idle_nodes": cluster_summary.get("idle_nodes", 0),
            },
        )

        if request.auto_start:
            executed = 0
            for action in campaign.plan:
                if action.status != "planned":
                    continue
                if executed >= request.max_actions:
                    break
                try:
                    result = self._execute_action(campaign, action)
                    action.status = "started"
                    action.instance_id = result.get("instance_id")
                    campaign.execution.append(
                        {
                            "action_id": action.id,
                            "kind": action.kind,
                            "status": "started",
                            "started_at": self._utc_now(),
                            **result,
                        }
                    )
                    executed += 1
                except Exception as exc:  # pragma: no cover - defensive fallback
                    action.status = "failed"
                    campaign.execution.append(
                        {
                            "action_id": action.id,
                            "kind": action.kind,
                            "status": "failed",
                            "started_at": self._utc_now(),
                            "error": str(exc),
                        }
                    )
            campaign.status = "running" if executed else "planned"
        campaign.updated_at = self._utc_now()
        self._save_campaign(campaign)
        return campaign

    def get_campaign(self, campaign_id: str) -> AutonomousCampaign:
        for campaign in self._load_campaigns():
            if campaign.campaign_id == campaign_id:
                return campaign
        raise FileNotFoundError(f"Autonomous campaign '{campaign_id}' not found")
