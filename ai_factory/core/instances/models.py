from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

InstanceType = Literal["prepare", "train", "finetune", "evaluate", "inference", "deploy", "report"]
InstanceStatus = Literal["pending", "running", "completed", "failed"]
EnvironmentKind = Literal["local", "cloud"]
DecisionAction = Literal["retrain", "finetune", "deploy", "re_evaluate", "open_inference"]
UserLevel = Literal["beginner", "hobbyist", "dev"]
OrchestrationMode = Literal["single", "local_parallel", "cloud_parallel", "hybrid"]
TrainingOrigin = Literal["existing_model", "from_scratch"]
LearningMode = Literal[
    "supervised",
    "unsupervised",
    "rlhf",
    "dpo",
    "orpo",
    "ppo",
    "lora",
    "qlora",
    "full_finetune",
]
DeploymentTarget = Literal["huggingface", "ollama", "lmstudio", "api", "openai_compatible_api", "custom_api"]
LifecycleStage = Literal["prepare", "train", "evaluate", "decide", "finetune", "infer", "publish"]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class PortForward(BaseModel):
    local_port: int
    remote_port: int
    bind_host: str = "127.0.0.1"
    description: str | None = None


class EnvironmentSpec(BaseModel):
    kind: EnvironmentKind = "local"
    profile_name: str | None = None
    host: str | None = None
    user: str | None = None
    port: int = 22
    key_path: str | None = None
    agent_socket: str | None = None
    remote_repo_root: str | None = None
    python_bin: str | None = None
    env: dict[str, str] = Field(default_factory=dict)
    port_forwards: list[PortForward] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_cloud_fields(self) -> EnvironmentSpec:
        if self.kind == "cloud" and not (self.profile_name or self.host):
            raise ValueError("cloud environments require either a profile_name or host")
        return self


class ExecutionHandle(BaseModel):
    backend: str
    pid: int | None = None
    remote_job_id: str | None = None
    started_at: str | None = None
    ended_at: str | None = None
    exit_code: int | None = None
    stdout_path: str | None = None
    stderr_path: str | None = None
    remote_stdout_path: str | None = None
    remote_stderr_path: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class MetricPoint(BaseModel):
    timestamp: str = Field(default_factory=utc_now_iso)
    name: str
    value: float | int | bool | None
    unit: str | None = None
    tags: dict[str, str] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class DecisionResult(BaseModel):
    action: DecisionAction
    rule: str
    thresholds: dict[str, float | int | None] = Field(default_factory=dict)
    summary: dict[str, Any] = Field(default_factory=dict)
    explanation: str


class ArchitectureSpec(BaseModel):
    family: str | None = None
    hidden_size: int | None = None
    num_layers: int | None = None
    num_attention_heads: int | None = None
    max_position_embeddings: int | None = None
    vocab_size: int | None = None
    notes: str | None = None


class EvaluationSuiteSpec(BaseModel):
    id: str | None = None
    label: str | None = None
    benchmark_config: str | None = None
    compare_to_models: list[str] = Field(default_factory=list)
    custom_dataset_paths: list[str] = Field(default_factory=list)
    notes: str | None = None


class LifecycleProfile(BaseModel):
    stage: LifecycleStage | None = None
    origin: TrainingOrigin | None = None
    learning_mode: LearningMode | None = None
    source_model: str | None = None
    architecture: ArchitectureSpec = Field(default_factory=ArchitectureSpec)
    evaluation_suite: EvaluationSuiteSpec | None = None
    deployment_targets: list[DeploymentTarget] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class FeedbackRecommendation(BaseModel):
    action: str
    reason: str
    priority: int = 1
    target_instance_type: InstanceType | None = None
    config_path: str | None = None
    deployment_target: DeploymentTarget | None = None
    command: list[str] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class InstanceError(BaseModel):
    code: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


class ProgressSnapshot(BaseModel):
    stage: str
    status_message: str | None = None
    completed_steps: int | None = None
    total_steps: int | None = None
    percent: float | None = None
    eta_seconds: float | None = None
    updated_at: str = Field(default_factory=utc_now_iso)
    metrics: dict[str, Any] = Field(default_factory=dict)


class InstanceManifest(BaseModel):
    id: str
    type: InstanceType
    status: InstanceStatus = "pending"
    environment: EnvironmentSpec = Field(default_factory=EnvironmentSpec)
    user_level: UserLevel = "hobbyist"
    orchestration_mode: OrchestrationMode = "single"
    lifecycle: LifecycleProfile = Field(default_factory=LifecycleProfile)
    name: str
    created_at: str = Field(default_factory=utc_now_iso)
    updated_at: str = Field(default_factory=utc_now_iso)
    parent_instance_id: str | None = None
    config_path: str | None = None
    config_snapshot_path: str | None = None
    artifact_refs: dict[str, Any] = Field(default_factory=dict)
    execution: ExecutionHandle | None = None
    metrics_summary: dict[str, Any] = Field(default_factory=dict)
    progress: ProgressSnapshot | None = None
    decision: DecisionResult | None = None
    recommendations: list[FeedbackRecommendation] = Field(default_factory=list)
    error: InstanceError | None = None
    orchestration_run_id: str | None = None
    task_summary: dict[str, Any] = Field(default_factory=dict)
    last_heartbeat_at: str | None = None
    active_agents: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def touch(self) -> None:
        self.updated_at = utc_now_iso()


class DeploymentReadiness(BaseModel):
    ready: bool = False
    blockers: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    suggested_target: DeploymentTarget | None = None
    artifact_path: str | None = None
    summary: dict[str, Any] = Field(default_factory=dict)


def check_deployment_readiness(
    manifest: InstanceManifest,
    *,
    min_accuracy: float = 0.0,
    required_status: str = "completed",
) -> DeploymentReadiness:
    blockers: list[str] = []
    warnings: list[str] = []

    if manifest.status != required_status:
        blockers.append(f"Instance status is '{manifest.status}', expected '{required_status}'.")

    if manifest.type not in {"train", "finetune", "evaluate", "inference"}:
        blockers.append(f"Instance type '{manifest.type}' is not a deployable source.")

    accuracy = manifest.metrics_summary.get("accuracy")
    if isinstance(accuracy, (int, float)) and accuracy < min_accuracy:
        blockers.append(f"Accuracy {accuracy:.3f} is below the deployment floor {min_accuracy:.3f}.")

    if manifest.decision and manifest.decision.action == "retrain":
        blockers.append("Decision engine recommends retraining before deployment.")

    if manifest.decision and manifest.decision.action == "re_evaluate":
        warnings.append("Decision engine recommends re-evaluation before deployment.")

    artifact_path = manifest.artifact_refs.get("source_artifact") or manifest.artifact_refs.get("run_dir")

    suggested_target: DeploymentTarget | None = None
    if manifest.lifecycle.deployment_targets:
        suggested_target = manifest.lifecycle.deployment_targets[0]

    ready = len(blockers) == 0
    return DeploymentReadiness(
        ready=ready,
        blockers=blockers,
        warnings=warnings,
        suggested_target=suggested_target,
        artifact_path=str(artifact_path) if artifact_path else None,
        summary={
            "instance_id": manifest.id,
            "instance_type": manifest.type,
            "status": manifest.status,
            "accuracy": accuracy if isinstance(accuracy, (int, float)) else None,
            "decision_action": manifest.decision.action if manifest.decision else None,
        },
    )


EnvironmentSpec.model_rebuild()
