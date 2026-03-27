from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from ai_factory.core.instances.models import (
    EnvironmentSpec,
    InstanceType,
    OrchestrationMode,
    PortForward,
    UserLevel,
)


class InstanceConfig(BaseModel):
    type: InstanceType
    name: str | None = None
    environment: EnvironmentSpec = Field(default_factory=EnvironmentSpec)
    parent_instance_id: str | None = None


class ExecutionConfig(BaseModel):
    backend: Literal["auto", "local", "ssh"] = "auto"
    cwd: str | None = None
    env: dict[str, str] = Field(default_factory=dict)
    retry_limit: int = 0
    timeout_s: int | None = None
    max_parallelism: int = 1


class MonitoringConfig(BaseModel):
    collect_gpu: bool = True
    write_timeseries: bool = True
    capture_system_metrics: bool = True
    heartbeat_interval_s: int = 30
    progress_window_size: int = 20


class UserExperienceConfig(BaseModel):
    level: UserLevel = "hobbyist"
    allow_command_override: bool | None = None
    allow_extra_args: bool | None = None
    allow_remote_shell: bool | None = None
    require_safe_defaults: bool | None = None
    max_parallel_sub_agents: int | None = None

    @model_validator(mode="after")
    def _apply_level_defaults(self) -> "UserExperienceConfig":
        presets = {
            "beginner": {
                "allow_command_override": False,
                "allow_extra_args": False,
                "allow_remote_shell": False,
                "require_safe_defaults": True,
                "max_parallel_sub_agents": 1,
            },
            "hobbyist": {
                "allow_command_override": False,
                "allow_extra_args": True,
                "allow_remote_shell": True,
                "require_safe_defaults": True,
                "max_parallel_sub_agents": 2,
            },
            "dev": {
                "allow_command_override": True,
                "allow_extra_args": True,
                "allow_remote_shell": True,
                "require_safe_defaults": False,
                "max_parallel_sub_agents": 8,
            },
        }
        defaults = presets[self.level]
        for key, value in defaults.items():
            if getattr(self, key) is None:
                setattr(self, key, value)
        return self


class RemoteAccessConfig(BaseModel):
    enable_ssh: bool = False
    sync_before_start: bool = False
    sync_mode: Literal["none", "git", "rsync"] = "none"
    agent_forwarding: bool = False
    ssh_keepalive_s: int = 30
    port_forwards: list[PortForward] = Field(default_factory=list)


class SubAgentConfig(BaseModel):
    enabled: bool = False
    max_parallelism: int = 1
    workloads: list[Literal["preprocess", "metrics", "evaluation", "finetune", "publish"]] = Field(
        default_factory=list
    )
    allow_nested: bool = True
    retry_limit: int = 1
    failure_budget: int = 1


class PublishHookConfig(BaseModel):
    target: Literal["huggingface", "ollama", "lmstudio", "api", "custom_api"]
    enabled: bool = True
    when: Literal["manual", "on_success", "after_evaluation"] = "manual"
    config_path: str | None = None
    provider_options: dict[str, Any] = Field(default_factory=dict)


class FeedbackLoopConfig(BaseModel):
    enabled: bool = True
    queue_follow_up_evaluation: bool = True
    auto_queue_finetune: bool = False
    auto_queue_retrain: bool = False
    suggest_failure_analysis: bool = True
    improvement_floor: float = 0.02
    max_recommendations: int = 4


class DecisionPolicy(BaseModel):
    min_accuracy: float = 0.75
    min_parse_rate: float = 0.7
    min_verifier_agreement: float = 0.6
    max_no_answer_rate: float = 0.15
    max_latency_s: float = 20.0
    finetune_accuracy_floor: float = 0.45
    auto_continue: bool = False


class PipelineConfig(BaseModel):
    auto_continue: bool = False
    max_auto_cycles: int = 1
    max_auto_children: int = 6
    default_prepare_config: str = "configs/prepare.yaml"
    default_train_config: str = "configs/train.yaml"
    default_eval_config: str = "configs/eval.yaml"
    default_deploy_config: str = "configs/deploy.yaml"
    default_finetune_config: str = "configs/finetune.yaml"
    default_report_config: str = "configs/report.yaml"


class SubsystemConfig(BaseModel):
    config_ref: str | None = None
    dry_run: bool = False
    extra_args: list[str] = Field(default_factory=list)
    model_variant: str | None = None
    serve_host: str = "127.0.0.1"
    serve_port: int = 8000
    provider: Literal["huggingface", "ollama", "lmstudio", "custom_api"] | None = None
    provider_options: dict[str, Any] = Field(default_factory=dict)
    source_instance_id: str | None = None
    source_artifact_ref: str | None = None
    output_dir_override: str | None = None
    command_override: list[str] | None = None
    labels: list[str] = Field(default_factory=list)
    shard_count: int = 1


class OrchestrationConfig(BaseModel):
    instance: InstanceConfig
    orchestration_mode: OrchestrationMode = "single"
    experience: UserExperienceConfig = Field(default_factory=UserExperienceConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    subsystem: SubsystemConfig
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    remote_access: RemoteAccessConfig = Field(default_factory=RemoteAccessConfig)
    sub_agents: SubAgentConfig = Field(default_factory=SubAgentConfig)
    decision_policy: DecisionPolicy = Field(default_factory=DecisionPolicy)
    feedback_loop: FeedbackLoopConfig = Field(default_factory=FeedbackLoopConfig)
    publish_hooks: list[PublishHookConfig] = Field(default_factory=list)
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)
    metadata: dict[str, Any] = Field(default_factory=dict)
    config_path: str | None = None
    resolved_subsystem_config_path: str | None = None
    resolved_subsystem_config: dict[str, Any] = Field(default_factory=dict)
