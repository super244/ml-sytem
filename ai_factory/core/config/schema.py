from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from ai_factory.core.instances.models import EnvironmentSpec, InstanceType


class InstanceConfig(BaseModel):
    type: InstanceType
    name: str | None = None
    environment: EnvironmentSpec = Field(default_factory=EnvironmentSpec)
    parent_instance_id: str | None = None


class ExecutionConfig(BaseModel):
    backend: Literal["auto", "local", "ssh"] = "auto"
    cwd: str | None = None
    env: dict[str, str] = Field(default_factory=dict)


class MonitoringConfig(BaseModel):
    collect_gpu: bool = True
    write_timeseries: bool = True


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
    default_eval_config: str = "configs/eval.yaml"
    default_deploy_config: str = "configs/deploy.yaml"
    default_finetune_config: str = "configs/finetune.yaml"


class SubsystemConfig(BaseModel):
    config_ref: str | None = None
    dry_run: bool = False
    extra_args: list[str] = Field(default_factory=list)
    model_variant: str | None = None
    serve_host: str = "127.0.0.1"
    serve_port: int = 8000
    provider: Literal["huggingface", "ollama", "lmstudio"] | None = None
    provider_options: dict[str, Any] = Field(default_factory=dict)
    source_instance_id: str | None = None
    source_artifact_ref: str | None = None
    output_dir_override: str | None = None
    command_override: list[str] | None = None


class OrchestrationConfig(BaseModel):
    instance: InstanceConfig
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    subsystem: SubsystemConfig
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    decision_policy: DecisionPolicy = Field(default_factory=DecisionPolicy)
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)
    metadata: dict[str, Any] = Field(default_factory=dict)
    config_path: str | None = None
    resolved_subsystem_config_path: str | None = None
    resolved_subsystem_config: dict[str, Any] = Field(default_factory=dict)
