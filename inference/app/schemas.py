from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from ai_factory.core.instances.models import (
    DeploymentTarget,
    EnvironmentSpec,
    InstanceManifest,
    LifecycleProfile,
    MetricPoint,
    UserLevel,
)
from ai_factory.core.orchestration.models import OrchestrationEvent, OrchestrationRun, OrchestrationTask

ModelVariant = str
Difficulty = Literal["easy", "medium", "hard", "olympiad"]
SolverMode = Literal["rigorous", "exam", "concise", "verification"]
OutputFormat = Literal["text", "json"]


class CandidateVerification(BaseModel):
    final_answer: str | None = None
    equivalent: bool | None = None
    step_correctness: float | None = None
    verifier_agreement: bool | None = None
    formatting_failure: bool = False
    arithmetic_slip: bool = False
    error_type: str = "unknown"


class Candidate(BaseModel):
    text: str
    display_text: str
    reasoning: str
    final_answer: str | None = None
    calculator_trace: list[dict[str, str]] = Field(default_factory=list)
    vote_count: int = 0
    score: float = 0.0
    verification: CandidateVerification | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


class GenerateRequest(BaseModel):
    question: str = Field(..., min_length=3)
    model_variant: ModelVariant = "finetuned"
    compare_to_base: bool = False
    compare_to_model: str | None = None
    prompt_preset: str = "atlas_rigorous"
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    top_p: float = Field(default=0.95, gt=0.0, le=1.0)
    max_new_tokens: int = Field(default=768, ge=64, le=4096)
    show_reasoning: bool = True
    difficulty_target: Difficulty = "hard"
    num_samples: int = Field(default=3, ge=1, le=16)
    use_calculator: bool = True
    solver_mode: SolverMode = "rigorous"
    output_format: OutputFormat = "text"
    use_cache: bool = True
    reference_answer: str | None = None
    step_checks: list[Any] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class StructuredAnswer(BaseModel):
    reasoning: str
    final_answer: str | None
    verification: CandidateVerification | None = None


class GenerateResult(BaseModel):
    model_variant: str
    prompt: str
    answer: str
    raw_text: str
    final_answer: str | None
    reasoning_steps: list[str]
    selected_score: float
    candidates: list[Candidate]
    verification: CandidateVerification | None = None
    structured: StructuredAnswer | None = None
    cache_hit: bool = False
    telemetry_id: str | None = None
    latency_s: float | None = None
    prompt_preset: str | None = None
    candidate_agreement: float = 0.0


class GenerateResponse(GenerateResult):
    comparison: GenerateResult | None = None


class CompareRequest(BaseModel):
    question: str = Field(..., min_length=3)
    primary_model: str = "finetuned"
    secondary_model: str = "base"
    prompt_preset: str = "atlas_rigorous"
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    top_p: float = Field(default=0.95, gt=0.0, le=1.0)
    max_new_tokens: int = Field(default=768, ge=64, le=4096)
    show_reasoning: bool = True
    difficulty_target: Difficulty = "hard"
    num_samples: int = Field(default=3, ge=1, le=16)
    use_calculator: bool = True
    solver_mode: SolverMode = "rigorous"
    output_format: OutputFormat = "text"
    use_cache: bool = True
    reference_answer: str | None = None
    step_checks: list[Any] = Field(default_factory=list)


class CompareResponse(BaseModel):
    primary: GenerateResult
    secondary: GenerateResult


class GenerateBatchRequest(BaseModel):
    requests: list[GenerateRequest] = Field(..., min_length=1, max_length=64)


class GenerateBatchResponse(BaseModel):
    results: list[GenerateResponse]


class VerifyRequest(BaseModel):
    candidate_answer: str
    reference_answer: str
    prediction_text: str | None = None
    step_checks: list[Any] = Field(default_factory=list)


class VerifyResponse(BaseModel):
    equivalent: bool
    step_correctness: float | None
    formatting_failure: bool = False
    arithmetic_slip: bool = False
    error_type: str = "unknown"
    details: dict[str, Any] = Field(default_factory=dict)


class InstanceCreateRequest(BaseModel):
    config_path: str = Field(..., min_length=1)
    start: bool = True
    environment: EnvironmentSpec | None = None
    parent_instance_id: str | None = None
    name: str | None = None
    user_level: UserLevel | None = None
    lifecycle: LifecycleProfile | None = None
    subsystem_overrides: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class InstanceEvaluateRequest(BaseModel):
    config_path: str | None = None
    start: bool = True


class InstanceInferenceRequest(BaseModel):
    config_path: str | None = None
    start: bool = True


class InstanceDeployRequest(BaseModel):
    target: Literal["huggingface", "ollama", "lmstudio", "api", "custom_api", "openai_compatible_api"]
    config_path: str | None = None
    start: bool = True


class InstanceLogsResponse(BaseModel):
    stdout: str = ""
    stderr: str = ""
    stdout_path: str | None = None
    stderr_path: str | None = None


class InstanceMetricsResponse(BaseModel):
    summary: dict[str, Any] = Field(default_factory=dict)
    points: list[MetricPoint] = Field(default_factory=list)


class InstanceActionDescriptor(BaseModel):
    action: str
    label: str
    description: str
    target_instance_type: str | None = None
    config_path: str | None = None
    deployment_target: DeploymentTarget | None = None


class InstanceDetail(InstanceManifest):
    config_snapshot: dict[str, Any] = Field(default_factory=dict)
    logs: InstanceLogsResponse | None = None
    metrics: InstanceMetricsResponse = Field(default_factory=InstanceMetricsResponse)
    children: list[InstanceManifest] = Field(default_factory=list)
    events: list[dict[str, Any]] = Field(default_factory=list)
    available_actions: list[InstanceActionDescriptor] = Field(default_factory=list)


class InstanceListResponse(BaseModel):
    instances: list[InstanceManifest]


class OrchestrationRunListResponse(BaseModel):
    runs: list[OrchestrationRun]


class OrchestrationTaskListResponse(BaseModel):
    tasks: list[OrchestrationTask | dict[str, Any]]


class OrchestrationEventListResponse(BaseModel):
    events: list[OrchestrationEvent | dict[str, Any]]


class OrchestrationRunDetail(BaseModel):
    run: OrchestrationRun | dict[str, Any]
    tasks: list[OrchestrationTask | dict[str, Any]] = Field(default_factory=list)
    events: list[OrchestrationEvent | dict[str, Any]] = Field(default_factory=list)
    summary: dict[str, Any] = Field(default_factory=dict)


class OrchestrationSummaryResponse(BaseModel):
    summary: dict[str, Any] = Field(default_factory=dict)
