from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from ai_factory.core.hashing import normalize_text, stable_question_fingerprint

Difficulty = Literal["easy", "medium", "hard", "olympiad"]
DatasetSplit = Literal["train", "eval", "test", "benchmark", "unspecified"]
ReasoningStyle = Literal["chain_of_thought", "proof", "exam", "verification", "mixed"]


class StepCheck(BaseModel):
    id: str | None = None
    kind: Literal["substring", "regex", "expr_equiv", "numeric_tolerance"] = "substring"
    value: str
    explanation: str | None = None
    weight: float = 1.0
    tolerance: float | None = None

    @field_validator("value")
    @classmethod
    def _non_empty(cls, v: str) -> str:
        cleaned = normalize_text(v)
        if not cleaned:
            raise ValueError("step check value must be non-empty")
        return cleaned

    @field_validator("weight")
    @classmethod
    def _weight_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("step check weight must be positive")
        return float(v)


class SourceLineage(BaseModel):
    dataset_id: str = "unknown"
    dataset_family: str = "unknown"
    origin_path: str | None = None
    loader: str = "local"
    source_url: str | None = None
    license: str | None = None
    filters: dict[str, Any] = Field(default_factory=dict)
    source_record_id: str | None = None
    notes: list[str] = Field(default_factory=list)


class GeneratorMetadata(BaseModel):
    generator_family: str
    template_id: str | None = None
    seed: int | None = None
    curriculum_bucket: str | None = None
    pedagogical_focus: list[str] = Field(default_factory=list)
    generation_profile: str | None = None


class ContaminationStatus(BaseModel):
    checked_against: list[str] = Field(default_factory=list)
    exact_match: bool = False
    near_match: bool = False
    max_similarity: float = 0.0
    notes: list[str] = Field(default_factory=list)


class ResourceSpec(BaseModel):
    """Resource specification for training jobs."""

    cpu_cores: int = 4
    memory_gb: int = 16
    gpu_count: int = 1
    gpu_memory_gb: int = 8
    disk_gb: int = 100
    custom_requirements: dict[str, Any] = Field(default_factory=dict)


class DatasetRecordV2(BaseModel):
    """Universal dataset record for any domain."""

    schema_version: Literal["v2"] = "v2"
    id: str | None = None
    question: str
    solution: str
    final_answer: str | None = None
    difficulty: Difficulty = "medium"
    domain: str = "general"  # NEW: Domain specification
    subdomain: str | None = None  # NEW: Subdomain within domain
    topic: str = "general"
    source: str = "unknown"
    dataset_split: DatasetSplit = "unspecified"
    step_checks: list[StepCheck] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    failure_case: bool = False
    quality_score: float = 0.0
    reasoning_style: ReasoningStyle | str = "mixed"
    pack_id: str | None = None
    contamination: ContaminationStatus = Field(default_factory=ContaminationStatus)
    lineage: SourceLineage = Field(default_factory=SourceLineage)
    generator: GeneratorMetadata | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class MathRecordV2(DatasetRecordV2):
    """Mathematics-specific dataset record (backward compatibility)."""

    domain: Literal["mathematics"] = "mathematics"

    @field_validator("step_checks", mode="before")
    @classmethod
    def _coerce_step_checks(cls, v: Any) -> list[Any]:
        if not v:
            return []
        if isinstance(v, str):
            return [{"kind": "substring", "value": item.strip()} for item in v.split("||") if item.strip()]
        coerced: list[Any] = []
        for item in v:
            if isinstance(item, dict):
                coerced.append(item)
            else:
                coerced.append({"kind": "substring", "value": str(item)})
        return coerced

    @field_validator("question", "solution")
    @classmethod
    def _trim_text(cls, v: str) -> str:
        text = str(v).strip()
        if not text:
            raise ValueError("text fields must be non-empty")
        return text

    @field_validator("topic", "source")
    @classmethod
    def _clean_simple(cls, v: str) -> str:
        cleaned = normalize_text(v)
        return cleaned or "unknown"

    @field_validator("tags")
    @classmethod
    def _tags_clean(cls, v: list[str]) -> list[str]:
        cleaned: list[str] = []
        for item in v:
            tag = normalize_text(item).lower()
            if tag and tag not in cleaned:
                cleaned.append(tag)
        return cleaned

    @field_validator("quality_score")
    @classmethod
    def _quality_range(cls, v: float) -> float:
        return max(0.0, min(1.0, float(v)))

    @model_validator(mode="after")
    def _fill_defaults(self) -> MathRecordV2:
        if not self.id:
            object.__setattr__(self, "id", stable_question_fingerprint(self.question))
        if not self.lineage.dataset_id:
            self.lineage.dataset_id = self.source
        if not self.lineage.dataset_family:
            self.lineage.dataset_family = self.source
        return self


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str

    @field_validator("content")
    @classmethod
    def _content_non_empty(cls, v: str) -> str:
        text = str(v).strip()
        if not text:
            raise ValueError("message content must be non-empty")
        return text


class PackagedMathRecordV2(MathRecordV2):
    messages: list[ChatMessage]
    weight: float = 1.0

    @field_validator("weight")
    @classmethod
    def _weight_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("weight must be positive")
        return float(v)


class DatasetBuildInfo(BaseModel):
    build_id: str
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    git_sha: str | None = None
    config_path: str | None = None
    config_sha256: str | None = None
    seed: int | None = None
    notes: list[str] = Field(default_factory=list)


class DatasetFileInfo(BaseModel):
    path: str
    sha256: str
    size_bytes: int
    num_rows: int


class DatasetManifest(BaseModel):
    schema_version: Literal["v2"] = "v2"
    manifest_type: Literal["dataset", "pack", "benchmark"] = "dataset"
    build: DatasetBuildInfo
    pack_id: str | None = None
    description: str | None = None
    inputs: list[DatasetFileInfo] = Field(default_factory=list)
    outputs: list[DatasetFileInfo] = Field(default_factory=list)
    source_lineage: list[SourceLineage] = Field(default_factory=list)
    stats: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class RunManifest(BaseModel):
    schema_version: Literal["v2"] = "v2"
    run_id: str
    run_name: str
    profile_name: str | None = None
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    base_dir: str
    model_name: str
    base_model: str
    config_path: str | None = None
    git_sha: str | None = None
    data_files: list[str] = Field(default_factory=list)
    metrics_files: list[str] = Field(default_factory=list)
    report_files: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


# Platform-level schemas for AI-Factory scaling and deployment


class DatasetSpec(BaseModel):
    """Dataset specification for domain registry."""

    name: str
    description: str
    path: str
    domain: str
    subdomain: str | None = None
    difficulty_range: list[str] = Field(default_factory=list)
    size: int = 0
    format: str = "jsonl"
    metadata: dict[str, Any] = Field(default_factory=dict)


class MetricSpec(BaseModel):
    """Metric specification for domain evaluation."""

    name: str
    description: str
    type: str  # accuracy, score, ranking, etc.
    domain: str
    subdomain: str | None = None
    range: list[float] | None = None  # min/max for score metrics
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvaluationSpec(BaseModel):
    """Evaluation specification for domain benchmarks."""

    name: str
    description: str
    domain: str
    subdomain: str | None = None
    datasets: list[str]
    metrics: list[str]
    splits: list[str] = Field(default_factory=lambda: ["test"])
    size: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)


class TrainingProfileSpec(BaseModel):
    """Training profile specification for domains."""

    name: str
    description: str
    domain: str
    subdomain: str | None = None
    training_method: str
    datasets: list[str]
    config_path: str
    curriculum_order: list[str] | None = None
    model_requirements: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ModelArtifact(BaseModel):
    """Model artifact for deployment."""

    name: str
    version: str
    path: str
    domain: str | None = None
    architecture: str
    parameters: int
    format: str  # pytorch, safetensors, gguf, etc.
    metadata: dict[str, Any] = Field(default_factory=dict)


class DeploymentTarget(BaseModel):
    """Deployment target specification."""

    name: str
    type: str  # huggingface, ollama, etc.
    description: str
    config: dict[str, Any] = Field(default_factory=dict)
    capabilities: list[str] = Field(default_factory=list)


class DeploymentSpec(BaseModel):
    """Deployment specification."""

    target: str
    model_name: str
    config: dict[str, Any] = Field(default_factory=dict)
    public: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class ScalingConfig(BaseModel):
    """Configuration for scaling manager."""

    max_nodes: int = 10
    default_resources: dict[str, Any] = Field(default_factory=dict)
    cluster_type: str = "local"  # local, slurm, kubernetes
    metadata: dict[str, Any] = Field(default_factory=dict)


class MonitoringConfig(BaseModel):
    """Configuration for monitoring manager."""

    collection_interval_seconds: float = 5.0
    storage_backend: str = "file"  # file, prometheus, influxdb
    alert_channels: list[str] = Field(default_factory=list)
    thresholds: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class TrainingJob(BaseModel):
    """Training job specification for scaling."""

    name: str
    profile: str
    resource_requirements: dict[str, Any] = Field(default_factory=dict)
    estimated_duration_hours: float = 0.0
    priority: str = "normal"  # low, normal, high, critical
    metadata: dict[str, Any] = Field(default_factory=dict)


class Alert(BaseModel):
    """Alert specification for monitoring."""

    id: str
    severity: str  # info, warning, critical
    message: str
    source: str
    timestamp: datetime
    metadata: dict[str, Any] = Field(default_factory=dict)


class MetricPoint(BaseModel):
    """Single metric data point."""

    timestamp: datetime
    name: str
    value: float
    labels: dict[str, str] = Field(default_factory=dict)


# Backward compatibility aliases
MathRecord = MathRecordV2
PackagedMathRecord = PackagedMathRecordV2


def path_to_file_info(path: Path, sha256: str, num_rows: int = 0) -> DatasetFileInfo:
    return DatasetFileInfo(
        path=str(path),
        sha256=sha256,
        size_bytes=path.stat().st_size if path.exists() else 0,
        num_rows=num_rows,
    )


class DomainType(str, Enum):
    """Domain categorization for models and datasets."""

    MATHEMATICS = "mathematics"
    CODING = "coding"
    REASONING = "reasoning"
    VISION = "vision"
    GENERAL = "general"


class ArchitectureSpec(BaseModel):
    """Specification of the model's architecture."""

    base_model: str
    context_window: int = 8192
    parameter_size_b: float | None = None
    quantization: Literal["4bit", "8bit", "16bit", "none"] = "none"
    lora_target_modules: list[str] | None = None
    lora_rank: int | None = None
    lora_alpha: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ResourceProfile(BaseModel):
    """Hardware resource requirements for running the model."""

    vram_required_gb: float
    cpu_cores_required: int = 4
    system_memory_gb: float = 16.0
    recommended_gpus: int = 1
    storage_gb: float = 50.0


class PerformanceProfile(BaseModel):
    """Expected performance metrics of the model."""

    throughput_tokens_per_sec: float | None = None
    latency_ms_per_token: float | None = None
    memory_footprint_gb: float | None = None
    power_consumption_w: float | None = None


class ModelLineage(BaseModel):
    """Historical lineage of the model's training and data."""

    parent_model: str | None = None
    training_dataset_ids: list[str] = Field(default_factory=list)
    training_run_ids: list[str] = Field(default_factory=list)
    creation_timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ModelCapability(BaseModel):
    """Specific evaluated capabilities of the model."""

    domain: DomainType
    score: float
    benchmark_name: str
    verified: bool = False


class UniversalModelSpec(BaseModel):
    """Universal specification for any trained/deployed model."""

    id: str
    name: str
    version: str
    domain: DomainType
    architecture: ArchitectureSpec
    resource_profile: ResourceProfile
    performance_profile: PerformanceProfile | None = None
    lineage: ModelLineage = Field(default_factory=ModelLineage)
    capabilities: list[ModelCapability] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class SearchSpaceSpec(BaseModel):
    """Configuration for hyperparameter search space."""

    hyperparameters: dict[str, list[Any]] = Field(default_factory=dict)
    architectures: list[str] | None = None
    datasets: list[str] | None = None


class OptimizationObjective(BaseModel):
    """Objective metric to optimize during experimentation."""

    metric: str
    maximize: bool = True
    target_value: float | None = None
    weight: float = 1.0


class IterationStrategy(BaseModel):
    """Strategy for experiment iterations."""

    max_iterations: int = 10
    early_stopping_patience: int = 3
    exploration_factor: float = 0.2
    batch_size: int = 1


class ResourceBudget(BaseModel):
    """Resource constraints for the experiment."""

    max_compute_hours: float | None = None
    max_cost_usd: float | None = None
    max_gpu_hours: float | None = None


class EvaluationCriterion(BaseModel):
    """Criteria for evaluating model success."""

    metric_name: str
    min_threshold: float
    critical: bool = True


class AutoDeploymentPolicy(BaseModel):
    """Policy for automated model deployment upon success."""

    enabled: bool = False
    targets: list[DeploymentTarget] = Field(default_factory=list)
    approval_required: bool = True
    rollback_on_failure: bool = True


class AutonomousExperimentConfig(BaseModel):
    """Configuration for autonomous model experimentation."""

    experiment_id: str
    name: str
    domains: list[DomainType]
    search_space: SearchSpaceSpec
    objectives: list[OptimizationObjective]
    strategy: IterationStrategy = Field(default_factory=IterationStrategy)
    budget: ResourceBudget = Field(default_factory=ResourceBudget)
    evaluation_criteria: list[EvaluationCriterion] = Field(default_factory=list)
    deployment_policy: AutoDeploymentPolicy = Field(default_factory=AutoDeploymentPolicy)
    metadata: dict[str, Any] = Field(default_factory=dict)
