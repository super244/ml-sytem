from __future__ import annotations

from datetime import datetime, timezone
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


class MathRecordV2(BaseModel):
    schema_version: Literal["v2"] = "v2"
    id: str | None = None
    question: str
    solution: str
    final_answer: str | None = None
    difficulty: Difficulty = "hard"
    topic: str = "general"
    subtopic: str | None = None
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
    def _fill_defaults(self) -> "MathRecordV2":
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


MathRecord = MathRecordV2
PackagedMathRecord = PackagedMathRecordV2


def path_to_file_info(path: Path, sha256: str, num_rows: int = 0) -> DatasetFileInfo:
    return DatasetFileInfo(
        path=str(path),
        sha256=sha256,
        size_bytes=path.stat().st_size if path.exists() else 0,
        num_rows=num_rows,
    )
