from datetime import datetime
from typing import Literal
from pydantic import BaseModel, ConfigDict


class PackSummary(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    source_distribution: dict[str, float]
    filter_pass_rates: dict[str, float]
    quality_histogram: list[float]
    dedup_removed: int
    toxicity_removed: int
    total_before_filter: int
    total_after_filter: int


class DatasetSummary(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: str
    name: str
    domain: Literal["math", "code", "reasoning", "creative", "mixed"]
    status: Literal["synthesizing", "filtering", "packing", "ready", "failed"]
    sample_count: int
    quality_score_mean: float
    quality_score_p10: float
    quality_score_p90: float
    size_mb: float
    content_hash: str
    pipeline_config_hash: str
    git_sha: str
    created_at: datetime
    pack_summary: PackSummary | None = None


class Sample(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: str
    content: str
    quality_score: float
    perplexity: float
    domain: str
    source: str


class DatasetSamplesResponse(BaseModel):
    samples: list[Sample]
    total: int
