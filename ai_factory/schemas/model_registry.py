from datetime import datetime
from typing import Literal
from pydantic import BaseModel, ConfigDict


class ModelSummary(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: str
    name: str
    base_model: str
    training_type: Literal["lora", "dpo", "rlhf", "full_finetune"]
    eval_scores: dict[str, float]
    children_count: int
    dataset_hash: str
    parent_model_id: str | None = None
    created_at: datetime
    deployed: bool
    size_gb: float


class LineageNodeSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: str
    type: Literal["dataset", "base_model", "checkpoint", "deployment"]
    label: str
    metadata: dict


class LineageEdgeSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    source: str
    target: str
    label: str


class LineageGraph(BaseModel):
    nodes: list[LineageNodeSchema]
    edges: list[LineageEdgeSchema]
