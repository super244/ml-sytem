from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, cast, Iterable

from pydantic import BaseModel, Field


class LineageRecord(BaseModel):
    """Immutable record of a model's clinical lineage."""

    id: str = Field(..., description="Unique hash of the record")
    parent_id: str | None = Field(None, description="Parent lineage record hash")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # Provenance
    base_model: str = Field(..., description="Base model name or hash")
    dataset_hash: str = Field(..., description="Content-addressed hash of the training pack")
    job_id: str | None = Field(None, description="ID of the training job that produced this")

    # Configuration snapshots
    training_config: dict[str, Any] = Field(default_factory=dict)
    snapshot_config: dict[str, Any] = Field(default_factory=dict)

    # Outcomes
    metrics: dict[str, float] = Field(default_factory=dict)
    eval_reports: list[str] = Field(default_factory=list, description="Paths to evaluation reports")

    # Documentation
    metadata: dict[str, Any] = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list)


class LineageGraph(BaseModel):
    """A collection of lineage records forming a tree or DAG."""

    root_id: str | None = None
    records: dict[str, LineageRecord] = Field(default_factory=dict)
