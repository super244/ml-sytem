from datetime import datetime, timezone
from uuid import uuid4
from sqlalchemy import JSON, String, Float, Integer, DateTime, Text
from sqlalchemy.orm import Mapped, mapped_column
from ai_factory.database import Base


class TrainingJob(Base):
    __tablename__ = "training_jobs"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid4()))
    name: Mapped[str] = mapped_column(String, nullable=False)
    type: Mapped[str] = mapped_column(String, nullable=False)
    status: Mapped[str] = mapped_column(String, default="queued")
    base_model: Mapped[str] = mapped_column(String, nullable=False)
    config_json: Mapped[dict] = mapped_column(JSON, default=dict)
    dataset_id: Mapped[str | None] = mapped_column(String, nullable=True)
    node_id: Mapped[str | None] = mapped_column(String, nullable=True)
    current_step: Mapped[int] = mapped_column(Integer, default=0)
    total_steps: Mapped[int] = mapped_column(Integer, default=10000)
    current_loss: Mapped[float | None] = mapped_column(Float, nullable=True)
    loss_history: Mapped[list] = mapped_column(JSON, default=list)
    step_history: Mapped[list] = mapped_column(JSON, default=list)
    gpu_utilization: Mapped[list] = mapped_column(JSON, default=list)
    vram_used_gb: Mapped[float] = mapped_column(Float, default=0.0)
    vram_total_gb: Mapped[float] = mapped_column(Float, default=80.0)
    eval_scores: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    lineage_node_id: Mapped[str | None] = mapped_column(String, nullable=True)
    parent_model_id: Mapped[str | None] = mapped_column(String, nullable=True)
    dataset_hash: Mapped[str | None] = mapped_column(String, nullable=True)
    logs_tail: Mapped[list] = mapped_column(JSON, default=list)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
