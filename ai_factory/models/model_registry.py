from datetime import datetime, timezone
from uuid import uuid4
from sqlalchemy import JSON, String, Float, Integer, DateTime, Boolean
from sqlalchemy.orm import Mapped, mapped_column
from ai_factory.database import Base


class ModelCheckpoint(Base):
    __tablename__ = "model_checkpoints"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid4()))
    name: Mapped[str] = mapped_column(String, nullable=False)
    base_model: Mapped[str] = mapped_column(String, nullable=False)
    training_type: Mapped[str] = mapped_column(String, nullable=False)
    checkpoint_path: Mapped[str] = mapped_column(String, default="")
    size_gb: Mapped[float] = mapped_column(Float, default=0.0)
    job_id: Mapped[str | None] = mapped_column(String, nullable=True)
    parent_model_id: Mapped[str | None] = mapped_column(String, nullable=True)
    dataset_hash: Mapped[str] = mapped_column(String, default="")
    eval_scores: Mapped[dict] = mapped_column(JSON, default=dict)
    children_count: Mapped[int] = mapped_column(Integer, default=0)
    deployed: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
