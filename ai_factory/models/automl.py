from datetime import datetime, timezone
from uuid import uuid4
from sqlalchemy import JSON, String, Float, Integer, DateTime
from sqlalchemy.orm import Mapped, mapped_column
from ai_factory.database import Base


class AutoMLSearch(Base):
    __tablename__ = "automl_searches"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid4()))
    strategy: Mapped[str] = mapped_column(String, nullable=False)
    status: Mapped[str] = mapped_column(String, default="running")
    search_space: Mapped[dict] = mapped_column(JSON, default=dict)
    best_loss: Mapped[float | None] = mapped_column(Float, nullable=True)
    best_config: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


class AutoMLRun(Base):
    __tablename__ = "automl_runs"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid4()))
    search_id: Mapped[str] = mapped_column(String, nullable=False)
    status: Mapped[str] = mapped_column(String, default="running")
    hyperparams: Mapped[dict] = mapped_column(JSON, default=dict)
    eval_loss: Mapped[float | None] = mapped_column(Float, nullable=True)
    composite_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    training_minutes: Mapped[float | None] = mapped_column(Float, nullable=True)
    step_pruned: Mapped[int | None] = mapped_column(Integer, nullable=True)
    prune_reason: Mapped[str | None] = mapped_column(String, nullable=True)
    job_id: Mapped[str | None] = mapped_column(String, nullable=True)
