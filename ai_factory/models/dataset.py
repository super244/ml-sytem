from datetime import datetime, timezone
from uuid import uuid4
from sqlalchemy import JSON, String, Float, Integer, DateTime
from sqlalchemy.orm import Mapped, mapped_column
from ai_factory.database import Base


class Dataset(Base):
    __tablename__ = "datasets"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid4()))
    name: Mapped[str] = mapped_column(String, nullable=False)
    domain: Mapped[str] = mapped_column(String, nullable=False)
    status: Mapped[str] = mapped_column(String, default="ready")
    sample_count: Mapped[int] = mapped_column(Integer, default=0)
    quality_score_mean: Mapped[float] = mapped_column(Float, default=0.0)
    quality_score_p10: Mapped[float] = mapped_column(Float, default=0.0)
    quality_score_p90: Mapped[float] = mapped_column(Float, default=0.0)
    size_mb: Mapped[float] = mapped_column(Float, default=0.0)
    content_hash: Mapped[str] = mapped_column(String, default="")
    pipeline_config_hash: Mapped[str] = mapped_column(String, default="")
    git_sha: Mapped[str] = mapped_column(String, default="")
    pack_summary_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    samples_json: Mapped[list] = mapped_column(JSON, default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
