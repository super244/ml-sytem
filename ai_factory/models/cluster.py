from datetime import datetime, timezone
from uuid import uuid4
from sqlalchemy import JSON, String, Float, DateTime
from sqlalchemy.orm import Mapped, mapped_column
from ai_factory.database import Base


class ClusterNode(Base):
    __tablename__ = "cluster_nodes"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid4()))
    name: Mapped[str] = mapped_column(String, nullable=False)
    type: Mapped[str] = mapped_column(String, nullable=False)
    status: Mapped[str] = mapped_column(String, default="online")
    gpus_json: Mapped[list] = mapped_column(JSON, default=list)
    cpu_utilization: Mapped[float] = mapped_column(Float, default=0.0)
    ram_used_gb: Mapped[float] = mapped_column(Float, default=0.0)
    ram_total_gb: Mapped[float] = mapped_column(Float, default=64.0)
    network_rx_mbps: Mapped[float] = mapped_column(Float, default=0.0)
    network_tx_mbps: Mapped[float] = mapped_column(Float, default=0.0)
    active_jobs: Mapped[list] = mapped_column(JSON, default=list)
    cost_per_hour: Mapped[float | None] = mapped_column(Float, nullable=True)
    last_seen: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
