from uuid import uuid4
from sqlalchemy import JSON, String
from sqlalchemy.orm import Mapped, mapped_column
from ai_factory.database import Base


class LineageNode(Base):
    __tablename__ = "lineage_nodes"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid4()))
    type: Mapped[str] = mapped_column(String, nullable=False)
    label: Mapped[str] = mapped_column(String, nullable=False)
    metadata_json: Mapped[dict] = mapped_column(JSON, default=dict)


class LineageEdge(Base):
    __tablename__ = "lineage_edges"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid4()))
    source_id: Mapped[str] = mapped_column(String, nullable=False)
    target_id: Mapped[str] = mapped_column(String, nullable=False)
    label: Mapped[str] = mapped_column(String, nullable=False)
