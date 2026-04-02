from __future__ import annotations

from pathlib import Path

from ai_factory.core.io import load_json, write_json
from ai_factory.core.lineage.models import LineageGraph, LineageRecord


class LineageRegistry:
    """Registry for persisting model lineage graphs."""

    def __init__(self, storage_path: str | Path):
        self._path = Path(storage_path)
        self._path.mkdir(parents=True, exist_ok=True)
        self._registry_file = self._path / "graph.json"

    def _load_graph(self) -> LineageGraph:
        data = load_json(self._registry_file, default=None)
        if data is None:
            return LineageGraph()
        return LineageGraph.model_validate(data)

    def _save_graph(self, graph: LineageGraph) -> None:
        write_json(self._registry_file, graph.model_dump(mode="json"))

    def record_lineage(self, record: LineageRecord) -> None:
        """Register a new lineage record."""
        graph = self._load_graph()
        if record.id in graph.records:
            return  # Already registered
        graph.records[record.id] = record
        if record.parent_id is None and graph.root_id is None:
            graph.root_id = record.id
        self._save_graph(graph)

    def get_lineage(self, record_id: str) -> LineageRecord | None:
        """Fetch a lineage record by its hash."""
        graph = self._load_graph()
        return graph.records.get(record_id)

    def list_lineage(self, *, limit: int = 100) -> list[LineageRecord]:
        """List lineage records sorted by recency."""
        graph = self._load_graph()
        records = list(graph.records.values())
        records.sort(key=lambda r: r.created_at, reverse=True)
        return records[:limit]

    def get_ancestors(self, record_id: str) -> list[LineageRecord]:
        """Walk up the lineage tree from a record to its root."""
        graph = self._load_graph()
        ancestors: list[LineageRecord] = []
        current_id: str | None = record_id
        while current_id:
            record = graph.records.get(current_id)
            if not record:
                break
            ancestors.append(record)
            current_id = record.parent_id
        return ancestors
