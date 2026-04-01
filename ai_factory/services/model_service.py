from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from ai_factory.models.model_registry import ModelCheckpoint
from ai_factory.models.lineage import LineageNode, LineageEdge
from ai_factory.schemas.model_registry import ModelSummary, LineageGraph, LineageNodeSchema, LineageEdgeSchema


class ModelService:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def list_models(self) -> list[ModelSummary]:
        result = await self.db.execute(select(ModelCheckpoint).order_by(ModelCheckpoint.created_at.desc()))
        models = result.scalars().all()
        return [self._to_summary(m) for m in models]

    async def get_lineage(self, model_id: str) -> LineageGraph | None:
        result = await self.db.execute(select(ModelCheckpoint).where(ModelCheckpoint.id == model_id))
        model = result.scalar_one_or_none()
        if not model:
            return None

        all_nodes_result = await self.db.execute(select(LineageNode))
        all_edges_result = await self.db.execute(select(LineageEdge))
        all_nodes = {n.id: n for n in all_nodes_result.scalars().all()}
        all_edges = list(all_edges_result.scalars().all())

        checkpoint_label = model.name
        root_ids = set()
        for n_id, n in all_nodes.items():
            if n.label == checkpoint_label or n_id == model_id:
                root_ids.add(n_id)

        reachable = set(root_ids)
        frontier = list(root_ids)
        for _ in range(5):
            next_frontier = []
            for edge in all_edges:
                if edge.source_id in frontier and edge.target_id not in reachable:
                    reachable.add(edge.target_id)
                    next_frontier.append(edge.target_id)
                if edge.target_id in frontier and edge.source_id not in reachable:
                    reachable.add(edge.source_id)
                    next_frontier.append(edge.source_id)
            frontier = next_frontier
            if not frontier:
                break

        if not reachable:
            reachable = set(all_nodes.keys())

        nodes = [
            LineageNodeSchema(id=n.id, type=n.type, label=n.label, metadata=n.metadata_json or {})
            for n_id, n in all_nodes.items() if n_id in reachable
        ]
        edges = [
            LineageEdgeSchema(source=e.source_id, target=e.target_id, label=e.label)
            for e in all_edges if e.source_id in reachable and e.target_id in reachable
        ]

        return LineageGraph(nodes=nodes, edges=edges)

    def _to_summary(self, m: ModelCheckpoint) -> ModelSummary:
        return ModelSummary(
            id=m.id,
            name=m.name,
            base_model=m.base_model,
            training_type=m.training_type,
            eval_scores=m.eval_scores or {},
            children_count=m.children_count,
            dataset_hash=m.dataset_hash or "",
            parent_model_id=m.parent_model_id,
            created_at=m.created_at,
            deployed=m.deployed,
            size_gb=m.size_gb,
        )
