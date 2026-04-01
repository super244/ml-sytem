from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from ai_factory.models.cluster import ClusterNode
from ai_factory.schemas.cluster import NodeStatus, GPUMetric


class ClusterService:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def list_nodes(self) -> list[NodeStatus]:
        result = await self.db.execute(select(ClusterNode))
        nodes = result.scalars().all()
        return [self._to_schema(n) for n in nodes]

    def _to_schema(self, node: ClusterNode) -> NodeStatus:
        gpus = []
        for g in (node.gpus_json or []):
            gpus.append(GPUMetric(**g))
        return NodeStatus(
            id=node.id,
            name=node.name,
            type=node.type,
            status=node.status,
            gpus=gpus,
            cpu_utilization=node.cpu_utilization,
            ram_used_gb=node.ram_used_gb,
            ram_total_gb=node.ram_total_gb,
            network_rx_mbps=node.network_rx_mbps,
            network_tx_mbps=node.network_tx_mbps,
            active_jobs=node.active_jobs or [],
            cost_per_hour=node.cost_per_hour,
            last_seen=node.last_seen,
        )
