"""Resource management for distributed training."""

import logging
from typing import Any

from ai_factory.core.schemas import ResourceSpec, ScalingConfig

logger = logging.getLogger(__name__)


class ResourceManager:
    """Manages resources across the training cluster."""

    def __init__(self, config: ScalingConfig) -> None:
        self.config = config
        metadata_resources = config.metadata.get("node_resources", {})
        self._resources: dict[str, dict[str, Any]] = (
            dict(metadata_resources) if isinstance(metadata_resources, dict) else {}
        )

    def select_optimal_node(self, suitable_nodes: list[str], requirements: ResourceSpec) -> str:
        """Select the optimal node from suitable candidates."""
        if not suitable_nodes:
            raise ValueError("No suitable nodes available")

        # Simple selection - in real implementation would consider load, latency, etc.
        return suitable_nodes[0]

    async def get_node_resources(self, node_id: str) -> dict[str, Any]:
        """Get current resource usage for a node."""
        node_resources = self._resources.get(node_id)
        if node_resources is None:
            return {
                "node_id": node_id,
                "status": "degraded",
                "errors": [f"No resource inventory is registered for node '{node_id}'."],
                "resources": {},
            }
        return {
            "node_id": node_id,
            "status": "available",
            "errors": [],
            "resources": node_resources,
        }

    async def get_cluster_resources(self) -> dict[str, Any]:
        """Get cluster-wide resource summary."""
        if not self._resources:
            return {
                "status": "degraded",
                "errors": ["Cluster resource inventory is unavailable."],
                "nodes": 0,
                "summary": {},
            }
        return {
            "status": "available",
            "errors": [],
            "nodes": len(self._resources),
            "summary": {
                "total_cpu_cores": sum(int(node.get("cpu_cores", 0) or 0) for node in self._resources.values()),
                "total_memory_gb": sum(float(node.get("memory_gb", 0) or 0) for node in self._resources.values()),
                "total_gpus": sum(int(node.get("gpu_count", 0) or 0) for node in self._resources.values()),
                "total_gpu_memory_gb": sum(
                    float(node.get("gpu_memory_gb", 0) or 0) for node in self._resources.values()
                ),
            },
        }

    async def reserve_resources(self, node_id: str, requirements: ResourceSpec) -> bool:
        """Reserve resources on a specific node."""
        del requirements
        if node_id not in self._resources:
            logger.warning("Cannot reserve resources on unknown node %s", node_id)
            return False
        logger.info("Reserving resources on node %s", node_id)
        return True

    async def release_resources(self, node_id: str, requirements: ResourceSpec) -> bool:
        """Release resources on a specific node."""
        del requirements
        if node_id not in self._resources:
            logger.warning("Cannot release resources on unknown node %s", node_id)
            return False
        logger.info("Releasing resources on node %s", node_id)
        return True
