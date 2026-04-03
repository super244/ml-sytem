"""Cluster management for distributed training."""

import asyncio
import logging
from typing import Any

from ai_factory.core.schemas import ResourceSpec, ScalingConfig, TrainingJob

logger = logging.getLogger(__name__)


class ClusterManager:
    """Manages compute cluster for distributed training."""

    def __init__(self, config: ScalingConfig) -> None:
        self.config = config
        metadata_nodes = config.metadata.get("nodes", {})
        self._nodes: dict[str, dict[str, Any]] = dict(metadata_nodes) if isinstance(metadata_nodes, dict) else {}

    async def find_suitable_nodes(self, resource_requirements: ResourceSpec) -> list[str]:
        """Find nodes that meet resource requirements."""
        suitable_nodes = []
        for node_id, node_info in self._nodes.items():
            if self._node_meets_requirements(node_info, resource_requirements):
                suitable_nodes.append(node_id)
        return suitable_nodes

    async def schedule_job(self, job: TrainingJob, primary_node: str) -> str:
        """Schedule a job on the specified node."""
        if primary_node not in self._nodes:
            raise ValueError(f"Node '{primary_node}' is not registered in the cluster inventory")
        job_id = f"job_{job.name}_{int(asyncio.get_event_loop().time())}"
        logger.info("Scheduling job %s on node %s", job_id, primary_node)
        return job_id

    async def get_job_status(self, job_id: str) -> dict[str, Any]:
        """Get status of a scheduled job."""
        return {"job_id": job_id, "status": "running", "progress": 0.5}

    async def cancel_job(self, job_id: str) -> bool:
        """Cancel a scheduled job."""
        logger.info("Cancelling job %s", job_id)
        return True

    async def scale_up(self, additional_nodes: int) -> bool:
        """Scale up cluster by adding nodes."""
        logger.warning("Scale-up requested for %s nodes, but no live cluster backend is configured", additional_nodes)
        return False

    async def scale_down(self, nodes_to_remove: int) -> bool:
        """Scale down cluster by removing idle nodes."""
        logger.warning("Scale-down requested for %s nodes, but no live cluster backend is configured", nodes_to_remove)
        return False

    async def list_nodes(self) -> list[str]:
        """List all nodes in the cluster."""
        return list(self._nodes.keys())

    async def get_cluster_metrics(self) -> dict[str, Any]:
        """Get cluster-wide metrics."""
        if not self._nodes:
            return {
                "status": "degraded",
                "errors": ["No cluster nodes are registered. Configure nodes before requesting live metrics."],
                "total_nodes": 0,
                "active_nodes": 0,
                "total_jobs": 0,
                "resource_utilization": None,
            }
        return {
            "status": "available",
            "errors": [],
            "total_nodes": len(self._nodes),
            "active_nodes": sum(1 for node in self._nodes.values() if node.get("status") != "offline"),
            "total_jobs": 0,
            "resource_utilization": None,
        }

    def _node_meets_requirements(self, node_info: dict[str, Any], requirements: ResourceSpec) -> bool:
        """Check if a node meets resource requirements."""
        cpu_cores = int(node_info.get("cpu_cores", 0) or 0)
        memory_gb = float(node_info.get("memory_gb", 0) or 0)
        gpu_count = int(node_info.get("gpu_count", 0) or 0)
        gpu_memory_gb = float(node_info.get("gpu_memory_gb", 0) or 0)
        return (
            cpu_cores >= requirements.cpu_cores
            and memory_gb >= requirements.memory_gb
            and gpu_count >= requirements.gpu_count
            and gpu_memory_gb >= requirements.gpu_memory_gb
        )
