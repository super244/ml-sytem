"""Scaling manager for distributed training and resource management."""

import logging
from pathlib import Path
from typing import Any

from ai_factory.core.schemas import ScalingConfig, TrainingJob

from .cluster import ClusterManager
from .resources import ResourceManager

logger = logging.getLogger(__name__)


class ScalingManager:
    """Manages scaling of AI-Factory workloads across multiple nodes."""

    def __init__(self, config: ScalingConfig, repo_root: Path):
        self.config = config
        self.repo_root = repo_root
        self.cluster_manager = ClusterManager(config)
        self.resource_manager = ResourceManager(config)
        self._active_jobs: dict[str, TrainingJob] = {}

    async def schedule_training_job(self, job: TrainingJob) -> str:
        """Schedule a training job across available resources."""
        # Check resource requirements
        required_resources = job.resource_requirements

        # Find suitable nodes
        suitable_nodes = await self.cluster_manager.find_suitable_nodes(required_resources)
        if not suitable_nodes:
            raise ValueError("No suitable nodes available for job")

        # Select primary node based on availability and performance
        primary_node = self.resource_manager.select_optimal_node(suitable_nodes, required_resources)

        # Schedule the job
        job_id = await self.cluster_manager.schedule_job(job, primary_node)
        self._active_jobs[job_id] = job

        logger.info(f"Scheduled training job {job_id} on node {primary_node}")
        return job_id

    async def scale_cluster(self, target_nodes: int) -> bool:
        """Scale the cluster to target number of nodes."""
        current_nodes = len(await self.cluster_manager.list_nodes())

        if target_nodes > current_nodes:
            # Scale up
            return await self.cluster_manager.scale_up(target_nodes - current_nodes)
        elif target_nodes < current_nodes:
            # Scale down (only idle nodes)
            return await self.cluster_manager.scale_down(current_nodes - target_nodes)

        return True

    async def get_job_status(self, job_id: str) -> dict[str, Any]:
        """Get status of a distributed training job."""
        if job_id not in self._active_jobs:
            raise ValueError(f"Job {job_id} not found")

        return await self.cluster_manager.get_job_status(job_id)

    async def cancel_job(self, job_id: str) -> bool:
        """Cancel a distributed training job."""
        if job_id not in self._active_jobs:
            raise ValueError(f"Job {job_id} not found")

        success = await self.cluster_manager.cancel_job(job_id)
        if success:
            del self._active_jobs[job_id]

        return success

    async def list_active_jobs(self) -> list[dict[str, Any]]:
        """List all active training jobs."""
        jobs = []
        for job_id in self._active_jobs:
            status = await self.get_job_status(job_id)
            jobs.append(status)
        return jobs

    async def get_cluster_metrics(self) -> dict[str, Any]:
        """Get cluster-wide metrics and resource usage."""
        return await self.cluster_manager.get_cluster_metrics()
