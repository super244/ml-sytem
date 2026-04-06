"""Distributed training orchestration with fault tolerance and monitoring."""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from ai_factory.core.exceptions import ClusterError, TimeoutError

logger = logging.getLogger(__name__)


class DistributedBackend(str, Enum):
    """Supported distributed training backends."""

    NCCL = "nccl"  # NVIDIA Collective Communication Library
    GLOO = "gloo"  # Facebook's collective library
    MPI = "mpi"  # Message Passing Interface


class NodeStatus(str, Enum):
    """Status of a distributed training node."""

    PENDING = "pending"
    CONNECTING = "connecting"
    READY = "ready"
    TRAINING = "training"
    FAILED = "failed"
    COMPLETED = "completed"


@dataclass
class NodeInfo:
    """Information about a distributed training node."""

    node_rank: int
    status: NodeStatus = NodeStatus.PENDING
    gpu_ids: list[int] = field(default_factory=list)
    hostname: str = "localhost"
    last_heartbeat: float = 0.0
    error_message: str | None = None


@dataclass
class DistributedConfig:
    """Configuration for distributed training with validation."""

    num_nodes: int = 1
    num_gpus_per_node: int = 1
    master_addr: str = "127.0.0.1"
    master_port: int = 29500
    node_rank: int = 0
    backend: DistributedBackend = DistributedBackend.NCCL
    timeout_seconds: float = 300.0  # Default timeout for operations
    max_restarts: int = 3  # Maximum number of restart attempts

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.num_nodes < 1:
            raise ValueError(f"num_nodes must be >= 1, got {self.num_nodes}")
        if self.num_gpus_per_node < 1:
            raise ValueError(f"num_gpus_per_node must be >= 1, got {self.num_gpus_per_node}")
        if not (1024 <= self.master_port <= 65535):
            raise ValueError(f"master_port must be in range [1024, 65535], got {self.master_port}")
        if not (0 <= self.node_rank < self.num_nodes):
            raise ValueError(f"node_rank must be in range [0, {self.num_nodes}), got {self.node_rank}")

    @property
    def world_size(self) -> int:
        """Total number of processes across all nodes."""
        return self.num_nodes * self.num_gpus_per_node


class DistributedTrainingOrchestrator:
    """
    Production-grade distributed training orchestrator with fault tolerance.

    Features:
    - Multi-node/multi-GPU training via torchrun
    - Automatic restart on failure (configurable)
    - Node health monitoring and heartbeats
    - Proper error handling with structured exceptions
    - Timeout handling for long-running operations
    """

    def __init__(self, config: DistributedConfig) -> None:
        self.config = config
        self._nodes: dict[int, NodeInfo] = {i: NodeInfo(node_rank=i) for i in range(config.num_nodes)}
        self._restart_count = 0
        self._start_time: float | None = None

    def _build_env(self) -> dict[str, str]:
        """Build environment variables for distributed training."""
        env = os.environ.copy()
        env["MASTER_ADDR"] = self.config.master_addr
        env["MASTER_PORT"] = str(self.config.master_port)
        env["WORLD_SIZE"] = str(self.config.world_size)
        env["NODE_RANK"] = str(self.config.node_rank)
        env["NCCL_DEBUG"] = os.environ.get("NCCL_DEBUG", "WARN")
        env["NCCL_IB_DISABLE"] = os.environ.get("NCCL_IB_DISABLE", "1")
        env["PYTHONUNBUFFERED"] = "1"
        return env

    def _build_torchrun_cmd(self, training_script: str, training_args: list[str]) -> list[str]:
        """Build the torchrun command for distributed execution."""
        cmd = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            f"--nnodes={self.config.num_nodes}",
            f"--nproc_per_node={self.config.num_gpus_per_node}",
            "--rdzv_id",
            f"ai_factory_job_{int(time.time())}",
            "--rdzv_backend",
            "c10d",
            f"--rdzv_endpoint={self.config.master_addr}:{self.config.master_port}",
            "--max_restarts",
            str(self.config.max_restarts),
        ]

        # Add timeout if specified
        if self.config.timeout_seconds > 0:
            cmd.extend(["--rdzv_timeout", str(int(self.config.timeout_seconds))])

        cmd.append(training_script)
        cmd.extend(training_args)
        return cmd

    def launch(self, training_script: str, training_args: list[str]) -> int:
        """
        Launch distributed training job with fault tolerance.

        Args:
            training_script: Path to the training script.
            training_args: Arguments to pass to the script.

        Returns:
            Exit code of the training process.

        Raises:
            ClusterError: If distributed training fails after all retries.
            TimeoutError: If training exceeds timeout.
        """
        self._start_time = time.time()

        while self._restart_count <= self.config.max_restarts:
            try:
                return self._launch_once(training_script, training_args)
            except subprocess.CalledProcessError as e:
                self._restart_count += 1
                elapsed = time.time() - self._start_time

                if self._restart_count > self.config.max_restarts:
                    logger.error(f"Distributed training failed after {self.config.max_restarts} restarts")
                    raise ClusterError(
                        f"Training failed after {self.config.max_restarts} restart attempts",
                        cluster_id=f"{self.config.master_addr}:{self.config.master_port}",
                        operation="distributed_training",
                    ) from e

                logger.warning(
                    f"Training failed with exit code {e.returncode}, "
                    f"attempting restart {self._restart_count}/{self.config.max_restarts}"
                )

                # Check timeout
                if elapsed > self.config.timeout_seconds:
                    raise TimeoutError(
                        "distributed_training",
                        timeout_seconds=self.config.timeout_seconds,
                    ) from e

                # Exponential backoff before retry
                backoff = min(2**self._restart_count, 30)
                logger.info(f"Waiting {backoff}s before restart...")
                time.sleep(backoff)

        return 1  # Should never reach here

    def _launch_once(self, training_script: str, training_args: list[str]) -> int:
        """Execute a single training launch attempt."""
        cmd = self._build_torchrun_cmd(training_script, training_args)
        env = self._build_env()

        logger.info(f"Launching distributed training: {' '.join(cmd)}")
        logger.info(f"Configuration: {self.config}")

        try:
            result = subprocess.run(
                cmd,
                env=env,
                stdout=sys.stdout,
                stderr=sys.stderr,
                timeout=self.config.timeout_seconds if self.config.timeout_seconds > 0 else None,
            )
            return result.returncode
        except subprocess.TimeoutExpired as e:
            raise TimeoutError(
                "distributed_training",
                timeout_seconds=self.config.timeout_seconds,
            ) from e

    def get_node_status(self, node_rank: int) -> NodeInfo | None:
        """Get status of a specific node."""
        return self._nodes.get(node_rank)

    def get_all_node_statuses(self) -> dict[int, NodeInfo]:
        """Get status of all nodes."""
        return self._nodes.copy()

    def update_node_status(self, node_rank: int, status: NodeStatus, error: str | None = None) -> None:
        """Update node status."""
        if node_rank in self._nodes:
            node = self._nodes[node_rank]
            node.status = status
            node.last_heartbeat = time.time()
            if error:
                node.error_message = error

    def get_elapsed_time(self) -> float | None:
        """Get elapsed time since training started."""
        if self._start_time is None:
            return None
        return time.time() - self._start_time

    def to_dict(self) -> dict[str, Any]:
        """Serialize orchestrator state to dictionary."""
        return {
            "config": {
                "num_nodes": self.config.num_nodes,
                "num_gpus_per_node": self.config.num_gpus_per_node,
                "world_size": self.config.world_size,
                "master_addr": self.config.master_addr,
                "master_port": self.config.master_port,
                "backend": self.config.backend.value,
            },
            "nodes": {
                rank: {
                    "status": node.status.value,
                    "hostname": node.hostname,
                    "last_heartbeat": node.last_heartbeat,
                }
                for rank, node in self._nodes.items()
            },
            "restart_count": self._restart_count,
            "elapsed_time": self.get_elapsed_time(),
        }
