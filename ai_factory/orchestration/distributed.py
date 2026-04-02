import logging
import os
import subprocess
import sys
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class DistributedConfig:
    """Configuration for distributed training."""

    num_nodes: int = 1
    num_gpus_per_node: int = 1
    master_addr: str = "127.0.0.1"
    master_port: int = 29500
    node_rank: int = 0
    backend: str = "nccl"


class DistributedTrainingOrchestrator:
    """
    Manages multi-node/multi-GPU training runs using torchrun.
    """

    def __init__(self, config: DistributedConfig) -> None:
        self.config = config
        self.world_size = self.config.num_nodes * self.config.num_gpus_per_node

    def _build_env(self) -> dict[str, str]:
        """Builds the environment variables for distributed training."""
        env = os.environ.copy()
        env["MASTER_ADDR"] = self.config.master_addr
        env["MASTER_PORT"] = str(self.config.master_port)
        env["WORLD_SIZE"] = str(self.world_size)
        env["NODE_RANK"] = str(self.config.node_rank)
        env["NCCL_DEBUG"] = "WARN"  # Reduce verbosity in production
        return env

    def get_torchrun_cmd(self, training_script: str, training_args: list[str]) -> list[str]:
        """Generates the torchrun command for distributed execution."""
        cmd = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            f"--nnodes={self.config.num_nodes}",
            f"--nproc_per_node={self.config.num_gpus_per_node}",
            "--rdzv_id=ai_factory_job",
            "--rdzv_backend=c10d",
            f"--rdzv_endpoint={self.config.master_addr}:{self.config.master_port}",
            training_script,
        ]
        cmd.extend(training_args)
        return cmd

    def launch(self, training_script: str, training_args: list[str], check: bool = True) -> int:
        """Launches the distributed training job.

        Args:
            training_script: The path to the script to run.
            training_args: Arguments to pass to the script.
            check: Whether to raise an exception on non-zero exit code.

        Returns:
            The exit code of the torchrun process.
        """
        cmd = self.get_torchrun_cmd(training_script, training_args)
        env = self._build_env()

        logger.info(f"Launching distributed training with command: {' '.join(cmd)}")
        # Log config at debug level only in development
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Distributed config: {self.config}")

        try:
            result = subprocess.run(
                cmd,
                env=env,
                check=check,
                stdout=sys.stdout,
                stderr=sys.stderr,
            )
            return result.returncode
        except subprocess.CalledProcessError as e:
            logger.error(f"Distributed training failed with exit code {e.returncode}")
            raise
