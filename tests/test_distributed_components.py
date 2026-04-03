from __future__ import annotations

import subprocess

import pytest

from ai_factory.core.distributed import DistributedProcessor
from ai_factory.orchestration.distributed import DistributedConfig, DistributedTrainingOrchestrator


def test_distributed_orchestrator_builds_env_and_command() -> None:
    config = DistributedConfig(num_nodes=2, num_gpus_per_node=4, master_addr="10.1.1.1", master_port=29400, node_rank=1)
    orchestrator = DistributedTrainingOrchestrator(config)

    env = orchestrator._build_env()
    cmd = orchestrator.get_torchrun_cmd("training/train.py", ["--epochs", "1"])

    assert env["MASTER_ADDR"] == "10.1.1.1"
    assert env["WORLD_SIZE"] == "8"
    assert "--nnodes=2" in cmd
    assert "--nproc_per_node=4" in cmd
    assert cmd[-3:] == ["training/train.py", "--epochs", "1"]


def test_distributed_orchestrator_launch_calls_subprocess(monkeypatch: pytest.MonkeyPatch) -> None:
    config = DistributedConfig(num_nodes=1, num_gpus_per_node=1)
    orchestrator = DistributedTrainingOrchestrator(config)
    called: dict[str, object] = {}

    def _fake_run(cmd, **kwargs):
        called["cmd"] = cmd
        called["kwargs"] = kwargs
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr("subprocess.run", _fake_run)
    code = orchestrator.launch("training/train.py", ["--dry-run"], check=True)

    assert code == 0
    assert called["cmd"] is not None


@pytest.mark.asyncio
async def test_distributed_processor_tracks_progress_and_aggregation() -> None:
    processor = DistributedProcessor()
    job_id = await processor.distribute_training_job({"id": "job-123", "dataset": [1, 2, 3, 4], "num_chunks": 2})

    first_status = await processor.aggregate_results(job_id)
    assert first_status["status"] == "running"
    assert first_status["progress"] == 0.0

    await processor.mark_subtask_complete(job_id, "job-123-chunk-0", {"loss": 0.4, "accuracy": 0.8})
    mid_status = await processor.aggregate_results(job_id)
    assert mid_status["status"] == "running"
    assert mid_status["progress"] > 0.0

    await processor.mark_subtask_complete(job_id, "job-123-chunk-1", {"loss": 0.2, "accuracy": 0.9})
    final_status = await processor.aggregate_results(job_id)
    assert final_status["status"] == "completed"
    assert final_status["progress"] == 1.0
    assert final_status["model_summary"]["avg_loss"] == pytest.approx(0.3)
