from __future__ import annotations

import subprocess
from types import SimpleNamespace

import pytest

from ai_factory.orchestration.distributed import DistributedConfig, DistributedTrainingOrchestrator


def test_distributed_orchestrator_build_env() -> None:
    config = DistributedConfig(num_nodes=2, num_gpus_per_node=4, master_addr="10.0.0.1", master_port=23456, node_rank=1)
    orchestrator = DistributedTrainingOrchestrator(config)
    env = orchestrator._build_env()

    assert env["MASTER_ADDR"] == "10.0.0.1"
    assert env["MASTER_PORT"] == "23456"
    assert env["WORLD_SIZE"] == "8"
    assert env["NODE_RANK"] == "1"


def test_distributed_orchestrator_builds_torchrun_command() -> None:
    config = DistributedConfig(num_nodes=2, num_gpus_per_node=2)
    orchestrator = DistributedTrainingOrchestrator(config)
    cmd = orchestrator.get_torchrun_cmd("train.py", ["--epochs", "1"])

    assert "torch.distributed.run" in cmd
    assert "--nnodes=2" in cmd
    assert "--nproc_per_node=2" in cmd
    assert cmd[-3:] == ["train.py", "--epochs", "1"]


def test_distributed_orchestrator_launch_success(monkeypatch: pytest.MonkeyPatch) -> None:
    config = DistributedConfig()
    orchestrator = DistributedTrainingOrchestrator(config)

    def _fake_run(*args, **kwargs):  # noqa: ANN002, ANN003
        assert kwargs["check"] is True
        assert kwargs["env"]["WORLD_SIZE"] == "1"
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("ai_factory.orchestration.distributed.subprocess.run", _fake_run)
    code = orchestrator.launch("train.py", ["--epochs", "1"], check=True)
    assert code == 0


def test_distributed_orchestrator_launch_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    config = DistributedConfig()
    orchestrator = DistributedTrainingOrchestrator(config)

    def _fake_run(*args, **kwargs):  # noqa: ANN002, ANN003
        raise subprocess.CalledProcessError(returncode=2, cmd=args[0])

    monkeypatch.setattr("ai_factory.orchestration.distributed.subprocess.run", _fake_run)
    with pytest.raises(subprocess.CalledProcessError):
        orchestrator.launch("train.py", ["--epochs", "1"], check=True)
