from __future__ import annotations

import subprocess

import pytest

from ai_factory.orchestration.distributed import DistributedConfig, DistributedTrainingOrchestrator


def test_build_env_populates_distributed_values(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EXISTING_KEY", "value")
    orchestrator = DistributedTrainingOrchestrator(
        DistributedConfig(num_nodes=2, num_gpus_per_node=4, master_addr="10.1.1.9", master_port=29701, node_rank=1)
    )

    env = orchestrator._build_env()

    assert env["EXISTING_KEY"] == "value"
    assert env["MASTER_ADDR"] == "10.1.1.9"
    assert env["MASTER_PORT"] == "29701"
    assert env["WORLD_SIZE"] == "8"
    assert env["NODE_RANK"] == "1"
    assert env["NCCL_DEBUG"] == "WARN"


def test_get_torchrun_cmd_includes_training_args() -> None:
    orchestrator = DistributedTrainingOrchestrator(DistributedConfig(num_nodes=2, num_gpus_per_node=2))

    command = orchestrator.get_torchrun_cmd("training/train.py", ["--config", "configs/train.yaml"])

    assert "--nnodes=2" in command
    assert "--nproc_per_node=2" in command
    assert command[-3:] == ["training/train.py", "--config", "configs/train.yaml"]


def test_launch_invokes_subprocess(monkeypatch: pytest.MonkeyPatch) -> None:
    orchestrator = DistributedTrainingOrchestrator(DistributedConfig(num_nodes=1, num_gpus_per_node=1))
    captured: dict[str, object] = {}

    def fake_run(cmd, env, check, stdout, stderr, timeout=None):  # noqa: ANN001
        captured["cmd"] = cmd
        captured["env"] = env
        captured["check"] = check
        captured["stdout"] = stdout
        captured["stderr"] = stderr
        captured["timeout"] = timeout
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = orchestrator.launch("train.py", ["--epochs", "1"], check=True)

    assert result == 0
    assert isinstance(captured["cmd"], list)
    assert captured["check"] is True
    assert captured["env"]["WORLD_SIZE"] == "1"  # type: ignore[index]


def test_launch_propagates_subprocess_error(monkeypatch: pytest.MonkeyPatch) -> None:
    orchestrator = DistributedTrainingOrchestrator(DistributedConfig())

    def fake_run(cmd, env, check, stdout, stderr, timeout=None):  # noqa: ANN001
        raise subprocess.CalledProcessError(returncode=2, cmd=cmd)

    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(subprocess.CalledProcessError):
        orchestrator.launch("train.py", [], check=True)
