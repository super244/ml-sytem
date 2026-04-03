from __future__ import annotations

import subprocess
import sys

from ai_factory.orchestration.distributed import DistributedConfig, DistributedTrainingOrchestrator


def test_distributed_orchestrator_builds_env_and_command() -> None:
    config = DistributedConfig(num_nodes=2, num_gpus_per_node=4, master_addr="10.0.0.5", master_port=29501, node_rank=1)
    orchestrator = DistributedTrainingOrchestrator(config)

    env = orchestrator._build_env()
    assert env["MASTER_ADDR"] == "10.0.0.5"
    assert env["MASTER_PORT"] == "29501"
    assert env["WORLD_SIZE"] == "8"
    assert env["NODE_RANK"] == "1"

    cmd = orchestrator.get_torchrun_cmd("training/train.py", ["--config", "configs/train.yaml"])
    assert cmd[:3] == [sys.executable, "-m", "torch.distributed.run"]
    assert "--nnodes=2" in cmd
    assert "--nproc_per_node=4" in cmd
    assert cmd[-3:] == ["training/train.py", "--config", "configs/train.yaml"]


def test_distributed_orchestrator_launch_invokes_subprocess(monkeypatch) -> None:
    orchestrator = DistributedTrainingOrchestrator(DistributedConfig(num_nodes=1, num_gpus_per_node=2))
    captured: dict[str, object] = {}

    def fake_run(cmd, env, check, stdout, stderr):
        captured["cmd"] = cmd
        captured["env"] = env
        captured["check"] = check
        captured["stdout"] = stdout
        captured["stderr"] = stderr
        return subprocess.CompletedProcess(args=cmd, returncode=0)

    monkeypatch.setattr(subprocess, "run", fake_run)

    exit_code = orchestrator.launch("training/train.py", ["--config", "configs/train.yaml"])
    assert exit_code == 0
    assert isinstance(captured["cmd"], list)
    assert captured["check"] is True
