from __future__ import annotations

from types import SimpleNamespace

import pytest

from ai_factory.platform.monitoring import hardware


def test_get_cluster_nodes_without_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(hardware, "_get_torch", lambda: None)
    monkeypatch.setattr(hardware, "get_system_ram_gb", lambda: 32)
    nodes = hardware.get_cluster_nodes()

    assert len(nodes) == 2
    assert nodes[0]["type"] == "CPU"
    assert nodes[0]["memory"] == "32GB"
    assert nodes[0]["usage"] == 0


def test_get_cluster_nodes_with_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Cuda:
        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def get_device_name(index: int) -> str:
            assert index == 0
            return "NVIDIA RTX"

        @staticmethod
        def memory_allocated(index: int) -> int:
            assert index == 0
            return 100

        @staticmethod
        def memory_reserved(index: int) -> int:
            assert index == 0
            return 200

    fake_torch = SimpleNamespace(cuda=_Cuda(), backends=SimpleNamespace(mps=None))
    monkeypatch.setattr(hardware, "_get_torch", lambda: fake_torch)
    monkeypatch.setattr(hardware, "get_system_ram_gb", lambda: 64)
    nodes = hardware.get_cluster_nodes()

    assert nodes[0]["type"] == "NVIDIA RTX"
    assert nodes[0]["usage"] == 50


def test_get_cluster_nodes_with_mps(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Cuda:
        @staticmethod
        def is_available() -> bool:
            return False

    class _Mps:
        @staticmethod
        def is_available() -> bool:
            return True

    fake_torch = SimpleNamespace(
        cuda=_Cuda(),
        backends=SimpleNamespace(mps=_Mps()),
        mps=SimpleNamespace(current_allocated_memory=lambda: 1024**3),
    )
    monkeypatch.setattr(hardware, "_get_torch", lambda: fake_torch)
    monkeypatch.setattr(hardware, "get_system_ram_gb", lambda: 8)
    nodes = hardware.get_cluster_nodes()

    assert nodes[0]["type"] == "MPS (Apple Silicon)"
    assert nodes[0]["usage"] > 0
