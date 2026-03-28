from __future__ import annotations

import importlib

import pytest


def _import_or_skip(module_name: str):
    try:
        return importlib.import_module(module_name)
    except Exception as exc:  # pragma: no cover - only used when surfaces are not wired yet
        pytest.skip(f"{module_name} is not available yet: {exc}", allow_module_level=True)


ai_factory_cli = _import_or_skip("ai_factory.cli")
_import_or_skip("inference.app.services.instance_service")
_import_or_skip("inference.app.routers.instances")
_import_or_skip("inference.app.routers.orchestration")


def test_cli_module_exposes_main():
    assert hasattr(ai_factory_cli, "main")


def test_cli_parse_args_supports_orchestration_commands(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        ["ai-factory", "tasks", "instance-001"],
    )
    args = ai_factory_cli.parse_args()
    assert args.command == "tasks"
    assert args.target == "instance-001"

    monkeypatch.setattr(
        "sys.argv",
        ["ai-factory", "watch", "instance-001", "--timeout", "5"],
    )
    args = ai_factory_cli.parse_args()
    assert args.command == "watch"
    assert args.timeout == 5.0


def test_cli_parse_args_supports_shared_backend_overrides(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        ["ai-factory", "--artifacts-dir", "/tmp/ai-factory", "--repo-root", "/workspace", "list"],
    )
    args = ai_factory_cli.parse_args()
    assert args.command == "list"
    assert args.artifacts_dir == "/tmp/ai-factory"
    assert args.repo_root == "/workspace"


def test_cli_parse_args_supports_tui_command(monkeypatch):
    monkeypatch.setattr("sys.argv", ["ai-factory", "tui", "--refresh-seconds", "1.5"])
    args = ai_factory_cli.parse_args()
    assert args.command == "tui"
    assert args.refresh_seconds == 1.5
