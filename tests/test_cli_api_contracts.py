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


def test_cli_module_exposes_main() -> None:
    assert hasattr(ai_factory_cli, "main")


def test_cli_parse_args_supports_orchestration_commands(monkeypatch) -> None:
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


def test_cli_parse_args_supports_shared_backend_overrides(monkeypatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        ["ai-factory", "--artifacts-dir", "/tmp/ai-factory", "--repo-root", "/workspace", "list"],
    )
    args = ai_factory_cli.parse_args()
    assert args.command == "list"
    assert args.artifacts_dir == "/tmp/ai-factory"
    assert args.repo_root == "/workspace"


def test_cli_parse_args_supports_tui_command(monkeypatch) -> None:
    monkeypatch.setattr("sys.argv", ["ai-factory", "tui", "--refresh-seconds", "1.5"])
    args = ai_factory_cli.parse_args()
    assert args.command == "tui"
    assert args.refresh_seconds == 1.5

    monkeypatch.setattr("sys.argv", ["ai-factory", "workspace", "--root", "/tmp/project"])
    args = ai_factory_cli.parse_args()
    assert args.command == "workspace"
    assert args.root == "/tmp/project"

    monkeypatch.setattr("sys.argv", ["ai-factory", "optimize", "detect", "--json"])
    args = ai_factory_cli.parse_args()
    assert args.command == "optimize"
    assert args.optimize_command == "detect"
    assert args.json is True


def test_cli_parse_args_supports_control_center_new_and_inference(monkeypatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "ai-factory",
            "new",
            "--config",
            "examples/orchestration/train.yaml",
            "--name",
            "scratch-branch",
            "--user-level",
            "dev",
            "--origin",
            "from_scratch",
            "--learning-mode",
            "supervised",
            "--architecture-family",
            "transformer",
            "--architecture-hidden-size",
            "768",
            "--deployment-target",
            "ollama",
        ],
    )
    args = ai_factory_cli.parse_args()
    assert args.command == "new"
    assert args.name == "scratch-branch"
    assert args.user_level == "dev"
    assert args.origin == "from_scratch"
    assert args.architecture_hidden_size == 768
    assert args.deployment_targets == ["ollama"]

    monkeypatch.setattr(
        "sys.argv",
        ["ai-factory", "inference", "instance-001", "--config", "examples/orchestration/inference.yaml"],
    )
    args = ai_factory_cli.parse_args()
    assert args.command == "inference"
    assert args.instance_id == "instance-001"

    monkeypatch.setattr(
        "sys.argv",
        ["ai-factory", "action", "instance-001", "finetune", "--config", "examples/orchestration/finetune.yaml", "--no-start"],
    )
    args = ai_factory_cli.parse_args()
    assert args.command == "action"
    assert args.instance_id == "instance-001"
    assert args.action == "finetune"
    assert args.no_start is True
