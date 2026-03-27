from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from ai_factory.core.config.schema import OrchestrationConfig
from ai_factory.core.execution.base import CommandSpec
from ai_factory.core.instances.models import InstanceManifest


class UnsupportedInstanceTypeError(RuntimeError):
    pass


def _python_bin(config: OrchestrationConfig) -> str:
    if config.instance.environment.kind == "local":
        return sys.executable
    return config.instance.environment.python_bin or "python3"


def _subsystem_config_ref(config: OrchestrationConfig) -> str | None:
    return config.resolved_subsystem_config_path or config.subsystem.config_ref


def _deploy_echo_command(payload: dict[str, Any]) -> list[str]:
    return [
        _python_bin_from_runtime(),
        "-c",
        (
            "import json; "
            f"print(json.dumps({json.dumps(payload)}, indent=2, sort_keys=True))"
        ),
    ]


def _python_bin_from_runtime() -> str:
    return sys.executable


def _build_finetune_command(config: OrchestrationConfig) -> CommandSpec:
    config_ref = _subsystem_config_ref(config)
    if not config_ref:
        raise ValueError("finetune instances require subsystem.config_ref")
    argv = [_python_bin(config), "-m", "training.train", "--config", config_ref]
    if config.subsystem.dry_run:
        argv.append("--dry-run")
    argv.extend(config.subsystem.extra_args)
    return CommandSpec(argv=argv, cwd=config.execution.cwd, env=config.execution.env)


def _build_evaluate_command(config: OrchestrationConfig) -> CommandSpec:
    config_ref = _subsystem_config_ref(config)
    if not config_ref:
        raise ValueError("evaluate instances require subsystem.config_ref")
    argv = [_python_bin(config), "-m", "evaluation.evaluate", "--config", config_ref]
    argv.extend(config.subsystem.extra_args)
    return CommandSpec(argv=argv, cwd=config.execution.cwd, env=config.execution.env)


def _build_inference_command(config: OrchestrationConfig) -> CommandSpec:
    if config.subsystem.command_override:
        return CommandSpec(
            argv=list(config.subsystem.command_override),
            cwd=config.execution.cwd,
            env=config.execution.env,
            long_running=True,
        )
    env = dict(config.execution.env)
    if config.subsystem.model_variant:
        env.setdefault("AI_FACTORY_MODEL_VARIANT", config.subsystem.model_variant)
    argv = [
        _python_bin(config),
        "-m",
        "uvicorn",
        "inference.app.main:app",
        "--host",
        config.subsystem.serve_host,
        "--port",
        str(config.subsystem.serve_port),
    ]
    argv.extend(config.subsystem.extra_args)
    return CommandSpec(
        argv=argv,
        cwd=config.execution.cwd,
        env=env,
        expected_artifacts={"telemetry": "artifacts/inference/telemetry/requests.jsonl"},
        long_running=True,
    )


def _build_deploy_command(config: OrchestrationConfig, manifest: InstanceManifest) -> CommandSpec:
    provider = config.subsystem.provider
    options = dict(config.subsystem.provider_options)
    if provider is None:
        raise ValueError("deploy instances require subsystem.provider")
    if config.subsystem.command_override:
        return CommandSpec(argv=list(config.subsystem.command_override), cwd=config.execution.cwd, env=config.execution.env)

    source_artifact = config.subsystem.source_artifact_ref
    dry_run = bool(options.get("dry_run", False))
    if dry_run:
        return CommandSpec(
            argv=_deploy_echo_command(
                {
                    "provider": provider,
                    "source_artifact": source_artifact,
                    "instance_id": manifest.id,
                    "mode": "dry_run",
                }
            ),
            cwd=config.execution.cwd,
            env=config.execution.env,
        )

    if provider == "huggingface":
        repo_id = options.get("repo_id")
        if not repo_id or not source_artifact:
            raise ValueError("huggingface deploys require repo_id and source_artifact_ref")
        return CommandSpec(
            argv=["hf", "upload-large-folder", repo_id, source_artifact],
            cwd=config.execution.cwd,
            env=config.execution.env,
        )

    if provider == "ollama":
        model_name = options.get("model_name")
        modelfile = options.get("modelfile")
        if not model_name or not modelfile:
            raise ValueError("ollama deploys require model_name and modelfile")
        return CommandSpec(
            argv=["ollama", "create", model_name, "-f", str(modelfile)],
            cwd=config.execution.cwd,
            env=config.execution.env,
        )

    if provider == "lmstudio":
        import_command = options.get("command")
        if isinstance(import_command, list) and import_command:
            return CommandSpec(argv=[str(part) for part in import_command], cwd=config.execution.cwd, env=config.execution.env)
        raise ValueError("lmstudio deploys require subsystem.provider_options.command for V1")

    raise ValueError(f"Unsupported deployment provider: {provider}")


def build_command(config: OrchestrationConfig, manifest: InstanceManifest) -> CommandSpec:
    if manifest.type == "train":
        raise UnsupportedInstanceTypeError(
            "V1 models train instances in schema/config, but execution is not implemented yet."
        )
    if manifest.type == "finetune":
        return _build_finetune_command(config)
    if manifest.type == "evaluate":
        return _build_evaluate_command(config)
    if manifest.type == "inference":
        return _build_inference_command(config)
    if manifest.type == "deploy":
        return _build_deploy_command(config, manifest)
    raise ValueError(f"Unknown instance type: {manifest.type}")
