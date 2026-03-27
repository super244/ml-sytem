from __future__ import annotations

import json
import sys
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


def _command_env(config: OrchestrationConfig) -> dict[str, str]:
    env = dict(config.execution.env)
    env.setdefault("AI_FACTORY_USER_LEVEL", config.experience.level)
    env.setdefault("AI_FACTORY_ORCHESTRATION_MODE", config.orchestration_mode)
    env.setdefault("AI_FACTORY_INSTANCE_TYPE", config.instance.type)
    return env


def _command_override(config: OrchestrationConfig, *, long_running: bool = False) -> CommandSpec | None:
    if not config.subsystem.command_override:
        return None
    return CommandSpec(
        argv=list(config.subsystem.command_override),
        cwd=config.execution.cwd,
        env=_command_env(config),
        long_running=long_running,
    )


def _build_prepare_command(config: OrchestrationConfig) -> CommandSpec:
    override = _command_override(config)
    if override is not None:
        return override
    config_ref = _subsystem_config_ref(config)
    if not config_ref:
        raise ValueError("prepare instances require subsystem.config_ref")
    argv = [_python_bin(config), "data/prepare_dataset.py", "--config", config_ref]
    argv.extend(config.subsystem.extra_args)
    return CommandSpec(argv=argv, cwd=config.execution.cwd, env=_command_env(config))


def _build_training_command(config: OrchestrationConfig) -> CommandSpec:
    override = _command_override(config)
    if override is not None:
        return override
    config_ref = _subsystem_config_ref(config)
    if not config_ref:
        raise ValueError(f"{config.instance.type} instances require subsystem.config_ref")
    argv = [_python_bin(config), "-m", "training.train", "--config", config_ref]
    if config.subsystem.dry_run:
        argv.append("--dry-run")
    argv.extend(config.subsystem.extra_args)
    return CommandSpec(argv=argv, cwd=config.execution.cwd, env=_command_env(config))


def _build_evaluate_command(config: OrchestrationConfig) -> CommandSpec:
    override = _command_override(config)
    if override is not None:
        return override
    config_ref = _subsystem_config_ref(config)
    if not config_ref:
        raise ValueError("evaluate instances require subsystem.config_ref")
    argv = [_python_bin(config), "-m", "evaluation.evaluate", "--config", config_ref]
    argv.extend(config.subsystem.extra_args)
    return CommandSpec(argv=argv, cwd=config.execution.cwd, env=_command_env(config))


def _build_inference_command(config: OrchestrationConfig) -> CommandSpec:
    override = _command_override(config, long_running=True)
    if override is not None:
        return override
    env = _command_env(config)
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


def _build_report_command(config: OrchestrationConfig) -> CommandSpec:
    override = _command_override(config)
    if override is not None:
        return override
    raise ValueError("report instances require subsystem.command_override")


def _build_deploy_command(config: OrchestrationConfig, manifest: InstanceManifest) -> CommandSpec:
    provider = config.subsystem.provider
    options = dict(config.subsystem.provider_options)
    source_artifact = config.subsystem.source_artifact_ref
    if provider is None:
        raise ValueError("deploy instances require subsystem.provider")
    override = _command_override(config)
    if override is not None:
        return override
    if provider in {"api", "custom_api"}:
        endpoint = options.get("endpoint")
        method = str(options.get("method", "POST")).upper()
        if not endpoint or not source_artifact:
            raise ValueError("custom_api deploys require endpoint and source_artifact_ref")
        payload = json.dumps(
            {
                "instance_id": manifest.id,
                "source_artifact": source_artifact,
                "provider": provider,
            }
        )
        return CommandSpec(
            argv=[
                _python_bin_from_runtime(),
                "-c",
                (
                    "import json, sys, urllib.request; "
                    f"payload = {json.dumps(payload)}.encode('utf-8'); "
                    f"request = urllib.request.Request({json.dumps(endpoint)}, data=payload, method={json.dumps(method)}); "
                    "request.add_header('Content-Type', 'application/json'); "
                    "response = urllib.request.urlopen(request); "
                    "body = response.read().decode('utf-8'); "
                    "sys.stdout.write(body)"
                ),
            ],
            cwd=config.execution.cwd,
            env=_command_env(config),
        )

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
            env=_command_env(config),
        )

    if provider == "huggingface":
        repo_id = options.get("repo_id")
        if not repo_id or not source_artifact:
            raise ValueError("huggingface deploys require repo_id and source_artifact_ref")
        return CommandSpec(
            argv=["hf", "upload-large-folder", repo_id, source_artifact],
            cwd=config.execution.cwd,
            env=_command_env(config),
        )

    if provider == "ollama":
        model_name = options.get("model_name")
        modelfile = options.get("modelfile")
        if not model_name or not modelfile:
            raise ValueError("ollama deploys require model_name and modelfile")
        return CommandSpec(
            argv=["ollama", "create", model_name, "-f", str(modelfile)],
            cwd=config.execution.cwd,
            env=_command_env(config),
        )

    if provider == "lmstudio":
        import_command = options.get("command")
        if isinstance(import_command, list) and import_command:
            return CommandSpec(argv=[str(part) for part in import_command], cwd=config.execution.cwd, env=_command_env(config))
        raise ValueError("lmstudio deploys require subsystem.provider_options.command for V1")

    raise ValueError(f"Unsupported deployment provider: {provider}")


def build_command(config: OrchestrationConfig, manifest: InstanceManifest) -> CommandSpec:
    if manifest.type == "prepare":
        return _build_prepare_command(config)
    if manifest.type == "train":
        return _build_training_command(config)
    if manifest.type == "finetune":
        return _build_training_command(config)
    if manifest.type == "evaluate":
        return _build_evaluate_command(config)
    if manifest.type == "inference":
        return _build_inference_command(config)
    if manifest.type == "deploy":
        return _build_deploy_command(config, manifest)
    if manifest.type == "report":
        return _build_report_command(config)
    raise ValueError(f"Unknown instance type: {manifest.type}")
