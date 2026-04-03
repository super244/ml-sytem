from __future__ import annotations

import json
import sys
from typing import TYPE_CHECKING, Any, cast

from ai_factory.core.execution.base import CommandSpec
from ai_factory.core.instances.models import InstanceManifest
from ai_factory.core.plugins.base import DeploymentProviderPlugin, InstanceHandlerPlugin, PluginDescriptor

if TYPE_CHECKING:
    from ai_factory.core.config.schema import OrchestrationConfig
    from ai_factory.core.plugins.registry import PluginRegistry


def _python_bin(config: OrchestrationConfig) -> str:
    if config.instance.environment.kind == "local":
        return sys.executable
    return config.instance.environment.python_bin or "python3"


def _python_bin_from_runtime() -> str:
    return sys.executable


def _subsystem_config_ref(config: OrchestrationConfig) -> str | None:
    return config.resolved_subsystem_config_path or config.subsystem.config_ref


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


def _deploy_echo_command(payload: dict[str, Any]) -> list[str]:
    return [
        _python_bin_from_runtime(),
        "-c",
        (f"import json; print(json.dumps({json.dumps(payload)}, indent=2, sort_keys=True))"),
    ]


class PrepareInstanceHandler:
    descriptor = PluginDescriptor(
        kind="instance_handler",
        name="builtin.prepare",
        label="Prepare instance handler",
        capabilities=["prepare"],
    )
    instance_types = ("prepare",)

    def build_command(
        self,
        config: OrchestrationConfig,
        manifest: InstanceManifest,
        registry: PluginRegistry,
    ) -> CommandSpec:
        override = _command_override(config)
        if override is not None:
            return override
        config_ref = _subsystem_config_ref(config)
        if not config_ref:
            raise ValueError("prepare instances require subsystem.config_ref")
        argv = [_python_bin(config), "data/prepare_dataset.py", "--config", config_ref]
        argv.extend(config.subsystem.extra_args)
        return CommandSpec(argv=argv, cwd=config.execution.cwd, env=_command_env(config))


class TrainingInstanceHandler:
    descriptor = PluginDescriptor(
        kind="instance_handler",
        name="builtin.training",
        label="Training instance handler",
        capabilities=["train", "finetune"],
    )
    instance_types = ("train", "finetune")

    def build_command(
        self,
        config: OrchestrationConfig,
        manifest: InstanceManifest,
        registry: PluginRegistry,
    ) -> CommandSpec:
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


class EvaluationInstanceHandler:
    descriptor = PluginDescriptor(
        kind="instance_handler",
        name="builtin.evaluation",
        label="Evaluation instance handler",
        capabilities=["evaluate"],
    )
    instance_types = ("evaluate",)

    def build_command(
        self,
        config: OrchestrationConfig,
        manifest: InstanceManifest,
        registry: PluginRegistry,
    ) -> CommandSpec:
        override = _command_override(config)
        if override is not None:
            return override
        config_ref = _subsystem_config_ref(config)
        if not config_ref:
            raise ValueError("evaluate instances require subsystem.config_ref")
        argv = [_python_bin(config), "-m", "evaluation.evaluate", "--config", config_ref]
        argv.extend(config.subsystem.extra_args)
        return CommandSpec(argv=argv, cwd=config.execution.cwd, env=_command_env(config))


class InferenceInstanceHandler:
    descriptor = PluginDescriptor(
        kind="instance_handler",
        name="builtin.inference",
        label="Inference instance handler",
        capabilities=["inference"],
    )
    instance_types = ("inference",)

    def build_command(
        self,
        config: OrchestrationConfig,
        manifest: InstanceManifest,
        registry: PluginRegistry,
    ) -> CommandSpec:
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


class ReportInstanceHandler:
    descriptor = PluginDescriptor(
        kind="instance_handler",
        name="builtin.report",
        label="Report instance handler",
        capabilities=["report"],
    )
    instance_types = ("report",)

    def build_command(
        self,
        config: OrchestrationConfig,
        manifest: InstanceManifest,
        registry: PluginRegistry,
    ) -> CommandSpec:
        override = _command_override(config)
        if override is not None:
            return override
        input_path = config.subsystem.source_artifact_ref
        output_path = config.subsystem.output_dir_override or "evaluation/results/failure_analysis.json"
        if not input_path:
            raise ValueError("report instances require subsystem.source_artifact_ref")
        argv = [
            _python_bin(config),
            "evaluation/analysis/analyze_failures.py",
            "--input",
            input_path,
            "--output",
            output_path,
        ]
        argv.extend(config.subsystem.extra_args)
        return CommandSpec(argv=argv, cwd=config.execution.cwd, env=_command_env(config))


class DeploymentInstanceHandler:
    descriptor = PluginDescriptor(
        kind="instance_handler",
        name="builtin.deployment",
        label="Deployment instance handler",
        capabilities=["deploy"],
    )
    instance_types = ("deploy",)

    def build_command(
        self,
        config: OrchestrationConfig,
        manifest: InstanceManifest,
        registry: PluginRegistry,
    ) -> CommandSpec:
        provider = config.subsystem.provider
        options = dict(config.subsystem.provider_options)
        source_artifact = config.subsystem.source_artifact_ref
        if provider is None:
            raise ValueError("deploy instances require subsystem.provider")
        override = _command_override(config)
        if override is not None:
            return override
        if bool(options.get("dry_run", False)):
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
        provider_plugin = registry.get_deployment_provider(provider)
        return provider_plugin.build_command(config, manifest, registry)


class ApiCompatibleDeploymentProvider:
    descriptor = PluginDescriptor(
        kind="deployment_provider",
        name="builtin.api_compatible",
        label="API-compatible deployment provider",
        capabilities=["api", "custom_api", "openai_compatible_api"],
    )
    provider_names = ("api", "custom_api", "openai_compatible_api")

    def build_command(
        self,
        config: OrchestrationConfig,
        manifest: InstanceManifest,
        registry: PluginRegistry,
    ) -> CommandSpec:
        options = dict(config.subsystem.provider_options)
        endpoint = options.get("endpoint")
        method = str(options.get("method", "POST")).upper()
        source_artifact = config.subsystem.source_artifact_ref
        if not endpoint or not source_artifact:
            raise ValueError("custom_api deploys require endpoint and source_artifact_ref")
        payload = json.dumps(
            {
                "instance_id": manifest.id,
                "source_artifact": source_artifact,
                "provider": config.subsystem.provider,
            }
        )
        return CommandSpec(
            argv=[
                _python_bin_from_runtime(),
                "-c",
                (
                    "import json, sys, urllib.request; "
                    f"payload = {json.dumps(payload)}.encode('utf-8'); "
                    "request = urllib.request.Request("
                    f"{json.dumps(endpoint)}, data=payload, method={json.dumps(method)}"
                    "); "
                    "request.add_header('Content-Type', 'application/json'); "
                    "response = urllib.request.urlopen(request); "
                    "body = response.read().decode('utf-8'); "
                    "sys.stdout.write(body)"
                ),
            ],
            cwd=config.execution.cwd,
            env=_command_env(config),
        )


class HuggingFaceDeploymentProvider:
    descriptor = PluginDescriptor(
        kind="deployment_provider",
        name="builtin.huggingface",
        label="Hugging Face deployment provider",
        capabilities=["huggingface"],
    )
    provider_names = ("huggingface",)

    def build_command(
        self,
        config: OrchestrationConfig,
        manifest: InstanceManifest,
        registry: PluginRegistry,
    ) -> CommandSpec:
        options = dict(config.subsystem.provider_options)
        repo_id = options.get("repo_id")
        source_artifact = config.subsystem.source_artifact_ref
        if not repo_id or not source_artifact:
            raise ValueError("huggingface deploys require repo_id and source_artifact_ref")
        return CommandSpec(
            argv=["hf", "upload-large-folder", repo_id, source_artifact],
            cwd=config.execution.cwd,
            env=_command_env(config),
        )


class OllamaDeploymentProvider:
    descriptor = PluginDescriptor(
        kind="deployment_provider",
        name="builtin.ollama",
        label="Ollama deployment provider",
        capabilities=["ollama"],
    )
    provider_names = ("ollama",)

    def build_command(
        self,
        config: OrchestrationConfig,
        manifest: InstanceManifest,
        registry: PluginRegistry,
    ) -> CommandSpec:
        options = dict(config.subsystem.provider_options)
        model_name = options.get("model_name")
        modelfile = options.get("modelfile")
        if not model_name or not modelfile:
            raise ValueError("ollama deploys require model_name and modelfile")
        return CommandSpec(
            argv=["ollama", "create", model_name, "-f", str(modelfile)],
            cwd=config.execution.cwd,
            env=_command_env(config),
        )


class LmStudioDeploymentProvider:
    descriptor = PluginDescriptor(
        kind="deployment_provider",
        name="builtin.lmstudio",
        label="LM Studio deployment provider",
        capabilities=["lmstudio"],
    )
    provider_names = ("lmstudio",)

    def build_command(
        self,
        config: OrchestrationConfig,
        manifest: InstanceManifest,
        registry: PluginRegistry,
    ) -> CommandSpec:
        options = dict(config.subsystem.provider_options)
        import_command = options.get("command")
        if isinstance(import_command, list) and import_command:
            return CommandSpec(
                argv=[str(part) for part in import_command],
                cwd=config.execution.cwd,
                env=_command_env(config),
            )
        raise ValueError("lmstudio deploys require subsystem.provider_options.command for V1")


BUILTIN_INSTANCE_HANDLERS = cast(
    tuple[InstanceHandlerPlugin, ...],
    (
        PrepareInstanceHandler(),
        TrainingInstanceHandler(),
        EvaluationInstanceHandler(),
        InferenceInstanceHandler(),
        ReportInstanceHandler(),
        DeploymentInstanceHandler(),
    ),
)

BUILTIN_DEPLOYMENT_PROVIDERS = cast(
    tuple[DeploymentProviderPlugin, ...],
    (
        ApiCompatibleDeploymentProvider(),
        HuggingFaceDeploymentProvider(),
        OllamaDeploymentProvider(),
        LmStudioDeploymentProvider(),
    ),
)
