from __future__ import annotations

from ai_factory.core.config.schema import OrchestrationConfig
from ai_factory.core.execution.base import CommandSpec
from ai_factory.core.instances.models import InstanceManifest
from ai_factory.core.plugins.registry import PluginRegistry, build_default_plugin_registry


class UnsupportedInstanceTypeError(Exception):
    pass


def build_command(
    config: OrchestrationConfig, manifest: InstanceManifest, *, plugin_registry: PluginRegistry | None = None
) -> CommandSpec:
    registry = plugin_registry or build_default_plugin_registry()
    handler = registry.get_instance_handler(manifest.type)
    command = handler.build_command(config, manifest, registry)
    if isinstance(command, CommandSpec):
        return command
    if hasattr(command, "model_dump"):
        return CommandSpec.model_validate(command.model_dump(mode="json"))
    return CommandSpec.model_validate(command)
