from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, Protocol

from pydantic import BaseModel, Field

from ai_factory.core.execution.base import CommandSpec
from ai_factory.core.instances.models import InstanceManifest

if TYPE_CHECKING:
    from ai_factory.core.config.schema import OrchestrationConfig
    from ai_factory.core.plugins.registry import PluginRegistry

PluginKind = Literal["instance_handler", "deployment_provider"]


class PluginDescriptor(BaseModel):
    kind: PluginKind
    name: str
    label: str
    version: str = "builtin"
    capabilities: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class InstanceHandlerPlugin(Protocol):
    descriptor: PluginDescriptor
    instance_types: tuple[str, ...]

    def build_command(
        self,
        config: OrchestrationConfig,
        manifest: InstanceManifest,
        registry: PluginRegistry,
    ) -> CommandSpec:
        raise NotImplementedError


class DeploymentProviderPlugin(Protocol):
    descriptor: PluginDescriptor
    provider_names: tuple[str, ...]

    def build_command(
        self,
        config: OrchestrationConfig,
        manifest: InstanceManifest,
        registry: PluginRegistry,
    ) -> CommandSpec:
        raise NotImplementedError
