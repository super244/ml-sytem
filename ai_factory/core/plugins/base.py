from __future__ import annotations

from typing import Any, Literal, Protocol

from pydantic import BaseModel, Field

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

    def build_command(self, config, manifest, registry) -> object:
        raise NotImplementedError


class DeploymentProviderPlugin(Protocol):
    descriptor: PluginDescriptor
    provider_names: tuple[str, ...]

    def build_command(self, config, manifest, registry) -> object:
        raise NotImplementedError
