from __future__ import annotations

import importlib

from ai_factory.core.execution.base import UnsupportedInstanceTypeError
from ai_factory.core.plugins.base import DeploymentProviderPlugin, InstanceHandlerPlugin, PluginDescriptor
from ai_factory.core.plugins.builtins import BUILTIN_DEPLOYMENT_PROVIDERS, BUILTIN_INSTANCE_HANDLERS


class PluginRegistry:
    def __init__(self):
        self._instance_handlers: dict[str, InstanceHandlerPlugin] = {}
        self._deployment_providers: dict[str, DeploymentProviderPlugin] = {}
        self._descriptors: list[PluginDescriptor] = []

    def register_instance_handler(self, plugin: InstanceHandlerPlugin) -> None:
        for instance_type in plugin.instance_types:
            self._instance_handlers[instance_type] = plugin
        self._descriptors.append(plugin.descriptor)

    def register_deployment_provider(self, plugin: DeploymentProviderPlugin) -> None:
        for provider_name in plugin.provider_names:
            self._deployment_providers[provider_name] = plugin
        self._descriptors.append(plugin.descriptor)

    def load_module(self, module_name: str) -> None:
        module = importlib.import_module(module_name)
        register = getattr(module, "register_plugins", None)
        if not callable(register):
            raise TypeError(f"Plugin module {module_name} must expose register_plugins(registry)")
        register(self)

    def get_instance_handler(self, instance_type: str) -> InstanceHandlerPlugin:
        try:
            return self._instance_handlers[instance_type]
        except KeyError as exc:
            raise UnsupportedInstanceTypeError(f"Unsupported instance type: {instance_type}") from exc

    def get_deployment_provider(self, provider_name: str) -> DeploymentProviderPlugin:
        try:
            return self._deployment_providers[provider_name]
        except KeyError as exc:
            raise ValueError(f"Unsupported deployment provider: {provider_name}") from exc

    def list_plugins(self) -> list[PluginDescriptor]:
        return list(self._descriptors)


def build_default_plugin_registry(plugin_modules: tuple[str, ...] = ()) -> PluginRegistry:
    registry = PluginRegistry()
    for plugin in BUILTIN_INSTANCE_HANDLERS:
        registry.register_instance_handler(plugin)
    for plugin in BUILTIN_DEPLOYMENT_PROVIDERS:
        registry.register_deployment_provider(plugin)
    for module_name in plugin_modules:
        registry.load_module(module_name)
    return registry
