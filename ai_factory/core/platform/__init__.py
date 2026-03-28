from __future__ import annotations

from importlib import import_module

__all__ = [
    "PlatformContainer",
    "PlatformSettings",
    "build_platform_container",
    "get_platform_settings",
]


def __getattr__(name: str) -> object:
    if name in {"PlatformSettings", "get_platform_settings"}:
        module = import_module("ai_factory.core.platform.settings")
        value = getattr(module, name)
        globals()[name] = value
        return value
    if name in {"PlatformContainer", "build_platform_container"}:
        module = import_module("ai_factory.core.platform.container")
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(name)
