from __future__ import annotations

from importlib import import_module

__all__ = [
    "DecisionResult",
    "EnvironmentSpec",
    "ExecutionHandle",
    "FileInstanceStore",
    "InstanceError",
    "InstanceManager",
    "InstanceManifest",
    "InstanceQueryService",
    "MetricPoint",
]

_EXPORTS = {
    "DecisionResult": "ai_factory.core.instances.models",
    "EnvironmentSpec": "ai_factory.core.instances.models",
    "ExecutionHandle": "ai_factory.core.instances.models",
    "FileInstanceStore": "ai_factory.core.instances.store",
    "InstanceError": "ai_factory.core.instances.models",
    "InstanceManager": "ai_factory.core.instances.manager",
    "InstanceManifest": "ai_factory.core.instances.models",
    "InstanceQueryService": "ai_factory.core.instances.queries",
    "MetricPoint": "ai_factory.core.instances.models",
}


def __getattr__(name: str) -> object:
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
