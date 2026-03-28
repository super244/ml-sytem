from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ai_factory.core.control.service import FactoryControlService
from ai_factory.core.instances.manager import InstanceManager
from ai_factory.core.instances.store import FileInstanceStore
from ai_factory.core.orchestration.service import OrchestrationService
from ai_factory.core.orchestration.sqlite import SqliteControlPlane
from ai_factory.core.platform.settings import PlatformSettings, get_platform_settings
from ai_factory.core.plugins.registry import PluginRegistry, build_default_plugin_registry


@dataclass
class PlatformContainer:
    settings: PlatformSettings
    store: FileInstanceStore
    control_plane: SqliteControlPlane
    plugin_registry: PluginRegistry
    orchestration: OrchestrationService
    manager: InstanceManager
    control_service: FactoryControlService


def build_platform_container(
    *,
    repo_root: str | Path | None = None,
    artifacts_dir: str | Path | None = None,
) -> PlatformContainer:
    settings = get_platform_settings(repo_root=repo_root, artifacts_dir=artifacts_dir)
    store = FileInstanceStore(settings.artifacts_dir)
    control_plane = SqliteControlPlane(settings.control_db_path)
    plugin_registry = build_default_plugin_registry(settings.plugin_modules)
    orchestration = OrchestrationService(control_plane, settings)
    manager = InstanceManager(
        store,
        orchestration=orchestration,
        platform_settings=settings,
        plugin_registry=plugin_registry,
    )
    control_service = FactoryControlService(
        manager=manager,
        store=store,
        settings=settings,
        plugin_registry=plugin_registry,
    )
    return PlatformContainer(
        settings=settings,
        store=store,
        control_plane=control_plane,
        plugin_registry=plugin_registry,
        orchestration=orchestration,
        manager=manager,
        control_service=control_service,
    )
