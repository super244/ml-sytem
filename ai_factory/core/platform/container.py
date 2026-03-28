from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ai_factory.core.instances.manager import InstanceManager
from ai_factory.core.instances.store import FileInstanceStore
from ai_factory.core.orchestration.service import OrchestrationService
from ai_factory.core.orchestration.sqlite import SqliteControlPlane
from ai_factory.core.platform.settings import PlatformSettings, get_platform_settings


@dataclass
class PlatformContainer:
    settings: PlatformSettings
    store: FileInstanceStore
    control_plane: SqliteControlPlane
    orchestration: OrchestrationService
    manager: InstanceManager


def build_platform_container(
    *,
    repo_root: str | Path | None = None,
    artifacts_dir: str | Path | None = None,
) -> PlatformContainer:
    settings = get_platform_settings(repo_root=repo_root, artifacts_dir=artifacts_dir)
    store = FileInstanceStore(settings.artifacts_dir)
    control_plane = SqliteControlPlane(settings.control_db_path)
    orchestration = OrchestrationService(control_plane, settings)
    manager = InstanceManager(store, orchestration=orchestration, platform_settings=settings)
    return PlatformContainer(
        settings=settings,
        store=store,
        control_plane=control_plane,
        orchestration=orchestration,
        manager=manager,
    )
