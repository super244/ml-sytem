"""Utility functions for platform management."""

import asyncio
from collections import Counter
from collections.abc import Coroutine
from pathlib import Path
from typing import Any, TypeVar

from ai_factory.core.instances.models import InstanceManifest
from ai_factory.core.platform.container import PlatformContainer, build_platform_container
from ai_factory.core.schemas import ScalingConfig

from .scaling.manager import ScalingManager

T = TypeVar("T")


def _build_container(
    repo_root: str | Path | None = None,
    artifacts_dir: str | Path | None = None,
) -> PlatformContainer:
    return build_platform_container(
        repo_root=Path(repo_root) if repo_root is not None else None,
        artifacts_dir=Path(artifacts_dir) if artifacts_dir is not None else None,
    )


def _run_async(coro: Coroutine[Any, Any, T]) -> T:
    return asyncio.run(coro)


def get_platform_status(
    *,
    repo_root: str | Path | None = None,
    artifacts_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Get live platform status from the shared control plane."""
    container = _build_container(repo_root=repo_root, artifacts_dir=artifacts_dir)
    instances = container.control_service.list_instances()
    status_counts = Counter(instance.status for instance in instances)
    type_counts = Counter(instance.type for instance in instances)
    orchestration = container.control_service.monitoring_summary()

    return {
        "foundation": container.control_service.describe_foundation().model_dump(mode="json"),
        "instances": {
            "total": len(instances),
            "status_counts": dict(status_counts),
            "type_counts": dict(type_counts),
        },
        "orchestration": orchestration,
        "paths": {
            "repo_root": str(container.settings.repo_root),
            "artifacts_dir": str(container.settings.artifacts_dir),
            "control_db_path": str(container.settings.control_db_path),
        },
    }


def scale_platform(
    target_nodes: int,
    *,
    repo_root: str | Path | None = None,
    artifacts_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Scale the cluster and report the observed cluster metrics."""
    container = _build_container(repo_root=repo_root, artifacts_dir=artifacts_dir)
    scaling_config = ScalingConfig(max_nodes=max(target_nodes, 1))
    scaling_manager = ScalingManager(scaling_config, container.settings.repo_root)

    success = _run_async(scaling_manager.scale_cluster(target_nodes))
    cluster_metrics = _run_async(scaling_manager.get_cluster_metrics())

    return {
        "success": success,
        "target_nodes": target_nodes,
        "cluster_metrics": cluster_metrics,
        "repo_root": str(container.settings.repo_root),
    }


def create_multi_domain_training(
    domains: list[str],
    config_path: str,
    start: bool,
    repo_root: str | Path,
    artifacts_dir: str | Path,
) -> InstanceManifest:
    """Create a real training instance annotated for multi-domain orchestration."""
    container = _build_container(repo_root=repo_root, artifacts_dir=artifacts_dir)
    resolved_config = Path(config_path)
    if not resolved_config.is_absolute():
        resolved_config = container.settings.repo_root / resolved_config
    if not resolved_config.exists():
        raise FileNotFoundError(f"Multi-domain config was not found: {resolved_config}")

    return container.control_service.create_instance(
        str(resolved_config),
        start=start,
        metadata_updates={
            "domains": domains,
            "multi_domain": True,
            "requested_via": "platform.multi_train",
        },
    )
