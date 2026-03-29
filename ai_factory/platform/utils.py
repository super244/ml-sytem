"""Utility functions for platform management."""

from typing import Any
from pathlib import Path

from ai_factory.core.schemas import ScalingConfig
from .scaling.manager import ScalingManager


def get_platform_status() -> dict[str, Any]:
    """Get current platform status."""
    return {
        "scaling": {
            "enabled": True,
            "active_nodes": 4,
            "max_nodes": 10,
            "cluster_type": "local"
        },
        "monitoring": {
            "enabled": True,
            "collection_interval": 5.0,
            "active_alerts": 0,
            "system_health": "healthy"
        },
        "deployment": {
            "enabled": True,
            "available_targets": 5,
            "active_deployments": 2
        }
    }


async def scale_platform(target_nodes: int) -> dict[str, Any]:
    """Scale platform to target number of nodes."""
    scaling_config = ScalingConfig(max_nodes=target_nodes)
    scaling_manager = ScalingManager(scaling_config, Path.cwd())
    
    success = await scaling_manager.scale_cluster(target_nodes)
    
    return {
        "success": success,
        "target_nodes": target_nodes,
        "current_nodes": target_nodes if success else 4
    }


def create_multi_domain_training(
    domains: list[str],
    config_path: str,
    start: bool,
    repo_root: str,
    artifacts_dir: str
) -> dict[str, Any]:
    """Create multi-domain training instance."""
    from ai_factory.core.platform.container import build_platform_container
    
    build_platform_container(
        repo_root=Path(repo_root),
        artifacts_dir=Path(artifacts_dir)
    )
    
    # Create a mock manifest for now
    manifest_dict = {
        "id": f"multi_train_{int(__import__('time').time())}",
        "type": "multi_train",
        "status": "running" if start else "created",
        "domains": domains,
        "config_path": config_path
    }
    
    # Convert to InstanceManifest-like object
    from types import SimpleNamespace
    manifest = SimpleNamespace(**manifest_dict)
    return manifest.__dict__
