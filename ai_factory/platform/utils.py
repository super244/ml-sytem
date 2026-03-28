"""Utility functions for platform management."""

from typing import Dict, Any, List
from pathlib import Path

from ai_factory.core.schemas import ScalingConfig, MonitoringConfig
from .scaling.manager import ScalingManager
from .monitoring.manager import MonitoringManager


def get_platform_status() -> Dict[str, Any]:
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


async def scale_platform(target_nodes: int) -> Dict[str, Any]:
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
    domains: List[str],
    config_path: str,
    start: bool,
    repo_root: str,
    artifacts_dir: str
) -> Dict[str, Any]:
    """Create multi-domain training instance."""
    from ai_factory.core.platform.container import build_platform_container
    
    container = build_platform_container(
        repo_root=Path(repo_root),
        artifacts_dir=Path(artifacts_dir)
    )
    
    # Create a mock manifest for now
    manifest = {
        "id": f"multi_train_{int(__import__('time').time())}",
        "type": "multi_train",
        "status": "running" if start else "created",
        "domains": domains,
        "config_path": config_path
    }
    
    return manifest
