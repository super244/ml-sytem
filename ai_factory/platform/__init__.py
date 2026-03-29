"""Platform layer for AI-Factory.

This package provides platform-level capabilities for scaling,
monitoring, and deployment of AI-Factory workloads.

Key components:
- scaling: Distributed training and resource management
- monitoring: Real-time metrics collection and alerting  
- deployment: Multi-target model deployment pipeline
"""

from .deployment import DeploymentManager
from .monitoring import MonitoringManager
from .scaling import ScalingManager
from .utils import create_multi_domain_training, get_platform_status, scale_platform

__all__ = [
    "ScalingManager",
    "MonitoringManager", 
    "DeploymentManager",
    "get_platform_status",
    "scale_platform",
    "create_multi_domain_training"
]
