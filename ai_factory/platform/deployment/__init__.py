"""Deployment capabilities for AI-Factory.

Multi-target model deployment pipeline supporting various platforms
and deployment scenarios.
"""

from .manager import DeploymentManager
from .models import DeploymentManifest, DeploymentRollbackReadiness, DeploymentRolloutStage, DeploymentVersionSummary
from .targets import CustomAPITarget, EdgeDeviceTarget, HuggingFaceTarget, LMStudioTarget, OllamaTarget

__all__ = [
    "DeploymentManager",
    "DeploymentManifest",
    "DeploymentVersionSummary",
    "DeploymentRolloutStage",
    "DeploymentRollbackReadiness",
    "HuggingFaceTarget",
    "OllamaTarget",
    "LMStudioTarget",
    "CustomAPITarget",
    "EdgeDeviceTarget",
]
