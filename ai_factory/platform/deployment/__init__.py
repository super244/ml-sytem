"""Deployment capabilities for AI-Factory.

Multi-target model deployment pipeline supporting various platforms
and deployment scenarios.
"""

from .manager import DeploymentManager
from .targets import (
    HuggingFaceTarget,
    OllamaTarget, 
    LMStudioTarget,
    CustomAPITarget,
    EdgeDeviceTarget
)

__all__ = [
    "DeploymentManager",
    "HuggingFaceTarget",
    "OllamaTarget",
    "LMStudioTarget", 
    "CustomAPITarget",
    "EdgeDeviceTarget"
]
