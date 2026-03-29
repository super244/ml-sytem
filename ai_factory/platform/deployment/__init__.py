"""Deployment capabilities for AI-Factory.

Multi-target model deployment pipeline supporting various platforms
and deployment scenarios.
"""

from .manager import DeploymentManager
from .targets import CustomAPITarget, EdgeDeviceTarget, HuggingFaceTarget, LMStudioTarget, OllamaTarget

__all__ = [
    "DeploymentManager",
    "HuggingFaceTarget",
    "OllamaTarget",
    "LMStudioTarget", 
    "CustomAPITarget",
    "EdgeDeviceTarget"
]
