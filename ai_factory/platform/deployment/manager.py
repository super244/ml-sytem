"""Deployment manager for multi-target model deployment."""

from typing import Dict, List, Any, Optional
from pathlib import Path
import asyncio
import logging
from enum import Enum

from ai_factory.core.schemas import DeploymentSpec, DeploymentTarget, ModelArtifact
from .targets import (
    HuggingFaceTarget,
    OllamaTarget,
    LMStudioTarget,
    CustomAPITarget,
    EdgeDeviceTarget
)


logger = logging.getLogger(__name__)


class DeploymentStatus(Enum):
    PENDING = "pending"
    BUILDING = "building"
    UPLOADING = "uploading"
    DEPLOYED = "deployed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class DeploymentManager:
    """Manages deployment of models to various targets."""
    
    def __init__(self, repo_root: Path):
        self.repo_root = repo_root
        self.targets = {
            "huggingface": HuggingFaceTarget(),
            "ollama": OllamaTarget(),
            "lmstudio": LMStudioTarget(),
            "custom_api": CustomAPITarget(),
            "edge_device": EdgeDeviceTarget()
        }
        self._active_deployments: Dict[str, Dict[str, Any]] = {}
    
    async def deploy_model(
        self, 
        model_artifact: ModelArtifact, 
        deployment_spec: DeploymentSpec
    ) -> str:
        """Deploy a model to the specified target."""
        deployment_id = f"deploy_{model_artifact.name}_{deployment_spec.target}_{int(asyncio.get_event_loop().time())}"
        
        try:
            # Record deployment start
            self._active_deployments[deployment_id] = {
                "status": DeploymentStatus.PENDING,
                "model": model_artifact.name,
                "target": deployment_spec.target,
                "started_at": asyncio.get_event_loop().time(),
                "spec": deployment_spec
            }
            
            # Get target handler
            target_handler = self.targets.get(deployment_spec.target)
            if not target_handler:
                raise ValueError(f"Unsupported deployment target: {deployment_spec.target}")
            
            # Update status to building
            self._active_deployments[deployment_id]["status"] = DeploymentStatus.BUILDING
            
            # Prepare model for deployment
            prepared_model = await target_handler.prepare_model(model_artifact, deployment_spec)
            
            # Update status to uploading
            self._active_deployments[deployment_id]["status"] = DeploymentStatus.UPLOADING
            
            # Deploy to target
            deployment_result = await target_handler.deploy(prepared_model, deployment_spec)
            
            # Update status to deployed
            self._active_deployments[deployment_id]["status"] = DeploymentStatus.DEPLOYED
            self._active_deployments[deployment_id]["result"] = deployment_result
            self._active_deployments[deployment_id]["completed_at"] = asyncio.get_event_loop().time()
            
            logger.info(f"Successfully deployed {model_artifact.name} to {deployment_spec.target}")
            return deployment_id
            
        except Exception as e:
            self._active_deployments[deployment_id]["status"] = DeploymentStatus.FAILED
            self._active_deployments[deployment_id]["error"] = str(e)
            self._active_deployments[deployment_id]["failed_at"] = asyncio.get_event_loop().time()
            logger.error(f"Failed to deploy {model_artifact.name} to {deployment_spec.target}: {e}")
            raise
    
    async def get_deployment_status(self, deployment_id: str) -> Dict[str, Any]:
        """Get status of a deployment."""
        if deployment_id not in self._active_deployments:
            raise ValueError(f"Deployment {deployment_id} not found")
        
        deployment = self._active_deployments[deployment_id]
        
        # If deployed, get live status from target
        if deployment["status"] == DeploymentStatus.DEPLOYED:
            target_handler = self.targets[deployment["target"]]
            live_status = await target_handler.get_deployment_status(deployment_id)
            deployment["live_status"] = live_status
        
        return deployment
    
    async def cancel_deployment(self, deployment_id: str) -> bool:
        """Cancel an active deployment."""
        if deployment_id not in self._active_deployments:
            raise ValueError(f"Deployment {deployment_id} not found")
        
        deployment = self._active_deployments[deployment_id]
        
        if deployment["status"] in [DeploymentStatus.DEPLOYED, DeploymentStatus.FAILED, DeploymentStatus.CANCELLED]:
            return False  # Cannot cancel completed deployments
        
        # Cancel the deployment
        target_handler = self.targets[deployment["target"]]
        cancelled = await target_handler.cancel_deployment(deployment_id)
        
        if cancelled:
            deployment["status"] = DeploymentStatus.CANCELLED
            deployment["cancelled_at"] = asyncio.get_event_loop().time()
        
        return cancelled
    
    async def list_deployments(
        self, 
        target: Optional[str] = None, 
        status: Optional[DeploymentStatus] = None
    ) -> List[Dict[str, Any]]:
        """List deployments with optional filtering."""
        deployments = []
        
        for deployment_id, deployment in self._active_deployments.items():
            if target and deployment["target"] != target:
                continue
            if status and deployment["status"] != status:
                continue
            
            deployments.append({
                "id": deployment_id,
                **deployment
            })
        
        return deployments
    
    async def get_available_targets(self) -> List[Dict[str, Any]]:
        """Get list of available deployment targets."""
        targets = []
        
        for target_id, target_handler in self.targets.items():
            target_info = {
                "id": target_id,
                "name": target_handler.name,
                "description": target_handler.description,
                "capabilities": target_handler.capabilities,
                "status": await target_handler.get_target_status()
            }
            targets.append(target_info)
        
        return targets
    
    async def validate_deployment_spec(self, deployment_spec: DeploymentSpec) -> List[str]:
        """Validate a deployment specification."""
        target_handler = self.targets.get(deployment_spec.target)
        if not target_handler:
            return [f"Unsupported deployment target: {deployment_spec.target}"]
        
        return await target_handler.validate_spec(deployment_spec)
