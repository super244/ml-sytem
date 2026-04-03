"""Deployment targets for AI-Factory models."""

import logging
from abc import ABC, abstractmethod
from typing import Any

from ai_factory.core.schemas import DeploymentSpec, ModelArtifact

logger = logging.getLogger(__name__)


class DeploymentTarget(ABC):
    """Base class for deployment targets."""

    def __init__(self) -> None:
        self.name = self.__class__.__name__
        self.description = "Base deployment target"
        self.capabilities: list[str] = []

    @abstractmethod
    async def prepare_model(self, model: ModelArtifact, spec: DeploymentSpec) -> ModelArtifact:
        """Prepare model for deployment."""
        pass

    @abstractmethod
    async def deploy(self, model: ModelArtifact, spec: DeploymentSpec) -> dict[str, Any]:
        """Deploy model to target."""
        pass

    @abstractmethod
    async def get_deployment_status(self, deployment_id: str) -> dict[str, Any]:
        """Get deployment status."""
        pass

    @abstractmethod
    async def cancel_deployment(self, deployment_id: str) -> bool:
        """Cancel deployment."""
        pass

    @abstractmethod
    async def validate_spec(self, spec: DeploymentSpec) -> list[str]:
        """Validate deployment specification."""
        pass

    async def get_target_status(self) -> dict[str, Any]:
        """Get target status."""
        return {
            "name": self.name,
            "status": "degraded",
            "capabilities": self.capabilities,
            "errors": [
                "Legacy platform deployment targets are informational only. "
                "Use managed deployment instances for production publishing."
            ],
        }


class HuggingFaceTarget(DeploymentTarget):
    """HuggingFace Hub deployment target."""

    def __init__(self) -> None:
        super().__init__()
        self.name = "HuggingFace"
        self.description = "Deploy models to HuggingFace Hub"
        self.capabilities = ["public", "private", "model_cards", "versioning"]

    async def prepare_model(self, model: ModelArtifact, spec: DeploymentSpec) -> ModelArtifact:
        """Prepare model for HuggingFace deployment."""
        logger.info(f"Preparing {model.name} for HuggingFace deployment")
        return model

    async def deploy(self, model: ModelArtifact, spec: DeploymentSpec) -> dict[str, Any]:
        """Deploy model to HuggingFace Hub."""
        del spec
        raise RuntimeError(
            f"HuggingFaceTarget is not wired to a live publisher for model '{model.name}'. "
            "Use a managed deploy instance instead."
        )

    async def get_deployment_status(self, deployment_id: str) -> dict[str, Any]:
        """Get HuggingFace deployment status."""
        return {
            "deployment_id": deployment_id,
            "status": "degraded",
            "errors": ["Legacy HuggingFace deployment tracking is unavailable."],
        }

    async def cancel_deployment(self, deployment_id: str) -> bool:
        """Cancel HuggingFace deployment."""
        logger.info(f"Cancelling HuggingFace deployment {deployment_id}")
        return True

    async def validate_spec(self, spec: DeploymentSpec) -> list[str]:
        """Validate HuggingFace deployment spec."""
        errors = []
        if not spec.config.get("repository"):
            errors.append("HuggingFace deployment requires 'repository' in config")
        return errors


class OllamaTarget(DeploymentTarget):
    """Ollama deployment target."""

    def __init__(self) -> None:
        super().__init__()
        self.name = "Ollama"
        self.description = "Deploy models to Ollama local registry"
        self.capabilities = ["local", "gguf", "quantization"]

    async def prepare_model(self, model: ModelArtifact, spec: DeploymentSpec) -> ModelArtifact:
        """Prepare model for Ollama deployment."""
        logger.info(f"Preparing {model.name} for Ollama deployment")
        # Convert to GGUF format if needed
        return model

    async def deploy(self, model: ModelArtifact, spec: DeploymentSpec) -> dict[str, Any]:
        """Deploy model to Ollama."""
        del spec
        raise RuntimeError(
            f"OllamaTarget is not wired to a live publisher for model '{model.name}'. "
            "Use a managed deploy instance instead."
        )

    async def get_deployment_status(self, deployment_id: str) -> dict[str, Any]:
        """Get Ollama deployment status."""
        return {
            "deployment_id": deployment_id,
            "status": "degraded",
            "errors": ["Legacy Ollama deployment tracking is unavailable."],
        }

    async def cancel_deployment(self, deployment_id: str) -> bool:
        """Cancel Ollama deployment."""
        logger.info(f"Cancelling Ollama deployment {deployment_id}")
        return True

    async def validate_spec(self, spec: DeploymentSpec) -> list[str]:
        """Validate Ollama deployment spec."""
        return []  # Ollama has minimal requirements


class LMStudioTarget(DeploymentTarget):
    """LM Studio deployment target."""

    def __init__(self) -> None:
        super().__init__()
        self.name = "LM Studio"
        self.description = "Export models for LM Studio import"
        self.capabilities = ["gguf", "quantization", "local_import"]

    async def prepare_model(self, model: ModelArtifact, spec: DeploymentSpec) -> ModelArtifact:
        """Prepare model for LM Studio."""
        logger.info(f"Preparing {model.name} for LM Studio")
        return model

    async def deploy(self, model: ModelArtifact, spec: DeploymentSpec) -> dict[str, Any]:
        """Export model for LM Studio."""
        del spec
        raise RuntimeError(
            f"LMStudioTarget is not wired to a live exporter for model '{model.name}'. "
            "Use a managed deploy instance instead."
        )

    async def get_deployment_status(self, deployment_id: str) -> dict[str, Any]:
        """Get LM Studio deployment status."""
        return {
            "deployment_id": deployment_id,
            "status": "degraded",
            "errors": ["Legacy LM Studio deployment tracking is unavailable."],
        }

    async def cancel_deployment(self, deployment_id: str) -> bool:
        """Cancel LM Studio deployment."""
        logger.info(f"Cancelling LM Studio deployment {deployment_id}")
        return True

    async def validate_spec(self, spec: DeploymentSpec) -> list[str]:
        """Validate LM Studio deployment spec."""
        return []


class CustomAPITarget(DeploymentTarget):
    """Custom API deployment target."""

    def __init__(self) -> None:
        super().__init__()
        self.name = "Custom API"
        self.description = "Deploy models to custom API endpoints"
        self.capabilities = ["rest_api", "grpc", "websocket"]

    async def prepare_model(self, model: ModelArtifact, spec: DeploymentSpec) -> ModelArtifact:
        """Prepare model for custom API deployment."""
        logger.info(f"Preparing {model.name} for custom API deployment")
        return model

    async def deploy(self, model: ModelArtifact, spec: DeploymentSpec) -> dict[str, Any]:
        """Deploy model to custom API."""
        del spec
        raise RuntimeError(
            f"CustomAPITarget is not wired to a live publisher for model '{model.name}'. "
            "Use a managed deploy instance instead."
        )

    async def get_deployment_status(self, deployment_id: str) -> dict[str, Any]:
        """Get custom API deployment status."""
        return {
            "deployment_id": deployment_id,
            "status": "degraded",
            "errors": ["Legacy custom API deployment tracking is unavailable."],
        }

    async def cancel_deployment(self, deployment_id: str) -> bool:
        """Cancel custom API deployment."""
        logger.info(f"Cancelling custom API deployment {deployment_id}")
        return True

    async def validate_spec(self, spec: DeploymentSpec) -> list[str]:
        """Validate custom API deployment spec."""
        errors = []
        if not spec.config.get("endpoint"):
            errors.append("Custom API deployment requires 'endpoint' in config")
        return errors


class EdgeDeviceTarget(DeploymentTarget):
    """Edge device deployment target."""

    def __init__(self) -> None:
        super().__init__()
        self.name = "Edge Device"
        self.description = "Deploy models to edge devices"
        self.capabilities = ["mobile", "iot", "embedded", "quantization"]

    async def prepare_model(self, model: ModelArtifact, spec: DeploymentSpec) -> ModelArtifact:
        """Prepare model for edge deployment."""
        logger.info(f"Preparing {model.name} for edge deployment")
        # Quantize and optimize for edge
        return model

    async def deploy(self, model: ModelArtifact, spec: DeploymentSpec) -> dict[str, Any]:
        """Deploy model to edge device."""
        del spec
        raise RuntimeError(
            f"EdgeDeviceTarget is not wired to a live publisher for model '{model.name}'. "
            "Use a managed deploy instance instead."
        )

    async def get_deployment_status(self, deployment_id: str) -> dict[str, Any]:
        """Get edge deployment status."""
        return {
            "deployment_id": deployment_id,
            "status": "degraded",
            "errors": ["Legacy edge deployment tracking is unavailable."],
        }

    async def cancel_deployment(self, deployment_id: str) -> bool:
        """Cancel edge deployment."""
        logger.info(f"Cancelling edge deployment {deployment_id}")
        return True

    async def validate_spec(self, spec: DeploymentSpec) -> list[str]:
        """Validate edge deployment spec."""
        errors = []
        if not spec.config.get("device_id"):
            errors.append("Edge deployment requires 'device_id' in config")
        return errors
