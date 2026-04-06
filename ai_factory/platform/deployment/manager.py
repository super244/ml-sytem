"""Deployment manager for multi-target model deployment."""

import asyncio
import logging
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from ai_factory.core.schemas import DeploymentSpec, ModelArtifact

from .models import (
    DeploymentManifest,
    DeploymentRollbackReadiness,
    DeploymentVersionSummary,
    assess_rollback_readiness,
    build_deployment_manifest,
    build_rollout_stages,
    summarize_model_version,
    validate_rollout_configuration,
)
from .targets import (
    CustomAPITarget,
    DeploymentTarget,
    EdgeDeviceTarget,
    HuggingFaceTarget,
    LMStudioTarget,
    OllamaTarget,
)

logger = logging.getLogger(__name__)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class DeploymentStatus(Enum):
    PENDING = "pending"
    BUILDING = "building"
    UPLOADING = "uploading"
    ROLLING_OUT = "rolling_out"
    PAUSED = "paused"
    DEPLOYED = "deployed"
    ROLLED_BACK = "rolled_back"
    FAILED = "failed"
    CANCELLED = "cancelled"


class DeploymentManager:
    """Manages deployment of models to various targets."""

    def __init__(self, repo_root: Path, *, targets: dict[str, DeploymentTarget] | None = None):
        self.repo_root = repo_root
        self.targets = targets or {
            "huggingface": HuggingFaceTarget(),
            "ollama": OllamaTarget(),
            "lmstudio": LMStudioTarget(),
            "custom_api": CustomAPITarget(),
            "edge_device": EdgeDeviceTarget(),
        }
        self._active_deployments: dict[str, dict[str, Any]] = {}

    def summarize_deployment_version(
        self, model_artifact: ModelArtifact, deployment_spec: DeploymentSpec | None = None
    ) -> DeploymentVersionSummary:
        return summarize_model_version(model_artifact, deployment_spec)

    def build_deployment_manifest(
        self,
        deployment_id: str,
        model_artifact: ModelArtifact,
        deployment_spec: DeploymentSpec,
        target_name: str,
        target_capabilities: list[str],
        target_supports_rollback: bool,
    ) -> DeploymentManifest:
        return build_deployment_manifest(
            deployment_id=deployment_id,
            model=model_artifact,
            spec=deployment_spec,
            target_name=target_name,
            target_capabilities=target_capabilities,
            target_supports_rollback=target_supports_rollback,
        )

    def assess_rollback_readiness(
        self,
        model_artifact: ModelArtifact,
        deployment_spec: DeploymentSpec,
        *,
        rollout_stages: list[Any] | None = None,
        target_supports_rollback: bool = False,
    ) -> DeploymentRollbackReadiness:
        stages = rollout_stages or build_rollout_stages(deployment_spec)
        return assess_rollback_readiness(
            model_artifact,
            deployment_spec,
            rollout_stages=stages,
            target_supports_rollback=target_supports_rollback,
        )

    def _deployment_id(self, model_artifact: ModelArtifact, deployment_spec: DeploymentSpec) -> str:
        return f"deploy_{model_artifact.name}_{deployment_spec.target}_{int(asyncio.get_event_loop().time())}"

    def _record_status(self, deployment: dict[str, Any], status: DeploymentStatus, message: str) -> None:
        deployment["status"] = status
        deployment["updated_at"] = _utc_now()
        history = list(deployment.get("status_history", []))
        history.append({"status": status.value, "timestamp": deployment["updated_at"], "message": message})
        deployment["status_history"] = history

    def _deployment_snapshot(self, deployment_id: str, deployment: dict[str, Any]) -> dict[str, Any]:
        snapshot = {
            "id": deployment_id,
            **deployment,
        }
        status = snapshot.get("status")
        if isinstance(status, DeploymentStatus):
            snapshot["status"] = status.value
        manifest = snapshot.get("manifest")
        if isinstance(manifest, dict):
            snapshot["manifest_summary"] = DeploymentManifest.model_validate(manifest).summary()
        return snapshot

    async def deploy_model(self, model_artifact: ModelArtifact, deployment_spec: DeploymentSpec) -> str:
        """Deploy a model to the specified target."""
        deployment_id = self._deployment_id(model_artifact, deployment_spec)

        target_handler = self.targets.get(deployment_spec.target)
        if not target_handler:
            raise ValueError(f"Unsupported deployment target: {deployment_spec.target}")

        manifest = self.build_deployment_manifest(
            deployment_id,
            model_artifact,
            deployment_spec,
            target_handler.name,
            target_handler.capabilities,
            getattr(target_handler, "supports_rollback", False),
        )
        deployment: dict[str, Any] = {
            "deployment_id": deployment_id,
            "status": DeploymentStatus.PENDING,
            "model": model_artifact.name,
            "target": deployment_spec.target,
            "target_name": target_handler.name,
            "started_at": _utc_now(),
            "spec": deployment_spec.model_dump(mode="json"),
            "manifest": manifest.model_dump(mode="json"),
            "manifest_summary": manifest.summary(),
            "version_summary": manifest.version_summary.model_dump(mode="json"),
            "rollout": [stage.model_dump(mode="json") for stage in manifest.rollout_stages],
            "rollback": manifest.rollback.model_dump(mode="json"),
            "target_capabilities": list(target_handler.capabilities),
            "status_history": [
                {
                    "status": DeploymentStatus.PENDING.value,
                    "timestamp": _utc_now(),
                    "message": "Deployment registered.",
                }
            ],
        }
        self._active_deployments[deployment_id] = deployment

        try:
            validation_errors = await self.validate_deployment_spec(deployment_spec)
            if validation_errors:
                raise ValueError("; ".join(validation_errors))

            self._record_status(deployment, DeploymentStatus.BUILDING, "Preparing model for deployment.")
            prepared_model = await target_handler.prepare_model(model_artifact, deployment_spec)

            self._record_status(deployment, DeploymentStatus.UPLOADING, "Uploading prepared model to target.")
            deployment_result = await target_handler.deploy(prepared_model, deployment_spec)

            self._record_status(deployment, DeploymentStatus.DEPLOYED, "Deployment completed successfully.")
            deployment["result"] = deployment_result
            deployment["completed_at"] = _utc_now()
            deployment["rollback"]["summary"]["last_known_result"] = deployment_result
            deployment["manifest"]["status"] = DeploymentStatus.DEPLOYED.value
            deployment["manifest"]["result"] = deployment_result

            logger.info("Successfully deployed %s to %s", model_artifact.name, deployment_spec.target)
            return deployment_id

        except Exception as exc:
            self._record_status(deployment, DeploymentStatus.FAILED, f"Deployment failed: {exc}")
            deployment["error"] = str(exc)
            deployment["failed_at"] = _utc_now()
            logger.error("Failed to deploy %s to %s: %s", model_artifact.name, deployment_spec.target, exc)
            raise

    async def get_deployment_status(self, deployment_id: str) -> dict[str, Any]:
        """Get status of a deployment."""
        if deployment_id not in self._active_deployments:
            raise ValueError(f"Deployment {deployment_id} not found")

        deployment = self._active_deployments[deployment_id]
        snapshot = self._deployment_snapshot(deployment_id, deployment)

        if deployment["status"] == DeploymentStatus.DEPLOYED:
            target_handler = self.targets[deployment["target"]]
            live_status = await target_handler.get_deployment_status(deployment_id)
            deployment["live_status"] = live_status
            snapshot["live_status"] = live_status

        return snapshot

    async def cancel_deployment(self, deployment_id: str) -> bool:
        """Cancel an active deployment."""
        if deployment_id not in self._active_deployments:
            raise ValueError(f"Deployment {deployment_id} not found")

        deployment = self._active_deployments[deployment_id]

        if deployment["status"] in [DeploymentStatus.DEPLOYED, DeploymentStatus.FAILED, DeploymentStatus.CANCELLED]:
            return False

        target_handler = self.targets[deployment["target"]]
        cancelled = await target_handler.cancel_deployment(deployment_id)

        if cancelled:
            self._record_status(deployment, DeploymentStatus.CANCELLED, "Deployment cancelled.")
            deployment["cancelled_at"] = _utc_now()
            deployment.setdefault("manifest", {})["status"] = DeploymentStatus.CANCELLED.value

        return cancelled

    async def list_deployments(
        self, target: str | None = None, status: DeploymentStatus | None = None
    ) -> list[dict[str, Any]]:
        """List deployments with optional filtering."""
        deployments = []

        for deployment_id, deployment in self._active_deployments.items():
            if target and deployment["target"] != target:
                continue
            if status and deployment["status"] != status:
                continue

            deployments.append(self._deployment_snapshot(deployment_id, deployment))

        return deployments

    async def get_available_targets(self) -> list[dict[str, Any]]:
        """Get list of available deployment targets."""
        targets = []

        for target_id, target_handler in self.targets.items():
            target_info = {
                "id": target_id,
                "name": target_handler.name,
                "description": target_handler.description,
                "capabilities": target_handler.capabilities,
                "status": await target_handler.get_target_status(),
            }
            targets.append(target_info)

        return targets

    async def validate_deployment_spec(self, deployment_spec: DeploymentSpec) -> list[str]:
        """Validate a deployment specification."""
        target_handler = self.targets.get(deployment_spec.target)
        if not target_handler:
            return [f"Unsupported deployment target: {deployment_spec.target}"]

        errors = list(await target_handler.validate_spec(deployment_spec))
        errors.extend(validate_rollout_configuration(deployment_spec))
        return errors

    def get_deployment_manifest(self, deployment_id: str) -> DeploymentManifest:
        if deployment_id not in self._active_deployments:
            raise ValueError(f"Deployment {deployment_id} not found")
        deployment = self._active_deployments[deployment_id]
        manifest = deployment.get("manifest")
        if not isinstance(manifest, dict):
            raise ValueError(f"Deployment {deployment_id} does not include a manifest")
        return DeploymentManifest.model_validate(manifest)
