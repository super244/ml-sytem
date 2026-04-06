"""Instance creation and management functionality."""

from __future__ import annotations

import uuid
from typing import Any

from ai_factory.core.config.loader import load_orchestration_config
from ai_factory.core.instances.models import (
    DeploymentTarget,
    EnvironmentSpec,
    InstanceManifest,
    LifecycleProfile,
    UserLevel,
    utc_now_iso,
)
from ai_factory.core.instances.store import FileInstanceStore


class InstanceCreationService:
    """Service for creating new instances."""

    def __init__(self, store: FileInstanceStore) -> None:
        self.store = store

    def create_instance(
        self,
        config_path: str,
        *,
        start: bool = True,
        environment_override: EnvironmentSpec | None = None,
        name_override: str | None = None,
        user_level_override: UserLevel | None = None,
        lifecycle_override: LifecycleProfile | None = None,
        parent_instance_id: str | None = None,
        metadata_updates: dict[str, Any] | None = None,
    ) -> InstanceManifest:
        """Create a new instance - simplified version that delegates to store."""
        del start
        config = load_orchestration_config(config_path)
        instance_id = f"instance-{uuid.uuid4().hex[:12]}"
        manifest = InstanceManifest(
            id=instance_id,
            type=config.instance.type,
            name=name_override or f"instance-{instance_id[:8]}",
            status="pending",
            created_at=utc_now_iso(),
            updated_at=utc_now_iso(),
            config_path=config_path,
            environment=environment_override or config.instance.environment,
            user_level=user_level_override or config.experience.level,
            lifecycle=lifecycle_override or config.lifecycle,
            parent_instance_id=parent_instance_id or config.instance.parent_instance_id,
            artifact_refs={},
            metadata=metadata_updates or {},
        )
        self.store.create(manifest, config.model_dump(mode="json"))
        return manifest

    def create_evaluation_instance(
        self,
        parent_instance_id: str,
        config_path: str,
        *,
        start: bool = True,
    ) -> InstanceManifest:
        """Create an evaluation instance from a parent."""
        del start
        instance_id = f"evaluate-{uuid.uuid4().hex[:12]}"

        # Create a minimal manifest
        manifest = InstanceManifest(
            id=instance_id,
            type="evaluate",
            name=f"eval-{instance_id[:8]}",
            status="pending",
            created_at=utc_now_iso(),
            updated_at=utc_now_iso(),
            config_path=config_path,
            parent_instance_id=parent_instance_id,
            artifact_refs={},
        )

        config = load_orchestration_config(config_path)
        self.store.create(manifest, config.model_dump(mode="json"))

        return manifest

    def create_deployment_instance(
        self,
        parent_instance_id: str,
        target: DeploymentTarget,
        config_path: str,
        *,
        start: bool = True,
    ) -> InstanceManifest:
        """Create a deployment instance."""
        del start
        instance_id = f"deploy-{uuid.uuid4().hex[:12]}"

        # Create a minimal manifest
        manifest = InstanceManifest(
            id=instance_id,
            type="deploy",
            name=f"deploy-{instance_id[:8]}",
            status="pending",
            created_at=utc_now_iso(),
            updated_at=utc_now_iso(),
            config_path=config_path,
            parent_instance_id=parent_instance_id,
            artifact_refs={},
        )

        config = load_orchestration_config(config_path)
        config.subsystem.provider = target
        self.store.create(manifest, config.model_dump(mode="json"))

        return manifest

    def create_inference_instance(
        self,
        parent_instance_id: str,
        config_path: str,
        *,
        start: bool = True,
    ) -> InstanceManifest:
        """Create an inference instance."""
        del start
        instance_id = f"inference-{uuid.uuid4().hex[:12]}"

        # Create a minimal manifest
        manifest = InstanceManifest(
            id=instance_id,
            type="inference",
            name=f"infer-{instance_id[:8]}",
            status="pending",
            created_at=utc_now_iso(),
            updated_at=utc_now_iso(),
            config_path=config_path,
            parent_instance_id=parent_instance_id,
            artifact_refs={},
        )

        config = load_orchestration_config(config_path)
        self.store.create(manifest, config.model_dump(mode="json"))

        return manifest
