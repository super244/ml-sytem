"""Instance creation and management functionality."""

from __future__ import annotations

import uuid
from typing import Any

from ai_factory.core.config.loader import load_orchestration_config
from ai_factory.core.instances.models import InstanceManifest, utc_now_iso
from ai_factory.core.instances.store import FileInstanceStore
from ai_factory.core.io import write_json


class InstanceCreationService:
    """Service for creating new instances."""

    def __init__(self, store: FileInstanceStore):
        self.store = store

    def create_instance(
        self,
        config_path: str,
        *,
        start: bool = True,
        environment_override=None,
        name_override: str | None = None,
        user_level_override=None,
        lifecycle_override=None,
        parent_instance_id: str | None = None,
        metadata_updates: dict[str, Any] | None = None,
    ) -> InstanceManifest:
        """Create a new instance - simplified version that delegates to store."""
        # This is a simplified version - the actual creation logic should stay in the manager
        # to avoid duplication and ensure consistency
        instance_id = f"instance-{uuid.uuid4().hex[:12]}"

        # Create a minimal manifest that will be properly populated by the manager
        manifest = InstanceManifest(
            id=instance_id,
            type="unknown",  # Will be set by manager
            name=name_override or f"instance-{instance_id[:8]}",
            status="pending",
            created_at=utc_now_iso(),
            updated_at=utc_now_iso(),
            config_path=config_path,
            artifact_refs={},
            metadata=metadata_updates or {},
        )

        # Save the basic manifest
        self.store.save(manifest)

        return manifest

    def create_evaluation_instance(
        self,
        parent_instance_id: str,
        config_path: str,
        *,
        start: bool = True,
    ) -> InstanceManifest:
        """Create an evaluation instance from a parent."""
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

        # Save the manifest
        self.store.save(manifest)

        # Save config snapshot
        config = load_orchestration_config(config_path)
        write_json(self.store.config_snapshot_path(instance_id), config.model_dump(mode="json"))

        return manifest

    def create_deployment_instance(
        self,
        parent_instance_id: str,
        target: str,
        config_path: str,
        *,
        start: bool = True,
    ) -> InstanceManifest:
        """Create a deployment instance."""
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

        # Save the manifest
        self.store.save(manifest)

        # Save config snapshot
        config = load_orchestration_config(config_path)
        config.subsystem.provider = target
        write_json(self.store.config_snapshot_path(instance_id), config.model_dump(mode="json"))

        return manifest

    def create_inference_instance(
        self,
        parent_instance_id: str,
        config_path: str,
        *,
        start: bool = True,
    ) -> InstanceManifest:
        """Create an inference instance."""
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

        # Save the manifest
        self.store.save(manifest)

        # Save config snapshot
        config = load_orchestration_config(config_path)
        write_json(self.store.config_snapshot_path(instance_id), config.model_dump(mode="json"))

        return manifest
