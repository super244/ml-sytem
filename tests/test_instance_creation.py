from __future__ import annotations

from pathlib import Path

from ai_factory.core.instances.creation import InstanceCreationService
from ai_factory.core.instances.store import FileInstanceStore


def test_instance_creation_service_creates_base_instance(tmp_path: Path) -> None:
    store = FileInstanceStore(artifacts_dir=tmp_path / "artifacts")
    service = InstanceCreationService(store)

    manifest = service.create_instance("configs/train.yaml")
    loaded = store.load(manifest.id)

    assert manifest.id.startswith("instance-")
    assert manifest.type == "train"
    assert loaded.id == manifest.id
    assert loaded.config_path == "configs/train.yaml"


def test_instance_creation_service_creates_child_instances(tmp_path: Path) -> None:
    store = FileInstanceStore(artifacts_dir=tmp_path / "artifacts")
    service = InstanceCreationService(store)
    parent = service.create_instance("configs/train.yaml")

    eval_manifest = service.create_evaluation_instance(parent.id, "configs/eval.yaml")
    infer_manifest = service.create_inference_instance(parent.id, "configs/inference.yaml")
    deploy_manifest = service.create_deployment_instance(parent.id, "ollama", "configs/deploy.yaml")

    assert eval_manifest.parent_instance_id == parent.id
    assert infer_manifest.parent_instance_id == parent.id
    assert deploy_manifest.parent_instance_id == parent.id
    assert store.config_snapshot_path(eval_manifest.id).exists()
    assert store.config_snapshot_path(infer_manifest.id).exists()
    assert store.config_snapshot_path(deploy_manifest.id).exists()

    deploy_snapshot = store.load_config_snapshot(deploy_manifest.id)
    assert deploy_snapshot["subsystem"]["provider"] == "ollama"
