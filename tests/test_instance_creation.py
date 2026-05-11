from __future__ import annotations

from pathlib import Path

from ai_factory.core.instances.creation import InstanceCreationService
from ai_factory.core.instances.store import FileInstanceStore

REPO_ROOT = Path(__file__).resolve().parents[1]
_TRAIN_CFG = str(REPO_ROOT / "examples" / "orchestration" / "train.yaml")
_EVAL_CFG = str(REPO_ROOT / "examples" / "orchestration" / "eval.yaml")
_INFERENCE_CFG = str(REPO_ROOT / "examples" / "orchestration" / "inference.yaml")
_DEPLOY_CFG = str(REPO_ROOT / "examples" / "orchestration" / "deploy.yaml")


def test_instance_creation_service_creates_base_instance(tmp_path: Path) -> None:
    store = FileInstanceStore(artifacts_dir=tmp_path / "artifacts")
    service = InstanceCreationService(store)

    manifest = service.create_instance(_TRAIN_CFG)
    loaded = store.load(manifest.id)

    assert manifest.id.startswith("instance-")
    assert manifest.type == "train"
    assert loaded.id == manifest.id
    assert loaded.config_path == _TRAIN_CFG


def test_instance_creation_service_creates_child_instances(tmp_path: Path) -> None:
    store = FileInstanceStore(artifacts_dir=tmp_path / "artifacts")
    service = InstanceCreationService(store)
    parent = service.create_instance(_TRAIN_CFG)

    eval_manifest = service.create_evaluation_instance(parent.id, _EVAL_CFG)
    infer_manifest = service.create_inference_instance(parent.id, _INFERENCE_CFG)
    deploy_manifest = service.create_deployment_instance(parent.id, "ollama", _DEPLOY_CFG)

    assert eval_manifest.parent_instance_id == parent.id
    assert infer_manifest.parent_instance_id == parent.id
    assert deploy_manifest.parent_instance_id == parent.id
    assert store.config_snapshot_path(eval_manifest.id).exists()
    assert store.config_snapshot_path(infer_manifest.id).exists()
    assert store.config_snapshot_path(deploy_manifest.id).exists()

    deploy_snapshot = store.load_config_snapshot(deploy_manifest.id)
    assert deploy_snapshot["subsystem"]["provider"] == "ollama"
