from __future__ import annotations

from pathlib import Path

from ai_factory.core.instances.creation import InstanceCreationService
from ai_factory.core.instances.models import EnvironmentSpec
from ai_factory.core.instances.store import FileInstanceStore


def _write_orchestration_config(tmp_path: Path, filename: str = "finetune.yaml") -> Path:
    training_profile = tmp_path / "training" / "configs" / "profiles" / "baseline_qlora.yaml"
    training_profile.parent.mkdir(parents=True, exist_ok=True)
    training_profile.write_text("run_name: baseline\n", encoding="utf-8")

    config_path = tmp_path / "configs" / filename
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        "\n".join(
            [
                "instance:",
                "  type: finetune",
                "  name: baseline",
                "  environment: local",
                "experience:",
                "  level: hobbyist",
                "subsystem:",
                "  config_ref: ../training/configs/profiles/baseline_qlora.yaml",
            ]
        ),
        encoding="utf-8",
    )
    return config_path


def test_create_instance_respects_overrides_and_persists_manifest(tmp_path: Path) -> None:
    config_path = _write_orchestration_config(tmp_path)
    store = FileInstanceStore(tmp_path / "artifacts")
    service = InstanceCreationService(store)

    manifest = service.create_instance(
        str(config_path),
        environment_override=EnvironmentSpec(kind="cloud", profile_name="primary"),
        name_override="custom-name",
        metadata_updates={"source": "test"},
    )

    loaded = store.load(manifest.id)

    assert manifest.id.startswith("instance-")
    assert loaded.name == "custom-name"
    assert loaded.environment.kind == "cloud"
    assert loaded.metadata["source"] == "test"
    assert loaded.type == "finetune"


def test_create_evaluation_instance_writes_snapshot(tmp_path: Path) -> None:
    config_path = _write_orchestration_config(tmp_path, "evaluate.yaml")
    store = FileInstanceStore(tmp_path / "artifacts")
    service = InstanceCreationService(store)

    manifest = service.create_evaluation_instance("finetune-123", str(config_path))
    snapshot = store.load_config_snapshot(manifest.id)

    assert manifest.id.startswith("evaluate-")
    assert manifest.parent_instance_id == "finetune-123"
    assert snapshot["instance"]["type"] == "finetune"


def test_create_deployment_instance_sets_provider(tmp_path: Path) -> None:
    config_path = _write_orchestration_config(tmp_path, "deploy.yaml")
    store = FileInstanceStore(tmp_path / "artifacts")
    service = InstanceCreationService(store)

    manifest = service.create_deployment_instance("evaluate-123", "ollama", str(config_path))
    snapshot = store.load_config_snapshot(manifest.id)

    assert manifest.id.startswith("deploy-")
    assert manifest.parent_instance_id == "evaluate-123"
    assert snapshot["subsystem"]["provider"] == "ollama"


def test_create_inference_instance_writes_snapshot(tmp_path: Path) -> None:
    config_path = _write_orchestration_config(tmp_path, "inference.yaml")
    store = FileInstanceStore(tmp_path / "artifacts")
    service = InstanceCreationService(store)

    manifest = service.create_inference_instance("deploy-123", str(config_path))
    snapshot = store.load_config_snapshot(manifest.id)

    assert manifest.id.startswith("inference-")
    assert manifest.parent_instance_id == "deploy-123"
    assert snapshot["instance"]["type"] == "finetune"
