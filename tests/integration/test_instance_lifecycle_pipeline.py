"""Integration test for instance lifecycle manifest creation."""

from __future__ import annotations

from pathlib import Path

from ai_factory.core.instances.creation import InstanceCreationService
from ai_factory.core.instances.store import FileInstanceStore


def _write(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")


def test_instance_lifecycle_pipeline_creates_linked_manifests(tmp_path: Path) -> None:
    training_profile = tmp_path / "training" / "configs" / "profiles" / "baseline_qlora.yaml"
    _write(training_profile, "run_name: baseline\ntraining:\n  artifacts_dir: artifacts\n")

    _write(
        tmp_path / "configs" / "train.yaml",
        "\n".join(
            [
                "instance:",
                "  type: train",
                "  name: integration-train",
                "subsystem:",
                "  config_ref: ../training/configs/profiles/baseline_qlora.yaml",
            ]
        ),
    )
    _write(
        tmp_path / "configs" / "eval.yaml",
        "\n".join(
            [
                "instance:",
                "  type: evaluate",
                "subsystem:",
                "  config_ref: ../training/configs/profiles/baseline_qlora.yaml",
            ]
        ),
    )
    _write(
        tmp_path / "configs" / "inference.yaml",
        "\n".join(
            [
                "instance:",
                "  type: inference",
                "subsystem:",
                "  model_variant: finetuned",
            ]
        ),
    )
    _write(
        tmp_path / "configs" / "deploy.yaml",
        "\n".join(
            [
                "instance:",
                "  type: deploy",
                "subsystem:",
                "  provider: huggingface",
            ]
        ),
    )

    store = FileInstanceStore(tmp_path / "artifacts")
    service = InstanceCreationService(store)

    train = service.create_instance(str(tmp_path / "configs" / "train.yaml"))
    evaluate = service.create_evaluation_instance(train.id, str(tmp_path / "configs" / "eval.yaml"))
    inference = service.create_inference_instance(train.id, str(tmp_path / "configs" / "inference.yaml"))
    deploy = service.create_deployment_instance(train.id, "ollama", str(tmp_path / "configs" / "deploy.yaml"))

    assert evaluate.parent_instance_id == train.id
    assert inference.parent_instance_id == train.id
    assert deploy.parent_instance_id == train.id

    assert store.config_snapshot_path(train.id).exists()
    assert store.config_snapshot_path(evaluate.id).exists()
    assert store.config_snapshot_path(inference.id).exists()
    assert store.config_snapshot_path(deploy.id).exists()

    deploy_snapshot = store.load_config_snapshot(deploy.id)
    assert deploy_snapshot["subsystem"]["provider"] == "ollama"
