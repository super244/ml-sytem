from __future__ import annotations

from pathlib import Path

from ai_factory.core.instances.creation import InstanceCreationService
from ai_factory.core.instances.store import FileInstanceStore


def _write(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body)


def test_instance_creation_service_writes_config_snapshots(tmp_path: Path) -> None:
    training_profile = tmp_path / "training" / "configs" / "profiles" / "baseline_qlora.yaml"
    _write(training_profile, "run_name: baseline\ntraining:\n  artifacts_dir: artifacts\n")
    eval_config = tmp_path / "evaluation" / "configs" / "base_vs_finetuned.yaml"
    _write(eval_config, "models:\n  primary_model: base\n")
    processing_config = tmp_path / "data" / "configs" / "processing.yaml"
    _write(processing_config, "sources: []\n")
    deploy_config = tmp_path / "configs" / "deploy.yaml"
    _write(
        deploy_config,
        "\n".join(
            [
                "instance:",
                "  type: deploy",
                "subsystem:",
                "  provider: huggingface",
            ]
        ),
    )
    finetune_config = tmp_path / "configs" / "finetune.yaml"
    _write(
        finetune_config,
        "\n".join(
            [
                "instance:",
                "  type: finetune",
                "  name: demo-finetune",
                "subsystem:",
                "  config_ref: ../training/configs/profiles/baseline_qlora.yaml",
            ]
        ),
    )
    eval_template = tmp_path / "configs" / "eval.yaml"
    _write(
        eval_template,
        "\n".join(
            [
                "instance:",
                "  type: evaluate",
                "subsystem:",
                "  config_ref: ../evaluation/configs/base_vs_finetuned.yaml",
            ]
        ),
    )
    inference_template = tmp_path / "configs" / "inference.yaml"
    _write(
        inference_template,
        "\n".join(
            [
                "instance:",
                "  type: inference",
                "subsystem:",
                "  model_variant: finetuned",
            ]
        ),
    )

    store = FileInstanceStore(tmp_path / "artifacts")
    service = InstanceCreationService(store)

    created = service.create_instance(str(finetune_config), start=False)
    assert created.type == "finetune"
    assert store.config_snapshot_path(created.id).exists()

    eval_instance = service.create_evaluation_instance(created.id, str(eval_template), start=False)
    assert eval_instance.parent_instance_id == created.id
    assert store.config_snapshot_path(eval_instance.id).exists()

    deploy_instance = service.create_deployment_instance(created.id, "huggingface", str(deploy_config), start=False)
    deploy_snapshot = store.load_config_snapshot(deploy_instance.id)
    assert deploy_snapshot["subsystem"]["provider"] == "huggingface"

    inference_instance = service.create_inference_instance(created.id, str(inference_template), start=False)
    assert inference_instance.parent_instance_id == created.id
    assert store.config_snapshot_path(inference_instance.id).exists()
