from __future__ import annotations

from pathlib import Path
from typing import Any

from ai_factory.core.artifacts import ArtifactLayout, current_git_sha, ensure_latest_pointer, write_json
from ai_factory.core.schemas import RunManifest
from training.src.analysis import write_run_summary
from training.src.config import ExperimentConfig
from training.src.modeling import export_merged_model


def write_run_manifest(
    layout: ArtifactLayout,
    config: ExperimentConfig,
    data_files: list[str],
    metrics_files: list[str],
    report_files: list[str],
    metadata: dict[str, Any],
) -> RunManifest:
    repo_root = Path(__file__).resolve().parents[2]
    data_files = [item for item in data_files if item]
    metrics_files = [item for item in metrics_files if item]
    report_files = [item for item in report_files if item]
    manifest = RunManifest(
        run_id=layout.run_id,
        run_name=config.run_name,
        profile_name=config.profile_name,
        base_dir=str(layout.root),
        model_name=config.packaging.publish_model_name,
        base_model=config.model.base_model_name,
        config_path=config.config_path,
        git_sha=current_git_sha(repo_root),
        data_files=data_files,
        metrics_files=metrics_files,
        report_files=report_files,
        metadata=metadata,
    )
    write_json(layout.manifests_dir / config.logging.manifest_filename, manifest.model_dump())
    return manifest


def publish_model_artifacts(
    layout: ArtifactLayout,
    config: ExperimentConfig,
    trainer: Any,
    tokenizer: Any,
) -> dict[str, str]:
    final_dir = layout.root / config.packaging.final_adapter_subdir
    final_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(final_dir))
    if config.packaging.save_tokenizer:
        tokenizer.save_pretrained(str(final_dir))

    outputs = {"final_adapter": str(final_dir)}
    if config.packaging.export_merged_model:
        merged_dir = layout.root / config.packaging.merged_model_subdir
        merged_dir.mkdir(parents=True, exist_ok=True)
        merged_result = export_merged_model(trainer.model, str(merged_dir))
        if merged_result:
            outputs["merged_model"] = merged_result

    model_dir = Path(config.training.artifacts_dir) / "models" / config.packaging.publish_model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    latest_pointer = model_dir / "LATEST_MODEL.json"
    if config.packaging.publish_latest:
        ensure_latest_pointer(
            latest_pointer,
            final_dir,
            metadata={"run_id": layout.run_id, "run_name": config.run_name},
        )
    return outputs


def write_training_summary(layout: ArtifactLayout, summary: dict[str, Any], filename: str) -> Path:
    output_path = layout.reports_dir / filename
    write_run_summary(output_path, summary)
    return output_path
