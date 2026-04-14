from __future__ import annotations

import argparse
import inspect
import json
import logging
import os
import sys
from pathlib import Path

import torch
from transformers import TrainingArguments, set_seed

# MPS/Metal optimizations for Apple Silicon
if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    # Enable tensor float-32 for faster computation on M1/M2/M3
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
    # Enable MPS memory optimization
    if hasattr(torch.backends.mps, "allow_tensor_float32"):
        torch.backends.mps.allow_tensor_float32 = True

from ai_factory.core.artifacts import prepare_run_layout, write_json
from ai_factory.orchestration.distributed import DistributedConfig, DistributedTrainingOrchestrator
from training.src.analysis import dataset_summary
from training.src.callbacks import JsonlMetricsCallback, TrackerCallback
from training.src.checkpoints import resolve_resume_checkpoint
from training.src.collators import WeightedDataCollator
from training.src.config import (
    ExperimentConfig,
    describe_experiment_config,
    load_experiment_config,
    validate_experiment_config,
)
from training.src.data import build_dataset
from training.src.environment import collect_environment_snapshot
from training.src.model_packaging import publish_model_artifacts, write_run_manifest, write_training_summary
from training.src.modeling import load_model_for_training, load_tokenizer, trainable_parameter_report
from training.src.optimization import HardwareDetector
from training.src.tracking import build_tracker
from training.src.ultimate_harness import (
    HarnessConfig,
    UltimateTrainingHarness,
    build_ultimate_trainer_with_harness,
)
from training.src.validation import run_dry_validation

# Set up structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)
_TRAINING_ARGUMENT_PARAMETERS = set(inspect.signature(TrainingArguments.__init__).parameters)


def _mps_available() -> bool:
    mps_backend = getattr(torch.backends, "mps", None)
    return bool(mps_backend and mps_backend.is_available())


def _resolve_precision_flags(config: ExperimentConfig) -> dict[str, bool]:
    training = config.training
    bf16_enabled = training.bf16
    fp16_enabled = training.fp16

    if torch.cuda.is_available():
        return {"bf16": bf16_enabled, "fp16": fp16_enabled, "use_cpu": False}

    if _mps_available():
        return {"bf16": False, "fp16": False, "use_cpu": False}

    # TrainingArguments validates mixed precision against the current runtime
    # during initialization, so CPU-only environments must disable GPU dtypes.
    return {"bf16": False, "fp16": False, "use_cpu": True}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train or fine-tune a causal language model with research-grade artifacts."
    )
    parser.add_argument("--config", required=True, help="Path to the YAML config file.")
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument(
        "--resume-from-latest-checkpoint",
        action="store_true",
        help="Resume from the newest checkpoint in the run directory when no explicit checkpoint is provided.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configs, tokenizer, datasets, and artifacts without training.",
    )
    parser.add_argument(
        "--validate-model-load",
        action="store_true",
        help="Load the model after dry validation and exit.",
    )
    parser.add_argument(
        "--distributed",
        action="store_true",
        help="Launch the script in distributed mode using DistributedTrainingOrchestrator.",
    )
    parser.add_argument("--num-gpus", type=int, default=1, help="Number of GPUs for distributed training.")
    return parser.parse_args()


def build_training_arguments(config: ExperimentConfig, layout) -> TrainingArguments:
    training = config.training
    optim = (
        "paged_adamw_8bit"
        if (config.model.use_4bit or config.model.use_8bit) and torch.cuda.is_available()
        else "adamw_torch"
    )
    precision_flags = _resolve_precision_flags(config)
    training_arguments = {
        "output_dir": str(layout.checkpoints_dir),
        "run_name": config.run_name,
        "num_train_epochs": training.num_train_epochs,
        "max_steps": training.max_steps,
        "learning_rate": training.learning_rate,
        "weight_decay": training.weight_decay,
        "warmup_ratio": training.warmup_ratio,
        "lr_scheduler_type": training.lr_scheduler_type,
        "per_device_train_batch_size": training.per_device_train_batch_size,
        "per_device_eval_batch_size": training.per_device_eval_batch_size,
        "gradient_accumulation_steps": training.gradient_accumulation_steps,
        "max_grad_norm": training.max_grad_norm,
        "logging_steps": training.logging_steps,
        "eval_steps": training.eval_steps,
        "save_steps": training.save_steps,
        "save_total_limit": training.save_total_limit,
        "bf16": precision_flags["bf16"],
        "fp16": precision_flags["fp16"],
        "use_cpu": precision_flags["use_cpu"],
        "save_strategy": training.save_strategy,
        "load_best_model_at_end": training.load_best_model_at_end,
        "report_to": config.logging.report_to,
        "remove_unused_columns": False,
        "optim": optim,
        "logging_dir": str(layout.logs_dir),
        "logging_first_step": config.logging.logging_first_step,
        "group_by_length": training.group_by_length,
        "save_safetensors": training.save_safetensors,
        "seed": config.seed,
        "deepspeed": (
            config.runtime.deepspeed_config
            if torch.cuda.is_available() or HardwareDetector._is_rocm_available()
            else None
        ),
        "torch_compile": config.runtime.torch_compile,
        "dataloader_num_workers": 0 if _mps_available() else training.dataloader_num_workers,
        "dataloader_pin_memory": not _mps_available(),
    }
    if "use_mps_device" in _TRAINING_ARGUMENT_PARAMETERS and _mps_available():
        training_arguments["use_mps_device"] = True
    if "eval_strategy" in _TRAINING_ARGUMENT_PARAMETERS:
        training_arguments["eval_strategy"] = training.evaluation_strategy
    else:
        training_arguments["evaluation_strategy"] = training.evaluation_strategy
    training_arguments = {
        key: value for key, value in training_arguments.items() if key in _TRAINING_ARGUMENT_PARAMETERS
    }
    return TrainingArguments(
        **training_arguments,
    )


def save_config_snapshot(config: ExperimentConfig, layout) -> Path:
    """
    Save a snapshot of the experiment configuration to the layout manifests directory.

    Args:
        config (ExperimentConfig): The experiment configuration to save.
        layout (Any): The initialized layout structure for the run.

    Returns:
        Path: The file path where the config snapshot was saved.
    """
    path = layout.manifests_dir / "config_snapshot.json"
    write_json(path, config.to_dict())
    return path


def write_config_report(
    config: ExperimentConfig,
    layout,
    *,
    warnings: list[str],
    resume_from_checkpoint: str | None,
) -> tuple[Path, dict[str, object]]:
    """
    Write a comprehensive configuration report including warnings and checkpoint data.

    Args:
        config (ExperimentConfig): The experiment configuration.
        layout (Any): The directory layout for the run.
        warnings (list[str]): List of warnings generated during config validation.
        resume_from_checkpoint (str | None): The checkpoint path used for resuming (if any).

    Returns:
        tuple[Path, dict[str, object]]: The path to the report file and the report data dictionary.
    """
    report = describe_experiment_config(config, warnings=warnings)
    report["resume_from_checkpoint"] = resume_from_checkpoint
    path = layout.manifests_dir / "config_report.json"
    write_json(path, report)
    return path, report


def summarize_environment(snapshot: dict[str, object]) -> dict[str, object]:
    """
    Summarize the full environment snapshot into key metrics for tracking.

    Args:
        snapshot (dict[str, object]): The raw environment snapshot dictionary.

    Returns:
        dict[str, object]: A summarized dictionary containing git SHA, Python/Platform versions,
        and CUDA availability.
    """
    python_info = snapshot.get("python") if isinstance(snapshot.get("python"), dict) else {}
    platform_info = snapshot.get("platform") if isinstance(snapshot.get("platform"), dict) else {}
    torch_info = snapshot.get("torch") if isinstance(snapshot.get("torch"), dict) else {}
    return {
        "git_sha": snapshot.get("git_sha"),
        "python_version": python_info.get("version"),
        "platform": platform_info.get("platform"),
        "cuda_available": torch_info.get("cuda_available"),
        "cuda_device_count": torch_info.get("cuda_device_count"),
    }


def build_dataset_artifacts(config: ExperimentConfig, tokenizer, layout):
    """
    Build the train and evaluation datasets, and generate a dataset report.

    Args:
        config (ExperimentConfig): The experiment configuration containing data paths.
        tokenizer (Any): The tokenizer to use for building the datasets.
        layout (Any): The run layout directory structure.

    Returns:
        tuple: A tuple containing (train_dataset, eval_dataset, dataset_report_dict).
    """
    logger.info("Building dataset artifacts.")
    train_dataset = build_dataset(
        file_path=config.data.train_file,
        tokenizer=tokenizer,
        data_config=config.data,
        split="train",
    )
    eval_dataset = None
    if config.data.eval_file:
        eval_dataset = build_dataset(
            file_path=config.data.eval_file,
            tokenizer=tokenizer,
            data_config=config.data,
            split="eval",
        )
    dataset_report = {
        "train": dataset_summary(config.data.train_file, split="train"),
        "eval": dataset_summary(config.data.eval_file, split="eval") if config.data.eval_file else None,
        "tokenized_train_rows": len(train_dataset),
        "tokenized_eval_rows": len(eval_dataset) if eval_dataset is not None else 0,
    }
    write_json(layout.metrics_dir / "dataset_report.json", dataset_report)
    return train_dataset, eval_dataset, dataset_report


def load_json_if_exists(path: Path) -> dict:
    """
    Load a JSON file into a dictionary if the file exists.

    Args:
        path (Path): The path to the JSON file.

    Returns:
        dict: The parsed JSON dictionary or an empty dict if the file is missing.
    """
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def main() -> None:
    """
    Main entry point for the training execution.

    Handles argument parsing, orchestrator setup for distributed training,
    configuration validation, artifact layout preparation, tokenizer/model loading,
    dry-run execution, full training execution, and metric/lineage reporting.
    """
    args = parse_args()

    # Orchestrator Integration: If distributed flag is set and we're not already the worker
    if args.distributed and "LOCAL_RANK" not in os.environ:
        logger.info("Distributed mode requested. Relaunching via DistributedTrainingOrchestrator.")
        orchestrator = DistributedTrainingOrchestrator(DistributedConfig(num_gpus_per_node=args.num_gpus))
        script_args = []
        # Filter out orchestrator-specific args
        skip_next = False
        for arg in sys.argv[1:]:
            if skip_next:
                skip_next = False
                continue
            if arg == "--distributed":
                continue
            if arg == "--num-gpus":
                skip_next = True
                continue
            if arg.startswith("--num-gpus="):
                continue
            script_args.append(arg)
        sys.exit(orchestrator.launch(sys.argv[0], script_args))

    logger.info("Loading experiment config.")
    config = load_experiment_config(args.config)
    validation_warnings = validate_experiment_config(config)
    for warning in validation_warnings:
        logger.warning("Config validation: %s", warning)
    set_seed(config.seed)

    layout = prepare_run_layout(config.training.artifacts_dir, config.run_name)
    resume_from_checkpoint, checkpoint_report = resolve_resume_checkpoint(
        layout.checkpoints_dir,
        explicit_checkpoint=args.resume_from_checkpoint,
        resume_from_latest=args.resume_from_latest_checkpoint,
        artifacts_dir=config.training.artifacts_dir,
        run_name=config.run_name,
        exclude_run_id=layout.run_id,
    )
    config_snapshot_path = save_config_snapshot(config, layout)
    config_report_path, config_report = write_config_report(
        config,
        layout,
        warnings=validation_warnings,
        resume_from_checkpoint=resume_from_checkpoint,
    )

    tracker = build_tracker(layout, config)
    tracker_summary: dict[str, float | int] = {}
    status = "failed"

    environment_snapshot_path: Path | None = None
    environment_summary: dict[str, object] | None = None
    if config.tracking.capture_environment:
        environment_snapshot_path, environment_snapshot = collect_environment_snapshot(config, layout)
        environment_summary = summarize_environment(environment_snapshot)
        tracker.log_params({"environment": environment_summary})
        tracker.log_artifact(environment_snapshot_path, name="manifests")

    tracker.log_params(
        {
            "run": {
                "run_name": config.run_name,
                "profile_name": config.profile_name,
                "seed": config.seed,
                "artifacts_dir": config.training.artifacts_dir,
                "resume_from_checkpoint": resume_from_checkpoint,
            },
            "model": config.model.__dict__,
            "data": {
                "train_file": config.data.train_file,
                "eval_file": config.data.eval_file,
                "pack_manifest": config.data.pack_manifest,
                "max_length": config.data.max_length,
            },
            "training": config.training.__dict__,
            "runtime": config.runtime.__dict__,
            "tracking": config.tracking.__dict__,
            "config_report": config_report,
        }
    )
    if config.tracking.log_config_artifact:
        tracker.log_artifact(config_snapshot_path, name="manifests")
        tracker.log_artifact(config_report_path, name="manifests")

    validation_report = {
        "warnings": validation_warnings,
        "resume_from_checkpoint": resume_from_checkpoint,
        "checkpoint_report": checkpoint_report,
        "config_report_path": str(config_report_path),
        "config_snapshot_path": str(config_snapshot_path),
    }
    validation_report_path = layout.manifests_dir / "validation_report.json"
    write_json(validation_report_path, validation_report)
    if config.tracking.log_config_artifact:
        tracker.log_artifact(validation_report_path, name="manifests")

    validate_model_load = args.validate_model_load or config.runtime.validate_model_load

    logger.info("Loading tokenizer.")
    require_local_tokenizer = config.model.initialization.lower() == "scratch" and not args.dry_run
    tokenizer = (
        load_tokenizer(config, require_local_path=require_local_tokenizer)
        if (not args.dry_run or validate_model_load)
        else None
    )

    dry_validation = run_dry_validation(config, tokenizer)
    write_json(layout.metrics_dir / "dry_run_validation.json", dry_validation)
    tracker.log_metrics(
        {
            "dry_validation_train_rows": dry_validation["train_dataset"]["num_rows"],
            "dry_validation_eval_rows": (
                dry_validation["eval_dataset"]["num_rows"] if dry_validation.get("eval_dataset") else 0
            ),
            "dry_validation_max_length": config.data.max_length,
        }
    )

    try:
        if args.dry_run:
            logger.info("Executing dry run mode.")
            if validate_model_load:
                logger.info("Validating model load.")
                tokenizer = tokenizer or load_tokenizer(config)
                model = load_model_for_training(config, tokenizer=tokenizer)
                parameter_report = trainable_parameter_report(model)
                write_json(layout.metrics_dir / "model_report.json", parameter_report)
                tracker.log_metrics({"trainable_ratio": parameter_report["trainable_ratio"]})
                if config.tracking.log_model_artifacts:
                    tracker.log_artifact(layout.metrics_dir / "model_report.json", name="metrics")

            manifest = write_run_manifest(
                layout,
                config,
                data_files=[config.data.train_file, config.data.eval_file or ""],
                metrics_files=[str(layout.metrics_dir / "dry_run_validation.json")],
                report_files=[],
                metadata={
                    "mode": "dry_run",
                    "resume_from_checkpoint": resume_from_checkpoint,
                    "config_report_path": str(config_report_path),
                    "validation_report_path": str(validation_report_path),
                    "environment_snapshot": str(environment_snapshot_path) if environment_snapshot_path else None,
                },
            )
            summary = {
                "run_name": config.run_name,
                "profile_name": config.profile_name,
                "base_model": config.model.base_model_name,
                "status": "dry_run_complete",
                "run_dir": str(layout.root),
                "validation": validation_report,
                "config": config_report,
                "train_rows": dry_validation["train_dataset"]["num_rows"],
                "eval_rows": dry_validation["eval_dataset"]["num_rows"] if dry_validation.get("eval_dataset") else 0,
                "parameter_report": load_json_if_exists(layout.metrics_dir / "model_report.json"),
                "dataset_report": dry_validation,
                "metrics": {
                    "status": "dry_run_complete",
                    "tokenizer_mode": dry_validation["train_dataset"].get("tokenizer_mode"),
                },
                "artifacts": {
                    "manifest": str(layout.manifests_dir / config.logging.manifest_filename),
                    "config_snapshot": str(config_snapshot_path),
                    "config_report": str(config_report_path),
                    "validation_report": str(validation_report_path),
                },
                "environment": environment_summary or {},
                "tracking": {
                    "provider": config.tracking.provider,
                    "project": config.tracking.project,
                    "experiment_name": config.tracking.experiment_name,
                },
                "resume_from_checkpoint": resume_from_checkpoint,
            }
            summary_path = write_training_summary(layout, summary, config.logging.summary_markdown_filename)
            tracker_summary = {
                "train_rows": summary["train_rows"],
                "eval_rows": summary["eval_rows"],
            }
            if config.tracking.log_summary_artifact:
                tracker.log_artifact(summary_path, name="reports")
            write_run_manifest(
                layout,
                config,
                data_files=manifest.data_files,
                metrics_files=manifest.metrics_files,
                report_files=[str(summary_path)],
                metadata={
                    "mode": "dry_run",
                    "resume_from_checkpoint": resume_from_checkpoint,
                    "validation": dry_validation,
                    "environment_snapshot": str(environment_snapshot_path) if environment_snapshot_path else None,
                },
            )
            status = "completed"
            logger.info(f"Dry run completed. Run directory: {layout.root}")
            return

        logger.info("Initializing full training run.")
        tokenizer = tokenizer or load_tokenizer(config)

        # Detect hardware and initialize ultimate harness
        logger.info("Detecting hardware and initializing ultimate optimization harness.")
        hardware = HardwareDetector.detect()
        harness_config = HarnessConfig(
            enable_mixed_precision=config.training.bf16 or config.training.fp16,
            enable_gradient_checkpointing=config.training.gradient_checkpointing or config.model.gradient_checkpointing,
            enable_torch_compile=config.runtime.torch_compile,
            enable_memory_profiling=os.environ.get("AI_FACTORY_MEMORY_PROFILE", "0") == "1",
        )
        harness = UltimateTrainingHarness(config, harness_config, hardware)
        harness.print_summary()

        # Load model first, then prepare it through the harness
        logger.info("Loading model for training.")
        tokenizer = tokenizer or load_tokenizer(config)
        model = load_model_for_training(config, tokenizer=tokenizer)

        # Prepare model with ultimate optimizations (apply LoRA, move to device, compile)
        model = harness.prepare_model(model)

        # Get optimized training arguments from harness
        training_args = harness.get_training_arguments(layout)

        parameter_report = trainable_parameter_report(model)
        write_json(layout.metrics_dir / "model_report.json", parameter_report)
        tracker.log_metrics({"trainable_ratio": parameter_report["trainable_ratio"]})
        if config.tracking.log_model_artifacts:
            tracker.log_artifact(layout.metrics_dir / "model_report.json", name="metrics")

        train_dataset, eval_dataset, dataset_report = build_dataset_artifacts(config, tokenizer, layout)
        tracker.log_metrics(
            {
                "tokenized_train_rows": dataset_report["tokenized_train_rows"],
                "tokenized_eval_rows": dataset_report["tokenized_eval_rows"],
            }
        )
        if config.tracking.log_dataset_artifacts:
            tracker.log_artifact(layout.metrics_dir / "dataset_report.json", name="metrics")

        data_collator = WeightedDataCollator(tokenizer=tokenizer, label_pad_token_id=-100, pad_to_multiple_of=8)

        # Build ultimate trainer with harness integration
        trainer = build_ultimate_trainer_with_harness(
            config=config,
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            tokenizer=tokenizer,
            callbacks=[
                JsonlMetricsCallback(layout.logs_dir / config.logging.jsonl_metrics_filename),
                TrackerCallback(tracker),
            ],
            layout=layout,
        )

        logger.info("Starting training loop with ultimate optimization.")
        print("[Ultimate Training Harness] Hardware-aware optimization active.")
        print(f"[Ultimate Training Harness] Backend: {harness.hardware.backend.upper()}")
        print(f"[Ultimate Training Harness] Device: {harness.hardware.device_name}")
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)

        logger.info("Training complete. Evaluating.")
        metrics = trainer.evaluate() if eval_dataset is not None else {}
        write_json(layout.metrics_dir / "metrics.json", metrics)
        tracker.log_metrics(metrics)
        tracker.log_artifact(layout.metrics_dir / "metrics.json", name="metrics")

        logger.info("Publishing artifacts.")
        published = publish_model_artifacts(layout, config, trainer, tokenizer)
        trainer.save_state()

        # Automatic Lineage Registration
        try:
            from ai_factory.core.lineage.models import LineageRecord
            from ai_factory.core.platform.container import build_platform_container

            container = build_platform_container(repo_root=Path.cwd())
            record = LineageRecord(
                id=layout.run_id,
                base_model=config.model.base_model_name,
                dataset_hash=config.data.pack_manifest or "unspecified",
                training_config=config.to_dict(),
                metrics={k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))},
                tags=config.tracking.tags + [config.adapter.method],
            )
            container.control_service.record_lineage(record)
            logger.info(f"Model lineage registered: {layout.run_id}")
        except Exception as e:
            logger.warning(f"Lineage registration failed: {e}")

        summary = {
            "run_name": config.run_name,
            "profile_name": config.profile_name,
            "base_model": config.model.base_model_name,
            "status": "completed",
            "run_dir": str(layout.root),
            "validation": validation_report,
            "config": config_report,
            "train_rows": dataset_report["tokenized_train_rows"],
            "eval_rows": dataset_report["tokenized_eval_rows"],
            "parameter_report": parameter_report,
            "dataset_report": dataset_report,
            "metrics": metrics,
            "artifacts": {
                "manifest": str(layout.manifests_dir / config.logging.manifest_filename),
                "config_snapshot": str(config_snapshot_path),
                "config_report": str(config_report_path),
                "validation_report": str(validation_report_path),
                "dataset_report": str(layout.metrics_dir / "dataset_report.json"),
                "model_report": str(layout.metrics_dir / "model_report.json"),
                "metrics": str(layout.metrics_dir / "metrics.json"),
                "summary": str(layout.reports_dir / config.logging.summary_markdown_filename),
            },
            "environment": environment_summary or {},
            "tracking": {
                "provider": config.tracking.provider,
                "project": config.tracking.project,
                "experiment_name": config.tracking.experiment_name,
            },
            "resume_from_checkpoint": resume_from_checkpoint,
            "published": published,
        }

        summary_path = write_training_summary(layout, summary, config.logging.summary_markdown_filename)
        tracker_summary = {
            "train_rows": summary["train_rows"],
            "eval_rows": summary["eval_rows"],
            **{key: value for key, value in metrics.items() if isinstance(value, (int, float))},
        }
        if config.tracking.log_summary_artifact:
            tracker.log_artifact(summary_path, name="reports")

        write_run_manifest(
            layout,
            config,
            data_files=[config.data.train_file, config.data.eval_file or "", config.data.pack_manifest or ""],
            metrics_files=[
                str(layout.metrics_dir / "dataset_report.json"),
                str(layout.metrics_dir / "model_report.json"),
                str(layout.metrics_dir / "metrics.json"),
            ],
            report_files=[str(summary_path)],
            metadata={
                "published": published,
                "resume_from_checkpoint": resume_from_checkpoint,
                "tracking": {
                    "provider": config.tracking.provider,
                    "project": config.tracking.project,
                    "experiment_name": config.tracking.experiment_name,
                },
                "environment_snapshot": str(environment_snapshot_path) if environment_snapshot_path else None,
            },
        )

        status = "completed"
        logger.info(f"Run completed successfully. Run directory: {layout.root}")
    except Exception as exc:
        logger.exception("An error occurred during execution.")
        tracker.log_params({"error": {"type": type(exc).__name__, "message": str(exc)}})
        raise
    finally:
        tracker.finalize(status=status, summary=tracker_summary)


if __name__ == "__main__":
    main()
