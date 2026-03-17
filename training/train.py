from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from transformers import TrainingArguments, set_seed

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ai_factory.core.artifacts import prepare_run_layout, write_json
from training.src.analysis import dataset_summary
from training.src.callbacks import JsonlMetricsCallback
from training.src.collators import WeightedDataCollator
from training.src.config import ExperimentConfig, load_experiment_config
from training.src.data import build_dataset
from training.src.modeling import load_model_for_training, load_tokenizer, trainable_parameter_report
from training.src.packaging import publish_model_artifacts, write_run_manifest, write_training_summary
from training.src.trainer import MathTrainer
from training.src.validation import run_dry_validation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune a math model with QLoRA and research-grade artifacts.")
    parser.add_argument("--config", required=True, help="Path to the YAML config file.")
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument("--dry-run", action="store_true", help="Validate configs, tokenizer, datasets, and artifacts without training.")
    parser.add_argument("--validate-model-load", action="store_true", help="Load the model after dry validation and exit.")
    return parser.parse_args()


def build_training_arguments(config: ExperimentConfig, layout) -> TrainingArguments:
    training = config.training
    optim = "paged_adamw_8bit" if (config.model.use_4bit or config.model.use_8bit) else "adamw_torch"
    return TrainingArguments(
        output_dir=str(layout.checkpoints_dir),
        run_name=config.run_name,
        num_train_epochs=training.num_train_epochs,
        max_steps=training.max_steps,
        learning_rate=training.learning_rate,
        weight_decay=training.weight_decay,
        warmup_ratio=training.warmup_ratio,
        lr_scheduler_type=training.lr_scheduler_type,
        per_device_train_batch_size=training.per_device_train_batch_size,
        per_device_eval_batch_size=training.per_device_eval_batch_size,
        gradient_accumulation_steps=training.gradient_accumulation_steps,
        max_grad_norm=training.max_grad_norm,
        logging_steps=training.logging_steps,
        eval_steps=training.eval_steps,
        save_steps=training.save_steps,
        save_total_limit=training.save_total_limit,
        bf16=training.bf16,
        fp16=training.fp16,
        evaluation_strategy=training.evaluation_strategy,
        save_strategy=training.save_strategy,
        load_best_model_at_end=training.load_best_model_at_end,
        report_to=config.logging.report_to,
        dataloader_num_workers=training.dataloader_num_workers,
        remove_unused_columns=False,
        optim=optim,
        logging_dir=str(layout.logs_dir),
        logging_first_step=config.logging.logging_first_step,
        group_by_length=training.group_by_length,
        save_safetensors=training.save_safetensors,
        seed=config.seed,
    )


def save_config_snapshot(config: ExperimentConfig, layout) -> None:
    write_json(layout.manifests_dir / "config_snapshot.json", config.to_dict())


def build_dataset_artifacts(config: ExperimentConfig, tokenizer, layout):
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
        "train": dataset_summary(config.data.train_file),
        "eval": dataset_summary(config.data.eval_file) if config.data.eval_file else None,
        "tokenized_train_rows": len(train_dataset),
        "tokenized_eval_rows": len(eval_dataset) if eval_dataset is not None else 0,
    }
    write_json(layout.metrics_dir / "dataset_report.json", dataset_report)
    return train_dataset, eval_dataset, dataset_report


def main() -> None:
    args = parse_args()
    config = load_experiment_config(args.config)
    set_seed(config.seed)

    layout = prepare_run_layout(config.training.artifacts_dir, config.run_name)
    save_config_snapshot(config, layout)

    validate_model_load = args.validate_model_load or config.runtime.validate_model_load
    tokenizer = load_tokenizer(config) if (not args.dry_run or validate_model_load) else None
    dry_validation = run_dry_validation(config, tokenizer)
    write_json(layout.metrics_dir / "dry_run_validation.json", dry_validation)

    if args.dry_run:
        if validate_model_load:
            tokenizer = tokenizer or load_tokenizer(config)
            model = load_model_for_training(config)
            parameter_report = trainable_parameter_report(model)
            write_json(layout.metrics_dir / "model_report.json", parameter_report)
        manifest = write_run_manifest(
            layout,
            config,
            data_files=[config.data.train_file, config.data.eval_file or ""],
            metrics_files=[str(layout.metrics_dir / "dry_run_validation.json")],
            report_files=[],
            metadata={"mode": "dry_run"},
        )
        summary = {
            "run_name": config.run_name,
            "profile_name": config.profile_name,
            "base_model": config.model.base_model_name,
            "train_rows": dry_validation["train_dataset"]["num_rows"],
            "eval_rows": dry_validation["eval_dataset"]["num_rows"] if dry_validation.get("eval_dataset") else 0,
            "parameter_report": load_json_if_exists(layout.metrics_dir / "model_report.json"),
            "metrics": {
                "status": "dry_run_complete",
                "tokenizer_mode": dry_validation["train_dataset"].get("tokenizer_mode"),
            },
        }
        summary_path = write_training_summary(layout, summary, config.logging.summary_markdown_filename)
        write_run_manifest(
            layout,
            config,
            data_files=manifest.data_files,
            metrics_files=manifest.metrics_files,
            report_files=[str(summary_path)],
            metadata={"mode": "dry_run", "validation": dry_validation},
        )
        print(json.dumps({"status": "dry_run_complete", "run_dir": str(layout.root)}, indent=2))
        return

    tokenizer = tokenizer or load_tokenizer(config)
    model = load_model_for_training(config)
    parameter_report = trainable_parameter_report(model)
    write_json(layout.metrics_dir / "model_report.json", parameter_report)

    train_dataset, eval_dataset, dataset_report = build_dataset_artifacts(config, tokenizer, layout)
    data_collator = WeightedDataCollator(tokenizer=tokenizer, label_pad_token_id=-100, pad_to_multiple_of=8)

    trainer = MathTrainer(
        model=model,
        args=build_training_arguments(config, layout),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        sequential_training=(config.data.curriculum_learning and config.data.sequential_curriculum),
        callbacks=[JsonlMetricsCallback(layout.logs_dir / config.logging.jsonl_metrics_filename)],
    )

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    metrics = trainer.evaluate() if eval_dataset is not None else {}
    write_json(layout.metrics_dir / "metrics.json", metrics)

    published = publish_model_artifacts(layout, config, trainer, tokenizer)
    trainer.save_state()
    summary = {
        "run_name": config.run_name,
        "profile_name": config.profile_name,
        "base_model": config.model.base_model_name,
        "train_rows": dataset_report["tokenized_train_rows"],
        "eval_rows": dataset_report["tokenized_eval_rows"],
        "parameter_report": parameter_report,
        "metrics": metrics,
    }
    summary_path = write_training_summary(layout, summary, config.logging.summary_markdown_filename)
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
        metadata={"published": published},
    )

    print(json.dumps({"metrics": metrics, "published": published, "run_dir": str(layout.root)}, indent=2))


def load_json_if_exists(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


if __name__ == "__main__":
    main()
