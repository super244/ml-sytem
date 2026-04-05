from __future__ import annotations

from typing import Any

from ai_factory.core.tokens import approximate_token_count
from training.src.analysis import dataset_summary
from training.src.config import DataConfig, ExperimentConfig
from training.src.data import build_dataset, build_messages, build_training_text, curriculum_sort, load_jsonl, render_chat


def build_validation_data_config(config: ExperimentConfig) -> DataConfig:
    return config.data.model_copy(
        update={
            "max_train_samples": min(
                config.training.max_validation_train_rows,
                config.data.max_train_samples or config.training.max_validation_train_rows,
            ),
            "max_eval_samples": min(
                config.training.max_validation_eval_rows,
                config.data.max_eval_samples or config.training.max_validation_eval_rows,
            ),
        }
    )


def run_dry_validation(config: ExperimentConfig, tokenizer: Any) -> dict[str, Any]:
    validation_config = build_validation_data_config(config)
    train_records = _load_validation_records(config.data.train_file, validation_config, split="train")
    eval_records = (
        _load_validation_records(config.data.eval_file, validation_config, split="eval")
        if config.data.eval_file
        else []
    )

    train_dataset = (
        build_dataset(
            file_path=config.data.train_file,
            tokenizer=tokenizer,
            data_config=validation_config,
            split="train",
        )
        if tokenizer is not None
        else None
    )
    eval_dataset = (
        build_dataset(
            file_path=config.data.eval_file,
            tokenizer=tokenizer,
            data_config=validation_config,
            split="eval",
        )
        if tokenizer is not None and config.data.eval_file
        else None
    )
    return {
        "train_dataset": {
            "path": config.data.train_file,
            "num_rows": len(train_records),
            "summary": dataset_summary(config.data.train_file),
            "tokenizer_mode": "loaded" if tokenizer is not None else "approximate_offline",
            "tokenized_rows": len(train_dataset) if train_dataset is not None else 0,
            "prompt_preview": _prompt_preview(train_records, validation_config, tokenizer),
        },
        "eval_dataset": {
            "path": config.data.eval_file,
            "num_rows": len(eval_records),
            "summary": dataset_summary(config.data.eval_file) if config.data.eval_file else None,
            "tokenizer_mode": "loaded" if tokenizer is not None else "approximate_offline",
            "tokenized_rows": len(eval_dataset) if eval_dataset is not None else 0,
            "prompt_preview": _prompt_preview(eval_records, validation_config, tokenizer),
        },
        "max_length": config.data.max_length,
    }


def _load_validation_records(file_path: str | None, data_config: DataConfig, split: str) -> list[dict[str, Any]]:
    if not file_path:
        return []
    records = load_jsonl(file_path)
    if split == "train" and data_config.max_train_samples:
        records = records[: data_config.max_train_samples]
    if split == "eval" and data_config.max_eval_samples:
        records = records[: data_config.max_eval_samples]
    if split == "train" and data_config.curriculum_learning:
        records = curriculum_sort(records, data_config.curriculum_phases)
    return records


def _prompt_preview(
    records: list[dict[str, Any]],
    data_config: DataConfig,
    tokenizer: Any | None,
    max_samples: int = 4,
) -> dict[str, Any]:
    previews: list[dict[str, Any]] = []
    for record in records[:max_samples]:
        if data_config.format == "pretraining_text":
            prompt_text = ""
            completion_text = build_training_text(record, data_config)
            full_text = completion_text
        else:
            messages = build_messages(record, data_config)
            prompt_text = render_chat(tokenizer, messages[:-1], add_generation_prompt=True)
            completion_text = messages[-1]["content"]
            full_text = render_chat(tokenizer, messages, add_generation_prompt=False)
        previews.append(
            {
                "id": record.get("id"),
                "topic": record.get("topic"),
                "difficulty": record.get("difficulty"),
                "prompt_tokens": approximate_token_count(prompt_text, tokenizer),
                "completion_tokens": approximate_token_count(completion_text, tokenizer),
                "total_tokens": approximate_token_count(full_text, tokenizer),
            }
        )
    return {
        "num_samples": len(previews),
        "samples": previews,
        "max_total_tokens": max((item["total_tokens"] for item in previews), default=0),
        "max_prompt_tokens": max((item["prompt_tokens"] for item in previews), default=0),
    }
