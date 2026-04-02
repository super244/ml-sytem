from __future__ import annotations

import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import yaml

from ai_factory.core.io import write_json
from data.builders.corpus_builder import ProcessingConfig, build_corpus

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-Math-1.5B-Instruct"
DEFAULT_CUSTOM_ROOT_CANDIDATES = (
    REPO_ROOT / "data" / "custom",
    REPO_ROOT / "datasets" / "custom",
)


@dataclass(frozen=True)
class WorkflowLayout:
    root: Path
    datasets_dir: Path
    configs_dir: Path
    reports_dir: Path


def slugify(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "-", value).strip("-").lower()
    return cleaned or "run"


def parse_csv_values(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def resolve_custom_data_root(explicit_root: str | Path | None = None) -> Path:
    if explicit_root is not None:
        root = Path(explicit_root)
        if not root.exists():
            raise FileNotFoundError(f"Custom dataset root was not found: {root}")
        return root
    for candidate in DEFAULT_CUSTOM_ROOT_CANDIDATES:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("No private dataset root was found. Checked data/custom and datasets/custom.")


def discover_private_categories(custom_root: str | Path | None = None) -> list[str]:
    root = resolve_custom_data_root(custom_root)
    return sorted(
        path.stem
        for path in root.glob("*.jsonl")
        if path.is_file()
    )


def resolve_private_category_paths(
    categories: list[str],
    *,
    custom_root: str | Path | None = None,
) -> list[Path]:
    if not categories:
        return []
    root = resolve_custom_data_root(custom_root)
    known = {path.stem: path for path in root.glob("*.jsonl") if path.is_file()}
    missing = [category for category in categories if category not in known]
    if missing:
        available = ", ".join(sorted(known))
        raise ValueError(f"Unknown private dataset categories: {', '.join(missing)}. Available: {available}")
    return [known[category] for category in categories]


def normalize_huggingface_dataset_reference(value: str) -> dict[str, str]:
    raw = value.strip()
    if not raw:
        raise ValueError("Empty Hugging Face dataset reference.")

    if raw.startswith("https://") or raw.startswith("http://"):
        parsed = urlparse(raw)
        if parsed.netloc not in {"huggingface.co", "www.huggingface.co"}:
            raise ValueError(f"Unsupported dataset host: {parsed.netloc}")
        parts = [part for part in parsed.path.split("/") if part]
        if len(parts) < 2 or parts[0] != "datasets":
            raise ValueError(f"Unsupported Hugging Face dataset URL: {raw}")
        dataset_parts: list[str] = []
        for part in parts[1:]:
            if part in {"tree", "blob"}:
                break
            dataset_parts.append(part)
        if not dataset_parts:
            raise ValueError(f"Could not resolve dataset id from URL: {raw}")
        query = parse_qs(parsed.query)
        split = (query.get("split") or ["train"])[0]
        revision = (query.get("revision") or [None])[0]
        payload = {"path": "/".join(dataset_parts), "split": split}
        if revision:
            payload["revision"] = revision
        return payload

    dataset_id = raw
    split = "train"
    revision: str | None = None
    if "#split=" in raw:
        dataset_id, split_fragment = raw.split("#split=", 1)
        split = split_fragment.strip() or "train"
    if "@" in dataset_id:
        dataset_id, revision = dataset_id.split("@", 1)
        dataset_id = dataset_id.strip()
        revision = revision.strip() or None
    payload = {"path": dataset_id, "split": split}
    if revision:
        payload["revision"] = revision
    return payload


def build_workflow_layout(workflow_name: str, run_name: str) -> WorkflowLayout:
    root = REPO_ROOT / "artifacts" / "workflows" / workflow_name / slugify(run_name)
    datasets_dir = root / "datasets"
    configs_dir = root / "configs"
    reports_dir = root / "reports"
    for path in (datasets_dir, configs_dir, reports_dir):
        path.mkdir(parents=True, exist_ok=True)
    return WorkflowLayout(root=root, datasets_dir=datasets_dir, configs_dir=configs_dir, reports_dir=reports_dir)


def build_source_specs(
    *,
    public_datasets: list[str] | None = None,
    private_categories: list[str] | None = None,
    local_datasets: list[str] | None = None,
    custom_root: str | Path | None = None,
) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []

    for public_dataset in public_datasets or []:
        normalized = normalize_huggingface_dataset_reference(public_dataset)
        specs.append(
            {
                "id": f"hf_{slugify(normalized['path'])}",
                "kind": "huggingface",
                "path": normalized["path"],
                "split": normalized.get("split", "train"),
                "revision": normalized.get("revision"),
            }
        )

    for category_path in resolve_private_category_paths(private_categories or [], custom_root=custom_root):
        specs.append(
            {
                "id": category_path.stem,
                "kind": "local",
                "path": str(category_path),
            }
        )

    for local_dataset in local_datasets or []:
        path = Path(local_dataset)
        specs.append(
            {
                "id": slugify(path.stem or path.name),
                "kind": "local",
                "path": str(path),
            }
        )

    if not specs:
        raise ValueError("No datasets were selected for the workflow.")
    return specs


def build_workflow_corpus(
    *,
    workflow_name: str,
    run_name: str,
    public_datasets: list[str] | None = None,
    private_categories: list[str] | None = None,
    local_datasets: list[str] | None = None,
    custom_root: str | Path | None = None,
    seed: int = 42,
    eval_ratio: float = 0.1,
    test_ratio: float = 0.05,
    max_samples: int | None = None,
    min_difficulty: str = "easy",
    system_prompt: str | None = None,
) -> dict[str, Any]:
    layout = build_workflow_layout(workflow_name, run_name)
    source_specs = build_source_specs(
        public_datasets=public_datasets,
        private_categories=private_categories,
        local_datasets=local_datasets,
        custom_root=custom_root,
    )
    config_payload = {
        "seed": seed,
        "eval_ratio": eval_ratio,
        "test_ratio": test_ratio,
        "min_difficulty": min_difficulty,
        "max_samples": max_samples,
        "output_dir": str(layout.datasets_dir),
        "sources": source_specs,
    }
    if system_prompt:
        config_payload["system_prompt"] = system_prompt
    config_path = layout.configs_dir / "processing_config.yaml"
    config_path.write_text(yaml.safe_dump(config_payload, sort_keys=False))
    result = build_corpus(ProcessingConfig(**config_payload), config_path)
    write_json(layout.reports_dir / "dataset_sources.json", {"sources": source_specs, "result": result})
    return {
        "layout": layout,
        "source_specs": source_specs,
        "processing_config_path": str(config_path),
        "dataset_dir": str(layout.datasets_dir),
        "train_file": str(layout.datasets_dir / "train.jsonl"),
        "eval_file": str(layout.datasets_dir / "eval.jsonl"),
        "test_file": str(layout.datasets_dir / "test.jsonl"),
        "manifest_path": str(layout.datasets_dir / "manifest.json"),
        **result,
    }


def build_training_config_payload(
    *,
    workflow_name: str,
    run_name: str,
    base_model_name: str,
    train_file: str,
    eval_file: str,
    test_file: str,
    pack_manifest: str,
    method: str = "qlora",
    seed: int = 42,
    artifacts_dir: str | Path | None = None,
    learning_rate: float = 1.5e-4,
    num_train_epochs: float = 2.0,
    per_device_train_batch_size: int = 2,
    gradient_accumulation_steps: int = 8,
    max_length: int = 2048,
    report_to: list[str] | None = None,
    run_description: str | None = None,
) -> dict[str, Any]:
    normalized_method = method.lower()
    if normalized_method not in {"qlora", "lora", "full", "sft"}:
        raise ValueError(f"Unsupported training method: {method}")

    use_4bit = normalized_method == "qlora"
    use_full_precision = normalized_method in {"full", "sft"}
    effective_learning_rate = learning_rate
    if normalized_method == "full":
        effective_learning_rate = min(learning_rate, 2.0e-5)
    elif normalized_method == "sft":
        effective_learning_rate = min(learning_rate, 5.0e-5)

    payload = {
        "profile_name": workflow_name,
        "run_name": run_name,
        "seed": seed,
        "model": {
            "name": slugify(base_model_name),
            "base_model_name": base_model_name,
            "trust_remote_code": True,
            "use_4bit": use_4bit,
            "use_8bit": False,
            "use_full_precision": use_full_precision,
            "bnb_4bit_quant_type": "nf4",
            "bnb_compute_dtype": "bfloat16",
            "double_quant": True,
            "gradient_checkpointing": True,
            "use_flash_attention": True,
            "device_map": "auto",
        },
        "data": {
            "train_file": train_file,
            "eval_file": eval_file,
            "test_file": test_file,
            "pack_manifest": pack_manifest,
            "max_length": max_length,
            "curriculum_learning": False,
            "sequential_curriculum": False,
            "oversample_hard_examples": False,
            "system_prompt": (
                "You are a rigorous assistant trained by AI-Factory. "
                "Answer using the dataset's input/output supervision and end with a clear final answer."
            ),
        },
        "training": {
            "artifacts_dir": str(artifacts_dir or (REPO_ROOT / "artifacts" / "workflow_runs")),
            "num_train_epochs": num_train_epochs,
            "max_steps": -1,
            "learning_rate": effective_learning_rate,
            "weight_decay": 0.01,
            "warmup_ratio": 0.05,
            "lr_scheduler_type": "cosine",
            "per_device_train_batch_size": per_device_train_batch_size,
            "per_device_eval_batch_size": per_device_train_batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "max_grad_norm": 1.0,
            "logging_steps": 10,
            "eval_steps": 50,
            "save_steps": 50,
            "save_total_limit": 2,
            "bf16": True,
            "fp16": False,
            "evaluation_strategy": "steps",
            "save_strategy": "steps",
            "load_best_model_at_end": True,
            "dataloader_num_workers": 0,
            "group_by_length": False,
            "save_safetensors": True,
            "max_validation_train_rows": 32,
            "max_validation_eval_rows": 16,
        },
        "adapter": {
            "method": normalized_method,
            "r": 32,
            "alpha": 64,
            "dropout": 0.05,
            "bias": "none",
            "task_type": "CAUSAL_LM",
            "target_modules": [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        },
        "runtime": {
            "profile_name": "hybrid_local",
            "accelerate_config": None,
            "deepspeed_config": None,
            "validate_tokenization_samples": 32,
            "validate_model_load": False,
            "torch_compile": False,
            "low_cpu_mem_usage": True,
        },
        "logging": {
            "report_to": report_to or ["none"],
            "logging_first_step": True,
            "jsonl_metrics_filename": "training_metrics.jsonl",
            "summary_markdown_filename": "run_summary.md",
            "manifest_filename": "run_manifest.json",
        },
        "tracking": {
            "provider": "jsonl",
            "project": "ai-factory",
            "experiment_name": workflow_name,
            "run_name": run_name,
            "tags": [workflow_name, normalized_method],
            "metadata": {"description": run_description or workflow_name},
            "capture_environment": True,
            "capture_installed_packages": True,
            "log_config_artifact": True,
            "log_dataset_artifacts": True,
            "log_model_artifacts": True,
            "log_summary_artifact": True,
        },
        "packaging": {
            "publish_model_name": slugify(run_name),
            "export_merged_model": normalized_method in {"full", "sft"},
            "publish_latest": True,
            "save_tokenizer": True,
            "final_adapter_subdir": "published/final_adapter",
            "merged_model_subdir": "published/merged_model",
        },
    }
    if normalized_method in {"full", "sft"}:
        payload["adapter"]["target_modules"] = []
    return payload


def write_training_config(
    *,
    layout: WorkflowLayout,
    filename: str,
    payload: dict[str, Any],
) -> Path:
    config_path = layout.configs_dir / filename
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False))
    return config_path


def launch_training(
    config_path: str | Path,
    *,
    dry_run: bool = False,
    validate_model_load: bool = False,
) -> subprocess.CompletedProcess[str]:
    command = [
        sys.executable,
        "training/train.py",
        "--config",
        str(config_path),
    ]
    if dry_run:
        command.append("--dry-run")
    if validate_model_load:
        command.append("--validate-model-load")
    return subprocess.run(command, cwd=REPO_ROOT, check=True, text=True)

