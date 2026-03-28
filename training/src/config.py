from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelConfig:
    name: str = "qwen25_math_1p5b"
    base_model_name: str = "Qwen/Qwen2.5-Math-1.5B-Instruct"
    tokenizer_name: str | None = None
    trust_remote_code: bool = True
    use_4bit: bool = True
    use_8bit: bool = False
    use_full_precision: bool = False
    bnb_4bit_quant_type: str = "nf4"
    bnb_compute_dtype: str = "bfloat16"
    double_quant: bool = True
    gradient_checkpointing: bool = True
    input_cost_per_million: float | None = None
    output_cost_per_million: float | None = None


@dataclass
class DataConfig:
    train_file: str = "data/processed/train.jsonl"
    eval_file: str | None = "data/processed/eval.jsonl"
    test_file: str | None = "data/processed/test.jsonl"
    pack_manifest: str | None = "data/processed/manifest.json"
    max_length: int = 2048
    curriculum_learning: bool = False
    sequential_curriculum: bool = False
    curriculum_phases: list[str] = field(default_factory=lambda: ["easy", "medium", "hard", "olympiad"])
    oversample_hard_examples: bool = False
    hard_weight: float = 2.0
    include_failure_tag: bool = True
    include_topic_prefix: bool = True
    include_source_tag: bool = True
    include_verification_tag: bool = True
    max_train_samples: int | None = None
    max_eval_samples: int | None = None
    system_prompt: str = (
        "You are an elite competition mathematician. Solve carefully and end with Final Answer: ..."
    )
    source_weights: dict[str, float] = field(default_factory=dict)
    difficulty_weights: dict[str, float] = field(default_factory=dict)
    failure_replay_boost: float = 1.35
    verification_boost: float = 1.15


@dataclass
class TrainingConfig:
    artifacts_dir: str = "artifacts"
    num_train_epochs: float = 2.0
    max_steps: int = -1
    learning_rate: float = 1.5e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    lr_scheduler_type: str = "cosine"
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    max_grad_norm: float = 1.0
    logging_steps: int = 10
    eval_steps: int = 50
    save_steps: int = 50
    save_total_limit: int = 2
    bf16: bool = True
    fp16: bool = False
    evaluation_strategy: str = "steps"
    save_strategy: str = "steps"
    load_best_model_at_end: bool = True
    dataloader_num_workers: int = 0
    group_by_length: bool = False
    save_safetensors: bool = True
    max_validation_train_rows: int = 32
    max_validation_eval_rows: int = 16


@dataclass
class AdapterConfig:
    method: str = "qlora"
    r: int = 32
    alpha: int = 64
    dropout: float = 0.05
    bias: str = "none"
    task_type: str = "CAUSAL_LM"
    target_modules: list[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )


@dataclass
class RuntimeConfig:
    profile_name: str = "hybrid_local"
    accelerate_config: str | None = None
    deepspeed_config: str | None = None
    validate_tokenization_samples: int = 32
    validate_model_load: bool = False
    torch_compile: bool = False
    low_cpu_mem_usage: bool = True


@dataclass
class LoggingConfig:
    report_to: list[str] = field(default_factory=lambda: ["none"])
    logging_first_step: bool = True
    jsonl_metrics_filename: str = "training_metrics.jsonl"
    summary_markdown_filename: str = "run_summary.md"
    manifest_filename: str = "run_manifest.json"


@dataclass
class TrackingConfig:
    provider: str = "none"
    project: str = "ai-factory"
    experiment_name: str = "default"
    run_name: str | None = None
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    mlflow_tracking_uri: str | None = None
    wandb_entity: str | None = None
    wandb_mode: str = "offline"
    capture_environment: bool = True
    capture_installed_packages: bool = True
    log_config_artifact: bool = True
    log_dataset_artifacts: bool = True
    log_model_artifacts: bool = True
    log_summary_artifact: bool = True


@dataclass
class PackagingConfig:
    publish_model_name: str = "atlas-math-failure-aware"
    export_merged_model: bool = False
    publish_latest: bool = True
    save_tokenizer: bool = True
    final_adapter_subdir: str = "published/final_adapter"
    merged_model_subdir: str = "published/merged_model"


@dataclass
class ExperimentConfig:
    run_name: str
    seed: int
    profile_name: str
    model: ModelConfig
    data: DataConfig
    training: TrainingConfig
    adapter: AdapterConfig
    runtime: RuntimeConfig
    logging: LoggingConfig
    tracking: TrackingConfig
    packaging: PackagingConfig
    config_path: str | None = None

    @property
    def lora(self) -> AdapterConfig:
        return self.adapter

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _deep_merge(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    merged = dict(left)
    for key, value in right.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _construct(dataclass_type: type[Any], payload: dict[str, Any] | None) -> Any:
    return dataclass_type(**(payload or {}))


def _load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text()) or {}


def _resolve_ref(config_path: Path, ref: str) -> Path:
    ref_path = Path(ref)
    if ref_path.is_absolute():
        return ref_path
    return (config_path.parent / ref_path).resolve()


def _apply_refs(config_path: Path, raw: dict[str, Any]) -> dict[str, Any]:
    refs = raw.get("refs") or {}
    merged: dict[str, Any] = {}
    for section, ref in refs.items():
        resolved = _resolve_ref(config_path, ref)
        merged[section] = _load_yaml(resolved)
    if "lora" in raw and "adapter" not in raw:
        raw["adapter"] = raw["lora"]
    for section in ("model", "data", "training", "adapter", "runtime", "logging", "tracking", "packaging"):
        if section in raw:
            merged[section] = _deep_merge(merged.get(section, {}), raw[section])
    if raw.get("overrides"):
        merged = _deep_merge(merged, raw["overrides"])
    merged["run_name"] = raw.get("run_name") or merged.get("run_name")
    merged["seed"] = raw.get("seed", merged.get("seed", 42))
    merged["profile_name"] = raw.get("profile_name", raw.get("name", config_path.stem))
    merged["config_path"] = str(config_path)
    return merged


def load_experiment_config(path: str) -> ExperimentConfig:
    config_path = Path(path).resolve()
    raw = _load_yaml(config_path)
    merged = _apply_refs(config_path, raw)
    return ExperimentConfig(
        run_name=merged["run_name"],
        seed=merged["seed"],
        profile_name=merged.get("profile_name", config_path.stem),
        model=_construct(ModelConfig, merged.get("model")),
        data=_construct(DataConfig, merged.get("data")),
        training=_construct(TrainingConfig, merged.get("training")),
        adapter=_construct(AdapterConfig, merged.get("adapter")),
        runtime=_construct(RuntimeConfig, merged.get("runtime")),
        logging=_construct(LoggingConfig, merged.get("logging")),
        tracking=_construct(TrackingConfig, merged.get("tracking")),
        packaging=_construct(PackagingConfig, merged.get("packaging")),
        config_path=str(config_path),
    )
