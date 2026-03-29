from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


class ConfigValidationError(ValueError):
    pass


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


SUPPORTED_ADAPTER_METHODS = {"qlora", "lora", "full", "sft"}
SUPPORTED_DTYPE_NAMES = {"bf16", "bfloat16", "fp16", "float16", "fp32", "float32"}
SUPPORTED_TRACKING_PROVIDERS = {"none", "jsonl", "mlflow", "wandb"}
SUPPORTED_STRATEGIES = {"no", "steps", "epoch"}


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


def _require(condition: bool, message: str, errors: list[str]) -> None:
    if not condition:
        errors.append(message)


def _find_project_root(start_path: Path) -> Path:
    """Find project root by looking for common project markers."""
    current = start_path
    while current != current.parent:
        if (current / "pyproject.toml").exists() or (current / "setup.py").exists():
            return current
        current = current.parent
    # Fallback to current directory if no markers found
    return start_path


def _path_exists(path_like: str | None, base_path: str | None = None) -> bool:
    """Check if a path exists, handling relative paths intelligently.
    
    Args:
        path_like: Path to check (relative or absolute)
        base_path: Base config file path for resolving relative paths
        
    Returns:
        True if path exists, False otherwise
    """
    if not path_like:
        return True
    
    path = Path(path_like)
    
    # If absolute, just check existence
    if path.is_absolute():
        return path.exists()
    
    # If no base path provided, check relative to current working directory
    if not base_path:
        return path.exists()
    
    base_dir = Path(base_path).parent
    
    # First try resolving relative to config file directory
    resolved = base_dir / path
    if resolved.exists():
        return True
    
    # For data/ paths, try resolving from project root
    # This handles the case where data files are relative to project root
    # but config files are in subdirectories
    if path_like.startswith("data/"):
        # Find project root dynamically
        project_root = _find_project_root(base_dir)
        resolved = project_root / path
        return resolved.exists()
    
    # Try other common relative patterns
    # Check relative to current working directory as fallback
    return path.exists()


def validate_experiment_config(config: ExperimentConfig) -> list[str]:
    errors: list[str] = []
    warnings: list[str] = []

    _require(bool(config.run_name.strip()), "run_name must be non-empty.", errors)
    _require(bool(config.profile_name.strip()), "profile_name must be non-empty.", errors)
    _require(bool(config.training.artifacts_dir.strip()), "training.artifacts_dir must be non-empty.", errors)
    _require(config.data.max_length > 0, "data.max_length must be positive.", errors)
    _require(
        config.training.per_device_train_batch_size > 0,
        "training.per_device_train_batch_size must be positive.",
        errors,
    )
    _require(
        config.training.per_device_eval_batch_size > 0,
        "training.per_device_eval_batch_size must be positive.",
        errors,
    )
    _require(
        config.training.gradient_accumulation_steps > 0,
        "training.gradient_accumulation_steps must be positive.",
        errors,
    )
    _require(config.training.max_grad_norm > 0, "training.max_grad_norm must be positive.", errors)
    _require(config.training.logging_steps > 0, "training.logging_steps must be positive.", errors)
    _require(config.training.eval_steps > 0, "training.eval_steps must be positive.", errors)
    _require(config.training.save_steps > 0, "training.save_steps must be positive.", errors)
    _require(config.training.save_total_limit >= 1, "training.save_total_limit must be at least 1.", errors)
    _require(
        config.training.max_validation_train_rows > 0,
        "training.max_validation_train_rows must be positive.",
        errors,
    )
    _require(
        config.training.max_validation_eval_rows > 0,
        "training.max_validation_eval_rows must be positive.",
        errors,
    )
    _require(0.0 <= config.training.warmup_ratio <= 1.0, "training.warmup_ratio must be between 0 and 1.", errors)
    _require(config.training.learning_rate > 0, "training.learning_rate must be positive.", errors)
    _require(config.training.weight_decay >= 0, "training.weight_decay must be non-negative.", errors)
    _require(
        config.training.num_train_epochs > 0 or config.training.max_steps > 0,
        "training requires positive num_train_epochs or max_steps.",
        errors,
    )
    _require(
        config.training.max_steps == -1 or config.training.max_steps > 0,
        "training.max_steps must be -1 or a positive integer.",
        errors,
    )
    _require(
        not (config.model.use_4bit and config.model.use_8bit),
        "model.use_4bit and model.use_8bit cannot both be enabled.",
        errors,
    )
    _require(
        not (config.model.use_full_precision and (config.model.use_4bit or config.model.use_8bit)),
        "model.use_full_precision cannot be combined with quantized loading.",
        errors,
    )
    _require(
        config.model.bnb_compute_dtype.lower() in SUPPORTED_DTYPE_NAMES,
        "model.bnb_compute_dtype must be one of bf16, fp16, or fp32.",
        errors,
    )
    _require(
        config.adapter.method.lower() in SUPPORTED_ADAPTER_METHODS,
        f"adapter.method must be one of {sorted(SUPPORTED_ADAPTER_METHODS)}.",
        errors,
    )
    _require(
        config.runtime.profile_name.strip() == config.runtime.profile_name,
        "runtime.profile_name must not contain leading or trailing whitespace.",
        errors,
    )
    _require(
        config.runtime.validate_tokenization_samples >= 0,
        "runtime.validate_tokenization_samples must be non-negative.",
        errors,
    )
    _require(config.runtime.low_cpu_mem_usage in {True, False}, "runtime.low_cpu_mem_usage must be boolean.", errors)
    _require(
        config.tracking.provider.lower() in SUPPORTED_TRACKING_PROVIDERS,
        f"tracking.provider must be one of {sorted(SUPPORTED_TRACKING_PROVIDERS)}.",
        errors,
    )
    _require(config.training.logging_steps >= 1, "training.logging_steps must be at least 1.", errors)
    _require(config.training.eval_steps >= 1, "training.eval_steps must be at least 1.", errors)
    _require(config.training.save_steps >= 1, "training.save_steps must be at least 1.", errors)
    _require(
        config.training.evaluation_strategy in SUPPORTED_STRATEGIES,
        f"training.evaluation_strategy must be one of {sorted(SUPPORTED_STRATEGIES)}.",
        errors,
    )
    _require(
        config.training.save_strategy in SUPPORTED_STRATEGIES,
        f"training.save_strategy must be one of {sorted(SUPPORTED_STRATEGIES)}.",
        errors,
    )
    _require(
        not (config.training.bf16 and config.training.fp16),
        "training.bf16 and training.fp16 cannot both be enabled.",
        errors,
    )
    _require(
        not (config.data.sequential_curriculum and not config.data.curriculum_learning),
        "data.sequential_curriculum requires data.curriculum_learning.",
        errors,
    )
    _require(
        _path_exists(config.data.train_file, config.config_path),
        f"training data file not found: {config.data.train_file}",
        errors,
    )
    _require(
        _path_exists(config.data.eval_file, config.config_path),
        f"eval data file not found: {config.data.eval_file}",
        errors,
    )
    _require(
        _path_exists(config.data.test_file, config.config_path),
        f"test data file not found: {config.data.test_file}",
        errors,
    )
    _require(
        _path_exists(config.data.pack_manifest, config.config_path),
        f"pack manifest not found: {config.data.pack_manifest}",
        errors,
    )
    _require(
        _path_exists(config.runtime.accelerate_config, config.config_path),
        f"accelerate config not found: {config.runtime.accelerate_config}",
        errors,
    )
    _require(
        _path_exists(config.runtime.deepspeed_config, config.config_path),
        f"deepspeed config not found: {config.runtime.deepspeed_config}",
        errors,
    )
    if config.training.load_best_model_at_end:
        _require(bool(config.data.eval_file), "load_best_model_at_end requires an evaluation file.", errors)
        _require(
            config.training.evaluation_strategy != "no",
            "load_best_model_at_end requires a non-'no' evaluation strategy.",
            errors,
        )
        _require(
            config.training.save_strategy == config.training.evaluation_strategy,
            "load_best_model_at_end works best when save_strategy matches evaluation_strategy.",
            errors,
        )
        if config.training.evaluation_strategy == "steps":
            _require(
                config.training.save_steps % config.training.eval_steps == 0,
                "save_steps should be a multiple of eval_steps when load_best_model_at_end is enabled.",
                errors,
            )
    if config.runtime.torch_compile and config.model.use_4bit:
        warnings.append("torch_compile with 4-bit quantization can be brittle; validate on a small run first.")
    if config.runtime.torch_compile and config.runtime.deepspeed_config:
        warnings.append("torch_compile with DeepSpeed should be validated carefully for your exact backend.")

    if errors:
        raise ConfigValidationError("\n".join(f"- {item}" for item in errors))
    return warnings


def describe_experiment_config(config: ExperimentConfig, *, warnings: list[str] | None = None) -> dict[str, Any]:
    model_mode = "full_precision"
    if config.model.use_4bit:
        model_mode = "4bit"
    elif config.model.use_8bit:
        model_mode = "8bit"
    elif config.model.use_full_precision:
        model_mode = "full_precision"
    return {
        "run": {
            "run_name": config.run_name,
            "profile_name": config.profile_name,
            "seed": config.seed,
            "config_path": config.config_path,
        },
        "model": {
            "name": config.model.name,
            "base_model_name": config.model.base_model_name,
            "tokenizer_name": config.model.tokenizer_name,
            "mode": model_mode,
            "adapter_method": config.adapter.method,
        },
        "data": {
            "train_file": config.data.train_file,
            "eval_file": config.data.eval_file,
            "test_file": config.data.test_file,
            "pack_manifest": config.data.pack_manifest,
            "max_length": config.data.max_length,
            "curriculum_learning": config.data.curriculum_learning,
            "sequential_curriculum": config.data.sequential_curriculum,
        },
        "training": {
            "artifacts_dir": config.training.artifacts_dir,
            "num_train_epochs": config.training.num_train_epochs,
            "max_steps": config.training.max_steps,
            "learning_rate": config.training.learning_rate,
            "batch_size": {
                "train": config.training.per_device_train_batch_size,
                "eval": config.training.per_device_eval_batch_size,
            },
            "gradient_accumulation_steps": config.training.gradient_accumulation_steps,
            "evaluation_strategy": config.training.evaluation_strategy,
            "save_strategy": config.training.save_strategy,
            "load_best_model_at_end": config.training.load_best_model_at_end,
        },
        "runtime": {
            "profile_name": config.runtime.profile_name,
            "accelerate_config": config.runtime.accelerate_config,
            "deepspeed_config": config.runtime.deepspeed_config,
            "validate_tokenization_samples": config.runtime.validate_tokenization_samples,
            "validate_model_load": config.runtime.validate_model_load,
            "torch_compile": config.runtime.torch_compile,
            "low_cpu_mem_usage": config.runtime.low_cpu_mem_usage,
        },
        "tracking": {
            "provider": config.tracking.provider,
            "project": config.tracking.project,
            "experiment_name": config.tracking.experiment_name,
            "tags": list(config.tracking.tags),
        },
        "packaging": {
            "publish_model_name": config.packaging.publish_model_name,
            "export_merged_model": config.packaging.export_merged_model,
            "publish_latest": config.packaging.publish_latest,
            "save_tokenizer": config.packaging.save_tokenizer,
        },
        "warnings": list(warnings or []),
    }


def load_experiment_config(path: str) -> ExperimentConfig:
    config_path = Path(path).resolve()
    raw = _load_yaml(config_path)
    merged = _apply_refs(config_path, raw)
    config = ExperimentConfig(
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
    validate_experiment_config(config)
    return config
