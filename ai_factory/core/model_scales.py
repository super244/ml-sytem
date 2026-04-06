from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelScaleSpec:
    scale: str
    parameter_count: int
    parameter_size_b: float
    tier: str
    runtime_profile: str
    recommended_quantization: str
    recommended_train_batch_size: int
    recommended_gradient_accumulation_steps: int
    recommended_context_length: int
    preferred_gpu_count: int

    @property
    def label(self) -> str:
        return self.scale.upper()


SUPPORTED_MODEL_SCALES: tuple[ModelScaleSpec, ...] = (
    ModelScaleSpec(
        scale="1b",
        parameter_count=1_000_000_000,
        parameter_size_b=1.0,
        tier="local",
        runtime_profile="hybrid_local",
        recommended_quantization="4bit",
        recommended_train_batch_size=4,
        recommended_gradient_accumulation_steps=4,
        recommended_context_length=4096,
        preferred_gpu_count=1,
    ),
    ModelScaleSpec(
        scale="2b",
        parameter_count=2_000_000_000,
        parameter_size_b=2.0,
        tier="local",
        runtime_profile="hybrid_local",
        recommended_quantization="4bit",
        recommended_train_batch_size=2,
        recommended_gradient_accumulation_steps=8,
        recommended_context_length=4096,
        preferred_gpu_count=1,
    ),
    ModelScaleSpec(
        scale="4b",
        parameter_count=4_000_000_000,
        parameter_size_b=4.0,
        tier="local",
        runtime_profile="hybrid_local",
        recommended_quantization="4bit",
        recommended_train_batch_size=1,
        recommended_gradient_accumulation_steps=16,
        recommended_context_length=4096,
        preferred_gpu_count=1,
    ),
    ModelScaleSpec(
        scale="7b",
        parameter_count=7_000_000_000,
        parameter_size_b=7.0,
        tier="scaleup",
        runtime_profile="hybrid_scaleup",
        recommended_quantization="4bit",
        recommended_train_batch_size=1,
        recommended_gradient_accumulation_steps=16,
        recommended_context_length=8192,
        preferred_gpu_count=2,
    ),
    ModelScaleSpec(
        scale="9b",
        parameter_count=9_000_000_000,
        parameter_size_b=9.0,
        tier="scaleup",
        runtime_profile="hybrid_scaleup",
        recommended_quantization="4bit",
        recommended_train_batch_size=1,
        recommended_gradient_accumulation_steps=16,
        recommended_context_length=8192,
        preferred_gpu_count=2,
    ),
    ModelScaleSpec(
        scale="12b",
        parameter_count=12_000_000_000,
        parameter_size_b=12.0,
        tier="scaleup",
        runtime_profile="hybrid_scaleup",
        recommended_quantization="4bit",
        recommended_train_batch_size=1,
        recommended_gradient_accumulation_steps=24,
        recommended_context_length=8192,
        preferred_gpu_count=2,
    ),
    ModelScaleSpec(
        scale="14b",
        parameter_count=14_000_000_000,
        parameter_size_b=14.0,
        tier="scaleup",
        runtime_profile="ultimate_cuda",
        recommended_quantization="none",
        recommended_train_batch_size=1,
        recommended_gradient_accumulation_steps=16,
        recommended_context_length=8192,
        preferred_gpu_count=2,
    ),
    ModelScaleSpec(
        scale="20b",
        parameter_count=20_000_000_000,
        parameter_size_b=20.0,
        tier="distributed",
        runtime_profile="hybrid_scaleup",
        recommended_quantization="4bit",
        recommended_train_batch_size=1,
        recommended_gradient_accumulation_steps=32,
        recommended_context_length=8192,
        preferred_gpu_count=4,
    ),
    ModelScaleSpec(
        scale="27b",
        parameter_count=27_000_000_000,
        parameter_size_b=27.0,
        tier="distributed",
        runtime_profile="hybrid_scaleup",
        recommended_quantization="4bit",
        recommended_train_batch_size=1,
        recommended_gradient_accumulation_steps=48,
        recommended_context_length=8192,
        preferred_gpu_count=4,
    ),
    ModelScaleSpec(
        scale="30b",
        parameter_count=30_000_000_000,
        parameter_size_b=30.0,
        tier="distributed",
        runtime_profile="hybrid_scaleup",
        recommended_quantization="4bit",
        recommended_train_batch_size=1,
        recommended_gradient_accumulation_steps=64,
        recommended_context_length=8192,
        preferred_gpu_count=8,
    ),
    ModelScaleSpec(
        scale="70b",
        parameter_count=70_000_000_000,
        parameter_size_b=70.0,
        tier="frontier",
        runtime_profile="hybrid_scaleup",
        recommended_quantization="4bit",
        recommended_train_batch_size=1,
        recommended_gradient_accumulation_steps=128,
        recommended_context_length=16384,
        preferred_gpu_count=8,
    ),
    ModelScaleSpec(
        scale="120b",
        parameter_count=120_000_000_000,
        parameter_size_b=120.0,
        tier="frontier",
        runtime_profile="hybrid_scaleup",
        recommended_quantization="4bit",
        recommended_train_batch_size=1,
        recommended_gradient_accumulation_steps=192,
        recommended_context_length=16384,
        preferred_gpu_count=16,
    ),
)

DEFAULT_MODEL_SCALE = "2b"
DEFAULT_MODEL_FAMILY = "qwen2"

_SUPPORTED_SCALE_INDEX = {spec.scale: spec for spec in SUPPORTED_MODEL_SCALES}


def normalize_model_scale_identifier(value: str | int | float) -> str:
    if isinstance(value, int):
        if value >= 1_000_000_000:
            return f"{int(round(value / 1_000_000_000))}b"
        return f"{value}m"
    if isinstance(value, float):
        normalized = f"{value:.3f}".rstrip("0").rstrip(".")
    else:
        normalized = str(value).strip().lower().replace("_", "")
    if not normalized:
        raise ValueError("Model scale cannot be empty.")
    suffix = normalized[-1]
    if suffix == "b":
        return f"{float(normalized[:-1]):g}b".replace(".0b", "b")
    if suffix == "m":
        return f"{float(normalized[:-1]):g}m".replace(".0m", "m")
    numeric = float(normalized)
    if numeric >= 1_000_000_000:
        return f"{numeric / 1_000_000_000:g}b".replace(".0b", "b")
    if numeric >= 1_000_000:
        return f"{numeric / 1_000_000:g}m".replace(".0m", "m")
    return f"{numeric:g}"


def list_model_scale_specs() -> list[ModelScaleSpec]:
    return list(SUPPORTED_MODEL_SCALES)


def is_supported_model_scale(value: str | int | float) -> bool:
    return normalize_model_scale_identifier(value) in _SUPPORTED_SCALE_INDEX


def get_model_scale_spec(value: str | int | float) -> ModelScaleSpec:
    normalized = normalize_model_scale_identifier(value)
    try:
        return _SUPPORTED_SCALE_INDEX[normalized]
    except KeyError as exc:
        supported = ", ".join(spec.scale for spec in SUPPORTED_MODEL_SCALES)
        raise ValueError(f"Unsupported model scale {value!r}. Supported scales: {supported}.") from exc


def default_foundation_model_ref(
    scale: str | int | float = DEFAULT_MODEL_SCALE,
    *,
    family: str = DEFAULT_MODEL_FAMILY,
) -> str:
    normalized = normalize_model_scale_identifier(scale)
    return f"artifacts/foundation/{family}-{normalized}"


def default_tokenizer_artifact(
    scale: str | int | float = DEFAULT_MODEL_SCALE,
    *,
    family: str = "qwen2_math",
) -> str:
    normalized = normalize_model_scale_identifier(scale)
    return f"artifacts/tokenizers/{family}_{normalized}"


DEFAULT_FOUNDATION_MODEL = default_foundation_model_ref()
