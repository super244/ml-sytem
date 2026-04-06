from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ai_factory.core.model_scales import list_model_scale_specs


def parse_parameter_count(value: str | int | float) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    normalized = str(value).strip().lower().replace("_", "")
    multipliers = {"k": 1_000, "m": 1_000_000, "b": 1_000_000_000}
    suffix = normalized[-1]
    if suffix in multipliers:
        return int(float(normalized[:-1]) * multipliers[suffix])
    return int(float(normalized))


def format_parameter_count(value: int) -> str:
    if value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.2f}B"
    if value >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if value >= 1_000:
        return f"{value / 1_000:.2f}K"
    return str(value)


def _round_to_multiple(value: float, multiple: int) -> int:
    return max(multiple, int(round(value / multiple)) * multiple)


def estimate_qwen2_dense_parameters(
    *,
    vocab_size: int,
    hidden_size: int,
    intermediate_size: int,
    num_hidden_layers: int,
    num_attention_heads: int,
    num_key_value_heads: int,
    tie_word_embeddings: bool = False,
) -> int:
    head_dim = hidden_size // num_attention_heads
    kv_dim = num_key_value_heads * head_dim
    embedding_params = vocab_size * hidden_size
    lm_head_params = 0 if tie_word_embeddings else vocab_size * hidden_size
    attention_params = (hidden_size * hidden_size) * 2 + (hidden_size * kv_dim) * 2
    mlp_params = hidden_size * intermediate_size * 3
    norm_params = hidden_size * 2
    final_norm_params = hidden_size
    return (
        embedding_params
        + lm_head_params
        + num_hidden_layers * (attention_params + mlp_params + norm_params)
        + final_norm_params
    )


@dataclass(frozen=True)
class ArchitectureCandidate:
    architecture: dict[str, Any]
    estimated_parameters: int
    relative_error: float


@dataclass(frozen=True)
class SearchPreference:
    preferred_layers: int
    preferred_hidden: int
    layer_min: int
    layer_max: int
    layer_step: int
    hidden_min: int
    hidden_max: int
    hidden_step: int = 128


_LARGE_SCALE_PREFERENCES: dict[int, SearchPreference] = {
    20_000_000_000: SearchPreference(
        preferred_layers=64,
        preferred_hidden=5632,
        layer_min=56,
        layer_max=72,
        layer_step=2,
        hidden_min=4096,
        hidden_max=8192,
    ),
    27_000_000_000: SearchPreference(
        preferred_layers=72,
        preferred_hidden=6144,
        layer_min=60,
        layer_max=84,
        layer_step=2,
        hidden_min=4608,
        hidden_max=9216,
    ),
    30_000_000_000: SearchPreference(
        preferred_layers=80,
        preferred_hidden=6656,
        layer_min=68,
        layer_max=92,
        layer_step=2,
        hidden_min=4608,
        hidden_max=9728,
    ),
    70_000_000_000: SearchPreference(
        preferred_layers=96,
        preferred_hidden=8960,
        layer_min=84,
        layer_max=108,
        layer_step=4,
        hidden_min=7168,
        hidden_max=12288,
    ),
    120_000_000_000: SearchPreference(
        preferred_layers=128,
        preferred_hidden=10240,
        layer_min=112,
        layer_max=140,
        layer_step=4,
        hidden_min=8192,
        hidden_max=14080,
    ),
}


def derive_qwen2_dense_architecture(
    *,
    target_parameters: int,
    vocab_size: int,
    max_position_embeddings: int = 4096,
    rope_theta: float = 1_000_000.0,
    tie_word_embeddings: bool = False,
    hidden_size_min: int = 1024,
    hidden_size_max: int = 6144,
    hidden_size_step: int = 128,
    layer_min: int = 12,
    layer_max: int = 48,
    layer_step: int = 2,
    head_dim: int = 128,
) -> ArchitectureCandidate:
    if target_parameters > 12_000_000_000:
        return _derive_qwen2_dense_architecture_large(
            target_parameters=target_parameters,
            vocab_size=vocab_size,
            max_position_embeddings=max_position_embeddings,
            rope_theta=rope_theta,
            tie_word_embeddings=tie_word_embeddings,
        )
    return _derive_qwen2_dense_architecture_exhaustive(
        target_parameters=target_parameters,
        vocab_size=vocab_size,
        max_position_embeddings=max_position_embeddings,
        rope_theta=rope_theta,
        tie_word_embeddings=tie_word_embeddings,
        hidden_size_min=hidden_size_min,
        hidden_size_max=hidden_size_max,
        hidden_size_step=hidden_size_step,
        layer_min=layer_min,
        layer_max=layer_max,
        layer_step=layer_step,
        head_dim=head_dim,
    )


def _derive_qwen2_dense_architecture_exhaustive(
    *,
    target_parameters: int,
    vocab_size: int,
    max_position_embeddings: int,
    rope_theta: float,
    tie_word_embeddings: bool,
    hidden_size_min: int,
    hidden_size_max: int,
    hidden_size_step: int,
    layer_min: int,
    layer_max: int,
    layer_step: int,
    head_dim: int,
) -> ArchitectureCandidate:
    candidates: list[ArchitectureCandidate] = []
    intermediate_ratios = (2.5, 2.6875, 2.75, 3.0, 3.5)
    kv_divisors = (2, 4, 8)

    for hidden_size in range(hidden_size_min, hidden_size_max + hidden_size_step, hidden_size_step):
        if hidden_size % head_dim != 0:
            continue
        num_attention_heads = hidden_size // head_dim
        if num_attention_heads < 8:
            continue
        for num_hidden_layers in range(layer_min, layer_max + layer_step, layer_step):
            for kv_divisor in kv_divisors:
                if num_attention_heads % kv_divisor != 0:
                    continue
                num_key_value_heads = max(1, num_attention_heads // kv_divisor)
                for ratio in intermediate_ratios:
                    intermediate_size = _round_to_multiple(hidden_size * ratio, 256)
                    estimate = estimate_qwen2_dense_parameters(
                        vocab_size=vocab_size,
                        hidden_size=hidden_size,
                        intermediate_size=intermediate_size,
                        num_hidden_layers=num_hidden_layers,
                        num_attention_heads=num_attention_heads,
                        num_key_value_heads=num_key_value_heads,
                        tie_word_embeddings=tie_word_embeddings,
                    )
                    relative_error = abs(estimate - target_parameters) / max(target_parameters, 1)
                    architecture = {
                        "vocab_size": vocab_size,
                        "hidden_size": hidden_size,
                        "intermediate_size": intermediate_size,
                        "num_hidden_layers": num_hidden_layers,
                        "num_attention_heads": num_attention_heads,
                        "num_key_value_heads": num_key_value_heads,
                        "max_position_embeddings": max_position_embeddings,
                        "rope_theta": rope_theta,
                        "tie_word_embeddings": tie_word_embeddings,
                    }
                    candidates.append(
                        ArchitectureCandidate(
                            architecture=architecture,
                            estimated_parameters=estimate,
                            relative_error=relative_error,
                        )
                    )

    if not candidates:
        raise ValueError("No qwen2 architecture candidates were generated for the requested parameter target.")

    return min(
        candidates,
        key=lambda candidate: (
            abs(candidate.architecture["num_hidden_layers"] - 24),
            round(candidate.relative_error, 2),
            abs(candidate.architecture["hidden_size"] - 2560),
            candidate.relative_error,
        ),
    )


def _closest_large_scale_preference(target_parameters: int) -> SearchPreference:
    preferences = sorted(_LARGE_SCALE_PREFERENCES.items(), key=lambda item: abs(item[0] - target_parameters))
    if preferences:
        return preferences[0][1]
    largest_scale = max(spec.parameter_count for spec in list_model_scale_specs())
    return _LARGE_SCALE_PREFERENCES[largest_scale]


def _derive_qwen2_dense_architecture_large(
    *,
    target_parameters: int,
    vocab_size: int,
    max_position_embeddings: int,
    rope_theta: float,
    tie_word_embeddings: bool,
) -> ArchitectureCandidate:
    preference = _closest_large_scale_preference(target_parameters)
    candidates: list[ArchitectureCandidate] = []
    intermediate_ratios = (2.5, 2.6875, 2.75, 3.0, 3.5)
    kv_divisors = (2, 4, 8)

    for hidden_size in range(
        preference.hidden_min, preference.hidden_max + preference.hidden_step, preference.hidden_step
    ):
        num_attention_heads = hidden_size // 128
        if hidden_size % 128 != 0 or num_attention_heads < 8:
            continue
        for num_hidden_layers in range(
            preference.layer_min, preference.layer_max + preference.layer_step, preference.layer_step
        ):
            for kv_divisor in kv_divisors:
                if num_attention_heads % kv_divisor != 0:
                    continue
                num_key_value_heads = max(1, num_attention_heads // kv_divisor)
                for ratio in intermediate_ratios:
                    intermediate_size = _round_to_multiple(hidden_size * ratio, 256)
                    estimate = estimate_qwen2_dense_parameters(
                        vocab_size=vocab_size,
                        hidden_size=hidden_size,
                        intermediate_size=intermediate_size,
                        num_hidden_layers=num_hidden_layers,
                        num_attention_heads=num_attention_heads,
                        num_key_value_heads=num_key_value_heads,
                        tie_word_embeddings=tie_word_embeddings,
                    )
                    relative_error = abs(estimate - target_parameters) / max(target_parameters, 1)
                    architecture = {
                        "vocab_size": vocab_size,
                        "hidden_size": hidden_size,
                        "intermediate_size": intermediate_size,
                        "num_hidden_layers": num_hidden_layers,
                        "num_attention_heads": num_attention_heads,
                        "num_key_value_heads": num_key_value_heads,
                        "max_position_embeddings": max_position_embeddings,
                        "rope_theta": rope_theta,
                        "tie_word_embeddings": tie_word_embeddings,
                    }
                    candidates.append(
                        ArchitectureCandidate(
                            architecture=architecture,
                            estimated_parameters=estimate,
                            relative_error=relative_error,
                        )
                    )

    if not candidates:
        raise ValueError("No qwen2 architecture candidates were generated for the requested large-scale target.")

    return min(
        candidates,
        key=lambda candidate: (
            round(candidate.relative_error, 4),
            abs(candidate.architecture["num_hidden_layers"] - preference.preferred_layers),
            abs(candidate.architecture["hidden_size"] - preference.preferred_hidden),
            candidate.relative_error,
        ),
    )


def resolve_scratch_architecture(
    *,
    model_type: str,
    architecture_overrides: dict[str, Any] | None = None,
    target_parameters: str | int | float | None = None,
) -> tuple[dict[str, Any], int | None]:
    overrides = dict(architecture_overrides or {})
    if not target_parameters:
        return overrides, None
    target = parse_parameter_count(target_parameters)
    normalized_model_type = model_type.strip().lower()
    if normalized_model_type != "qwen2":
        raise ValueError(
            f"Automatic scratch scaling is currently only supported for model_type='qwen2', got {model_type!r}."
        )
    candidate = derive_qwen2_dense_architecture(
        target_parameters=target,
        vocab_size=int(overrides.get("vocab_size", 50257)),
        max_position_embeddings=int(overrides.get("max_position_embeddings", 4096)),
        rope_theta=float(overrides.get("rope_theta", 1_000_000.0)),
        tie_word_embeddings=bool(overrides.get("tie_word_embeddings", False)),
    )
    resolved = {**candidate.architecture, **overrides}
    estimate = estimate_qwen2_dense_parameters(
        vocab_size=int(resolved["vocab_size"]),
        hidden_size=int(resolved["hidden_size"]),
        intermediate_size=int(resolved["intermediate_size"]),
        num_hidden_layers=int(resolved["num_hidden_layers"]),
        num_attention_heads=int(resolved["num_attention_heads"]),
        num_key_value_heads=int(resolved["num_key_value_heads"]),
        tie_word_embeddings=bool(resolved.get("tie_word_embeddings", False)),
    )
    return resolved, estimate
