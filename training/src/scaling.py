from __future__ import annotations

from dataclasses import dataclass
from typing import Any


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
    return embedding_params + lm_head_params + num_hidden_layers * (
        attention_params + mlp_params + norm_params
    ) + final_norm_params


@dataclass(frozen=True)
class ArchitectureCandidate:
    architecture: dict[str, Any]
    estimated_parameters: int
    relative_error: float


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
        raise ValueError(f"Automatic scratch scaling is currently only supported for model_type='qwen2', got {model_type!r}.")
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
