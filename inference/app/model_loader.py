"""Backward-compatible facade for `ai_factory.core.math_stack.model_loader`."""

from __future__ import annotations

from ai_factory.core.math_stack.model_loader import (
    LoadedModel,
    MathModelRegistry,
    MathModelRuntime,
    ModelSpec,
    build_quant_config,
    load_registry_from_yaml,
    resolve_dtype,
)

__all__ = [
    "LoadedModel",
    "MathModelRegistry",
    "MathModelRuntime",
    "ModelSpec",
    "build_quant_config",
    "load_registry_from_yaml",
    "resolve_dtype",
]
