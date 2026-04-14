"""Shared math generation, model registry, and prompt utilities (boundary-safe across subsystems)."""

from __future__ import annotations

from ai_factory.core.math_stack.generation import MathGenerator
from ai_factory.core.math_stack.model_catalog import (
    list_model_catalog,
    normalize_model_record,
    summarize_model_catalog,
)
from ai_factory.core.math_stack.model_loader import (
    LoadedModel,
    MathModelRegistry,
    MathModelRuntime,
    ModelSpec,
    build_quant_config,
    load_registry_from_yaml,
    resolve_dtype,
)
from ai_factory.core.math_stack.parameters import GenerationParameters
from ai_factory.core.math_stack.prompts import (
    DEFAULT_PROMPT_PRESET_ID,
    PromptPreset,
    build_user_prompt,
    load_prompt_presets,
)

__all__ = [
    "DEFAULT_PROMPT_PRESET_ID",
    "GenerationParameters",
    "LoadedModel",
    "MathGenerator",
    "MathModelRegistry",
    "MathModelRuntime",
    "ModelSpec",
    "PromptPreset",
    "build_quant_config",
    "build_user_prompt",
    "list_model_catalog",
    "load_prompt_presets",
    "load_registry_from_yaml",
    "normalize_model_record",
    "resolve_dtype",
    "summarize_model_catalog",
]
