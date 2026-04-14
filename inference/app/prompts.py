"""Backward-compatible facade for `ai_factory.core.math_stack.prompts`."""

from __future__ import annotations

from ai_factory.core.math_stack.prompts import (
    DEFAULT_PROMPT_PRESET_ID,
    PromptPreset,
    build_user_prompt,
    load_prompt_presets,
)

__all__ = [
    "DEFAULT_PROMPT_PRESET_ID",
    "PromptPreset",
    "build_user_prompt",
    "load_prompt_presets",
]
