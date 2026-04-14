"""Backward-compatible facade for `ai_factory.core.math_stack.model_catalog`."""

from __future__ import annotations

from ai_factory.core.math_stack.model_catalog import (
    list_model_catalog,
    normalize_model_record,
    summarize_model_catalog,
)

__all__ = ["list_model_catalog", "normalize_model_record", "summarize_model_catalog"]
