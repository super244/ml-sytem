from __future__ import annotations

from pathlib import Path

import pytest

from evaluation.evaluate import validate_model_availability
from inference.app.model_loader import MathModelRegistry, ModelSpec


def test_validate_model_availability_raises_for_missing_adapter_path(tmp_path: Path) -> None:
    registry = MathModelRegistry(
        {
            "finetuned": ModelSpec(
                name="finetuned",
                base_model="Qwen/Qwen2.5-Math-1.5B-Instruct",
                adapter_path=str(tmp_path / "missing-adapter"),
            )
        }
    )

    with pytest.raises(FileNotFoundError, match="Model artifacts required for evaluation are missing"):
        validate_model_availability(registry, ["finetuned"])


def test_validate_model_availability_accepts_existing_adapter_path(tmp_path: Path) -> None:
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir(parents=True)
    registry = MathModelRegistry(
        {
            "finetuned": ModelSpec(
                name="finetuned",
                base_model="Qwen/Qwen2.5-Math-1.5B-Instruct",
                adapter_path=str(adapter_dir),
            )
        }
    )

    validate_model_availability(registry, ["finetuned"])
