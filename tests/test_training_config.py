import subprocess
import sys
from pathlib import Path

import pytest

from training.src.config import ConfigValidationError, load_experiment_config, validate_experiment_config
from training.src.modeling import resolve_attention_implementation
from training.src.validation import build_validation_data_config


def test_profile_config_merges_components() -> None:
    config = load_experiment_config("training/configs/profiles/failure_aware.yaml")
    assert config.profile_name == "failure_aware"
    assert config.model.base_model_name.endswith("1.5B-Instruct")
    assert config.data.failure_replay_boost > 1.0
    assert config.packaging.publish_model_name == "atlas-math-failure-aware"


def test_validation_rejects_conflicting_training_flags() -> None:
    config = load_experiment_config("training/configs/profiles/baseline_qlora.yaml")
    config.training.fp16 = True

    with pytest.raises(ConfigValidationError):
        validate_experiment_config(config)


def test_build_validation_data_config_preserves_model_type() -> None:
    config = load_experiment_config("training/configs/profiles/baseline_qlora.yaml")
    config.data.max_train_samples = 8
    config.data.max_eval_samples = 4

    validation_config = build_validation_data_config(config)

    assert validation_config.max_train_samples == 8
    assert validation_config.max_eval_samples == 4
    assert validation_config.model_dump() != {}


def test_resolve_attention_implementation_falls_back_when_flash_attention_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_experiment_config("training/configs/profiles/baseline_qlora.yaml")
    monkeypatch.setattr("training.src.modeling.is_flash_attn_2_available", lambda: False)

    assert resolve_attention_implementation(config) is None


def test_resolve_attention_implementation_preserves_flash_attention_when_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_experiment_config("training/configs/profiles/baseline_qlora.yaml")
    monkeypatch.setattr("training.src.modeling.is_flash_attn_2_available", lambda: True)

    assert resolve_attention_implementation(config) == "flash_attention_2"


def test_build_training_arguments_uses_installed_transformers_api(tmp_path: Path) -> None:
    script = f"""
from ai_factory.core.artifacts import prepare_run_layout
from training.train import build_training_arguments
from training.src.config import load_experiment_config

config = load_experiment_config("training/configs/profiles/baseline_qlora.yaml")
layout = prepare_run_layout(r"{tmp_path}", "unit-training-arguments")
arguments = build_training_arguments(config, layout)
print(arguments.eval_strategy)
"""
    completed = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=True)

    assert "steps" in completed.stdout.lower()


def test_all_training_profiles_load_and_validate() -> None:
    profile_dir = Path("training/configs/profiles")

    for profile_path in sorted(profile_dir.glob("*.yaml")):
        config = load_experiment_config(str(profile_path))
        warnings = validate_experiment_config(config)

        assert isinstance(warnings, list), profile_path.name


def test_standard_trainer_module_imports_without_trl_dependency() -> None:
    script = """
from training.src.trainer import build_ultimate_trainer
print(callable(build_ultimate_trainer))
"""
    completed = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=True)

    assert "True" in completed.stdout
