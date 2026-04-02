import pytest

from training.src.config import ConfigValidationError, load_experiment_config, validate_experiment_config


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
