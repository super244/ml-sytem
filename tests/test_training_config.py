from training.src.config import load_experiment_config


def test_profile_config_merges_components():
    config = load_experiment_config("training/configs/profiles/failure_aware.yaml")
    assert config.profile_name == "failure_aware"
    assert config.model.base_model_name.endswith("1.5B-Instruct")
    assert config.data.failure_replay_boost > 1.0
    assert config.packaging.publish_model_name == "atlas-math-failure-aware"
