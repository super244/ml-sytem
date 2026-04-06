"""Test mathematics domain configuration."""

from ai_factory.domains.mathematics.config import MathDomainConfig


def test_math_domain_config_defaults():
    """Test default configuration values."""
    config = MathDomainConfig()
    assert config.name == "mathematics"
    assert config.version == "1.0.0"
    assert "calculus" in config.subdomains
    assert "algebra" in config.subdomains
    assert "derivatives" in config.dataset_families
    assert "integrals" in config.dataset_families


def test_math_domain_config_subdomains():
    """Test subdomains are properly configured."""
    config = MathDomainConfig()
    expected_subdomains = ["calculus", "algebra", "geometry", "olympiad", "statistics", "linear_algebra"]
    assert config.subdomains == expected_subdomains


def test_math_domain_config_dataset_families():
    """Test dataset families are properly configured."""
    config = MathDomainConfig()
    expected_families = [
        "derivatives",
        "integrals",
        "limits_series",
        "multivariable",
        "odes_optimization",
        "olympiad_reasoning",
    ]
    assert config.dataset_families == expected_families


def test_math_domain_config_default_models():
    """Test default models are properly configured."""
    config = MathDomainConfig()
    assert len(config.default_models) >= 1
    assert any(model == "Qwen/Qwen2.5-Math-1.5B-Instruct" for model in config.default_models)
    assert any(model.endswith("artifacts/foundation/qwen2-12b") for model in config.default_models)


def test_math_domain_config_benchmarks():
    """Test benchmarks are properly configured."""
    config = MathDomainConfig()
    assert len(config.benchmarks) >= 1
    assert "mathematics_benchmark" in config.benchmarks


def test_math_domain_config_custom_values():
    """Test custom configuration values."""
    config = MathDomainConfig(
        subdomains=["calculus", "algebra"],
        dataset_families=["derivatives"],
    )
    assert config.subdomains == ["calculus", "algebra"]
    assert config.dataset_families == ["derivatives"]


def test_math_domain_config_serialization():
    """Test configuration can be serialized."""
    config = MathDomainConfig()
    data = config.model_dump()
    assert data["name"] == "mathematics"
    assert isinstance(data["subdomains"], list)
    assert isinstance(data["dataset_families"], list)


def test_math_domain_config_metrics_default_empty():
    """Test that metrics default to empty list."""
    config = MathDomainConfig()
    assert config.metrics == []
