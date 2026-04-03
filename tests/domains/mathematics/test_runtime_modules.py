"""Runtime tests for mathematics domain registries and utilities."""

from __future__ import annotations

from pathlib import Path

import pytest

from ai_factory.domains.mathematics.datasets import MathDatasetRegistry
from ai_factory.domains.mathematics.evaluation import MathEvaluationSuite
from ai_factory.domains.mathematics.training import MathTrainingProfiles
from ai_factory.domains.utils import get_domain_info, list_available_domains


def test_math_dataset_registry_lists_and_filters() -> None:
    registry = MathDatasetRegistry(Path("."))
    names = registry.list_datasets()

    assert set(names) == {"derivatives", "integrals", "limits_series", "olympiad_reasoning"}

    calculus = registry.get_datasets_by_subdomain("calculus")
    olympiad = registry.get_datasets_by_subdomain("olympiad")
    assert len(calculus) == 3
    assert len(olympiad) == 1
    assert olympiad[0].name == "olympiad_reasoning"


def test_math_dataset_registry_get_dataset_errors() -> None:
    registry = MathDatasetRegistry(Path("."))
    derivatives = registry.get_dataset("derivatives")

    assert derivatives.domain == "mathematics"
    assert derivatives.subdomain == "calculus"

    with pytest.raises(ValueError, match="not found"):
        registry.get_dataset("does-not-exist")


def test_math_evaluation_suite_accessors() -> None:
    suite = MathEvaluationSuite(Path("."))

    metrics = suite.list_metrics()
    benchmarks = suite.list_benchmarks()
    assert "mathematical_accuracy" in metrics
    assert "mathematics_benchmark" in benchmarks

    metric = suite.get_metric("verification_score")
    benchmark = suite.get_benchmark("calculus_specialist")
    assert metric.type == "score"
    assert benchmark.domain == "mathematics"
    assert benchmark.subdomain == "calculus"

    with pytest.raises(ValueError, match="not found"):
        suite.get_metric("unknown-metric")
    with pytest.raises(ValueError, match="not found"):
        suite.get_benchmark("unknown-benchmark")


def test_math_training_profiles_accessors() -> None:
    profiles = MathTrainingProfiles(Path("."))
    names = profiles.list_profiles()
    assert "baseline_qlora" in names
    assert "olympiad_reasoning" in names

    profile = profiles.get_profile("baseline_qlora")
    assert profile.training_method == "qlora"
    assert profile.domain == "mathematics"

    qlora_profiles = profiles.get_profiles_by_method("qlora")
    methods = {entry.training_method for entry in qlora_profiles}
    assert methods == {"qlora"}

    with pytest.raises(ValueError, match="not found"):
        profiles.get_profile("missing-profile")


def test_domain_utils_return_expected_metadata() -> None:
    available = list_available_domains()
    assert any(config.name == "mathematics" for config in available)

    info = get_domain_info("mathematics")
    assert info["name"] == "mathematics"
    assert "calculus" in info["subdomains"]
    assert "mathematics_benchmark" in info["benchmarks"]

    with pytest.raises(ValueError, match="Unknown domain"):
        get_domain_info("unknown")
