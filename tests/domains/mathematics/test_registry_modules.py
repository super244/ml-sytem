from __future__ import annotations

from pathlib import Path

import pytest

from ai_factory.domains.mathematics.datasets import MathDatasetRegistry
from ai_factory.domains.mathematics.evaluation import MathEvaluationSuite
from ai_factory.domains.mathematics.training import MathTrainingProfiles


def test_math_dataset_registry_lists_and_filters() -> None:
    registry = MathDatasetRegistry(Path("."))
    names = registry.list_datasets()
    assert {"derivatives", "integrals", "limits_series", "olympiad_reasoning"} <= set(names)

    calculus = registry.get_datasets_by_subdomain("calculus")
    assert len(calculus) >= 3
    assert all(item.subdomain == "calculus" for item in calculus)

    olympiad = registry.get_dataset("olympiad_reasoning")
    assert olympiad.subdomain == "olympiad"

    with pytest.raises(ValueError, match="not found"):
        registry.get_dataset("missing")


def test_math_evaluation_suite_lists_and_retrieves_specs() -> None:
    suite = MathEvaluationSuite(Path("."))

    metrics = suite.list_metrics()
    benchmarks = suite.list_benchmarks()
    assert "mathematical_accuracy" in metrics
    assert "mathematics_benchmark" in benchmarks

    metric = suite.get_metric("verification_score")
    assert metric.type == "score"

    benchmark = suite.get_benchmark("calculus_specialist")
    assert benchmark.subdomain == "calculus"
    assert "calculus_fluency" in benchmark.metrics

    with pytest.raises(ValueError, match="not found"):
        suite.get_metric("unknown_metric")
    with pytest.raises(ValueError, match="not found"):
        suite.get_benchmark("unknown_benchmark")


def test_math_training_profiles_lists_and_filters() -> None:
    profiles = MathTrainingProfiles(Path("."))
    names = profiles.list_profiles()
    assert "baseline_qlora" in names
    assert "olympiad_reasoning" in names

    qlora_profiles = profiles.get_profiles_by_method("qlora")
    assert qlora_profiles
    assert all(item.training_method == "qlora" for item in qlora_profiles)

    olympiad = profiles.get_profile("olympiad_reasoning")
    assert olympiad.training_method == "full_finetune"

    with pytest.raises(ValueError, match="not found"):
        profiles.get_profile("unknown_profile")
