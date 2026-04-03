from __future__ import annotations

from pathlib import Path

import pytest

from ai_factory.domains.mathematics.datasets import MathDatasetRegistry
from ai_factory.domains.mathematics.evaluation import MathEvaluationSuite
from ai_factory.domains.mathematics.training import MathTrainingProfiles
from ai_factory.domains.utils import get_domain_info, list_available_domains


def test_math_dataset_registry_lists_and_filters(tmp_path: Path) -> None:
    registry = MathDatasetRegistry(tmp_path)
    datasets = registry.list_datasets()
    assert "derivatives" in datasets
    assert "integrals" in datasets
    assert registry.get_dataset("derivatives").domain == "mathematics"
    assert registry.get_datasets_by_subdomain("calculus")

    with pytest.raises(ValueError, match="not found"):
        registry.get_dataset("unknown")


def test_math_training_profiles_getters(tmp_path: Path) -> None:
    profiles = MathTrainingProfiles(tmp_path)
    names = profiles.list_profiles()
    assert "baseline_qlora" in names
    assert profiles.get_profile("baseline_qlora").training_method == "qlora"
    assert profiles.get_profiles_by_method("qlora")

    with pytest.raises(ValueError, match="not found"):
        profiles.get_profile("unknown")


def test_math_evaluation_suite_getters(tmp_path: Path) -> None:
    suite = MathEvaluationSuite(tmp_path)
    assert "mathematical_accuracy" in suite.list_metrics()
    assert "mathematics_benchmark" in suite.list_benchmarks()
    assert suite.get_metric("mathematical_accuracy").domain == "mathematics"
    assert suite.get_benchmark("calculus_specialist").subdomain == "calculus"

    with pytest.raises(ValueError, match="not found"):
        suite.get_metric("unknown")
    with pytest.raises(ValueError, match="not found"):
        suite.get_benchmark("unknown")


def test_domain_utils_expose_math_domain() -> None:
    domains = list_available_domains()
    assert any(domain.name == "mathematics" for domain in domains)
    info = get_domain_info("mathematics")
    assert info["name"] == "mathematics"
    assert "subdomains" in info

    with pytest.raises(ValueError, match="Unknown domain"):
        get_domain_info("unknown")
