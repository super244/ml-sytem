"""Mathematics domain dataset registry."""

from pathlib import Path

from ai_factory.core.schemas import DatasetSpec


class MathDatasetRegistry:
    """Registry for mathematics-specific datasets."""

    def __init__(self, repo_root: Path):
        self.repo_root = repo_root
        self._datasets = self._load_math_datasets()

    def _load_math_datasets(self) -> dict[str, DatasetSpec]:
        """Load mathematics dataset specifications."""
        datasets = {}

        # Synthetic calculus datasets
        datasets.update(
            {
                "derivatives": DatasetSpec(
                    name="derivatives",
                    description="Synthetic calculus derivative problems",
                    path="data/custom/custom_derivative_mastery.jsonl",
                    domain="mathematics",
                    subdomain="calculus",
                    difficulty_range=["easy", "medium", "hard"],
                    size=10000,
                    format="jsonl",
                ),
                "integrals": DatasetSpec(
                    name="integrals",
                    description="Synthetic calculus integration problems",
                    path="data/custom/custom_integral_mastery.jsonl",
                    domain="mathematics",
                    subdomain="calculus",
                    difficulty_range=["easy", "medium", "hard"],
                    size=10000,
                    format="jsonl",
                ),
                "limits_series": DatasetSpec(
                    name="limits_series",
                    description="Synthetic limits and series problems",
                    path="data/custom/custom_limits_series.jsonl",
                    domain="mathematics",
                    subdomain="calculus",
                    difficulty_range=["easy", "medium", "hard"],
                    size=8000,
                    format="jsonl",
                ),
                "olympiad_reasoning": DatasetSpec(
                    name="olympiad_reasoning",
                    description="Olympiad-level mathematical reasoning problems",
                    path="data/custom/custom_olympiad_reasoning.jsonl",
                    domain="mathematics",
                    subdomain="olympiad",
                    difficulty_range=["medium", "hard", "expert"],
                    size=5000,
                    format="jsonl",
                ),
            }
        )

        return datasets

    def list_datasets(self) -> list[str]:
        """List all available mathematics datasets."""
        return list(self._datasets.keys())

    def get_dataset(self, name: str) -> DatasetSpec:
        """Get a specific mathematics dataset."""
        if name not in self._datasets:
            raise ValueError(f"Mathematics dataset '{name}' not found")
        return self._datasets[name]

    def get_datasets_by_subdomain(self, subdomain: str) -> list[DatasetSpec]:
        """Get datasets for a specific mathematics subdomain."""
        return [dataset for dataset in self._datasets.values() if dataset.subdomain == subdomain]
